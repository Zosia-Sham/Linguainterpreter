"""
Read-only monitoring tools for the ReAct execution watcher.

The execution watcher decides whether a running subprocess is making real
progress or should be killed. It needs to inspect:

- The subprocess itself (CPU %, RSS, thread count, state) via psutil.
- The host (overall CPU, memory pressure, disk I/O) via psutil.
- Optional GPU utilization via ``nvidia-smi`` (no heavy Python bindings).
- The stdout/stderr produced so far (post-warning-filter, trimmed by the agent).
- The code that's running and the task description it's supposed to fulfil.
- Timing context (elapsed, prediction, budget) — computed in Python.

All tools are pure read. None of them can kill the process or modify state.
The kill decision is made by the parent bash_agent thread based on the
watcher's FINAL JSON verdict.

Warnings-filter regex is also exported here so bash_agent uses the same rule
as any downstream log consumer — Python warnings (``<file>.py:<line>: <X>Warning: ...``
plus the indented source line) are noise, not progress.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Warning detection (shared with bash_agent reader)
# ---------------------------------------------------------------------------
# Matches standard Python warning headers like:
#   /path/file.py:178: UserWarning: Boolean Series key will be reindexed...
#   C:\foo\bar.py:42: FutureWarning: ...
# The warning class is any identifier ending in ``Warning``.
WARNING_HEADER_RE = re.compile(
    r"""^\s*                      # optional leading whitespace
        \S+?\.py                  # a *.py path (no whitespace inside; colons OK for Win drive)
        :\d+:\s*                  # :lineno: + spaces
        \w*Warning:               # any *Warning: class name
    """,
    re.VERBOSE,
)


def is_warning_header(line: str) -> bool:
    """True if ``line`` looks like a stdlib warning header."""
    if not line:
        return False
    return bool(WARNING_HEADER_RE.match(line))


# ---------------------------------------------------------------------------
# Watcher context
# ---------------------------------------------------------------------------
@dataclass
class WatcherCtx:
    """
    Live state a ReAct watcher can read. Constructed by bash_agent per-tick.

    - ``get_stdout`` / ``get_stderr`` return the current filtered buffers
      (warnings already stripped at ingress).
    - ``task_text`` / ``code`` describe WHAT the process is supposed to do.
    - ``prediction`` is whatever execution_predictor_agent returned, or {}.
    - ``get_timing`` returns a fresh timing dict on each call.
    - ``pid`` is the target subprocess PID (None if not yet spawned).
    """

    get_stdout: Callable[[], str]
    get_stderr: Callable[[], str]
    task_text: str = ""
    code: str = ""
    prediction: Dict[str, Any] = field(default_factory=dict)
    get_timing: Callable[[], Dict[str, Any]] = field(default_factory=lambda: (lambda: {}))
    pid: Optional[int] = None
    # Internal snapshot for disk I/O delta calculation.
    _io_snapshot: Dict[str, Any] = field(default_factory=dict)


class WatcherToolError(Exception):
    """Raised on bad args or tool failure."""


# ---------------------------------------------------------------------------
# psutil lazy import
# ---------------------------------------------------------------------------
def _psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _fmt_mb(n_bytes: float) -> str:
    try:
        return f"{n_bytes / (1024 * 1024):.1f}MB"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# System tools
# ---------------------------------------------------------------------------
def tool_sys_cpu(ctx: WatcherCtx, interval: float = 1.0) -> str:
    """Overall CPU %, per-core average. `interval` (0.1..3.0) is blocking sample."""
    ps = _psutil()
    if ps is None:
        return "psutil_unavailable"
    try:
        iv = float(interval)
    except Exception:
        iv = 1.0
    iv = max(0.1, min(iv, 3.0))
    pct = ps.cpu_percent(interval=iv)
    per_core = ps.cpu_percent(interval=None, percpu=True)
    return json.dumps({
        "cpu_percent_total": pct,
        "cpu_count": ps.cpu_count(logical=True),
        "per_core_sample": per_core[:16],
        "sample_interval_sec": iv,
    })


def tool_sys_mem(ctx: WatcherCtx) -> str:
    ps = _psutil()
    if ps is None:
        return "psutil_unavailable"
    vm = ps.virtual_memory()
    sm = ps.swap_memory()
    return json.dumps({
        "total_mb": int(vm.total / (1024 * 1024)),
        "available_mb": int(vm.available / (1024 * 1024)),
        "used_percent": vm.percent,
        "swap_used_mb": int(sm.used / (1024 * 1024)),
        "swap_total_mb": int(sm.total / (1024 * 1024)),
    })


def tool_sys_io(ctx: WatcherCtx) -> str:
    """Disk I/O rate (MB/s read/write) since the last call in this run."""
    ps = _psutil()
    if ps is None:
        return "psutil_unavailable"
    try:
        io = ps.disk_io_counters()
    except Exception as e:
        return f"disk_io_unavailable: {e}"
    now = time.time()
    prev = ctx._io_snapshot or {}
    ctx._io_snapshot = {
        "ts": now,
        "read_bytes": io.read_bytes,
        "write_bytes": io.write_bytes,
    }
    if not prev:
        return json.dumps({
            "note": "first sample — call again to see rate",
            "read_bytes_total": io.read_bytes,
            "write_bytes_total": io.write_bytes,
        })
    dt = max(1e-3, now - prev.get("ts", now))
    dr = max(0, io.read_bytes - prev.get("read_bytes", io.read_bytes))
    dw = max(0, io.write_bytes - prev.get("write_bytes", io.write_bytes))
    return json.dumps({
        "read_mb_per_sec": round(dr / dt / (1024 * 1024), 3),
        "write_mb_per_sec": round(dw / dt / (1024 * 1024), 3),
        "window_sec": round(dt, 2),
    })


def tool_gpu_stats(ctx: WatcherCtx) -> str:
    """Per-GPU utilization & memory via nvidia-smi, or `available: false`."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return json.dumps({"available": False, "reason": "nvidia-smi not found"})
    try:
        cp = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"available": True, "error": "nvidia-smi timeout"})
    except Exception as e:
        return json.dumps({"available": True, "error": f"nvidia-smi failed: {e}"})
    if cp.returncode != 0:
        return json.dumps({"available": True, "error": cp.stderr.strip()[:400]})
    gpus: List[Dict[str, Any]] = []
    for line in (cp.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "gpu_util_percent": float(parts[2]),
                "mem_util_percent": float(parts[3]),
                "mem_used_mb": float(parts[4]),
                "mem_total_mb": float(parts[5]),
                "power_w": float(parts[6]) if parts[6] not in ("", "N/A") else None,
            })
        except Exception:
            continue
    return json.dumps({"available": True, "gpus": gpus})


def tool_proc_stats(ctx: WatcherCtx) -> str:
    """CPU %, RSS, thread count, state of the target subprocess (+ children)."""
    ps = _psutil()
    if ps is None:
        return "psutil_unavailable"
    pid = ctx.pid
    if pid is None:
        return "no_pid"
    try:
        p = ps.Process(int(pid))
    except Exception as e:
        return f"pid_lookup_failed: {e}"
    try:
        # cpu_percent wants a small blocking window on first call.
        p.cpu_percent(interval=None)
        time.sleep(0.3)
        cpu_pct = p.cpu_percent(interval=None)
        info = {
            "pid": p.pid,
            "status": p.status(),
            "cpu_percent": cpu_pct,
            "num_threads": p.num_threads(),
            "rss_mb": int(p.memory_info().rss / (1024 * 1024)),
        }
        children: List[Dict[str, Any]] = []
        try:
            for c in p.children(recursive=True):
                try:
                    c.cpu_percent(interval=None)
                except Exception:
                    continue
        except Exception:
            pass
        # One more short pass so children have a cpu_percent basis.
        time.sleep(0.2)
        try:
            for c in p.children(recursive=True):
                try:
                    children.append({
                        "pid": c.pid,
                        "name": c.name(),
                        "cpu_percent": c.cpu_percent(interval=None),
                        "rss_mb": int(c.memory_info().rss / (1024 * 1024)),
                        "status": c.status(),
                    })
                except Exception:
                    continue
        except Exception:
            pass
        info["children"] = children[:12]
        info["children_cpu_sum"] = round(sum(c.get("cpu_percent", 0.0) for c in children), 1)
        info["children_rss_mb_sum"] = sum(c.get("rss_mb", 0) for c in children)
        return json.dumps(info)
    except Exception as e:
        return f"proc_stats_failed: {e}"


# ---------------------------------------------------------------------------
# Output / code / task tools
# ---------------------------------------------------------------------------
_MAX_TOOL_TEXT = 20000  # hard cap to keep scratchpad bounded


def _clip(s: str, n: int) -> str:
    if s is None:
        return ""
    try:
        n = int(n)
    except Exception:
        n = 4000
    n = max(200, min(n, _MAX_TOOL_TEXT))
    if len(s) <= n:
        return s
    return s[-n:]


def _clip_head(s: str, n: int) -> str:
    if s is None:
        return ""
    try:
        n = int(n)
    except Exception:
        n = 2000
    n = max(200, min(n, _MAX_TOOL_TEXT))
    if len(s) <= n:
        return s
    return s[:n]


def tool_tail_stdout(ctx: WatcherCtx, n_chars: int = 6000) -> str:
    return _clip(ctx.get_stdout() or "", n_chars)


def tool_tail_stderr(ctx: WatcherCtx, n_chars: int = 4000) -> str:
    return _clip(ctx.get_stderr() or "", n_chars)


def tool_head_stdout(ctx: WatcherCtx, n_chars: int = 2000) -> str:
    return _clip_head(ctx.get_stdout() or "", n_chars)


def tool_code_excerpt(ctx: WatcherCtx, n_chars: int = 6000) -> str:
    return _clip_head(ctx.code or "", n_chars)


def tool_task_text(ctx: WatcherCtx) -> str:
    return (ctx.task_text or "(no task text provided)")[:_MAX_TOOL_TEXT]


def tool_timing(ctx: WatcherCtx) -> str:
    try:
        t = ctx.get_timing() or {}
    except Exception as e:
        return f"timing_error: {e}"
    t = dict(t)
    pred = ctx.prediction or {}
    if pred:
        t["prediction_expected_sec"] = pred.get("expected_time_sec")
        t["prediction_task_kind"] = pred.get("task_kind")
        t["prediction_expected_cpu_load"] = pred.get("expected_cpu_load")
        t["prediction_expected_gpu_load"] = pred.get("expected_gpu_load")
        t["prediction_rationale"] = pred.get("rationale") or pred.get("reasoning")
    return json.dumps(t, default=str)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "sys_cpu": tool_sys_cpu,
    "sys_mem": tool_sys_mem,
    "sys_io": tool_sys_io,
    "gpu_stats": tool_gpu_stats,
    "proc_stats": tool_proc_stats,
    "tail_stdout": tool_tail_stdout,
    "tail_stderr": tool_tail_stderr,
    "head_stdout": tool_head_stdout,
    "code_excerpt": tool_code_excerpt,
    "task_text": tool_task_text,
    "timing": tool_timing,
}


TOOLS_HELP = """Watcher tools (read-only; one decision at the end):

System / resource:
- sys_cpu(interval=1.0)         : overall CPU % + per-core sample
- sys_mem()                     : RAM + swap usage
- sys_io()                      : disk MB/s read/write since last call
- gpu_stats()                   : per-GPU util & memory (via nvidia-smi; may be unavailable)
- proc_stats()                  : target PID CPU%, RSS, threads, state, children

Output / context:
- tail_stdout(n_chars=6000)     : tail of filtered stdout (warnings stripped)
- tail_stderr(n_chars=4000)     : tail of filtered stderr
- head_stdout(n_chars=2000)     : head of stdout (what was printed at start)
- code_excerpt(n_chars=6000)    : code being executed (head)
- task_text()                   : the task description from the plan
- timing()                      : elapsed, prediction, task_allocated, savings, budget

Use as few calls as possible. If evidence is clear after 1-2 calls, decide
immediately. You have up to MAX_STEPS calls."""


def list_tool_names() -> List[str]:
    return list(_TOOL_REGISTRY.keys())


def call_tool(ctx: WatcherCtx, name: str, args: Dict[str, Any]) -> str:
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        raise WatcherToolError(
            f"unknown tool: {name!r}. Available: {', '.join(_TOOL_REGISTRY)}"
        )
    kwargs = dict(args or {})
    try:
        return fn(ctx, **kwargs)
    except WatcherToolError:
        raise
    except TypeError as e:
        raise WatcherToolError(f"bad arguments to {name}: {e}")
    except Exception as e:
        raise WatcherToolError(f"{name} failed: {e}")


def parse_args_line(raw: str) -> Dict[str, Any]:
    """Forgiving arg parser. Accepts JSON object, key=value pairs, or empty."""
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s or s in ("{}", "()"):
        return {}
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    if "=" in s:
        out: Dict[str, Any] = {}
        for part in s.split():
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                try:
                    if "." in v:
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
                except Exception:
                    out[k] = v
        if out:
            return out
    return {}
