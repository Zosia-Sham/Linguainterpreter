"""
Read-only filesystem tools for Lead Agent ReAct loop.

All tools are anchored at ``orch.project_root`` with strict path-traversal
protection. None of these tools can write, delete, or execute anything — the
Lead Agent is allowed to inspect state only. Results are truncated so the
ReAct loop stays within the LLM context budget.

Tool catalogue
--------------
- ls(path)                : directory listing with size/type, max 200 entries
- exists(path)            : boolean existence
- stat(path)              : size, mtime, is_dir, is_file
- find(glob)              : recursive glob search, max 100 results
- parquet_schema(path)    : columns, dtypes, row_count via polars
- csv_head(path, n=5)     : first n rows via polars
- read_text(path, n_bytes): head of a text file
- grep(pattern, path)     : substring search in a text file
"""

from __future__ import annotations

import fnmatch
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_MAX_TEXT = 2000          # Per-call result cap (characters).
_MAX_LIST_ENTRIES = 200   # Max entries ls() can return.
_MAX_FIND_HITS = 100      # Max hits find() can return.
_MAX_READ_BYTES = 8000    # Absolute ceiling on read_text/csv_head/grep reads.


class LeadToolError(Exception):
    """Raised when a tool call is invalid (bad args, traversal, missing path)."""


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def _project_root(orch: Any) -> Path:
    root = getattr(orch, "project_root", None)
    if root is None:
        raise LeadToolError("orch has no project_root")
    return Path(root).resolve()


def _safe_resolve(orch: Any, rel_path: str) -> Path:
    """Resolve ``rel_path`` under project_root, rejecting traversal."""
    if rel_path is None:
        raise LeadToolError("path is required")
    p = str(rel_path).strip().strip('"').strip("'")
    if not p:
        raise LeadToolError("path is empty")
    # Reject obvious traversal tokens early — even before resolve — so the
    # error message is explicit rather than a silent "file not found".
    normalized = p.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise LeadToolError(f"path traversal blocked: {p!r}")
    root = _project_root(orch)
    candidate = (root / p) if not os.path.isabs(p) else Path(p)
    try:
        resolved = candidate.resolve()
    except Exception as e:
        raise LeadToolError(f"cannot resolve {p!r}: {e}")
    try:
        resolved.relative_to(root)
    except ValueError:
        raise LeadToolError(f"path escapes project_root: {p!r}")
    return resolved


def _truncate(text: str, limit: int = _MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - 80]
    return head + f"\n... [truncated {len(text) - len(head)} chars]"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}T"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def tool_ls(orch: Any, path: str = ".") -> str:
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise LeadToolError(f"not found: {path}")
    if not target.is_dir():
        raise LeadToolError(f"not a directory: {path}")
    rows: List[str] = []
    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    truncated = False
    if len(entries) > _MAX_LIST_ENTRIES:
        entries = entries[:_MAX_LIST_ENTRIES]
        truncated = True
    for entry in entries:
        try:
            st = entry.stat()
            size = _fmt_bytes(st.st_size) if entry.is_file() else "-"
            kind = "dir" if entry.is_dir() else "file"
            rows.append(f"{kind:<4} {size:>8} {entry.name}")
        except OSError as e:
            rows.append(f"?    ?        {entry.name}  [stat error: {e}]")
    if truncated:
        rows.append(f"... [showing first {_MAX_LIST_ENTRIES} entries]")
    header = f"ls {target.relative_to(_project_root(orch)) or '.'}"
    return _truncate(header + "\n" + "\n".join(rows))


def tool_exists(orch: Any, path: str) -> str:
    target = _safe_resolve(orch, path)
    return "true" if target.exists() else "false"


def tool_stat(orch: Any, path: str) -> str:
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise LeadToolError(f"not found: {path}")
    st = target.stat()
    info = {
        "path": str(target.relative_to(_project_root(orch))) or ".",
        "is_dir": target.is_dir(),
        "is_file": target.is_file(),
        "size_bytes": st.st_size,
        "size_human": _fmt_bytes(st.st_size),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
    }
    return "\n".join(f"{k}: {v}" for k, v in info.items())


def tool_find(orch: Any, glob: str) -> str:
    if not glob or not isinstance(glob, str):
        raise LeadToolError("glob is required")
    root = _project_root(orch)
    # rglob handles ``**`` naturally; bare patterns become recursive by default.
    pattern = glob.strip()
    try:
        hits: List[Path] = []
        # Split into directory prefix and file pattern if slash present.
        if "/" in pattern or "\\" in pattern:
            norm = pattern.replace("\\", "/")
            if norm.startswith("./"):
                norm = norm[2:]
            hits = list(root.glob(norm))
        else:
            hits = list(root.rglob(pattern))
    except Exception as e:
        raise LeadToolError(f"glob error: {e}")
    hits = sorted(hits)
    truncated = False
    if len(hits) > _MAX_FIND_HITS:
        hits = hits[:_MAX_FIND_HITS]
        truncated = True
    lines: List[str] = []
    for h in hits:
        try:
            rel = h.relative_to(root)
        except ValueError:
            continue
        try:
            size = _fmt_bytes(h.stat().st_size) if h.is_file() else "-"
        except OSError:
            size = "?"
        lines.append(f"{size:>8}  {rel.as_posix()}")
    if not lines:
        return f"no matches for {pattern!r}"
    if truncated:
        lines.append(f"... [truncated to {_MAX_FIND_HITS} hits]")
    return _truncate("\n".join(lines))


def tool_parquet_schema(orch: Any, path: str) -> str:
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise LeadToolError(f"not found: {path}")
    if not target.is_file():
        raise LeadToolError(f"not a file: {path}")
    try:
        import polars as pl
    except ImportError as e:
        raise LeadToolError(f"polars not available: {e}")
    try:
        lf = pl.scan_parquet(str(target))
        schema = lf.collect_schema()
        # row_count via lazy count to avoid loading data.
        n_rows = int(lf.select(pl.len()).collect().item())
    except Exception as e:
        raise LeadToolError(f"parquet read failed: {e}")
    lines = [
        f"file: {target.relative_to(_project_root(orch)).as_posix()}",
        f"rows: {n_rows}",
        f"columns: {len(schema)}",
        "",
        "column                              dtype",
    ]
    for name, dtype in schema.items():
        lines.append(f"{str(name):<36} {dtype}")
    return _truncate("\n".join(lines))


def tool_csv_head(orch: Any, path: str, n: int = 5) -> str:
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise LeadToolError(f"not found: {path}")
    if not target.is_file():
        raise LeadToolError(f"not a file: {path}")
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        n_int = 5
    n_int = max(1, min(n_int, 20))
    try:
        import polars as pl
    except ImportError as e:
        raise LeadToolError(f"polars not available: {e}")
    try:
        df = pl.read_csv(str(target), n_rows=n_int, infer_schema_length=200)
    except Exception as e:
        raise LeadToolError(f"csv read failed: {e}")
    out = (
        f"file: {target.relative_to(_project_root(orch)).as_posix()}\n"
        f"columns ({len(df.columns)}): {df.columns}\n"
        f"head({n_int}):\n{df}"
    )
    return _truncate(out)


def tool_read_text(orch: Any, path: str, n_bytes: int = 3000) -> str:
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise LeadToolError(f"not found: {path}")
    if not target.is_file():
        raise LeadToolError(f"not a file: {path}")
    try:
        limit = int(n_bytes)
    except (TypeError, ValueError):
        limit = 3000
    limit = max(200, min(limit, _MAX_READ_BYTES))
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(limit + 1)
    except Exception as e:
        raise LeadToolError(f"read failed: {e}")
    truncated = len(data) > limit
    data = data[:limit]
    header = f"--- {target.relative_to(_project_root(orch)).as_posix()} (first {limit}B)"
    if truncated:
        header += " [truncated]"
    return _truncate(header + "\n" + data)


def tool_grep(orch: Any, pattern: str, path: str) -> str:
    if not pattern:
        raise LeadToolError("pattern is required")
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise LeadToolError(f"not found: {path}")
    if not target.is_file():
        raise LeadToolError(f"not a file: {path}")
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(_MAX_READ_BYTES * 4)  # cap total scan
    except Exception as e:
        raise LeadToolError(f"read failed: {e}")
    needle = str(pattern)
    hits: List[str] = []
    for i, line in enumerate(data.splitlines(), start=1):
        if needle in line:
            hits.append(f"{i}: {line}")
            if len(hits) >= 50:
                hits.append("... [truncated to 50 matches]")
                break
    if not hits:
        return f"no matches for {needle!r} in {path}"
    return _truncate("\n".join(hits))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
# Lazy import of artifact tools — they're optional (git may be absent) but
# exposing them here gives the Lead ReAct loop a single unified namespace.
def _load_artifact_tool(name: str) -> Callable[..., str]:
    from . import artifact_tools as _at  # local import avoids cycles

    tool = _at._TOOL_REGISTRY.get(name)
    if tool is None:
        raise LeadToolError(f"artifact tool not available: {name}")
    return tool


def _wrap_artifact_tool(name: str) -> Callable[..., str]:
    def _runner(orch: Any, **kwargs: Any) -> str:
        try:
            return _load_artifact_tool(name)(orch, **kwargs)
        except Exception as e:
            # Re-wrap so the ReAct loop sees a LeadToolError (uniform).
            raise LeadToolError(f"{name} failed: {e}")

    _runner.__name__ = f"artifact_{name}"
    return _runner


_TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "ls": tool_ls,
    "exists": tool_exists,
    "stat": tool_stat,
    "find": tool_find,
    "parquet_schema": tool_parquet_schema,
    "csv_head": tool_csv_head,
    "read_text": tool_read_text,
    "grep": tool_grep,
    # Artifact-scoped tools — writes sandboxed to artifacts_dir, git-anchored
    # ground truth for "what was actually saved". See src/artifact_tools.py.
    "save_artifact": _wrap_artifact_tool("save_artifact"),
    "list_artifacts": _wrap_artifact_tool("list_artifacts"),
    "read_artifact": _wrap_artifact_tool("read_artifact"),
    "artifacts_diff": _wrap_artifact_tool("artifacts_diff"),
}


TOOLS_HELP = """Available tools (anchored at project_root, writes only via save_artifact):

Read-only (project_root scope):
- ls(path)                       : directory listing
- exists(path)                   : true/false
- stat(path)                     : size, mtime, is_dir, is_file
- find(glob)                     : recursive glob (e.g. "*.parquet")
- parquet_schema(path)           : columns + dtypes + row_count
- csv_head(path, n=5)            : first n rows of a CSV
- read_text(path, n_bytes=3000)  : head of a text file
- grep(pattern, path)            : substring search in a text file

Artifact-scoped (artifacts_dir only, git-anchored):
- save_artifact(path, content)   : WRITE a file under artifacts_dir. Only
                                   sanctioned write path. Returns sha256+size.
- list_artifacts(subdir='.')     : enumerate artifacts_dir with size + sha256.
                                   Use this instead of fabricating shapes.
- read_artifact(path, n_bytes)   : read a file under artifacts_dir (head).
- artifacts_diff(since=<sha>)    : files added/modified/deleted since a
                                   reference commit. Ground-truth for "was it
                                   actually saved?".

All paths are relative to project_root (or artifacts_dir for artifact tools).
Traversal (`..`) is blocked. Only save_artifact can write; every other write
attempt is rejected."""


def list_tool_names() -> List[str]:
    return list(_TOOL_REGISTRY.keys())


def call_tool(orch: Any, name: str, args: Dict[str, Any]) -> str:
    """Dispatch a single tool call. Returns plain-text result or raises LeadToolError."""
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        raise LeadToolError(
            f"unknown tool: {name!r}. Available: {', '.join(_TOOL_REGISTRY)}"
        )
    kwargs = dict(args or {})
    try:
        return fn(orch, **kwargs)
    except LeadToolError:
        raise
    except TypeError as e:
        raise LeadToolError(f"bad arguments to {name}: {e}")
    except Exception as e:
        raise LeadToolError(f"{name} failed: {e}")


def parse_args_line(raw: str) -> Dict[str, Any]:
    """
    Very forgiving arg parser for the ReAct loop.

    Accepts:
      - JSON:        {"path": "artifacts/x.parquet"}
      - key=value:   path=artifacts/x.parquet n=10
      - positional:  artifacts/x.parquet   (stored under ``path``)
    """
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    # JSON first.
    if s.startswith("{") and s.endswith("}"):
        try:
            import json
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # key=value pairs.
    if "=" in s:
        out: Dict[str, Any] = {}
        # naive split on whitespace — adequate for short paths/globs.
        parts: List[str] = []
        buf = ""
        in_q: Optional[str] = None
        for ch in s:
            if in_q:
                if ch == in_q:
                    in_q = None
                else:
                    buf += ch
            elif ch in ("'", '"'):
                in_q = ch
            elif ch.isspace():
                if buf:
                    parts.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            parts.append(buf)
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
            else:
                out.setdefault("path", part)
        if out:
            return out
    # Fallback: treat as a single positional ``path`` argument.
    return {"path": s.strip().strip('"').strip("'")}
