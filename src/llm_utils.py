from __future__ import annotations
import json
import time
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from langchain_core.messages import ToolMessage, AIMessage

try:
    from colorama import Fore, Style
except Exception:
    class Fore:
        CYAN = ""
        GREEN = ""
        YELLOW = ""
        MAGENTA = ""
    class Style:
        RESET_ALL = ""

_TIMING_ORCH: Any = None

# (match_substring_lower, input_usd_per_1k, output_usd_per_1k). Filled by load_llm_pricing().
_LLM_PRICE_ROWS: List[Tuple[str, float, float]] = []


def _default_llm_price_rows() -> List[Tuple[str, float, float]]:
    """Minimal built-in fallback when llm_model_pricing.json is missing (USD per 1k tokens)."""
    return [
        ("gpt-4o-mini", 0.00015, 0.0006),
        ("gpt-4o", 0.0025, 0.01),
        ("claude-3-5-sonnet", 0.003, 0.015),
        ("claude-3-haiku", 0.00025, 0.00125),
        ("gemini-2.5-flash", 0.0003, 0.0025),
        ("gemini-2.5-pro", 0.00125, 0.01),
        ("kimi-k2", 0.0006, 0.0025),
        ("deepseek-chat", 0.00028, 0.00042),
    ]


def _parse_pricing_payload(raw: Any) -> List[Tuple[str, float, float]]:
    if not isinstance(raw, dict):
        raise ValueError("pricing root must be an object")
    unit = str(raw.get("unit", "per_million_tokens")).lower().replace(" ", "_").replace("-", "_")
    entries = raw.get("entries") or raw.get("models") or []
    rows: List[Tuple[str, float, float]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        m = str(e.get("match", "")).strip().lower()
        if not m:
            continue
        inp = float(e.get("input", 0) or 0)
        out = float(e.get("output", 0) or 0)
        if unit in ("per_million_tokens", "per_million", "million", "usd_per_million"):
            inp /= 1000.0
            out /= 1000.0
        elif unit in ("per_1k_tokens", "per_1k", "1k"):
            pass
        else:
            inp /= 1000.0
            out /= 1000.0
        rows.append((m, inp, out))
    if not rows:
        raise ValueError("no valid entries")
    return rows


def load_llm_pricing(path: str | Path | None) -> None:
    """
    Load USD pricing for token usage estimates. Path is typically repo_root/llm_model_pricing.json.
    Uses longest substring match on the model name (see _model_price_usd_per_1k).
    """
    global _LLM_PRICE_ROWS
    if not path:
        _LLM_PRICE_ROWS = list(_default_llm_price_rows())
        return
    p = Path(path)
    if not p.is_file():
        try:
            print(f"{Fore.YELLOW}[LLM] Pricing file not found: {p}; using built-in defaults{Style.RESET_ALL}")
        except Exception:
            print(f"[LLM] Pricing file not found: {p}; using built-in defaults")
        _LLM_PRICE_ROWS = list(_default_llm_price_rows())
        return
    try:
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            import yaml  # type: ignore

            raw = yaml.safe_load(text) or {}
        rows = _parse_pricing_payload(raw)
        _LLM_PRICE_ROWS = rows
        print(f"{Fore.CYAN}[LLM] Loaded {len(rows)} model pricing rules from {p}{Style.RESET_ALL}")
    except Exception as ex:
        try:
            print(f"{Fore.YELLOW}[LLM] Failed to load pricing file {p}: {ex}; using built-in defaults{Style.RESET_ALL}")
        except Exception:
            print(f"[LLM] Failed to load pricing file {p}: {ex}; using built-in defaults")
        _LLM_PRICE_ROWS = list(_default_llm_price_rows())


def set_timing_orchestrator(orch: Any) -> None:
    global _TIMING_ORCH
    _TIMING_ORCH = orch


def _is_transient_llm_error(exc: Exception) -> bool:
    s = str(exc).lower()
    transient_markers = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "520",
        "522",
        "524",
        "rate limit",
        "ratelimit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "internalservererror",
        "provider internal",
        "no healthy upstream",
        "bad gateway",
        "gateway timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection error",
        "too many requests",
        "overloaded",
        "server error",
        "upstream",
        "engine is currently overloaded",
        "unexpected eof",
        "broken pipe",
        "reset by peer",
        "network is unreachable",
    ]
    non_transient_markers = [
        "invalid api key",
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid request",
        "bad request",
        "not found",
    ]
    if any(m in s for m in non_transient_markers):
        return False
    # Also check the exception class name — openai raises typed exceptions
    # like InternalServerError, APIConnectionError, etc.
    cls_name = type(exc).__name__.lower()
    if any(m in cls_name for m in ("internal", "server", "connection", "timeout", "gateway", "overloaded")):
        return True
    return any(m in s for m in transient_markers)


def _sleep_with_accounting(seconds: float) -> None:
    orch = _TIMING_ORCH
    if orch is not None and hasattr(orch, "sleep_with_pause_accounting"):
        orch.sleep_with_pause_accounting(seconds)
    else:
        time.sleep(seconds)


def _current_run_root() -> Path:
    orch = _TIMING_ORCH
    if orch is not None and hasattr(orch, "project_root"):
        return Path(getattr(orch, "project_root")).resolve()
    return Path(os.getcwd()).resolve()


def _agents_log_dir() -> Path:
    orch = _TIMING_ORCH
    root = _current_run_root()
    artifacts_dir = "artifacts"
    if orch is not None and hasattr(orch, "cfg"):
        try:
            artifacts_dir = str(getattr(orch.cfg.paths, "artifacts_dir", "artifacts"))
        except Exception:
            artifacts_dir = "artifacts"
    base_dir = root / artifacts_dir / "agents"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _retry_settings() -> Tuple[int, float, float]:
    # User requirement: long exponential retry for transient provider limits.
    max_attempts = 30
    first_delay_sec = 2.0
    max_delay_sec = 300.0
    orch = _TIMING_ORCH
    if orch is not None and hasattr(orch, "cfg"):
        try:
            max_attempts = int(getattr(orch.cfg.runtime, "llm_retry_attempts", max_attempts))
            first_delay_sec = float(getattr(orch.cfg.runtime, "llm_retry_initial_delay_sec", first_delay_sec))
            max_delay_sec = float(getattr(orch.cfg.runtime, "llm_retry_max_delay_sec", max_delay_sec))
        except Exception:
            pass
    return max(1, max_attempts), max(0.1, first_delay_sec), max(1.0, max_delay_sec)


def _invoke_with_retry(invoke_fn, agent_name: str):
    max_attempts, first_delay_sec, max_delay_sec = _retry_settings()
    delay = float(first_delay_sec)
    attempt = 1
    while True:
        try:
            return invoke_fn()
        except Exception as e:
            if attempt >= max_attempts or not _is_transient_llm_error(e):
                raise
            print(
                f"[LLM RETRY] agent={agent_name} attempt={attempt}/{max_attempts} "
                f"sleep={int(delay)}s reason={e}"
            )
            _sleep_with_accounting(delay)
            delay = min(max_delay_sec, delay * 2.0)
            attempt += 1


def _extract_usage_dict(res: Any) -> Dict[str, Any]:
    usage = {}
    for attr in ("usage_metadata",):
        v = getattr(res, attr, None)
        if isinstance(v, dict):
            usage.update(v)
    md = getattr(res, "response_metadata", None)
    if isinstance(md, dict):
        for k in ("token_usage", "usage", "usage_metadata"):
            maybe = md.get(k)
            if isinstance(maybe, dict):
                usage.update(maybe)
        if "model_name" in md and "model_name" not in usage:
            usage["model_name"] = md.get("model_name")
    for k in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens"):
        v = getattr(res, k, None)
        if isinstance(v, int):
            usage[k] = v
    return usage


def _normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    in_tok = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    out_tok = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tok = int(usage.get("total_tokens", 0) or 0)
    if total_tok <= 0:
        total_tok = in_tok + out_tok
    return {"input_tokens": max(0, in_tok), "output_tokens": max(0, out_tok), "total_tokens": max(0, total_tok)}


def _model_price_usd_per_1k(model_name: str) -> Dict[str, float]:
    """
    USD per 1k input/output tokens. Longest `match` substring wins (avoids gpt-4o matching gpt-4o-mini).
    Populated by load_llm_pricing() from llm_model_pricing.json (or defaults).
    """
    m = (model_name or "").lower()
    rows = _LLM_PRICE_ROWS or _default_llm_price_rows()
    best: Optional[Tuple[str, float, float]] = None
    best_len = -1
    for match_s, inp, out in rows:
        if match_s in m and len(match_s) > best_len:
            best = (match_s, inp, out)
            best_len = len(match_s)
    if best:
        return {"input": best[1], "output": best[2]}
    return {"input": 0.0, "output": 0.0}


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _fmt_cost(cost: float, known_price: bool) -> str:
    if not known_price:
        return "n/a"
    return f"${cost:.6f}"


def _log_token_usage(agent_name: str, res: Any) -> None:
    try:
        usage_raw = _extract_usage_dict(res)
        usage = _normalize_usage(usage_raw)
        model_name = str(
            usage_raw.get("model_name")
            or getattr(res, "model_name", "")
            or getattr(getattr(res, "response_metadata", {}), "get", lambda *_: "")("model_name")
            or "unknown"
        )
        prices = _model_price_usd_per_1k(model_name)
        known_price = (prices.get("input", 0.0) > 0.0) or (prices.get("output", 0.0) > 0.0)
        estimated_cost_usd = (
            (usage["input_tokens"] / 1000.0) * prices["input"]
            + (usage["output_tokens"] / 1000.0) * prices["output"]
        )
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": agent_name,
            "model": model_name,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": round(estimated_cost_usd, 8),
            "known_price": known_price,
            "price_per_1k": prices,
        }
        base_dir = _agents_log_dir()
        with open(base_dir / "token_usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = _update_token_summary(base_dir, record)
        total = summary.get("total", {}) if isinstance(summary, dict) else {}
        this_cost = _fmt_cost(float(record.get("estimated_cost_usd", 0.0)), bool(record.get("known_price", False)))
        cum_known_price = any(
            (
                (isinstance(v, dict) and float(v.get("estimated_cost_usd", 0.0)) > 0.0)
                for v in (summary.get("by_model", {}) if isinstance(summary, dict) else {}).values()
            )
        )
        cum_cost = _fmt_cost(float(total.get("estimated_cost_usd", 0.0)), cum_known_price)
        print(
            f"{Fore.CYAN}[LLM]{Style.RESET_ALL} "
            f"{Fore.GREEN}{record['agent']}{Style.RESET_ALL} | "
            f"{Fore.MAGENTA}{record['model']}{Style.RESET_ALL} | "
            f"in {_fmt_int(record['input_tokens'])} | "
            f"out {_fmt_int(record['output_tokens'])} | "
            f"tot {_fmt_int(record['total_tokens'])} | "
            f"cost {this_cost}  "
            f"{Fore.YELLOW}|| cum: in {_fmt_int(total.get('input_tokens', 0))}, "
            f"out {_fmt_int(total.get('output_tokens', 0))}, "
            f"tot {_fmt_int(total.get('total_tokens', 0))}, "
            f"cost {cum_cost}{Style.RESET_ALL}"
        )
    except Exception as e:
        print(f"Failed to write token usage log: {e}")


def _update_token_summary(base_dir: Path, rec: Dict[str, Any]) -> Dict[str, Any]:
    summary_path = base_dir / "token_usage_summary.json"
    summary: Dict[str, Any] = {"by_model": {}, "by_agent": {}, "total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}}
    try:
        if summary_path.exists():
            cur = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(cur, dict):
                summary = cur
    except Exception:
        pass

    def _acc(bucket: Dict[str, Any], key: str) -> None:
        row = bucket.get(key) or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
        row["input_tokens"] = int(row.get("input_tokens", 0)) + int(rec.get("input_tokens", 0))
        row["output_tokens"] = int(row.get("output_tokens", 0)) + int(rec.get("output_tokens", 0))
        row["total_tokens"] = int(row.get("total_tokens", 0)) + int(rec.get("total_tokens", 0))
        row["estimated_cost_usd"] = float(row.get("estimated_cost_usd", 0.0)) + float(rec.get("estimated_cost_usd", 0.0))
        bucket[key] = row

    by_model = summary.setdefault("by_model", {})
    by_agent = summary.setdefault("by_agent", {})
    total = summary.setdefault("total", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0})
    _acc(by_model, str(rec.get("model", "unknown")))
    _acc(by_agent, str(rec.get("agent", "unknown")))
    total["input_tokens"] = int(total.get("input_tokens", 0)) + int(rec.get("input_tokens", 0))
    total["output_tokens"] = int(total.get("output_tokens", 0)) + int(rec.get("output_tokens", 0))
    total["total_tokens"] = int(total.get("total_tokens", 0)) + int(rec.get("total_tokens", 0))
    total["estimated_cost_usd"] = float(total.get("estimated_cost_usd", 0.0)) + float(rec.get("estimated_cost_usd", 0.0))
    summary["total"] = total
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _log_to_jsonl(agent_name: str, inputs: Any, output: str, is_error: bool = False):
    """
    Append a single JSONL record for a given agent.
    Logs are written under {project_root}/{artifacts_dir}/agents/ (see _agents_log_dir).
    """
    try:
        base_dir = _agents_log_dir()

        log_entry = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "agent": agent_name,
            "is_error": is_error,
            "inputs": str(inputs)[:50000],
            "output": str(output)[:50000],
        }

        # One JSONL file per agent
        agent_log_path = base_dir / f"{agent_name}.jsonl"
        with open(agent_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to write agent log: {e}")

def invoke_with_tools(llm: Any, prompt: Any, params: Dict[str, Any], tools: Optional[List[Any]] = None, agent_name: str = None, max_steps: int = 15) -> Any:
    if agent_name is None:
        agent_name = sys._getframe(1).f_code.co_name

    if not tools:
        res = _invoke_with_retry(lambda: (prompt | llm).invoke(params), agent_name=agent_name)
        _log_to_jsonl(agent_name, safe_json_dumps(params), getattr(res, 'content', str(res)))
        _log_token_usage(agent_name, res)
        return res

    llm_with_tools = llm.bind_tools(tools)
    messages = prompt.format_messages(**params)

    for step in range(max(1, max_steps)):
        res = _invoke_with_retry(lambda: llm_with_tools.invoke(messages), agent_name=agent_name)
        if not hasattr(res, 'tool_calls') or not res.tool_calls:
            _log_to_jsonl(agent_name, safe_json_dumps(params), getattr(res, 'content', str(res)))
            _log_token_usage(agent_name, res)
            return res
            
        messages.append(res)
        for tool_call in res.tool_calls:
            tool = next((t for t in tools if t.name == tool_call['name']), None)
            if tool:
                try:
                    tool_res = tool.invoke(tool_call['args'])
                    content = str(tool_res)
                except Exception as e:
                    content = f"Error executing tool: {e}"
            else:
                content = f"Tool {tool_call['name']} not found."
                
            messages.append(ToolMessage(
                name=tool_call['name'],
                tool_call_id=tool_call['id'],
                content=content
            ))
            
    content_str = str(getattr(res, 'content', res))
    _log_to_jsonl(agent_name, params, content_str + "\n[Max steps reached]")
    _log_token_usage(agent_name, res)
    return res

def safe_json_dumps(obj):
    """Safely convert object to JSON string."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def invoke_and_log(llm: Any, prompt: Any, params: Dict[str, Any], agent_name: str = None) -> Any:
    if agent_name is None:
        agent_name = sys._getframe(1).f_code.co_name
        
    try:
        res = _invoke_with_retry(lambda: (prompt | llm).invoke(params), agent_name=agent_name)
        _log_to_jsonl(agent_name, safe_json_dumps(params), getattr(res, 'content', str(res)))
        _log_token_usage(agent_name, res)
        return res
    except Exception as e:
        _log_to_jsonl(agent_name, safe_json_dumps(params), str(e), is_error=True)
        raise


def log_agent_trace(agent_name: str, stage: str, detail: Any) -> None:
    """
    Append one line to artifacts/agents/{agent_name}.jsonl without an LLM call.
    Use for LangGraph ReAct / custom loops that bypass invoke_and_log.
    """
    try:
        body = detail if isinstance(detail, str) else safe_json_dumps(detail)
        _log_to_jsonl(agent_name, stage, body[:50000])
    except Exception:
        pass
