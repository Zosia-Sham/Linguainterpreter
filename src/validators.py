from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional

_DL_MARKERS = (
    r"\btorch\b", r"\btensorflow\b", r"\bkeras\b", r"\bpytorch_lightning\b", r"\balbumentations\b",
)
_ML_MARKERS = (
    r"\bxgboost\b", r"\blightgbm\b", r"\bcatboost\b",
    r"\bsklearn\.ensemble\b", r"\bsklearn\.svm\b", r"\bsklearn\.linear_model\b",
)

_METRICS_RE = re.compile(r"METRICS_JSON\s*:\s*(\{.*\})", re.DOTALL)


def _balanced_json_object_from(text: str, open_brace: int) -> Optional[str]:
    """Extract one `{...}` object from text starting at open_brace (brace-aware, strings respected)."""
    depth = 0
    in_str = False
    esc = False
    for i in range(open_brace, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]
    return None


def _iter_metrics_json_payloads(stdout: str):
    """Yield candidate JSON strings after each METRICS_JSON: marker (for last-wins parsing)."""
    if not stdout:
        return
    pos = 0
    needle = "METRICS_JSON"
    while True:
        i = stdout.find(needle, pos)
        if i < 0:
            break
        colon = stdout.find(":", i + len(needle))
        if colon < 0:
            pos = i + 1
            continue
        j = colon + 1
        while j < len(stdout) and stdout[j] in " \t":
            j += 1
        if j < len(stdout) and stdout[j] == "{":
            blob = _balanced_json_object_from(stdout, j)
            if blob:
                yield blob
        pos = i + 1


def detect_mixed_stacks(code_text: str) -> bool:
    """Грубая эвристика: одновременно видны DL и классический ML импорты в одном скрипте."""
    t = code_text or ""
    has_dl = any(re.search(p, t) for p in _DL_MARKERS)
    has_ml = any(re.search(p, t) for p in _ML_MARKERS)
    return has_dl and has_ml

def parse_metrics_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    """Ищем блок METRICS_JSON:{...} в stdout и парсим JSON (последний валидный wins — часто после эпох логов)."""
    if not stdout:
        return None
    last_ok: Optional[Dict[str, Any]] = None
    for raw in _iter_metrics_json_payloads(stdout):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("type") in ("calculated", "skipped"):
                last_ok = obj
        except Exception:
            continue
    if last_ok is not None:
        return last_ok
    m = _METRICS_RE.search(stdout)
    if not m:
        return None
    raw = m.group(1)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        try:
            last = raw.rfind("}")
            if last > 0:
                obj = json.loads(raw[: last + 1])
                return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def validate_recovered_metrics(obj: Optional[Dict[str, Any]], spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Validate LLM-recovered metrics. Returns None to fall through to verifier script.
    """
    if not obj or not isinstance(obj, dict):
        return None
    t = obj.get("type")
    if t == "skipped":
        reason = str(obj.get("reason", "")).lower()
        if "no_parseable" in reason:
            return None
        return obj
    if t not in ("calculated", None):
        return None
    obj = dict(obj)
    obj.setdefault("type", "calculated")
    pm = (spec or {}).get("primary_metric") or {}
    exp_name = pm.get("name")
    if exp_name is not None and str(obj.get("name", "")).lower() != str(exp_name).lower():
        return None
    try:
        float(obj.get("primary"))
    except (TypeError, ValueError):
        return None
    if not isinstance(obj.get("maximize"), bool):
        return None
    return obj
