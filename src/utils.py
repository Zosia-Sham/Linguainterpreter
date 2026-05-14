from __future__ import annotations

import datetime
import hashlib
import json
import platform
import re
import time
from pathlib import Path
from shutil import which
from typing import Optional, List, Dict, Any, Union
import sys

import ast
import yaml
from colorama import Fore, Style, init

def _which(cmd: str) -> Optional[str]:
    return which(cmd)

def detect_os():
    sysname = platform.system()
    # Always use the current interpreter for spawning Python code, to avoid relying on
    # 'python'/'python3' being on PATH or differing between environments.
    if sysname == "Windows":
        return {"os": sysname, "shell": "powershell", "python_exec": sys.executable}
    return {"os": sysname, "shell": "bash", "python_exec": sys.executable}

def format_spec_constraints_block(spec: Optional[Dict[str, Any]]) -> str:
    """Human-readable competition rules for LLM prompts (planner, ordering, code agents)."""
    if not spec:
        return ""
    c = spec.get("constraints")
    if not isinstance(c, dict):
        return ""
    lines = [
        "COMPETITION CONSTRAINTS (from spec.json — obey in plans and code):",
        f"  internet_allowed: {c.get('internet_allowed', True)}",
        f"  pretrained_allowed: {c.get('pretrained_allowed', True)}",
        f"  external_data_allowed: {c.get('external_data_allowed', True)}",
        f"  external_data_requires_tag: {c.get('external_data_requires_tag', False)}",
    ]
    notes = c.get("notes")
    if notes:
        lines.append(f"  notes: {notes}")
    lines.append(
        "  (pip/conda packages are typically preinstalled by the orchestrator — "
        "when internet_allowed is false, avoid runtime HTTP, hub downloads, and "
        "loading **external** pretrained weights; use pretrained=False or weights under ./artifacts/.)"
    )
    return "\n".join(lines) + "\n\n"


def shorten_string_middle(text: str | None, max_length: int) -> str:
    if text is None: return ""
    if len(text) <= max_length: return text
    keep = max_length - 3
    left = keep // 2
    right = keep - left
    return text[:left] + "..." + text[-right:]


class YAMLParseError(Exception):
    pass


def _split_top_level_brace_blocks(s: str) -> List[str]:
    """
    Split a string into top-level `{...}` segments, ignoring braces inside '...' and "..." strings.
    Used when the LLM concatenates Python dict reprs instead of YAML.
    """
    out: List[str] = []
    i = 0
    n = len(s)
    depth = 0
    start = -1
    while i < n:
        c = s[i]
        if c in "'\"":
            quote = c
            i += 1
            while i < n:
                if s[i] == "\\":
                    i += 2
                    continue
                if s[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(s[start : i + 1])
                start = -1
            i += 1
            continue
        i += 1
    return out


def _extract_fenced_block(text: str, lang: str = "yaml") -> Optional[str]:
    """Extract content from a fenced code block (```yaml ... ```) anywhere in text.
    Returns the content inside the LAST matching block, or None if not found."""
    pattern = re.compile(
        r"```" + re.escape(lang) + r"\s*\n(.*?)```",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    # Also try generic ``` blocks that look like YAML (start with tasks: or - task:)
    generic = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    for m in reversed(generic.findall(text)):
        stripped = m.strip()
        if stripped.startswith("tasks:") or stripped.startswith("- task:"):
            return stripped
    return None


def _try_parse_tasks_non_yaml(llm_output: str) -> Optional[List[Any]]:
    """
    Fallback when yaml.safe_load fails: JSON `{"tasks":[...]}`, Python list of dicts,
    or concatenated dict reprs `{'task':...}{'task':...}` (common model mistake).
    """
    s = str(llm_output).strip()
    if not s:
        return None
    # Try extracting fenced block first
    fenced = _extract_fenced_block(s, "yaml") or _extract_fenced_block(s, "json")
    if fenced:
        s = fenced
    else:
        if s.startswith("```yaml"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    if s.startswith("{") and '"tasks"' in s[:800]:
        try:
            data = json.loads(s)
            if isinstance(data, dict) and isinstance(data.get("tasks"), list):
                return _tasks_list_from_parsed(data["tasks"])
        except Exception:
            pass

    if s.startswith("[") and "task" in s[:1200]:
        try:
            data = ast.literal_eval(s)
            if isinstance(data, list) and data and isinstance(data[0], dict) and "task" in data[0]:
                return _tasks_list_from_parsed(data)
        except Exception:
            pass

    first = s.find("{")
    if first < 0:
        return None
    blocks = _split_top_level_brace_blocks(s[first:])
    out: List[Any] = []
    for b in blocks:
        try:
            obj = ast.literal_eval(b.strip())
            if isinstance(obj, dict) and "task" in obj:
                out.append(obj)
        except Exception:
            continue
    return out if out else None


def _tasks_list_from_parsed(tasks_list: List[Any]) -> List[Any]:
    processed: List[Any] = []
    for item in tasks_list:
        if isinstance(item, dict) and "task" in item:
            processed.append(item)
        elif isinstance(item, dict):
            processed.append(" ".join([f"{k} {v}" for k, v in item.items()]))
        else:
            processed.append(str(item))
    return processed


def _parse_yaml_tasks(llm_output: Any, raise_on_error: bool = False) -> List[Any]:
    if isinstance(llm_output, list):
        # Could be LangChain blocks
        if all(isinstance(x, dict) and 'text' in x for x in llm_output):
            llm_output = "".join(str(x['text']) for x in llm_output)
        elif all(isinstance(x, str) for x in llm_output):
            llm_output = "".join(llm_output)
        else:
            return llm_output  # Might be already parsed tasks
            
    content = str(llm_output).strip()

    # Try to extract a fenced ```yaml block from anywhere in the text
    fenced = _extract_fenced_block(content, "yaml")
    if fenced:
        llm_output = fenced
    else:
        # Fallback: strip leading/trailing fences if the whole thing is wrapped
        if content.startswith("```yaml"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        llm_output = content.strip()

    try:
        data = yaml.safe_load(llm_output)
        if isinstance(data, dict) and 'tasks' in data:
            tasks_list = data['tasks']
            if isinstance(tasks_list, list):
                if not tasks_list:
                    raise YAMLParseError("parsed YAML has empty 'tasks' list")
                # We want to return list of dicts if they have 'task' and 'time_budget_sec'
                processed = []
                for item in tasks_list:
                    if isinstance(item, dict) and "task" in item:
                        # New format with time budgets
                        processed.append(item)
                    elif isinstance(item, dict):
                        # Old fallback
                        processed.append(" ".join([f"{k} {v}" for k, v in item.items()]))
                    else:
                        processed.append(str(item))
                return processed
            else:
                fb = _try_parse_tasks_non_yaml(str(llm_output))
                if fb:
                    print(Fore.YELLOW + "[TASKS] YAML 'tasks' not a list; recovered non-YAML fallback.")
                    return fb
                msg = f"'tasks' is not a list (got {type(tasks_list)})"
                if raise_on_error: raise YAMLParseError(msg)
                print(Fore.RED + "Ошибка: " + msg); return []
        else:
            fb = _try_parse_tasks_non_yaml(str(llm_output))
            if fb:
                print(Fore.YELLOW + "[TASKS] YAML missing valid root; recovered non-YAML fallback.")
                return fb
            msg = f"YAML lacks root key 'tasks'. Parsed: {type(data)}"
            if raise_on_error: raise YAMLParseError(msg)
            print(Fore.RED + "Ошибка: " + msg); return []
    except Exception as e:
        fb = _try_parse_tasks_non_yaml(str(llm_output))
        if fb:
            print(Fore.YELLOW + "[TASKS] YAML failed; recovered non-YAML task list (dict repr / JSON fallback).")
            return fb
        if raise_on_error:
            raise YAMLParseError(f"YAML parse failed: {e}\n---\n{llm_output}\n---")
        print(Fore.RED + f"Ошибка: Не удалось разобрать YAML. Ошибка: {e}")
        print(Fore.CYAN + f"Содержимое, которое не удалось разобрать:\n---\n{llm_output}\n---")
        return []

def _as_bool(x) -> bool:
    try:
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in ("true", "1", "yes", "y", "ok")
    except Exception:
        return False


def _parse_check_summary(stdout: str) -> Dict[str,Any]:
    """
    Ищем финальную строку CHECK_SUMMARY: {...}
    + fallback по строкам CHECK PASS/FAIL.
    """
    summary = {"total": 0, "failed": 0, "fail_names": [], "pass_names": []}
    if not stdout:
        return summary

    # Comment translated to English.
    m = re.search(r'CHECK_SUMMARY:\s*(\{.*\})', stdout, re.DOTALL)
    if m:
        try:
            js = json.loads(m.group(1))
            for k in ("total","failed","fail_names","pass_names"):
                if k in js: summary[k] = js[k]
            if not summary["total"]:
                # Comment translated to English.
                summary["total"] = int(len(summary.get("fail_names",[])) + len(summary.get("pass_names",[])))
            return summary
        except Exception:
            pass

    # Comment translated to English.
    fails = re.findall(r'^CHECK FAIL:\s*(.+)$', stdout, flags=re.MULTILINE)
    passes = re.findall(r'^CHECK PASS:\s*(.+)$', stdout, flags=re.MULTILINE)
    summary["fail_names"] = [x.strip() for x in fails]
    summary["pass_names"] = [x.strip() for x in passes]
    summary["total"] = len(fails) + len(passes)
    summary["failed"] = len(fails)
    return summary

def _slug(s: str, maxlen: int = 64) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:maxlen] or "part"

def _validate_and_normalize_metrics(m: Any) -> tuple[bool, Dict[str, Any], str]:
    """
    Приводит метрики к **единому канону** для артефактов, ledger и best/last:

    **calculated (успешное обучение/оценка):**
      {"type": "calculated", "primary": float, "name": str, "maximize": bool, "extras"?: dict}

    **skipped** (EDA и т.п.) — только для in-memory / контекста; в ledger через _update_best не пишется:
      {"type": "skipped", "reason": str}

    Возвращает (ok, norm, reason_if_not_ok)
    """
    if not isinstance(m, dict):
        return False, {}, "metrics is not a dict"
    if m.get("type") == "skipped":
        return True, {
            "type": "skipped",
            "reason": str(m.get("reason", "task_does_not_produce_metrics")),
        }, ""

    out: Dict[str, Any] = {}
    name = m.get("name", "primary")
    maximize = m.get("maximize", True)
    primary = m.get("primary", None)
    try:
        primary = float(primary)
    except Exception:
        return False, {}, "primary is missing or non-numeric"

    out["type"] = "calculated"
    out["primary"] = primary
    out["name"] = str(name) if isinstance(name, str) else "primary"
    out["maximize"] = bool(maximize)
    if isinstance(m.get("extras"), dict):
        out["extras"] = m["extras"]
    return True, out, ""

def _now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def _bool_from_text(t: str) -> bool | None:
    if not isinstance(t, str):
        return None
    s = t.strip().lower()
    if s in ("true", "false"):
        return s == "true"
    return None


def _rel_better(new: Dict[str, Any], old: Dict[str, Any], rel_thr: float) -> bool:
    """
    True, если new лучше old с учётом относительного порога rel_thr.
    - Для метрик, где maximize=True: требуем nv >= ov * (1 + rel_thr)
    - Для метрик, где maximize=False: требуем nv <= ov * (1 - rel_thr)
    Пустой old => считаем улучшением.
    """
    if not new:
        return False
    if not old:
        return True
    maximize = bool(new.get("maximize", old.get("maximize", True)))
    nv = float(new.get("primary", 0.0) or 0.0)
    ov = float(old.get("primary", 0.0) or 0.0)

    if maximize:
        # Comment translated to English.
        return nv >= (ov * (1.0 + rel_thr)) if ov != 0.0 else nv >= 0.0
    else:
        # Comment translated to English.
        return nv <= (ov * (1.0 - rel_thr)) if ov != 0.0 else nv < ov


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _stable_id(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()[:10]


def clean_specs(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    """
    Рекурсивно удаляет "пустые" значения из словаря или списка словарей.
    Функция изменяет структуру данных на месте.

    "Пустыми" считаются:
    - None
    - '' (пустая строка)
    - [] (пустой список)
    - {} (пустой словарь)

    :param data: Словарь или список для очистки.
    :return: Тот же объект словаря или списка, измененный на месте.
    """
    if isinstance(data, dict):
        # Comment translated to English.
        # Comment translated to English.
        keys_to_delete = []
        for key, value in data.items():
            # Comment translated to English.
            if isinstance(value, (dict, list)):
                clean_specs(value)

            # Comment translated to English.
            # Comment translated to English.
            # Comment translated to English.
            if value is None or value == '' or value == [] or value == {}:
                keys_to_delete.append(key)

        # Comment translated to English.
        for key in keys_to_delete:
            del data[key]

    elif isinstance(data, list):
        # Comment translated to English.
        # Comment translated to English.
        items_to_keep = []
        for item in data:
            # Comment translated to English.
            if isinstance(item, (dict, list)):
                clean_specs(item)

            # Comment translated to English.
            if item is not None and item != '' and item != [] and item != {}:
                items_to_keep.append(item)

        # Comment translated to English.
        # Comment translated to English.
        data.clear()
        data.extend(items_to_keep)

    return data
