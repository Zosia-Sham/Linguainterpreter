from __future__ import annotations

import json
import re
import ast
import subprocess
import sys
from typing import Any, List, Dict, Optional, Union
from langchain_core.prompts import ChatPromptTemplate

from src.utils import _parse_yaml_tasks, YAMLParseError, shorten_string_middle, _bool_from_text
from src.llm_utils import invoke_with_tools, invoke_and_log
from src.prompts.templates import (
    DS_META_PLANNER_SYS,
    PROBLEM_SPEC_SYS, METRICS_RECOVER_FROM_STDOUT_SYS, CHECKER_CODE_SYS, TASK_COMPLEXITY_SYS,
    PERFORM_TASK_PYTHON_SYS, DATAPATH_SYS, DATAPATH_CONSISTENCY_CHECK_SYS, FINETUNE_CODE_SYS,
    CHECKS_GEN_SYS, TASKS_GEN_SYS, TASK_ORDERING_SYS,
    AGGREGATE_ANSWERS_SYS, CHECK_ANSWER_SYS, FIX_ANSWER_SYS,
    VERIFICATION_CODE_GEN_SYS, IMPLEMENT_CHANGES_SYS,
    LEAD_AGENT_SYS, LEAD_INCIDENT_MANAGER_SYS, ERROR_TRIAGE_SYS, IMPROVEMENT_TASKS_SYS,
    RUNTIME_OUTPUT_OK_SYS, EXECUTION_PREDICTOR_SYS, EXECUTION_WATCHER_SYS,
    REPLANNING_SYS, ARTIFACT_REVIEWER_SYS, IMPROVEMENT_REPLANNING_SYS, IMPROVER_HEAD_SYS,
    REACT_IMPROVER_META_PLANNER_SYS, REACT_ARTIFACTS_COLLECTOR_SYS,
    LOG_UPDATE_SYS, FINAL_METRIC_SELECTOR_SYS, REACT_PREEXEC_AUDITOR_SYS, # NEW
    SUBMISSION_SANITY_SYS,
    CURATOR_SYS,
)

# NEW: Meta-Planner Agent
def meta_planner_agent(
    llm_strong,
    task: str,
    spec: Optional[Dict[str, Any]],
    *,
    max_stages: int = 7,
    remaining_sec: int = 0,
    total_budget_sec: int = 0,
    constraints_block: str = "",
) -> List[Any]:
    """
    Generates a high-level, skeletal plan for a data science project.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", DS_META_PLANNER_SYS),
        ("user",
         "{constraints_block}"
         "TASK: {task}\n\nSPEC:\n{spec}\n\n"
         "PLANNING CONSTRAINTS:\n"
         "- MAX_STAGES: {max_stages}  (hard ceiling — output at most this many YAML list items)\n"
         "- REMAINING_TIME_SEC: {remaining_sec}\n"
         "- TOTAL_BUDGET_SEC: {total_budget_sec}\n")
    ])
    res = invoke_and_log(llm_strong, prompt, {
        "constraints_block": constraints_block or "",
        "task": task,
        "spec": json.dumps(spec, indent=2),
        "max_stages": max(1, int(max_stages)),
        "remaining_sec": max(0, int(remaining_sec)),
        "total_budget_sec": max(0, int(total_budget_sec)),
    })
    return _parse_yaml_tasks(_strip_think(getattr(res, "content", "")))


def react_improver_meta_planner_agent(
    llm_strong,
    orch: Any,
    *,
    task: str,
    spec: Dict[str, Any],
    metrics_summary: Dict[str, Any] | str,
    recent_summaries: str,
    depth: int,
    max_depth: int,
    remaining_improve_sec: int,
    artifacts_hint: str = "",
    graph_hint: str = "",
    attempt_idx: int = 0,
) -> Dict[str, Any]:
    """
    ReAct meta-planner for the improver loop.
    Returns a JSON dict matching the contract in `REACT_IMPROVER_META_PLANNER_SYS`.
    """
    from .parsers import extract_json

    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:
        StructuredTool = None

    if StructuredTool is None:
        return {
            "meta_summary": "react_meta_planner skipped: StructuredTool unavailable",
            "metrics_findings": [],
            "bottlenecks": [],
            "high_level_plan": [],
            "next_investigation_order": [],
            "anti_patterns": [],
            "report_markdown": "StructuredTool unavailable; no meta-plan produced.",
        }

    # Tool wrappers must be read-focused and avoid obvious destructive shell operations.
    def _is_unsafe_command(cmd: str) -> bool:
        s = (cmd or "").lower()
        blocked = [
            " rm ", " rm -", " del ", " rmdir ", "mkdir -p --", "format ", "mkfs ",
            "shutdown", "reboot", "poweroff",
            "git reset", "git clean", "hg purge",
            "remove-item", "removeitem", "new-item",  # allow new-item? keep conservative: block
        ]
        # Allow listing/reading even if user writes e.g. "dir /b"
        if any(x in s for x in blocked):
            return True
        return False

    def bash_exec(command: str, timeout_sec: int = 60) -> str:
        """Execute a shell command (read-only when possible) and return stdout/stderr."""
        cmd = (command or "").strip()
        if not cmd:
            return "error: empty command"
        if _is_unsafe_command(cmd):
            return "error: blocked unsafe command"
        timeout_sec = int(timeout_sec or 60)
        timeout_sec = max(5, min(timeout_sec, 300))
        res = orch.bash.run(cmd, timeout=timeout_sec, stream=False)
        stdout = (res.get("stdout", "") or "")[:8000]
        stderr = (res.get("stderr", "") or "")[:4000]
        ec = res.get("exit_code", 1)
        return f"exit_code={ec}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    def python_exec(code: str, timeout_sec: int = 60) -> str:
        """Run a short python snippet via orch.run_python_code and return stdout/stderr."""
        c = (code or "").strip()
        if not c:
            return "error: empty python code"

        # Heuristic safety: block obvious destructive ops and file writes.
        low = c.lower()
        forbidden = [
            "shutil.rmtree", "os.remove", "os.unlink", "pathlibpath", ".unlink(",
            "subprocess.", "requests.", "urllib.",
        ]
        if any(x in low for x in forbidden):
            return "error: blocked unsafe python operations"
        # Block explicit write modes when present.
        if "open(" in low and ("'w'" in low or '"w"' in low or "mode='w'" in low or "mode=\"w\"" in low):
            return "error: blocked python write mode"

        timeout_sec = int(timeout_sec or 60)
        timeout_sec = max(5, min(timeout_sec, 120))
        fname = f"react_meta_{int(attempt_idx)}.py"
        res = orch.run_python_code(c, filename=fname, timeout=timeout_sec)
        out = (res.get("output", "") or "")[:8000]
        err = (res.get("errors", "") or "")[:4000]
        ec = res.get("exit_code", 1)
        return f"exit_code={ec}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

    tools = [
        StructuredTool.from_function(
            bash_exec,
            name="bash_exec",
            description="Read-only oriented shell execution. Use for ls/dir and reading small files (type/Get-Content).",
        ),
        StructuredTool.from_function(
            python_exec,
            name="python_exec",
            description="Short python snippet execution. Use for parsing JSON/CSV and computing small aggregates.",
        ),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REACT_IMPROVER_META_PLANNER_SYS),
            (
                "user",
                "TASK:\n{task}\n\n"
                "SPEC:\n{spec_json}\n\n"
                "CURRENT_METRICS_SUMMARY (may be partial):\n{metrics_summary}\n\n"
                "RECENT_EXECUTION_SUMMARIES:\n{recent_summaries}\n\n"
                "TREE_CONTEXT: depth={depth} / max_depth={max_depth}\n"
                "REMAINING_IMPROVE_BUDGET_SEC: {remaining_improve_sec}\n\n"
                "GRAPH_HINT (optional):\n{graph_hint}\n\n"
                "ARTIFACTS_HINT (optional):\n{artifacts_hint}\n\n"
                "Now produce the meta-plan JSON (contract keys only).",
            ),
        ]
    )

    metrics_payload = metrics_summary
    if isinstance(metrics_summary, dict):
        metrics_payload = json.dumps(metrics_summary, ensure_ascii=False, indent=2)

    res = invoke_with_tools(
        llm_strong,
        prompt,
        {
            "task": task,
            "spec_json": json.dumps(spec or {}, ensure_ascii=False, indent=2),
            "metrics_summary": str(metrics_payload),
            "recent_summaries": recent_summaries or "",
            "depth": int(depth or 0),
            "max_depth": int(max_depth or 0),
            "remaining_improve_sec": int(remaining_improve_sec or 0),
            "graph_hint": graph_hint or "",
            "artifacts_hint": artifacts_hint or "",
            "attempt_idx": int(attempt_idx or 0),
        },
        tools=tools,
        agent_name="react_improver_meta_planner",
    )

    obj = extract_json(getattr(res, "content", "") or "")
    if not isinstance(obj, dict):
        return {
            "meta_summary": "meta-planner fallback: could not parse JSON",
            "metrics_findings": [],
            "bottlenecks": [],
            "high_level_plan": [
                {
                    "task": "Execute direct artifact-driven recovery to produce metrics and submission",
                    "time_budget_sec": max(0, int(remaining_improve_sec)),
                    "rationale": "Fallback meta-plan when parsed JSON is invalid.",
                    "deep_tasks": [
                        {
                            "task": "Inspect artifacts/checkpoints/preds, run best available model or fast baseline, and write canonical submission.csv plus METRICS_JSON",
                            "time_budget_sec": max(120, min(1200, int(remaining_improve_sec))),
                            "acceptance_checks": [
                                "artifacts/metrics.json exists",
                                "canonical submission.csv exists and is non-empty",
                            ],
                        }
                    ],
                }
            ],
            "next_investigation_order": [0],
            "anti_patterns": ["planning-only loops", "duplication of already-tried tasks"],
            "report_markdown": "Fallback meta-plan: JSON parsing failed.",
        }

    # Minimal coercion to contract types.
    if "high_level_plan" in obj and not isinstance(obj["high_level_plan"], list):
        obj["high_level_plan"] = []
    if "next_investigation_order" in obj and not isinstance(obj["next_investigation_order"], list):
        obj["next_investigation_order"] = []
    if "metrics_findings" in obj and not isinstance(obj["metrics_findings"], list):
        obj["metrics_findings"] = []
    if "bottlenecks" in obj and not isinstance(obj["bottlenecks"], list):
        obj["bottlenecks"] = []
    if "anti_patterns" in obj and not isinstance(obj["anti_patterns"], list):
        obj["anti_patterns"] = []
    if "report_markdown" not in obj:
        obj["report_markdown"] = ""

    return obj

# NEW: Secretary Agent
def log_update_agent(llm_fast, task: str, status: str, summary: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", LOG_UPDATE_SYS),
        ("user", "TASK: {task}\nSTATUS: {status}\nSUMMARY:\n{summary}")
    ])
    res = invoke_and_log(llm_fast, prompt, {
        "task": task,
        "status": status,
        "summary": summary
    })
    return getattr(res, "content", "")

# NEW: Archivist Agent
def artifact_reviewer_agent(llm_fast, file_path: str, file_content_preview: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ARTIFACT_REVIEWER_SYS),
        ("user", "FILE_PATH: {file_path}\n\nFILE_PREVIEW:\n{preview}")
    ])
    res = invoke_and_log(llm_fast, prompt, {
        "file_path": file_path,
        "preview": file_content_preview
    })
    return getattr(res, "content", "")


def replanning_agent(
    llm_strong,
    task: str,
    project_log: str,
    remaining_tasks: List[Any],
    remaining_time: int,
    *,
    max_tail_tasks: int | None = None,
    extra_budget_sec: int = 0,
) -> Dict[str, Any]:
    cap = max_tail_tasks if max_tail_tasks is not None else len(remaining_tasks)
    cap = max(0, int(cap))
    # Relaxation policy: when extra_budget is substantial relative to remaining time,
    # allow the agent to add exploratory tasks beyond HARD_CAP_REMAINING.
    relax = False
    relax_cap = cap
    try:
        _ratio = float(extra_budget_sec) / max(1.0, float(remaining_time))
    except Exception:
        _ratio = 0.0
    if extra_budget_sec >= 600 and _ratio >= 0.2:
        relax = True
        relax_cap = cap + max(2, int(extra_budget_sec // 1800))  # +1 task per ~30min saved, min +2
    extra_line = ""
    if extra_budget_sec > 0:
        extra_line = (
            f"\nEXTRA_BUDGET_SEC: {extra_budget_sec}\n"
            "(Tasks finished early and saved this many seconds. You can distribute this to remaining tasks — spend it all, don't be conservative.)\n"
        )
    relax_line = (
        f"BUDGET_RELAXATION_ALLOWED: {'true' if relax else 'false'}\n"
        f"MAX_NEW_TASKS_IF_RELAXED: {relax_cap}\n"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPLANNING_SYS),
        ("user",
         "ORIGINAL TASK:\n{task}\n\nPROJECT LOG:\n{log}\n\nREMAINING TASKS:\n{remaining}\n\n"
         "REMAINING TOTAL TIME BUDGET (sec): {time}\n"
         "{extra_budget}"
         "HARD_CAP_REMAINING: {cap}\n"
         "{relax_block}")
    ])
    from .parsers import extract_json
    res = invoke_and_log(llm_strong, prompt, {
        "task": task,
        "log": project_log,
        "remaining": json.dumps(remaining_tasks, ensure_ascii=False, indent=2),
        "time": remaining_time,
        "extra_budget": extra_line,
        "cap": cap,
        "relax_block": relax_line,
    })
    return extract_json(getattr(res, "content", "")) or {"updated_remaining_tasks": remaining_tasks, "reasoning": "Fallback"}

def review_artifacts_agent(llm_strong, task: str, metrics: Dict[str, Any], code: str, artifacts_summary: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ARTIFACT_REVIEWER_SYS),
        ("user", "ORIGINAL TASK:\n{task}\n\nCURRENT METRICS:\n{metrics}\n\nLATEST CODE:\n{code}\n\nARTIFACTS SNAPSHOT:\n{artifacts}")
    ])
    res = invoke_and_log(llm_strong, prompt, {
        "task": task,
        "metrics": json.dumps(metrics, ensure_ascii=False, indent=2),
        "code": shorten_string_middle(code, 15000),
        "artifacts": artifacts_summary
    })
    return _strip_think(getattr(res, "content", ""))

def improvement_replanning_agent(
    llm_strong,
    task: str,
    project_log: str,
    remaining_tasks: List[Any],
    depth: int,
    max_depth: int,
    remaining_time: int = 3600,
    head_notes: str = "",
    *,
    max_tail_tasks: int | None = None,
    extra_budget_sec: int = 0,
) -> Dict[str, Any]:
    cap = max_tail_tasks if max_tail_tasks is not None else len(remaining_tasks)
    cap = max(0, int(cap))
    extra_line = ""
    if extra_budget_sec > 0:
        extra_line = (
            f"\nEXTRA_BUDGET_SEC: {extra_budget_sec}\n"
            "(Saved from early-finishing tasks. Spend it all on remaining tasks — don't be conservative.)\n"
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVEMENT_REPLANNING_SYS),
        ("user", "ORIGINAL TASK:\n{task}\n\nPROJECT LOG:\n{log}\n\nREMAINING TASKS:\n{remaining}\n\n"
                 "ITERATION DEPTH: {depth} / {max_depth}\n\nREMAINING_IMPROVE_TIME_SEC: {remaining_time}\n\n"
                 "{extra_budget}"
                 "HARD_CAP_REMAINING: {cap}\n\n"
                 "IMPROVER_HEAD_NOTES:\n{head_notes}\n")
    ])
    from .parsers import extract_json
    res = invoke_and_log(llm_strong, prompt, {
        "task": task,
        "log": project_log,
        "remaining": json.dumps(remaining_tasks, ensure_ascii=False, indent=2),
        "depth": depth,
        "max_depth": max_depth,
        "remaining_time": remaining_time,
        "extra_budget": extra_line,
        "cap": cap,
        "head_notes": head_notes or "(none)",
    })
    return extract_json(getattr(res, "content", "")) or {"updated_remaining_tasks": remaining_tasks, "reasoning": "Fallback"}


def improver_head_agent(
    llm_fast,
    *,
    task: str,
    spec: Dict[str, Any],
    metrics_summary: str,
    recent_summaries: str,
    depth: int,
    max_depth: int,
    remaining_improve_sec: int,
    graph_hint: str = "",
    main_pipeline_artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # Build main pipeline context block
    main_ctx = ""
    if main_pipeline_artifacts:
        parts = []
        if main_pipeline_artifacts.get("best_metrics"):
            parts.append(f"best_primary: {main_pipeline_artifacts['best_metrics'].get('primary', 'unknown')}")
        if main_pipeline_artifacts.get("data_schema"):
            parts.append(f"data_schema: {json.dumps(main_pipeline_artifacts['data_schema'], ensure_ascii=False)[:2000]}")
        if main_pipeline_artifacts.get("version_history"):
            parts.append(f"version_history: {json.dumps(main_pipeline_artifacts['version_history'], ensure_ascii=False)[:1500]}")
        if parts:
            main_ctx = "\n\nMAIN_PIPELINE_CONTEXT:\n" + "\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVER_HEAD_SYS),
        ("user", "GOAL:\n{task}\n\nSPEC_PRIMARY:\n{spec_primary}\n\nCURRENT_METRICS:\n{metrics}\n\n"
                 "RECENT_SUMMARIES:\n{recent}\n\nDEPTH: {depth} / {max_depth}\n\n"
                 "REMAINING_IMPROVE_SEC: {rem}\n\nGRAPH_HINT:\n{hint}\n{main_ctx}")
    ])
    from .parsers import extract_json
    pm = spec.get("primary_metric", {}) if isinstance(spec, dict) else {}
    res = invoke_and_log(llm_fast, prompt, {
        "task": task,
        "spec_primary": json.dumps(pm, ensure_ascii=False),
        "metrics": metrics_summary[:6000],
        "recent": recent_summaries[:8000],
        "depth": depth,
        "max_depth": max_depth,
        "rem": remaining_improve_sec,
        "hint": graph_hint[:4000],
        "main_ctx": main_ctx,
    })
    return extract_json(getattr(res, "content", "")) or {
        "verdict": "continue",
        "stuck": False,
        "metric_trend": "unknown",
        "reasoning": "fallback",
        "notes_for_replanner": "",
    }


# ... utilities ...


def react_artifacts_collector_agent(
    llm_strong,
    orch: Any,
    *,
    initial_scan: Dict[str, Any],
    spec: Dict[str, Any],
    task: str,
    previous_iteration_context: Dict[str, Any] | None = None,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """
    ReAct agent that deeply analyzes project artifacts like a Kaggle Grandmaster.
    Receives initial_scan from _collect_main_pipeline_artifacts, then reads files itself
    to verify and enrich the context. Returns enriched artifacts + improvement suggestions.
    """
    from .parsers import extract_json

    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        StructuredTool = None

    if StructuredTool is None:
        # Fallback: return initial scan as-is
        return initial_scan

    def _is_unsafe_command(cmd: str) -> bool:
        s = (cmd or "").lower()
        blocked = [
            " rm ", " rm -", " del ", " rmdir ", "format ", "mkfs ",
            "shutdown", "reboot", "poweroff",
            "git reset", "git clean", "hg purge",
            "remove-item", "removeitem",
        ]
        return any(x in s for x in blocked)

    def bash_exec(command: str, timeout_sec: int = 60) -> str:
        """Execute a shell command (read-only) and return stdout/stderr."""
        cmd = (command or "").strip()
        if not cmd:
            return "error: empty command"
        if _is_unsafe_command(cmd):
            return "error: blocked unsafe command"
        timeout_sec = max(5, min(int(timeout_sec or 60), 300))
        res = orch.bash.run(cmd, timeout=timeout_sec, stream=False)
        stdout = (res.get("stdout", "") or "")[:8000]
        stderr = (res.get("stderr", "") or "")[:4000]
        ec = res.get("exit_code", 1)
        return f"exit_code={ec}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    def python_exec(code: str, timeout_sec: int = 60) -> str:
        """Run a Python snippet for data analysis (read-only). Use for parsing JSON/CSV, computing stats."""
        c = (code or "").strip()
        if not c:
            return "error: empty python code"
        low = c.lower()
        forbidden = ["shutil.rmtree", "os.remove", "os.unlink", "subprocess.", "requests.", "urllib."]
        if any(x in low for x in forbidden):
            return "error: blocked unsafe python operations"
        if "open(" in low and ("'w'" in low or '"w"' in low or "mode='w'" in low or 'mode="w"' in low):
            return "error: blocked python write mode"
        timeout_sec = max(5, min(int(timeout_sec or 60), 120))
        res = orch.run_python_code(c, filename="react_artifact_scan.py", timeout=timeout_sec)
        out = (res.get("output", "") or "")[:8000]
        err = (res.get("errors", "") or "")[:4000]
        ec = res.get("exit_code", 1)
        return f"exit_code={ec}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

    tools = [
        StructuredTool.from_function(
            bash_exec,
            name="bash_exec",
            description="Read-only shell execution. Use for ls/dir, reading files (type/cat), inspecting directories.",
        ),
        StructuredTool.from_function(
            python_exec,
            name="python_exec",
            description="Python snippet execution for data analysis. Use for parsing JSON/CSV, loading dataframes, computing statistics, inspecting model code.",
        ),
    ]

    # Build initial scan summary for the agent
    scan_parts = []
    for k, v in initial_scan.items():
        if isinstance(v, str):
            scan_parts.append(f"--- {k} ---\n{v[:2000]}")
        elif isinstance(v, (dict, list)):
            scan_parts.append(f"--- {k} ---\n{json.dumps(v, ensure_ascii=False)[:2000]}")
    initial_scan_text = "\n\n".join(scan_parts) if scan_parts else "(no initial scan data)"

    prev_ctx_text = ""
    if previous_iteration_context:
        prev_ctx_text = (
            "\n\nPREVIOUS_ITERATION_CONTEXT:\n"
            + json.dumps(previous_iteration_context, ensure_ascii=False)[:4000]
        )

    art_dir = str(orch.project_root / orch.cfg.paths.artifacts_dir) if hasattr(orch, 'project_root') else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", REACT_ARTIFACTS_COLLECTOR_SYS),
        ("user",
         "TASK:\n{task}\n\n"
         "ARTIFACTS_DIR: {art_dir}\n"
         "PROJECT_ROOT: {project_root}\n\n"
         "INITIAL_SCAN (basic file scan — verify and enrich this):\n{initial_scan}\n"
         "{prev_ctx}\n\n"
         "SPEC_PRIMARY_METRIC:\n{spec_primary}\n\n"
         "Now use your tools to deeply analyze the project. Read the actual files. "
         "Understand the data, the model, the score trajectory. "
         "Then return the enriched JSON analysis.")
    ])

    pm = spec.get("primary_metric", {}) if isinstance(spec, dict) else {}
    res = invoke_with_tools(
        llm_strong,
        prompt,
        {
            "task": task,
            "art_dir": art_dir,
            "project_root": str(getattr(orch, 'project_root', '')),
            "initial_scan": initial_scan_text,
            "prev_ctx": prev_ctx_text,
            "spec_primary": json.dumps(pm, ensure_ascii=False),
        },
        tools=tools,
        agent_name="react_artifacts_collector",
        max_steps=max_steps,
    )

    obj = extract_json(getattr(res, "content", "") or "")
    if not isinstance(obj, dict):
        # Fallback: merge initial scan with whatever text we got
        initial_scan["_react_analysis"] = (getattr(res, "content", "") or "")[:4000]
        return initial_scan

    # Merge: keep initial scan data, overlay with agent's enriched data
    merged = dict(initial_scan)
    merged.update(obj)
    return merged


def execution_predictor_agent(llm_fast, code_text: str, spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXECUTION_PREDICTOR_SYS),
        ("user", "SPEC (meta):\n{spec_meta}\n\nDATA_SIZE_HINT:\n{data_size_hint}\n\nCODE:\n{code}")
    ])
    from .parsers import extract_json
    safe_spec = spec or {}
    # Build data size hint from csv_summaries
    data_size_hint = ""
    meta = (safe_spec.get("data", {}) or {}).get("meta", {}) or {}
    csv_s = meta.get("csv_summaries", {})
    if isinstance(csv_s, dict):
        for name, info in csv_s.items():
            if isinstance(info, dict):
                cols = info.get("columns", [])
                dtypes = info.get("dtypes", {})
                data_size_hint += f"  {name}: {len(cols)} columns, dtypes={list(dtypes.values())[:5]}\n"
    if meta.get("total_files_seen"):
        data_size_hint += f"  total_files: {meta['total_files_seen']}\n"
    if not data_size_hint:
        data_size_hint = "(no data size info available)"
    res = invoke_and_log(llm_fast, prompt, {
        "spec_meta": json.dumps(meta, ensure_ascii=False),
        "data_size_hint": data_size_hint,
        "code": code_text
    })
    out = extract_json(getattr(res, "content", "")) or {}
    if not isinstance(out, dict):
        out = {}
    out.setdefault("expected_time_sec", 300)
    out.setdefault("task_kind", "other")
    out.setdefault("expected_cpu_load", "medium")
    out.setdefault("expected_gpu_load", "none")
    out.setdefault("resource_intensity", "medium")
    if "rationale" not in out and "reasoning" in out:
        out["rationale"] = out["reasoning"]
    out.setdefault("rationale", "Fallback prediction")
    return out

def execution_watcher_agent(llm_fast, ctx: Any, max_steps: int = 5) -> Dict[str, Any]:
    """
    ReAct-style watcher. Receives a ``WatcherCtx`` (see src/watcher_tools.py)
    with read-only tools for CPU/RAM/GPU/process/disk stats plus stdout/stderr
    tails, code excerpt, task text, and timing. It issues up to ``max_steps``
    tool calls, then emits FINAL JSON:

        {"status": "...", "action": "continue|kill|warn", "reason": "..."}

    Fast-path: if elapsed already exceeds effective_timeout + extra_budget,
    return overtime/kill immediately without spending a ReAct turn.
    """
    from .parsers import extract_json
    from .watcher_tools import call_tool, parse_args_line, TOOLS_HELP, WatcherToolError

    # Fast-path overtime check — no LLM needed.
    try:
        t = ctx.get_timing() or {}
    except Exception:
        t = {}
    elapsed = float(t.get("elapsed_sec", 0) or 0)
    eff_to = float(t.get("effective_timeout_sec", 0) or 0)
    extra = float(t.get("extra_budget_sec", 0) or 0)
    hard_budget = eff_to + extra
    if hard_budget > 0 and elapsed >= hard_budget:
        return {
            "status": "overtime",
            "action": "kill",
            "reason": (
                f"overtime: elapsed={int(elapsed)}s >= effective_timeout={int(eff_to)}s "
                f"+ extra_budget={int(extra)}s"
            ),
        }

    system_prompt = EXECUTION_WATCHER_SYS + "\n\n" + TOOLS_HELP + (
        "\n\nReAct protocol (exactly one of per step):\n"
        "  - Tool request:\n"
        "      THOUGHT: <one short sentence>\n"
        "      TOOL: <tool_name>\n"
        "      ARGS: <json or key=value or empty>\n"
        "  - Final decision — a line that starts with FINAL: followed by a JSON "
        "object {\"status\": \"...\", \"action\": \"continue|kill|warn\", "
        "\"reason\": \"short explanation\"}.\n"
        f"- You have at most {max_steps} tool calls. Decide as soon as evidence is clear."
    )

    user_template = (
        "You are watching a running subprocess. Use the tools to gather "
        "evidence, then decide continue/warn/kill.\n\n"
        "SCRATCHPAD (previous tool calls + observations):\n{scratchpad}\n\n"
        "Emit your next step now."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_template),
    ])

    scratch_lines: List[str] = []
    final_json: Optional[Dict[str, Any]] = None
    for step in range(1, max_steps + 1):
        try:
            res = invoke_and_log(
                llm_fast, prompt,
                {"scratchpad": "\n".join(scratch_lines) or "(empty)"},
                agent_name="execution_watcher_react",
            )
        except Exception as e:
            return {
                "status": "normal",
                "action": "continue",
                "reason": f"watcher LLM error: {e}",
            }
        text = _strip_think(getattr(res, "content", "") or "")
        parsed = _parse_react_reply(text)
        if parsed.get("kind") == "final":
            body = parsed.get("final", "")
            obj = extract_json(body)
            if isinstance(obj, dict):
                final_json = obj
            else:
                # Accept loose finals.
                final_json = {
                    "status": "normal",
                    "action": "continue",
                    "reason": (body or "")[:400] or "no json in final",
                }
            break
        tool = parsed.get("tool") or ""
        args_raw = parsed.get("args_raw", "")
        args = parse_args_line(args_raw)
        try:
            obs = call_tool(ctx, tool, args)
        except WatcherToolError as e:
            obs = f"[tool error] {e}"
        except Exception as e:
            obs = f"[tool error] {e}"
        # Keep scratchpad bounded.
        if len(obs) > 4000:
            obs = obs[:4000] + "\n... [obs truncated]"
        scratch_lines.append(
            f"STEP {step}\nTOOL: {tool}\nARGS: {args_raw}\nOBSERVATION:\n{obs}\n"
        )

    if final_json is None:
        # Out of steps — force a final pass.
        scratch_lines.append(
            "\nSTEP budget exhausted — emit FINAL with a JSON decision now.\n"
        )
        try:
            res = invoke_and_log(
                llm_fast, prompt,
                {"scratchpad": "\n".join(scratch_lines)},
                agent_name="execution_watcher_react_final",
            )
            text = _strip_think(getattr(res, "content", "") or "")
            parsed = _parse_react_reply(text)
            obj = extract_json(parsed.get("final", "") or text)
            if isinstance(obj, dict):
                final_json = obj
        except Exception as e:
            final_json = {
                "status": "normal",
                "action": "continue",
                "reason": f"watcher final error: {e}",
            }

    if not isinstance(final_json, dict):
        final_json = {
            "status": "normal",
            "action": "continue",
            "reason": "watcher returned no decision — default continue",
        }
    final_json.setdefault("status", "normal")
    final_json.setdefault("action", "continue")
    final_json.setdefault("reason", "")
    return final_json

# Comment translated to English.
def _normalize_secondary_metrics(sec: Any) -> List[str]:
    """Canonicalize secondary_metrics to list[str]."""
    raw: List[Any]
    if isinstance(sec, list):
        raw = sec
    elif isinstance(sec, str) and sec.strip():
        raw = [sec.strip()]
    else:
        raw = []

    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        name = ""
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        elif isinstance(item, str):
            s = item.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = json.loads(s)
                    if isinstance(d, dict):
                        name = str(d.get("name", "")).strip()
                except Exception:
                    try:
                        d = ast.literal_eval(s)
                        if isinstance(d, dict):
                            name = str(d.get("name", "")).strip()
                    except Exception:
                        name = s
            else:
                name = s
        else:
            name = str(item).strip()

        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _strip_think(text: Any) -> str:
    if isinstance(text, list):
        parts = []
        for p in text:
            if isinstance(p, dict) and 'text' in p:
                parts.append(str(p['text']))
            else:
                parts.append(str(p))
        text = "".join(parts)
    if isinstance(text, dict):
        text = str(text)

    if not text: return ""

    return (str(text).replace("<think>","").replace("</think>","")
                .replace("<THINK>","").replace("</THINK>","")).strip()

def _extract_code_block(text: str) -> str:
    text = _strip_think(text)
    # Use regex to find content between ```python and ```
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback for cases where markdown is missing but code is present
    if "```" not in text and ("import " in text or "def " in text):
        return text.strip()
        
    # Fallback for simple ```...``` blocks
    if text.count("```") >= 2:
        parts = text.split("```")
        # Return the largest block of text between ```
        return max(parts[1::2], key=len).strip()

    return text.strip() # Return the text as-is if no block is found

def extract_code(resp) -> str:
    return _extract_code_block(getattr(resp, "content", resp) or "")

def extract_boolean(resp) -> bool:
    t = _strip_think(getattr(resp, "content", resp) or "")
    if isinstance(t, list):
        t = ' '.join(t)
    t = t.lower()
    if "true" in t and "false" not in t: return True
    if "false" in t and "true" not in t: return False
    t0 = t.split()[0] if t.split() else t
    return t0 in ("true","yes","pass")

def extract_numbered_list(resp) -> List[str]:
    txt = _strip_think(getattr(resp, "content", resp) or "")
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    out = []
    for l in lines:
        if l[0].isdigit():
            i=0
            while i < len(l) and l[i].isdigit(): i+=1
            while i < len(l) and l[i] in ('.',')',' '): i+=1
            out.append(l[i:].strip())
        else:
            out.append(l)
    return out


def default_spec_skeleton() -> Dict[str, Any]:
    """Safe baseline when the LLM did not return JSON or spec generation raised."""
    return {
        "modalities": ["image"],
        "primary_metric": {"name": "f2_micro", "maximize": True},
        "secondary_metrics": ["precision", "recall", "f1_micro"],
        "submission": {"columns": ["id", "attribute_ids"], "delimiter": ","},
        "validation": {"strategy": "stratified_kfold", "n_splits": 5, "seed": 42},
        "ensemble_allowed": True,
        "single_stack_per_stage": True,
        "baseline_required": True,
        "constraints": {
            "internet_allowed": True,
            "pretrained_allowed": True,
            "external_data_allowed": True,
            "external_data_requires_tag": False,
            "notes": "",
        },
    }


# Comment translated to English.
def problem_spec_from_text(llm_strong, task_text: str, tools: Optional[List[Any]] = None) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROBLEM_SPEC_SYS),
        (
            "user",
            "TASK DESCRIPTION — extract modalities, metrics, validation, and **all competition constraints** "
            "(internet, pretrained weights, external data, kernels-only, train-from-scratch, etc.):\n\n{task}",
        ),
    ])
    res = invoke_with_tools(llm_strong, prompt, {"task": task_text}, tools=tools)

    from .parsers import extract_json
    spec = extract_json(getattr(res, "content", ""))
    if not spec:
        spec = default_spec_skeleton()
    # Self-heal mixed/invalid metric formats from model outputs.
    if not isinstance(spec, dict):
        spec = {}
    spec["secondary_metrics"] = _normalize_secondary_metrics(spec.get("secondary_metrics"))
    return spec


def metrics_recover_from_stdout(
    llm_fast,
    stdout: str,
    spec: Dict[str, Any],
    *,
    subtask: str = "",
    max_chars: int = 14000,
) -> Optional[Dict[str, Any]]:
    """
    LLM fallback when METRICS_JSON regex parsing fails. Uses stdout tail only.
    Returns a dict with type calculated|skipped or None if unparseable.
    """
    if not (stdout or "").strip():
        return None
    tail = stdout if len(stdout) <= max_chars else stdout[-max_chars:]
    spec_min = json.dumps(
        {
            "primary_metric": spec.get("primary_metric") or {},
            "constraints": spec.get("constraints") or {},
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", METRICS_RECOVER_FROM_STDOUT_SYS),
            (
                "user",
                "SPEC (primary_metric + constraints only):\n{spec_min}\n\n"
                "SUB_TASK:\n{subtask}\n\nSTDOUT_TAIL:\n{tail}\n",
            ),
        ]
    )
    res = invoke_and_log(
        llm_fast,
        prompt,
        {
            "spec_min": spec_min,
            "subtask": subtask or "(not provided)",
            "tail": tail,
        },
        agent_name="metrics_recover_from_stdout",
    )
    from .parsers import extract_json

    obj = extract_json(getattr(res, "content", "") or "")
    if not isinstance(obj, dict):
        return None
    t = obj.get("type")
    if t == "skipped":
        return obj
    if t == "calculated":
        return obj
    return None


def checker_code_agent(
    llm_fast,
    task: str,
    spec: Dict[str, Any],
    code_summary: str = "",
    *,
    final_answer: str = "",
    metrics_json: str = "",
    stdout_tail: str = "",
    stderr_tail: str = "",
    improvement_summary: str = "",
) -> str:
    spec_str = json.dumps(spec, ensure_ascii=False, indent=2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHECKER_CODE_SYS),
        ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nCODE SUMMARY:\n{code}\n\nFINAL ANSWER (report):\n{final_answer}\n\nCURRENT METRICS JSON (raw string):\n{metrics_json}\n\nSTDOUT TAIL:\n{stdout_tail}\n\nSTDERR TAIL:\n{stderr_tail}\n\nIMPROVEMENT SUMMARY:\n{improve_summary}\n")
    ])

    res = invoke_and_log(llm_fast, prompt, {
        "task": task,
        "spec": spec_str,
        "code": code_summary or "",
        "final_answer": final_answer or "",
        "metrics_json": metrics_json or "",
        "stdout_tail": stdout_tail or "",
        "stderr_tail": stderr_tail or "",
        "improve_summary": improvement_summary or "",
    })
    return extract_code(getattr(res, "content", ""))


# Comment translated to English.
def task_complexity_check(
    llm_fast,
    task: str,
    main_task: str,
    previously_task: str = None,
    tree_depth: int | None = None,
    *,
    remaining_total_sec: int | None = None,
    min_split_sec: int = 600,
    tree_max_depth: int | None = None,
) -> str:
    if remaining_total_sec is not None and int(remaining_total_sec) < int(min_split_sec):
        return "False"
    tree_level = tree_depth
    headroom: int | str = "unknown"
    if tree_max_depth is not None and tree_level is not None:
        try:
            headroom = max(0, int(tree_max_depth) - int(tree_level))
        except Exception:
            headroom = "unknown"
    if isinstance(headroom, int) and headroom <= 0:
        return "False"
    prompt = ChatPromptTemplate.from_messages([
        ("system", TASK_COMPLEXITY_SYS),
        ("user",
         "SUB - Task Description: {input}\nTREE_LEVEL: {tree_level}\n"
         "HEADROOM_LEVELS: {headroom}\n"
         "REMAINING_TOTAL_TIME_SEC: {remaining_total}\nMIN_SPLIT_SEC: {min_split_sec}\n"
         "Main Task - {main_task}\nPrevious task: {previously_task}")
    ])
    res = invoke_and_log(
        llm_fast,
        prompt,
        {
            "input": task,
            "tree_level": tree_level if tree_level is not None else "unknown",
            "headroom": headroom,
            "remaining_total": remaining_total_sec if remaining_total_sec is not None else "unknown",
            "min_split_sec": int(min_split_sec),
            "main_task": main_task,
            "previously_task": previously_task,
        },
    )
    return "True" if extract_boolean(res) else "False"


def perform_task_python_v2(code_llm, subtask: str, spec: Dict[str, Any], previous_code: str = "",
                           context: str = "", tools: Optional[List[Any]] = None,
                           orch: Any = None, schema_snapshot: str = "") -> str:
    effective_tools = list(tools or [])

    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:
        StructuredTool = None

    def _tool_names(seq: List[Any]) -> set[str]:
        names: set[str] = set()
        for t in seq:
            try:
                n = getattr(t, "name", None)
                if isinstance(n, str) and n:
                    names.add(n)
            except Exception:
                continue
        return names

    if StructuredTool is not None and "generate_and_execute" not in _tool_names(effective_tools):
        def generate_and_execute(python_code: str, timeout_sec: int = 20) -> str:
            """
            Run a short Python snippet for fast validation and return exit_code/stdout/stderr.
            Intended for quick hypothesis checks, not heavy training.
            """
            code = (python_code or "").strip()
            if not code:
                return "error: empty python_code"

            low = code.lower()
            forbidden = [
                "shutil.rmtree", "os.remove", "os.unlink", "pathlib.path.unlink",
                "subprocess.", "os.system(", "requests.", "urllib.",
                "open(", ".write_text(", ".write_bytes(",
            ]
            if any(x in low for x in forbidden):
                return "error: blocked potentially unsafe operation in python_code"

            try:
                compile(code, "<generate_and_execute>", "exec")
            except Exception as e:
                return f"exit_code=1\nSTDOUT:\n\nSTDERR:\nsyntax_error: {e}"

            t = max(5, min(int(timeout_sec or 20), 120))
            try:
                # Use venv python when available so validation matches actual execution environment.
                venv_py = None
                if orch is not None:
                    vpy = getattr(getattr(orch, "project", None), "vpy", None)
                    if vpy is not None and Path(str(vpy)).exists():
                        venv_py = str(vpy)
                exec_py = venv_py or sys.executable
                res = subprocess.run(
                    [exec_py, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=t,
                )
                stdout = (res.stdout or "")[:8000]
                stderr = (res.stderr or "")[:4000]
                return f"exit_code={res.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            except subprocess.TimeoutExpired:
                return f"exit_code=124\nSTDOUT:\n\nSTDERR:\ntimeout after {t}s"
            except Exception as e:
                return f"exit_code=1\nSTDOUT:\n\nSTDERR:\n{type(e).__name__}: {e}"

        effective_tools.append(
            StructuredTool.from_function(
                generate_and_execute,
                name="generate_and_execute",
                description=(
                    "Execute a short Python snippet for fast validation/recovery and return "
                    "exit_code/stdout/stderr. Use only lightweight checks."
                ),
            )
        )

    def _invoke_planner(_subtask: str, _spec: Dict[str, Any], _ctx: str, _prev: str) -> str:
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are PLANNER_AGENT in a multi-agent ReAct coding loop.\n"
             "Goal: produce an actionable plan for writing ONE executable Python script for the sub-task.\n"
             "You may use available tools aggressively (bash/python/file readers, generate_and_execute if available) "
             "to inspect code, artifacts, and markdown docs before planning.\n"
             "Return concise JSON with keys: intent, required_inputs, artifacts_to_read, execution_steps, risks, done_criteria.\n"
             "Do not output Python code."),
            ("user",
             "SUBTASK:\n{subtask}\n\nSPEC:\n{spec}\n\nPROJECT CONTEXT:\n{ctx}\n\nPREVIOUS CODE:\n{prev}")
        ])
        res = invoke_with_tools(code_llm, planner_prompt, {
            "subtask": _subtask,
            "spec": _spec or {},
            "ctx": _ctx or "N/A",
            "prev": _prev or "N/A",
        }, tools=effective_tools, agent_name="planner_agent")
        return _strip_think(getattr(res, "content", "") or "")

    def _invoke_reviewer(_code: str, _subtask: str, _spec: Dict[str, Any], _ctx: str) -> str:
        reviewer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are REVIEWER_AGENT in a multi-agent ReAct coding loop.\n"
             "Audit the proposed script for: executable correctness, metrics signaling contract, "
             "submission contract, anti-hardcode spec policy, and path safety.\n"
             "Use tools when needed to verify against actual files/docs/spec.\n"
             # Literal braces must be doubled for LangChain (else {\"pass\"} is parsed as a template var).
             "Return ONLY JSON: {{\"pass\": true|false, \"issues\": [\"...\"], \"must_fix\": [\"...\"]}}."),
            ("user",
             "SUBTASK:\n{subtask}\n\nSPEC:\n{spec}\n\nCONTEXT:\n{ctx}\n\nCANDIDATE_CODE:\n```python\n{code}\n```")
        ])
        res = invoke_with_tools(code_llm, reviewer_prompt, {
            "subtask": _subtask,
            "spec": _spec or {},
            "ctx": _ctx or "N/A",
            "code": _code or "",
        }, tools=effective_tools, agent_name="reviewer_agent")
        return _strip_think(getattr(res, "content", "") or "")

    max_retries = 3
    try:
        planner_notes = _invoke_planner(subtask, spec, context, previous_code)
    except Exception as e:
        print(f"[PERFORM_TASK] planner_agent LLM exhausted: {type(e).__name__}: {e}")
        planner_notes = "{}"

    for attempt in range(max_retries):
        prompt = ChatPromptTemplate.from_messages([
            ("system", PERFORM_TASK_PYTHON_SYS),
            ("user", "PROJECT LOG:\n{ctx}\n\nSPEC JSON:\n{spec}\n\nDATA SCHEMA (authoritative — use these exact column names / file structures):\n{schema}\n\nSUB-TASK:\n{subtask}\n\nPREVIOUS CODE (if any):\n{prev}\n\nPLANNER_AGENT OUTPUT:\n{plan}\n\n"
                     "MULTI-AGENT EXECUTION POLICY:\n"
                     "- You are CODER_AGENT.\n"
                     "- If tools are available, use them to inspect relevant source files and markdown docs before finalizing code.\n"
                     "- Before any merge/join/groupby on tabular data or loading of non-tabular artifacts, consult DATA SCHEMA above. If it is missing or ambiguous for a file you need, call `inspect_artifact` tool.\n"
                     "- Never assume a column (e.g. `TeamID`) exists — the winner/loser convention uses `WTeamID`/`LTeamID`. For images/audio/text check probe output (folder classes, sample shapes) via `inspect_artifact` before coding the loader.\n"
                     "- If `generate_and_execute` tool exists, you may call it for fast validation/recovery.\n"
                     "- Produce one final standalone script only.")
        ])
        try:
            res = invoke_with_tools(code_llm, prompt, {
                "spec": spec,
                "subtask": subtask,
                "prev": previous_code or "N/A",
                "ctx": context or "N/A",
                "plan": planner_notes or "{}",
                "schema": schema_snapshot or "N/A",
            }, tools=effective_tools, agent_name="coder_agent")
        except Exception as e:
            print(f"[PERFORM_TASK] coder_agent LLM exhausted (attempt {attempt+1}): {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                continue
            return previous_code or ""

        generated_code = extract_code(res)
        review_payload = _invoke_reviewer(generated_code, subtask, spec, context)
        review_ok = False
        review_text = str(review_payload or "")
        review_low = review_text.lower()
        if "\"pass\": true" in review_low or "'pass': true" in review_low or "\"pass\":true" in review_low:
            review_ok = True

        try:
            # NEW: Syntax validation loop
            compile(generated_code, '<string>', 'exec')
            if review_ok:
                return generated_code  # Return code if it is valid and reviewer approved
            context += (
                "\n\n[REVIEWER FEEDBACK] Reviewer did not approve current script. "
                "Apply mandatory fixes and regenerate full executable code.\n"
                + review_text
            )
            previous_code = generated_code
        except SyntaxError as e:
            print(f"Syntax error in generated code (attempt {attempt + 1}): {e}")
            # Add the error to the context and retry
            context += f"\n\n[SYNTAX ERROR] The previous code failed with a syntax error: {e}. Please fix the syntax and provide the complete, correct script."
            previous_code = generated_code # Use the broken code as the base for the next attempt
            if attempt == max_retries - 1:
                print("Max retries for syntax correction reached. Returning last broken code.")
                return generated_code # Return the broken code after last attempt

    return "" # Should not be reached

def datapath_agent(llm_fast, task_text: str, filetree: str, os_name: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", DATAPATH_SYS),
        ("user", "OS: {os}\n\nTASK:\n{task}\n\nFILETREE:\n{tree}")
    ])
    res = invoke_and_log(llm_fast, prompt, {"os": os_name, "task": task_text, "tree": filetree})
    return getattr(res, "content", "")


def datapath_consistency_check_agent(llm_fast, proposed_data: Dict[str, Any], filetree: str) -> Dict[str, Any]:
    """
    Fast LLM validator to prevent FILETREE->spec.data hallucinations (e.g., labels_csv).
    """
    from .parsers import extract_json

    prompt = ChatPromptTemplate.from_messages([
        ("system", DATAPATH_CONSISTENCY_CHECK_SYS),
        ("user", "FILETREE:\n{tree}\n\nPROPOSED_DATA:\n{proposed}"),
    ])
    res = invoke_and_log(
        llm_fast,
        prompt,
        {
            "tree": filetree,
            "proposed": json.dumps(proposed_data or {}, ensure_ascii=False, indent=2),
        },
    )
    content = getattr(res, "content", "") or ""
    parsed = extract_json(content) if isinstance(content, str) else {}
    if not isinstance(parsed, dict):
        return {"ok": False, "data": proposed_data or {}, "reason": "Failed to parse checker output"}
    return parsed

def finetune_code_v2(code_llm, task: str, code: str, spec: Dict[str,Any], error: str = "There is no error", tools: Optional[List[Any]] = None) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", FINETUNE_CODE_SYS),
        ("user", "SPEC:\n{spec}\n\nTASK:\n{task}\n\nCODE:\n{code}\n\nLAST ERROR:\n{error}")
    ])
    res = invoke_with_tools(code_llm, prompt, {"spec": spec, "task": task, "code": code, "error": error}, tools=tools)
    return extract_code(res)

def checks_generation(llm_fast, task: str, spec: Dict[str,Any]) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHECKS_GEN_SYS),
        ("user", "TASK:\n{task}\n\nSPEC:\n{spec}")
    ])
    res = invoke_and_log(llm_fast, prompt, {"task": task, "spec": spec})
    return _strip_think(getattr(res,"content",""))


def tasks_generation(
    llm_strong,
    task: str,
    spec: Optional[Dict[str, Any]],
    project_log: str = "",
    tasks_history: str = "",
    extra_context: str = "",
    *,
    remaining_total_sec: int | None = None,
    total_budget_sec: int | None = None,
    min_split_sec: int = 600,
    constraints_block: str = "",
) -> List[str]:
    time_block = ""
    if remaining_total_sec is not None:
        time_block = (
            f"\nREMAINING_TOTAL_TIME_SEC: {int(remaining_total_sec)}\n"
            f"TOTAL_BUDGET_SEC: {int(total_budget_sec) if total_budget_sec is not None else 'unknown'}\n"
            f"MIN_SPLIT_SEC: {int(min_split_sec)}\n"
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", TASKS_GEN_SYS),
        ("user", "{constraints_block}DECOMPOSE THIS CURRENT TASK:\n{task}\n\nPROJECT SPEC:\n{spec}\n\nPROJECT LOG:\n{log}\n\nTASKS HISTORY (DO NOT DUPLICATE):\n{tasks_history}\n\nADDITIONAL CONTEXT:\n{extra_context}{time_block}\n\nAnalyze scope, then generate subtasks following the rules.\nOutput ONLY YAML."),
    ])

    res = invoke_and_log(llm_strong, prompt, {
        "constraints_block": constraints_block or "",
        "task": task,
        "spec": spec,
        "log": project_log,
        "tasks_history": tasks_history,
        "extra_context": extra_context,
        "time_block": time_block,
    })

    return _parse_yaml_tasks(_strip_think(getattr(res, "content", "")))

def task_ordering(
    llm_strong,
    task: str,
    sub_tasks: Union[List[str], str],
    spec: Optional[Dict[str, Any]],
    *,
    overall_time_limit_sec: int | None = None,
    constraints_block: str = "",
) -> List[str]:
    ot = int(overall_time_limit_sec) if overall_time_limit_sec is not None else 0
    prompt = ChatPromptTemplate.from_messages([
        ("system", TASK_ORDERING_SYS),
        ("user", "{constraints_block}TASK:\n{task}\n\nSPEC:\n{spec}\n\nOVERALL_TIME_LIMIT_SEC:\n{overall}\n\nSUBTASKS:\n{sub_tasks}")
    ])
    res = invoke_and_log(llm_strong, prompt, {
        "constraints_block": constraints_block or "",
        "task": task,
        "spec": spec,
        "overall": ot,
        "sub_tasks": sub_tasks,
    })
    return _parse_yaml_tasks(_strip_think(getattr(res,"content","")))

def aggregate_answers(
    llm_fast,
    task: str,
    project_log: str,
    spec: Dict[str, Any],
    *,
    verified_artifacts: Optional[List[str]] = None,
    claimed_but_missing: Optional[List[str]] = None,
) -> str:
    """
    Generate the cross-task summary report.

    ``verified_artifacts`` lists paths that the pipeline has confirmed exist
    on disk. ``claimed_but_missing`` lists paths that appeared in stdout as
    "saved to X" but were never actually written. The agent is instructed
    (in AGGREGATE_ANSWERS_SYS) to treat verified_artifacts as the SINGLE
    source of truth for the "Stack & Artifacts" section and to flag the
    missing ones as failures in section 4.
    """
    verified_block = "\n".join(f"- {p}" for p in (verified_artifacts or [])) or "(none)"
    missing_block = "\n".join(f"- {p}" for p in (claimed_but_missing or [])) or "(none)"
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATE_ANSWERS_SYS),
        ("user",
         "TASK:\n{task}\n\n"
         "SPEC:\n{spec}\n\n"
         "VERIFIED_ARTIFACTS (exist on disk — use ONLY these when claiming saved files):\n{verified}\n\n"
         "CLAIMED_BUT_MISSING (mentioned in stdout as saved but NOT on disk — report as failures):\n{missing}\n\n"
         "PROJECT LOG:\n{log}")
    ])
    res = invoke_and_log(llm_fast, prompt, {
        "task": task,
        "spec": spec,
        "verified": verified_block,
        "missing": missing_block,
        "log": project_log,
    })
    return _strip_think(getattr(res, "content", ""))

def check_answer(llm_fast, task: str, answer: str, check: str, spec: Dict[str,Any]) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", CHECK_ANSWER_SYS),
        ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nANSWER:\n{ans}\n\nCHECK:\n{chk}")
    ])
    res = invoke_and_log(llm_fast, prompt, {"task":task,"spec":spec,"ans":answer,"chk":check})
    return "True" if extract_boolean(res) else "False"

def fix_answer(llm_fast, task: str, answer: str, check: str, spec: Dict[str,Any]) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", FIX_ANSWER_SYS),
        ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nANSWER:\n{ans}\n\nFAILED CHECK:\n{chk}")
    ])
    res = invoke_and_log(llm_fast, prompt, {"task":task,"spec":spec,"ans":answer,"chk":check})
    return extract_code(res)

def verification_code_gen(code_llm, spec: Optional[Dict[str,Any]], context: str = "") -> str:
    # Generate verifier code that ALWAYS loads spec dynamically from artifacts/spec.json.
    # This avoids anti-hardcode violations in downstream checkers.
    prompt = ChatPromptTemplate.from_messages([
        ("system", VERIFICATION_CODE_GEN_SYS),
        ("user", "CONTEXT (what's already run / file names):\n{ctx}\n\n"
                 "MANDATORY:\n"
                 "1) Do NOT embed SPEC as dict/JSON literal in code.\n"
                 "2) Load spec ONLY from ./artifacts/spec.json (fallback: ../artifacts/spec.json).\n"
                 "3) Print exactly one line marker: "
                 "SPEC_SOURCE_JSON: {{\"spec_path\":\"<path>\",\"loaded\":true|false}}\n"
                 "4) Then compute and print METRICS_JSON.\n")
    ])
    res = invoke_and_log(code_llm, prompt, {"ctx": context or "N/A"})
    generated_code = extract_code(res)

    # Post-process: inject robust dynamic spec loader if model omitted it.
    required_probe = "SPEC_SOURCE_JSON"
    if required_probe not in generated_code or "artifacts/spec.json" not in generated_code:
        header = (
            "import json\n"
            "from pathlib import Path\n"
            "root = Path('.').resolve()\n"
            "candidates = [root / 'artifacts' / 'spec.json', root.parent / 'artifacts' / 'spec.json']\n"
            "spec = {}\n"
            "spec_path = ''\n"
            "for p in candidates:\n"
            "    if p.exists():\n"
            "        spec_path = str(p)\n"
            "        try:\n"
            "            spec = json.loads(p.read_text(encoding='utf-8'))\n"
            "        except Exception:\n"
            "            spec = {}\n"
            "        break\n"
            "print('SPEC_SOURCE_JSON: ' + json.dumps({'spec_path': spec_path, 'loaded': bool(spec)}, ensure_ascii=False))\n"
        )
        generated_code = header + "\n" + generated_code

    return generated_code

def implement_changes_agent(llm, suggestions: str, original_code: str, task: str, spec: Dict[str, Any]) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPLEMENT_CHANGES_SYS),
        ("user", "TECH LEAD SUGGESTIONS:\n---\n{suggestions}\n---\n\nORIGINAL CODE TO MODIFY:\n---\n```python\n{original_code}\n```\n---\n\nORIGINAL TASK (for context): {task}\nSPECIFICATION (for context): {spec}\n\nNow, provide the complete, corrected Python code.")
    ])
    res = invoke_and_log(llm, prompt, {
        "suggestions": suggestions,
        "original_code": original_code,
        "task": task,
        "spec": spec
    })
    return extract_code(res)

def _lead_react_step(llm, system_prompt: str, scratchpad: str, user_context: Dict[str, Any]) -> str:
    """
    Single ReAct turn for the Lead Agent. The model either emits
    ``TOOL: <name>`` / ``ARGS: <...>`` to request a tool call, or
    ``FINAL:`` followed by the Problem Analysis + Proposed Solution.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user",
         "LEAD REASON:\n{lead_reason}\n\n"
         "ORIGINAL TASK:\n{task}\n\n"
         "SPECIFICATION:\n{spec}\n\n"
         "LAST EXECUTION OUTPUT:\n{last_stdout}\n\n"
         "CURRENT CODE TO BE FIXED:\n```python\n{code}\n```\n\n"
         "SCRATCHPAD (previous tool calls + observations):\n{scratchpad}\n\n"
         "Decide your next step. Either:\n"
         "  - Request ONE tool call in the exact format:\n"
         "      THOUGHT: <one short sentence>\n"
         "      TOOL: <tool_name>\n"
         "      ARGS: <json or key=value>\n"
         "  - OR emit the final answer starting with a line that begins with FINAL:\n"
         "    followed by the Problem Analysis and Proposed Solution.\n"),
    ])
    res = invoke_and_log(llm, prompt, {
        "lead_reason": user_context.get("lead_reason", ""),
        "task": user_context.get("task", ""),
        "spec": user_context.get("spec", ""),
        "last_stdout": user_context.get("last_stdout", ""),
        "code": user_context.get("code", ""),
        "scratchpad": scratchpad or "(empty)",
    })
    return _strip_think(getattr(res, "content", ""))


def _parse_react_reply(text: str) -> Dict[str, Any]:
    """Parse a single ReAct reply into {'kind', 'tool', 'args', 'final'}."""
    if not text:
        return {"kind": "final", "final": ""}
    # FINAL marker can appear anywhere in the first ~5 lines.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("FINAL:"):
            final_body = "\n".join(lines[i:])
            # Strip the leading "FINAL:" token from the first line.
            first = final_body.split("\n", 1)
            head = first[0]
            head = head[head.upper().index("FINAL:") + len("FINAL:"):].lstrip()
            rest = first[1] if len(first) > 1 else ""
            body = (head + ("\n" + rest if rest else "")).strip()
            return {"kind": "final", "final": body}
    tool_name = None
    args_raw = ""
    for line in lines:
        s = line.strip()
        if s.upper().startswith("TOOL:"):
            tool_name = s.split(":", 1)[1].strip()
        elif s.upper().startswith("ARGS:"):
            args_raw = s.split(":", 1)[1].strip()
    if tool_name:
        return {"kind": "tool", "tool": tool_name, "args_raw": args_raw}
    # No structured markers — treat the whole reply as a final answer.
    return {"kind": "final", "final": text.strip()}


def lead_agent_propose_changes_react(
    llm,
    *,
    orch: Any,
    lead_reason: str,
    task: str,
    spec: Dict[str, Any],
    code: str,
    last_stdout: str,
    max_steps: int = 6,
) -> str:
    """
    ReAct version of the Lead Agent. Gives the model access to a small set of
    read-only filesystem tools (see src/lead_tools.py) so it can verify
    claimed artifacts before proposing changes. The FINAL output format
    (Problem Analysis + Proposed Solution) is unchanged so downstream
    ``implement_changes_agent`` keeps working without modification.
    """
    from src.lead_tools import call_tool, parse_args_line, TOOLS_HELP, LeadToolError

    system_prompt = LEAD_AGENT_SYS + "\n\n" + TOOLS_HELP + (
        "\n\nReAct protocol:\n"
        "- On each step, emit either a tool request (THOUGHT/TOOL/ARGS) "
        "or a FINAL answer.\n"
        "- Verify assumptions about filesystem state (does this file exist? "
        "what columns does it have?) BEFORE proposing code changes.\n"
        "- When you have enough evidence, reply with FINAL: followed by the "
        "required Problem Analysis + Proposed Solution — same format as "
        "before.\n"
        "- You have at most {max_steps} tool calls. Budget them carefully."
    ).replace("{max_steps}", str(max_steps))

    user_context = {
        "lead_reason": lead_reason,
        "task": task,
        "spec": json.dumps(spec, ensure_ascii=False, indent=2)[:4000] if isinstance(spec, dict) else str(spec)[:4000],
        "last_stdout": (last_stdout or "")[:4000],
        "code": (code or "")[:6000],
    }

    scratch_lines: List[str] = []
    final_answer = ""
    for step in range(1, max_steps + 1):
        reply = _lead_react_step(llm, system_prompt, "\n".join(scratch_lines), user_context)
        parsed = _parse_react_reply(reply)
        if parsed["kind"] == "final":
            final_answer = parsed["final"]
            break
        tool = parsed["tool"]
        args_raw = parsed.get("args_raw", "")
        args = parse_args_line(args_raw)
        try:
            observation = call_tool(orch, tool, args)
        except LeadToolError as e:
            observation = f"[tool error] {e}"
        # Trim each observation so the scratchpad stays bounded.
        obs_trimmed = observation if len(observation) <= 1500 else observation[:1500] + "\n... [obs truncated]"
        scratch_lines.append(
            f"STEP {step}\n"
            f"TOOL: {tool}\n"
            f"ARGS: {args_raw}\n"
            f"OBSERVATION:\n{obs_trimmed}\n"
        )
    if not final_answer:
        # Force a final pass: tell the model to stop tool-calling and summarise.
        scratch_lines.append(
            "\nSTEP budget exhausted — you MUST emit a FINAL answer now, "
            "using whatever evidence you already gathered.\n"
        )
        reply = _lead_react_step(llm, system_prompt, "\n".join(scratch_lines), user_context)
        parsed = _parse_react_reply(reply)
        final_answer = parsed.get("final", reply.strip())
    return final_answer


def lead_agent_propose_changes(
    llm,
    lead_reason: str,
    task: str,
    spec: Dict[str, Any],
    code: str,
    last_stdout: str,
    *,
    orch: Any = None,
) -> str:
    """
    Backwards-compatible entry point.

    If ``orch`` is supplied, runs the ReAct loop with filesystem tools.
    Otherwise falls back to the original single-shot prompt so any caller
    that does not wire ``orch`` (tests, legacy code) keeps working.
    """
    if orch is not None:
        return lead_agent_propose_changes_react(
            llm,
            orch=orch,
            lead_reason=lead_reason,
            task=task,
            spec=spec,
            code=code,
            last_stdout=last_stdout,
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", LEAD_AGENT_SYS),
        ("user", "LEAD REASON:\n{lead_reason}\n\nORIGINAL TASK:\n{task}\n\nSPECIFICATION:\n{spec}\n\nLAST EXECUTION OUTPUT \n{last_stdout}\n\nCURRENT CODE TO BE FIXED:\n```python\n{code}\n```\n\nProvide your analysis and proposed solution.")
    ])
    res = invoke_and_log(llm, prompt, {
        "lead_reason": lead_reason,
        "task": task,
        "spec": spec,
        "last_stdout": last_stdout,
        "code": code
    })
    return _strip_think(getattr(res, "content", ""))


def lead_incident_manager_agent(
        llm_fast,
        *,
        task: str,
        spec: Dict[str, Any],
        triage_plan: Dict[str, Any],
        attempts: List[Dict[str, Any]],
        stderr_tail: str,
        stdout_tail: str,
        code_head: str,
) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", LEAD_INCIDENT_MANAGER_SYS),
        ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nTRIAGE_PLAN:\n{triage}\n\nATTEMPTS:\n{attempts}\n\nSTDERR_TAIL:\n{stderr}\n\nSTDOUT_TAIL:\n{stdout}\n\nCODE_HEAD:\n{code}")
    ])
    from .parsers import extract_json
    res = invoke_and_log(llm_fast, prompt, {
        "task": task,
        "spec": json.dumps(spec, ensure_ascii=False, indent=2),
        "triage": json.dumps(triage_plan or {}, ensure_ascii=False, indent=2),
        "attempts": json.dumps((attempts or [])[-25:], ensure_ascii=False, indent=2),
        "stderr": (stderr_tail or "")[-4000:],
        "stdout": (stdout_tail or "")[-4000:],
        "code": (code_head or "")[:4000],
    })
    plan = extract_json(getattr(res, "content", "")) or {}
    if not isinstance(plan, dict):
        return {"route": "coding", "packages": [], "bash_cmds": [], "pip_extra": "", "reason": "fallback", "notes": ""}
    plan.setdefault("route", "coding")
    plan.setdefault("packages", [])
    plan.setdefault("bash_cmds", [])
    plan.setdefault("pip_extra", "")
    plan.setdefault("notes", "")
    plan.setdefault("reason", "")
    return plan

def error_triage_agent(llm_fast, stderr: str, stdout: str, spec: Dict[str,Any], code_text: str, os_name: str = "Windows",
                       consecutive_spec_updates: int = 0, last_spec_update_reason: str = "") -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ERROR_TRIAGE_SYS),
        ("user", "OS: {os}\n\nSPEC (data):\n{spec_data}\n\n"
                 "CONSECUTIVE_SPEC_UPDATES: {n_updates} (how many times in a row spec_update was chosen without a successful run in between)\n"
                 "LAST_SPEC_UPDATE_REASON: {last_reason}\n"
                 "If CONSECUTIVE_SPEC_UPDATES >= 2, DO NOT route to spec_update again. Route to `coding` or `bash` and fix the symptom in code.\n\n"
                 "STDERR:\n{stderr}\n\nSTDOUT (tail):\n{stdout}\n\nCODE (head):\n{code}")
    ])
    res = invoke_and_log(llm_fast, prompt, {
        "os": os_name,
        "spec_data": json.dumps(spec.get("data",{}), ensure_ascii=False, indent=2),
        "stderr": stderr or "N/A",
        "stdout": (stdout if stdout else "N/A"),
        "code": (code_text if code_text else "N/A"),
        "n_updates": int(consecutive_spec_updates or 0),
        "last_reason": (last_spec_update_reason or "N/A")[:400],
    })
    return getattr(res, "content", "")

def generate_tasks_with_retry(
    llm_strong,
    task: str,
    spec: Optional[Dict[str, Any]],
    project_log: str = "",
    tasks_history: str = "",
    max_retries: int = 3,
    extra_context: str = "",
    *,
    remaining_total_sec: int | None = None,
    total_budget_sec: int | None = None,
    min_split_sec: int = 600,
    constraints_block: str = "",
) -> List[str]:
    err: str = ""
    for i in range(max_retries):
        raw = tasks_generation(
            llm_strong, task, spec, project_log, tasks_history, extra_context,
            remaining_total_sec=remaining_total_sec,
            total_budget_sec=total_budget_sec,
            min_split_sec=min_split_sec,
            constraints_block=constraints_block,
        )
        try:
            return _parse_yaml_tasks(_strip_think(getattr(raw,"content", raw) or ""), raise_on_error=True)
        except YAMLParseError as e:
            err = str(e)
            fix_prompt = ChatPromptTemplate.from_messages([
                ("system", "Your previous output was not valid YAML or lacked the root key 'tasks'. "
                 "Do NOT paste Python dict repr or concatenated {{'task':...}}{{'task':...}}. "
                 "Return ONLY a ```yaml fenced block: root key tasks:, each item with task: and time_budget_sec:.\n"
                 "Example:\n```yaml\ntasks:\n  - task: \"Step one description\"\n    time_budget_sec: 400\n```"),
                ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nPARSER ERROR:\n{err}")
            ])
            raw = invoke_and_log(
                llm_strong,
                fix_prompt,
                {"task": task, "spec": spec, "err": err},
                agent_name="tasks_yaml_fix",
            )
            try:
                return _parse_yaml_tasks(getattr(raw,"content",""), raise_on_error=True)
            except YAMLParseError:
                continue
    raise YAMLParseError(f"Failed to get valid YAML after {max_retries} attempts. Last error: {err}")

def improvement_tasks_generation(
    llm_strong,
    task: str,
    spec: Dict[str, Any],
    code_bank: List[str],
    metrics: Dict[str, Any],
    max_tasks: int = 7,
    *,
    constraints_block: str = "",
) -> List[Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVEMENT_TASKS_SYS),
        ("user", "{constraints_block}ORIGINAL TASK:\n{task}\n\nSPEC:\n{spec}\n\nCURRENT METRICS:\n{metrics}\n\nCURRENT CODE:\n{code}\n\n"
                 "MAX_SUBSTANTIVE_TASKS (not counting mandatory final submission line): {max_tasks}\n")
    ])
    code_summary = "\n".join(code_bank)
    res = invoke_and_log(llm_strong, prompt, {
        "constraints_block": constraints_block or "",
        "task": task,
        "spec": json.dumps(spec, ensure_ascii=False, indent=2) if isinstance(spec, dict) else str(spec),
        "metrics": json.dumps(metrics, ensure_ascii=False, indent=2),
        "code": code_summary,
        "max_tasks": max(1, int(max_tasks)),
    })
    return _parse_yaml_tasks(_strip_think(getattr(res, "content", "")))

def runtime_output_ok_agent(llm_fast, *, stdout: str, stderr: str, spec: dict, code_text: str = "", additional_context: str = ""):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RUNTIME_OUTPUT_OK_SYS),
        ("user", "\nAdditional context:\n{additional_context}\nSPEC:\n{spec}\n\nCODE:\n{code}\n\nSTDERR:\n{stderr}\n\nSTDOUT:\n{stdout}\n")
    ])
    res = invoke_and_log(llm_fast, prompt, {
        "spec": spec,
        "code": code_text or "",
        "stderr": stderr or "",
        "stdout": stdout or "",
        "additional_context": additional_context or ""
    })
    return _strip_think(getattr(res, "content", ""))

def evaluate_run_ok_with_retry(llm_fast, stdout: str, stderr: str, spec: Dict[str, Any], code_text: str, additional_context: str = "") -> bool:
    ans = runtime_output_ok_agent(llm_fast, stdout=stdout, stderr=stderr, spec=spec, code_text=code_text, additional_context=additional_context)
    val = _bool_from_text(ans)
    if val is not None:
        return val

    text = f"{stderr}\n{stdout}".lower()
    markers = ("traceback", "exception", "error:", "valueerror", "keyerror", "cuda out of memory", "no such file or directory", "shape mismatch", "nan", " inf", "fail", "failed", "not found", "metric not computed")
    for m in markers:
        if m in text:
            return False
    return True

def order_tasks_with_retry(
    llm_strong,
    task: str,
    sub_tasks: List[str],
    spec: Optional[Dict[str, Any]],
    max_retries: int = 3,
    *,
    overall_time_limit_sec: int | None = None,
    constraints_block: str = "",
) -> List[str]:
    err: str = ""
    for i in range(max_retries):
        raw = task_ordering(
            llm_strong, task, "\n".join(sub_tasks), spec,
            overall_time_limit_sec=overall_time_limit_sec,
            constraints_block=constraints_block,
        )
        try:
            return _parse_yaml_tasks(_strip_think(getattr(raw,"content", raw) or ""), raise_on_error=True)
        except YAMLParseError as e:
            err = str(e)
            fix_prompt = ChatPromptTemplate.from_messages([
                ("system", "Reformat these subtasks as a minimal, logically ordered YAML list under 'tasks'. Return ONLY a YAML code block."),
                ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nSUBTASKS (raw):\n{sub_tasks}\n\nPARSER ERROR:\n{err}")
            ])
            raw = invoke_and_log(
                llm_strong,
                fix_prompt,
                {"task": task, "spec": spec, "sub_tasks": "\n".join(sub_tasks), "err": err},
                agent_name="tasks_order_yaml_fix",
            )
            try:
                return _parse_yaml_tasks(getattr(raw,"content",""), raise_on_error=True)
            except YAMLParseError:
                continue
    raise YAMLParseError(f"Failed to order tasks with valid YAML after {max_retries} attempts. Last error: {err}")


def final_metric_selector_agent(
    llm_fast,
    task: str,
    spec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Pick best candidate/run using all metrics (stability-biased), not only `primary`.
    """
    # Keep prompt payload small: only what the selector needs.
    trimmed: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates[:12]):
        m = (c.get("metrics") or {}) if isinstance(c.get("metrics"), dict) else {}
        extras = m.get("extras")
        if isinstance(extras, dict):
            # Trim extras to avoid token blowups.
            extras_items = list(extras.items())[:20]
            trimmed_extras: Dict[str, Any] = {}
            for k, v in extras_items:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    sv = v
                else:
                    sv = str(v)
                if isinstance(sv, str) and len(sv) > 200:
                    sv = sv[:200] + "..."
                trimmed_extras[k] = sv
            extras = trimmed_extras
        entry: Dict[str, Any] = {
            "candidate_idx": i,
            "tag": c.get("tag", ""),
            "ts": c.get("ts", ""),
            "metrics": {
                "type": m.get("type"),
                "primary": m.get("primary"),
                "name": m.get("name"),
                "maximize": m.get("maximize"),
                "extras": extras,
            },
            "paths": {
                "submission": (c.get("submission_rel") or c.get("paths", {}).get("submission") or ""),
                "code": (c.get("code_rel") or c.get("paths", {}).get("code") or ""),
                "metrics": (c.get("metrics_rel") or c.get("paths", {}).get("metrics") or ""),
            },
        }
        if c.get("_leakage_warning"):
            entry["leakage_warning"] = (
                "Validation metrics suspiciously perfect (may be data leakage in metric calc). "
                f"Signals: {c['_leakage_warning']}. "
                "Check submission diversity — if submission predictions are diverse and in-range, candidate may still be valid."
            )
        trimmed.append(entry)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", FINAL_METRIC_SELECTOR_SYS),
            (
                "user",
                "TASK:\n{task}\n\nPRIMARY METRIC SEMANTICS:\n{primary_semantics}\n\nCANDIDATES_JSON:\n{candidates_json}",
            ),
        ]
    )

    primary_semantics = {
        "name": (spec.get("primary_metric") or {}).get("name"),
        "maximize": (spec.get("primary_metric") or {}).get("maximize"),
    }

    res = invoke_and_log(
        llm_fast,
        prompt,
        {
            "task": task,
            "primary_semantics": json.dumps(primary_semantics, ensure_ascii=False),
            "candidates_json": json.dumps(trimmed, ensure_ascii=False),
        },
        agent_name="final_metric_selector",
    )

    content = getattr(res, "content", "") or ""
    # Parse JSON; fallback to primary-based selection.
    try:
        from .parsers import extract_json

        parsed = extract_json(content)
        if isinstance(parsed, dict) and "chosen_candidate_idx" in parsed:
            return parsed
    except Exception:
        pass

    # Fallback: choose max/min by primary.
    maximize = bool(primary_semantics.get("maximize", True))
    best_idx = 0
    best_val = None
    for i, c in enumerate(trimmed):
        p = (((c.get("metrics") or {}) or {}).get("primary"))
        try:
            pv = float(p)
        except Exception:
            continue
        if best_val is None:
            best_val = pv
            best_idx = i
        else:
            if (pv > best_val) if maximize else (pv < best_val):
                best_val = pv
                best_idx = i

    return {
        "chosen_candidate_idx": best_idx,
        "chosen_tag": trimmed[best_idx].get("tag", ""),
        "reasoning": "fallback: invalid selector output; chose by primary metric",
    }


def react_preexec_auditor_agent(
    llm_fast,
    orch: Any,
    *,
    task: str,
    spec: Dict[str, Any],
    code_text: str,
    context: str = "",
) -> Dict[str, Any]:
    """
    Agentic preflight auditor with lightweight read-only tools.
    """
    from .parsers import extract_json
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:
        return {
            "allow_run": True,
            "planning_only": False,
            "issues": ["structured_tool_unavailable"],
            "required_fixes": [],
            "evidence": [],
        }

    def _is_unsafe_command(cmd: str) -> bool:
        s = (cmd or "").lower()
        blocked = [" rm ", " del ", " rmdir ", "shutdown", "reboot", "poweroff", "git reset", "git clean", "remove-item", "new-item"]
        return any(x in s for x in blocked)

    def bash_exec(command: str, timeout_sec: int = 30) -> str:
        cmd = (command or "").strip()
        if not cmd:
            return "error: empty command"
        if _is_unsafe_command(cmd):
            return "error: blocked unsafe command"
        res = orch.bash.run(cmd, timeout=max(5, min(int(timeout_sec or 30), 120)), stream=False)
        return (
            f"exit_code={res.get('exit_code', 1)}\n"
            f"STDOUT:\n{(res.get('stdout', '') or '')[:8000]}\n"
            f"STDERR:\n{(res.get('stderr', '') or '')[:4000]}"
        )

    def python_exec(code: str, timeout_sec: int = 30) -> str:
        c = (code or "").strip()
        if not c:
            return "error: empty python code"
        low = c.lower()
        forbidden = ["shutil.rmtree", "os.remove", "os.unlink", "subprocess.", ".write_text(", ".write_bytes("]
        if any(x in low for x in forbidden):
            return "error: blocked unsafe python operations"
        res = orch.run_python_code(c, filename=f"preexec_{__import__('uuid').uuid4().hex[:6]}.py", timeout=max(5, min(int(timeout_sec or 30), 120)))
        return (
            f"exit_code={res.get('exit_code', 1)}\n"
            f"STDOUT:\n{(res.get('output', '') or '')[:8000]}\n"
            f"STDERR:\n{(res.get('errors', '') or '')[:4000]}"
        )

    tools = [
        StructuredTool.from_function(func=bash_exec, name="bash_exec", description="Read-only shell for filesystem/log inspection."),
        StructuredTool.from_function(func=python_exec, name="python_exec", description="Run short read-only python checks/parsers."),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", REACT_PREEXEC_AUDITOR_SYS),
        ("user", "TASK:\n{task}\n\nSPEC:\n{spec}\n\nCONTEXT:\n{context}\n\nCODE:\n{code}")
    ])
    res = invoke_with_tools(
        llm_fast,
        prompt,
        {
            "task": task or "",
            "spec": json.dumps(spec or {}, ensure_ascii=False, indent=2),
            "context": context or "",
            "code": code_text or "",
        },
        tools=tools,
        agent_name="react_preexec_auditor",
    )
    content = _strip_think(getattr(res, "content", "") or "")
    parsed = extract_json(content)
    if isinstance(parsed, dict):
        return parsed

    # First parse failed — try a JSON-reformatting retry with a simpler prompt
    try:
        reformat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a JSON formatter. Output ONLY valid JSON, no markdown, no extra text."),
            ("user", (
                "Convert this pre-execution audit assessment into a valid JSON object with exactly "
                "these keys: allow_run (bool), planning_only (bool), issues (list of strings), "
                "required_fixes (list of strings), evidence (list of strings).\n\n"
                "Assessment:\n{assessment}"
            )),
        ])
        retry_res = invoke_and_log(
            llm_fast,
            reformat_prompt,
            {"assessment": content[:3000]},
            agent_name="preexec_audit_json_reformat",
        )
        retry_content = _strip_think(getattr(retry_res, "content", "") or "")
        retry_parsed = extract_json(retry_content)
        if isinstance(retry_parsed, dict):
            return retry_parsed
    except Exception:
        pass

    # Both parses failed — default to allow_run=True to avoid infinite block loops.
    # Log the raw response for debugging; do NOT inject a blocking issue.
    from colorama import Fore
    print(Fore.YELLOW + f"[PREFLIGHT] Auditor response could not be parsed as JSON (both attempts). "
                        f"Defaulting allow_run=True. Raw snippet: {str(content)[:200]}")
    return {
        "allow_run": True,
        "planning_only": False,
        "issues": [],
        "required_fixes": [],
        "evidence": [f"audit_json_parse_failed: {str(content)[:300]}"],
    }

__all__ = [
    "problem_spec_from_text",
    "metrics_recover_from_stdout",
    "default_spec_skeleton",
    "task_complexity_check", "generate_tasks_with_retry", "order_tasks_with_retry",
    "checker_code_agent", "perform_task_python_v2", "error_triage_agent", "finetune_code_v2", "datapath_agent",
    "checks_generation", "tasks_generation", "task_ordering", "aggregate_answers", "check_answer", "fix_answer",
    "verification_code_gen", "extract_code", "extract_boolean", "extract_numbered_list", "improvement_tasks_generation",
    "execution_predictor_agent", "execution_watcher_agent", "replanning_agent",
    "review_artifacts_agent", "improvement_replanning_agent", "improver_head_agent",
    "meta_planner_agent", "react_improver_meta_planner_agent",
    "log_update_agent", "artifact_reviewer_agent",
    "final_metric_selector_agent", "react_preexec_auditor_agent",
    "submission_sanity_agent",
]


def submission_sanity_agent(
    llm_fast,
    task_text: str,
    submission_head: str,
    submission_stats: Dict[str, Any],
    sample_target_head: str,
    sample_target_stats: Dict[str, Any],
    metrics_json_current: str,
) -> Dict[str, Any]:
    """
    Decides whether ``artifacts/submission.csv`` is valid for the current
    competition. Returns a dict with keys ``verdict`` ("valid"/"invalid"/
    "suspicious"), ``reasons``, ``must_rebuild``, ``rebuild_hint``.

    Never raises — on parse failure returns ``{"verdict": "suspicious", ...}``
    so the caller can decide whether to block finalization.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUBMISSION_SANITY_SYS),
        ("user",
         "TASK TEXT:\n{task}\n\n"
         "SUBMISSION_HEAD:\n{sub_head}\n\n"
         "SUBMISSION_STATS:\n{sub_stats}\n\n"
         "SAMPLE_TARGET_HEAD:\n{tgt_head}\n\n"
         "SAMPLE_TARGET_STATS:\n{tgt_stats}\n\n"
         "METRICS_JSON_CURRENT:\n{metrics}\n"),
    ])
    from src.parsers import extract_json
    try:
        res = invoke_and_log(llm_fast, prompt, {
            "task": shorten_string_middle(task_text or "", 6000),
            "sub_head": shorten_string_middle(submission_head or "(missing)", 2000),
            "sub_stats": json.dumps(submission_stats or {}, ensure_ascii=False),
            "tgt_head": shorten_string_middle(sample_target_head or "(missing)", 2000),
            "tgt_stats": json.dumps(sample_target_stats or {}, ensure_ascii=False),
            "metrics": shorten_string_middle(metrics_json_current or "(missing)", 1500),
        })
        parsed = extract_json(getattr(res, "content", "")) or {}
    except Exception as e:
        return {
            "verdict": "suspicious",
            "reasons": [f"sanity_agent_error: {e}"],
            "must_rebuild": False,
            "rebuild_hint": "sanity agent failed; defer decision",
        }
    if not isinstance(parsed, dict) or "verdict" not in parsed:
        return {
            "verdict": "suspicious",
            "reasons": ["sanity_agent_returned_unparseable_output"],
            "must_rebuild": False,
            "rebuild_hint": "rerun sanity check with stricter output contract",
        }
    parsed.setdefault("reasons", [])
    parsed.setdefault("must_rebuild", parsed.get("verdict") == "invalid")
    parsed.setdefault("rebuild_hint", "")
    return parsed


# ---------------------------------------------------------------------------
# Knowledge Curator — ReAct supervisor called synchronously before every main
# agent (role=coder/planner/replanner/...) and after every run/prune/heartbeat.
# It owns five canonical .md files under artifacts/curator/ and serves tailored
# context to consumer agents while keeping the files in sync with reality.
# ---------------------------------------------------------------------------
def knowledge_curator_agent(
    llm_fast,
    llm_strong,
    orch,
    role: str,
    task_hint: str = "",
    trigger: str = "before",
    event_payload: Optional[Dict[str, Any]] = None,
    char_budget: int = 5000,
) -> str:
    """
    Synchronously invoke the knowledge curator.

    BEFORE-mode (trigger='before'): returns a short markdown context block for
    the given consumer role, capped at char_budget. The caller prepends this
    block to its own prompt.

    AFTER-mode (any other trigger): returns a short JSON log of updates the
    curator performed; callers can ignore the return value.
    """
    try:
        from src.artifact_tools import build_curator_tools
    except Exception as e:
        return f"[CURATOR CONTEXT for role={role}]\n(curator unavailable: {e})\n"

    try:
        tools = build_curator_tools(orch) or []
    except Exception as e:
        return f"[CURATOR CONTEXT for role={role}]\n(curator tools unavailable: {e})\n"

    llm = llm_strong if trigger in ("bootstrap", "finalize") else llm_fast

    # Compact state snapshot so the curator can reason about freshness/budgets.
    snapshot: Dict[str, Any] = {}
    try:
        snapshot["global_remaining_sec"] = int(
            float(getattr(orch, "global_deadline_sec",
                          orch.cfg.orchestration.total_budget_sec))
            - orch.effective_elapsed_sec()
        )
    except Exception:
        snapshot["global_remaining_sec"] = None
    try:
        snapshot["artifacts_dir"] = str(
            getattr(orch, "project_root", ".") / orch.cfg.paths.artifacts_dir
        )
    except Exception:
        pass

    payload_json = json.dumps(event_payload or {}, ensure_ascii=False, indent=2)
    # Hard cap the payload to protect context.
    if len(payload_json) > 12000:
        payload_json = payload_json[:12000] + "\n... <truncated>"

    user_tmpl = (
        "TRIGGER: {trigger}\n"
        "ROLE: {role}\n"
        "TASK_HINT: {task_hint}\n"
        "CHAR_BUDGET: {char_budget}\n"
        "STATE_SNAPSHOT:\n{snapshot}\n\n"
        "EVENT_PAYLOAD (trigger-specific, may be empty):\n{payload}\n\n"
        "If TRIGGER == 'before': return ONE markdown block starting with "
        "'[CURATOR CONTEXT for role={role}]' and no more than {char_budget} chars.\n"
        "Otherwise: inspect+patch canonical files via write_md_section / "
        "append_md_line and return ONLY the JSON log."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", CURATOR_SYS),
        ("user", user_tmpl),
    ])
    try:
        res = invoke_with_tools(
            llm, prompt,
            {
                "trigger": trigger,
                "role": role,
                "task_hint": (task_hint or "")[:1500],
                "char_budget": int(char_budget),
                "snapshot": json.dumps(snapshot, ensure_ascii=False),
                "payload": payload_json,
            },
            tools=tools,
            agent_name="knowledge_curator",
        )
        text = getattr(res, "content", "") or ""
    except Exception as e:
        print(f"[CURATOR] invocation exception: {type(e).__name__}: {e}")
        return f"[CURATOR CONTEXT for role={role}]\n(curator invocation failed: {e})\n"

    if trigger == "before":
        if not text.strip():
            return f"[CURATOR CONTEXT for role={role}]\n(curator returned empty)\n"
        # Safety trim.
        if len(text) > char_budget * 2:
            text = text[: char_budget * 2]
        if not text.lstrip().startswith("[CURATOR CONTEXT"):
            text = f"[CURATOR CONTEXT for role={role}]\n" + text
        return text + ("\n" if not text.endswith("\n") else "")
    return text or "{\"updated\": []}"
