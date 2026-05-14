from __future__ import annotations

import os
import platform
import re
import uuid
import time
import json
import ast
import csv
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

from colorama import Fore, init as colorama_init

from src.helpers import _detect_and_store_submissions, _copy_text_file, _update_best_from_candidate, \
    _finalize_single_submission, _enforce_stack_guardrails, snapshot_data_tree, _deep_merge, validate_final_submission, \
    run_final_output_gate, _finalize_single_submission_by_all_metrics_llm, materialize_project_root_submission_csv
from src.utils import (shorten_string_middle, YAMLParseError, _parse_check_summary,
                       _rel_better, _ensure_dir, _slug, \
                       _validate_and_normalize_metrics, clean_specs, format_spec_constraints_block)
from src.hardware import attach_hardware_to_spec
from src.optimizer import optimize_metrics
from src.data_meta import build_data_meta
from src.parsers import extract_json
from src.router import ErrorRouter
from src.bootstrap import bootstrap_gpu_stack
from src.orchestrator import GlobalOrchestrator
from src.validators import parse_metrics_from_stdout, validate_recovered_metrics
from src.dataset_checker import probe_dataset_with_bash
from src.prompts_agents import (
    problem_spec_from_text,
    default_spec_skeleton,
    task_complexity_check,
    perform_task_python_v2,
    finetune_code_v2,
    checks_generation,
    aggregate_answers,
    check_answer,
    fix_answer,
    verification_code_gen,
    datapath_agent, datapath_consistency_check_agent, generate_tasks_with_retry, order_tasks_with_retry, checker_code_agent, improvement_tasks_generation,
    evaluate_run_ok_with_retry, lead_agent_propose_changes, lead_incident_manager_agent, implement_changes_agent,
    execution_predictor_agent, execution_watcher_agent, replanning_agent,
    review_artifacts_agent, improvement_replanning_agent, improver_head_agent,
    meta_planner_agent, react_improver_meta_planner_agent,
    log_update_agent, artifact_reviewer_agent,
    metrics_recover_from_stdout, react_preexec_auditor_agent,
    react_artifacts_collector_agent,
    knowledge_curator_agent,
)
from src.verification import FormalVerifier, VerificationSpec
from src.artifact_tools import (
    ensure_artifacts_repo,
    snapshot_artifacts,
    artifacts_diff_since,
    build_schema_snapshot,
    build_structured_artifact_tools,
    write_artifacts_index,
    CURATOR_MD_SCHEMA,
)

colorama_init(autoreset=True)


def _global_remaining_sec(orch: GlobalOrchestrator) -> int:
    deadline = float(
        getattr(orch, "global_deadline_sec", orch.cfg.orchestration.total_budget_sec)
    )
    return max(0, int(deadline - orch.effective_elapsed_sec()))



def _json_roundtrip_safe(obj: Any) -> Any:
    """
    Convert objects to a JSON-serializable structure via roundtrip.
    Paths and other custom objects are stringified.
    """
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Knowledge Curator call wrappers — synchronous "before" / fire-and-log "after".
# These are the ONLY sanctioned paths to read/update artifacts/curator/*.md.
# ---------------------------------------------------------------------------
def _curator_before(orch: GlobalOrchestrator, llm_fast, llm_strong, role: str,
                    task_hint: str = "", char_budget: int = 4500) -> str:
    print(Fore.MAGENTA + f"[CURATOR] before(role={role}) invoked")
    try:
        out = knowledge_curator_agent(
            llm_fast, llm_strong, orch,
            role=role, task_hint=task_hint,
            trigger="before", char_budget=char_budget,
        )
        print(Fore.MAGENTA + f"[CURATOR] before(role={role}) returned {len(out or '')} chars")
        return out
    except Exception as e:
        print(Fore.YELLOW + f"[CURATOR] before({role}) failed: {e}")
        return ""


def _curator_after(orch: GlobalOrchestrator, llm_fast, llm_strong,
                   role: str, task_hint: str, trigger: str,
                   payload: Optional[Dict[str, Any]] = None) -> None:
    print(Fore.MAGENTA + f"[CURATOR] {trigger}(role={role}) invoked")
    try:
        knowledge_curator_agent(
            llm_fast, llm_strong, orch,
            role=role, task_hint=task_hint,
            trigger=trigger, event_payload=payload or {},
        )
        print(Fore.MAGENTA + f"[CURATOR] {trigger}(role={role}) done")
    except Exception as e:
        print(Fore.YELLOW + f"[CURATOR] {trigger}({role}) failed: {e}")


def _audit_generated_code_policy(code_text: str) -> tuple[bool, list[str]]:
    """
    Fast static policy audit for generated scripts.
    Blocks obviously destructive code and spec-hardcoding patterns.
    """
    txt = str(code_text or "")
    low = txt.lower()
    issues: list[str] = []

    dangerous_snippets = [
        "shutil.rmtree(",
        "os.remove(",
        "os.unlink(",
        ".unlink(",
        "subprocess.run(",
        "subprocess.popen(",
        "git reset",
        "git clean",
        "remove-item",
        "rmdir ",
    ]
    for s in dangerous_snippets:
        if s in low:
            issues.append(f"dangerous_op:{s}")

    # Canonical curator .md files — only the Knowledge Curator agent may write.
    for _cf in CURATOR_MD_SCHEMA.keys():
        if _cf in low and ("write_text" in low or "open(" in low or "write(" in low):
            issues.append(f"curator_md_write_forbidden:{_cf}")

    # Protected pipeline control files — must never be written by generated code.
    _protected_files = ("tree.json", "task_graph_events.jsonl", "spec.json")
    import re as _re
    for _pf in _protected_files:
        # Match open()/write_text()/write()/json.dump targeting the protected file.
        if _pf in low:
            # open("tree.json", "w") / open("tree.json", "wb") patterns
            _open_write = _re.search(
                r'open\s*\([^)]*' + _re.escape(_pf) + r'[^)]*["\'],\s*["\']w',
                low,
            )
            # write_text() always writes — no mode string needed
            _write_text = _re.search(r'write_text\s*\(', low)
            # json.dump() only writes to file when a 'w' mode file handle is present
            _json_dump = _re.search(r'json\.dump\s*\(', low) and (
                "'w'" in low or '"w"' in low
            )
            if _open_write or _write_text or _json_dump:
                issues.append(f"protected_pipeline_file_write:{_pf}")

    # Spec hardcoding guardrail:
    # If script appears to define full spec-like dict, require dynamic load from artifacts/spec.json.
    looks_like_hardcoded_spec = (
        ("primary_metric" in low and "submission" in low and "secondary_metrics" in low and "spec" in low)
        and ("spec = {" in low or "spec={" in low or "spec = dict(" in low)
    )
    if "spec_json = '''" in low or "spec_json='''" in low or "spec_json = \"\"\"" in low:
        looks_like_hardcoded_spec = True
    if "spec_json" in low and "json.loads(spec_json" in low:
        looks_like_hardcoded_spec = True
    if "spec_json" in low and "primary_metric" in low and "submission" in low:
        looks_like_hardcoded_spec = True
    has_dynamic_spec_read = ("artifacts/spec.json" in low) or ("spec.json" in low and "json.load" in low)
    if looks_like_hardcoded_spec and not has_dynamic_spec_read:
        issues.append("spec_hardcoded_without_dynamic_load")

    # Anti-mocking guardrail (prevents agent from inventing data to bypass loading errors)
    if "np.random." in low or "random.random" in low or "random.randint" in low:
        # If it looks like it's creating a fake dataset instead of loading
        if "pd.DataFrame(" in low and not (".read_csv(" in low or ".read_pickle(" in low or ".read_parquet(" in low):
            issues.append("potential_data_mocking_detected")
    
    # Check for hardcoded metrics
    if "METRICS_JSON" in low and ": {" in low:
        if '"primary":' in low and not any(x in low for x in ("score", "metric", "accuracy", "rmse", "val_", "loss")):
            # Very loose check, but might catch literal hardcoding like {"primary": 0.85}
            import re
            if re.search(r'"primary":\s*[0-9]\.[0-9]+', low):
                if not re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[0-9]\.[0-9]+', low): # no variable assignment
                     issues.append("potential_hardcoded_metrics")

    # Narrative/Large string literal check (blocks giant embedded MD reports that cause SyntaxErrors)
    if '"""' in code_text or "'''" in code_text:
        import re
        # Find triple-quoted blocks longer than 3000 chars
        large_blocks = re.findall(r'\"{3}[\s\S]{3000,}\"{3}|\'{3}[\s\S]{3000,}\'{3}', code_text)
        if large_blocks:
            issues.append("excessive_narrative_string_literals_detected")

    return (len(issues) == 0), issues


def _pipeline_effective_depth(orch: GlobalOrchestrator, depth: int, node_id: Optional[str]) -> int:
    """Max(pipeline depth param, persisted tree depth). Fixes --resume where depth resets but tree is deep."""
    try:
        d0 = int(depth)
    except (TypeError, ValueError):
        d0 = 0
    if not node_id:
        return max(0, d0)
    try:
        td = int((orch.tree_node(node_id) or {}).get("depth") or 0)
    except (TypeError, ValueError):
        td = 0
    return max(d0, td)


def _persist_last_code_artifact(orch: GlobalOrchestrator, code_text: str) -> None:
    """
    Persist latest executable code to artifacts/last/code.py even when metrics are missing.
    This prevents final gate failure for tasks requiring code artifact submission.
    """
    if not str(code_text or "").strip():
        return
    try:
        root = Path(orch.project_root)
        last_dir = root / orch.cfg.paths.artifacts_dir / "last"
        _ensure_dir(last_dir)
        orch.write_file(str(last_dir / "code.py"), code_text)
    except Exception:
        pass


def _agentic_repair_verifier_code(
    code_llm,
    spec: Dict[str, Any],
    initial_code: str,
    context: str,
    mcp_tools: Optional[List[Any]],
    max_attempts: int = 3,
) -> str:
    """
    Agent-first verifier repair loop.
    Instead of static fallback code, ask coder agent to fix verifier code using
    policy issues + task context + tool access.
    """
    candidate = str(initial_code or "")
    for attempt in range(max(1, int(max_attempts))):
        ok, issues = _audit_generated_code_policy(candidate)
        if ok and candidate.strip():
            return candidate

        repair_task = (
            "Repair this Python verifier script so it is executable and policy-compliant. "
            "Must dynamically read artifacts/spec.json (no hardcoded spec), inspect existing artifacts, "
            "and print exactly one METRICS_JSON line. "
            "If spec.json is malformed/missing, recover robustly by reading existing metrics/checkpoints and "
            "emit valid skipped metrics only as last resort. "
            "Use canonical project-root artifacts paths only."
        )
        repair_ctx = (
            f"Verifier repair attempt {attempt + 1}/{max_attempts}\n"
            f"Policy issues: {issues}\n"
            f"{context}\n"
            "The script may inspect filesystem with tools first, then rewrite full standalone verifier code."
        )
        candidate = perform_task_python_v2(
            code_llm,
            repair_task,
            spec or {},
            previous_code=candidate,
            context=repair_ctx,
            tools=mcp_tools,
        )
        candidate = str(candidate or "").replace("```python", "").replace("```", "").strip()
    return candidate


def _attempt_output_gate_recovery(
    orch: GlobalOrchestrator,
    llm_fast,
    code_llm,
    spec: Dict[str, Any],
    task: str,
    gate_errors: List[str],
    mcp_tools: Optional[List[Any]],
) -> Dict[str, Any]:
    """
    Last-chance autonomous recovery before hard-failing output gate:
    - generate and run a focused recovery script via prompt+tools
    - try metrics recovery
    - materialize canonical submission and rerun gate
    """
    print(Fore.YELLOW + "[FINAL][RECOVERY] output gate failed, attempting autonomous recovery...")
    rec_task = (
        "FINAL RECOVERY: produce canonical submission.csv and metrics artifacts for this run. "
        "Read spec dynamically from artifacts/spec.json; never hardcode spec dict; "
        "write submission to canonical location and print METRICS_JSON."
    )
    rec_ctx = (
        "Output gate errors:\n- " + "\n- ".join(gate_errors or []) + "\n\n"
        "Use available artifacts/checkpoints/preds if they exist. "
        "If no trained model exists, run the fastest valid baseline to produce a real submission. "
        "Do not create fake helper submissions."
    )
    rec_code = perform_task_python_v2(
        code_llm,
        rec_task,
        spec or {},
        previous_code="",
        context=rec_ctx + f"\nOriginal task: {task}",
        tools=mcp_tools,
        orch=orch,
    )
    clean_rec_code = str(rec_code or "").replace("```python", "").replace("```", "").strip()
    _persist_last_code_artifact(orch, clean_rec_code)

    if clean_rec_code:
        rec_res = orch.code_executor(clean_rec_code, spec=spec or {})
        rec_stdout = rec_res.get("output", "") or ""
        rec_metrics = parse_metrics_from_stdout(rec_stdout) or {}
        if not rec_metrics:
            try:
                verif_code = verification_code_gen(
                    code_llm,
                    spec,
                    context="Recover METRICS_JSON from existing artifacts after final recovery attempt.",
                )
                _v_ok, _v_issues = _audit_generated_code_policy(verif_code)
                if not _v_ok:
                    print(Fore.YELLOW + f"[FINAL][RECOVERY] verifier code blocked by policy: {_v_issues}; invoking agentic verifier repair")
                    verif_code = _agentic_repair_verifier_code(
                        code_llm=code_llm,
                        spec=spec or {},
                        initial_code=verif_code,
                        context=(
                            "Recover METRICS_JSON from existing artifacts after output-gate failure. "
                            "Do not hardcode spec; fix verifier code using filesystem evidence."
                        ),
                        mcp_tools=mcp_tools,
                    )
                verif_res = orch.code_executor(verif_code, "metric_check.py")
                rec_metrics = parse_metrics_from_stdout(verif_res.get("output", "") or "") or {}
            except Exception:
                rec_metrics = {}
        if rec_metrics and rec_metrics.get("type") != "skipped":
            try:
                _update_best_from_candidate(
                    orch,
                    candidate_metrics=rec_metrics,
                    code_text=clean_rec_code,
                    tag="final_recovery",
                    spec=spec,
                )
            except Exception:
                pass

    try:
        materialize_project_root_submission_csv(orch)
    except Exception:
        pass
    return run_final_output_gate(orch, spec or {}, task_txt_root=Path(orch.project_root))


def _reconcile_submission_columns_from_sample(orch: GlobalOrchestrator, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer real sample_submission header over LLM guess for spec.submission.columns.
    This prevents one-column hallucinations like ['Insult'] when sample has ['Comment','Insult'].
    """
    if not isinstance(spec, dict):
        return spec
    root = Path(orch.project_root)
    data = spec.get("data", {}) if isinstance(spec.get("data"), dict) else {}
    candidates: List[Path] = []
    raw = str(data.get("sample_submission_csv") or data.get("sample_submission") or "").strip()
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else (root / p))
    resolved_root = str(data.get("resolved_root") or "").strip()
    if resolved_root:
        candidates.append(Path(resolved_root) / "sample_submission.csv")
    candidates.append(root / "data" / "sample_submission.csv")

    sample_path: Optional[Path] = None
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                sample_path = p
                break
        except Exception:
            continue
    if sample_path is None:
        return spec

    try:
        with sample_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = [str(h).strip() for h in (next(reader, None) or [])]
    except Exception:
        return spec
    if not header:
        return spec

    sub = spec.get("submission", {}) if isinstance(spec.get("submission"), dict) else {}
    old_cols = [str(c).strip() for c in (sub.get("columns") or [])]
    if old_cols != header:
        sub["columns"] = header
        sub.setdefault("delimiter", ",")
        spec["submission"] = sub
        print(Fore.YELLOW + f"[SPEC] submission.columns reconciled from sample_submission header: {header}")
    return spec


def _normalize_plan_tail_entries(raw: List[Any], default_tb: int) -> List[Dict[str, Any]]:
    """Replan / ordering may return dicts or bare strings."""
    out: List[Dict[str, Any]] = []
    db = max(60, int(default_tb))
    for x in raw or []:
        if isinstance(x, dict) and x.get("task"):
            try:
                tb = int(x.get("time_budget_sec", db) or db)
            except Exception:
                tb = db
            out.append({"task": str(x["task"]).strip(), "time_budget_sec": max(30, tb)})
        elif isinstance(x, str) and x.strip():
            out.append({"task": x.strip(), "time_budget_sec": db})
    return out


def _sanitize_triage_bash_cmds(cmds: List[Any]) -> List[str]:
    """
    Block dangerous triage bash commands that can kill the orchestrator itself.
    Keep only minimally safe filesystem/process-inspection commands.
    """
    safe: List[str] = []
    for raw in (cmds or []):
        cmd = str(raw or "").strip()
        if not cmd:
            continue
        low = cmd.lower()

        # Hard block broad Python process kills that can terminate this very run.
        blocked_patterns = (
            "get-process python | stop-process",
            "stop-process -name python",
            "taskkill /im python",
            "pkill -f python",
            "killall python",
        )
        if any(p in low for p in blocked_patterns):
            print(Fore.YELLOW + f"[TRIAGE/BASH] blocked unsafe command: {cmd}")
            continue

        # Allow explicit PID kill only for non-current process.
        if ("stop-process" in low or "taskkill" in low or " kill " in f" {low} ") and ("-id" in low or "/pid" in low):
            safe.append(cmd)
            continue

        # Common safe commands for triage.
        safe_prefixes = (
            "dir", "ls", "get-childitem", "test-path", "resolve-path", "where", "which",
            "expand-archive", "new-item", "move-item", "copy-item", "python ", "powershell ",
        )
        if low.startswith(safe_prefixes):
            safe.append(cmd)
            continue

        # Unknown process/system command -> skip.
        if any(x in low for x in ("stop-process", "taskkill", "pkill", "killall", "kill ")):
            print(Fore.YELLOW + f"[TRIAGE/BASH] blocked process-kill command: {cmd}")
            continue
        safe.append(cmd)
    return safe


def _verify_dependency_claims(orch: "GlobalOrchestrator", issues: List[str]) -> List[str]:
    """
    Verify "Missing dependency" claims by actually importing the package.
    Returns the filtered list with false positives removed.
    """
    import re as _re
    # Matches: "Missing critical dependency: pandas", "No module named numpy",
    # "Missing package: sklearn", "pandas is not installed"
    _dep_patterns = [
        _re.compile(r"[Mm]issing\b.*?(?:dependency|package|module|import)[:\s]+(\w[\w.-]*)", _re.IGNORECASE),
        _re.compile(r"[Nn]o module named ['\"]?(\w[\w.-]*)", _re.IGNORECASE),
        _re.compile(r"(\w[\w.-]*)\s+is not installed", _re.IGNORECASE),
    ]
    verified: List[str] = []
    for issue in issues:
        m = None
        for pat in _dep_patterns:
            m = pat.search(issue)
            if m:
                break
        if not m:
            verified.append(issue)
            continue
        pkg_name = m.group(1).strip().rstrip(".")
        # Map common import names to their actual import module
        _import_name = {
            "scikit-learn": "sklearn",
            "opencv-python": "cv2",
            "pillow": "PIL",
            "pyyaml": "yaml",
            "python-dotenv": "dotenv",
            "beautifulsoup4": "bs4",
        }.get(pkg_name, pkg_name)
        try:
            res = orch.run_python_code(
                f"import {_import_name}; print('ok')",
                filename=f"_dep_check_{_import_name}.py",
                timeout=15,
            )
            if res.get("exit_code", 1) == 0 and "ok" in (res.get("output", "") or ""):
                print(Fore.GREEN + f"[PREFLIGHT/VERIFY] '{pkg_name}' IS available — removing false positive")
                continue  # Package works fine, skip this issue
        except Exception:
            pass
        verified.append(issue)
    if len(verified) < len(issues):
        print(Fore.GREEN + f"[PREFLIGHT/VERIFY] Removed {len(issues) - len(verified)} false dependency claims")
    return verified


_DEFAULT_CLASSIFICATION_SECONDARY: tuple[str, ...] = (
    "f1_macro",
    "f1_weighted",
    "cohen_kappa",
    "precision_macro",
    "recall_macro",
    "confusion_matrix",
)

_DEFAULT_REGRESSION_SECONDARY: tuple[str, ...] = (
    "rmse",
    "mae",
    "medae",
    "r2",
    "explained_variance",
    "max_error",
)

_DEFAULT_MULTILABEL_SECONDARY: tuple[str, ...] = (
    "f1_macro",
    "f1_micro",
    "hamming_loss",
    "subset_accuracy",
    "precision_macro",
    "recall_macro",
)


def merge_default_secondary_metrics(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Union standard secondary metrics by inferred problem type (spec contract)."""
    if not isinstance(spec, dict):
        return spec
    pm = spec.get("primary_metric") or {}
    if not isinstance(pm, dict):
        pm = {}
    pname = str(pm.get("name", "")).lower().strip()
    raw_sec = spec.get("secondary_metrics")

    def _normalize_metric_name(v: Any) -> str:
        if isinstance(v, dict):
            return str(v.get("name", "")).strip()
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ""
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = json.loads(s)
                    if isinstance(d, dict):
                        return str(d.get("name", "")).strip()
                except Exception:
                    try:
                        d = ast.literal_eval(s)
                        if isinstance(d, dict):
                            return str(d.get("name", "")).strip()
                    except Exception:
                        pass
            return s
        return str(v).strip()

    existing: List[str] = []
    if isinstance(raw_sec, list):
        existing = [_normalize_metric_name(x) for x in raw_sec]
        existing = [x for x in existing if x]
    elif isinstance(raw_sec, str) and raw_sec.strip():
        single = _normalize_metric_name(raw_sec)
        existing = [single] if single else []
    existing_l = [x.lower() for x in existing]

    is_cls = (
        pname
        in (
            "accuracy",
            "f1",
            "f1_macro",
            "f1_weighted",
            "balanced_accuracy",
            "cohen_kappa",
            "log_loss",
        )
        or pname in ("auc", "roc_auc", "pr_auc", "average_precision")
        or any("confusion" in x for x in existing_l)
        or any(x.startswith("f1") for x in existing_l)
        or any(x.startswith("precision") for x in existing_l)
        or any(x.startswith("recall") for x in existing_l)
    )
    is_regr = (
        pname in ("rmse", "mse", "mae", "r2", "r_squared", "mape", "rmsle", "msle", "smape")
        or "rmse" in pname
        or "mae" in pname
        or pname.endswith("_error")
        or any(x in ("rmse", "mae", "r2", "rmsle") for x in existing_l)
    )
    is_mll = (
        any(x in ("hamming_loss", "subset_accuracy") for x in existing_l)
        or "multilabel" in json.dumps(spec.get("modalities", [])).lower()
    )

    defaults: tuple[str, ...] = ()
    if is_cls:
        defaults = _DEFAULT_CLASSIFICATION_SECONDARY
    elif is_mll:
        defaults = _DEFAULT_MULTILABEL_SECONDARY
    elif is_regr:
        defaults = _DEFAULT_REGRESSION_SECONDARY
    else:
        return spec

    merged: List[str] = []
    seen: set[str] = set()
    for k in list(existing) + list(defaults):
        ks = str(k).strip()
        if not ks:
            continue
        kl = ks.lower()
        if kl in seen:
            continue
        seen.add(kl)
        merged.append(ks)
    # Canonical format: list[str] only.
    spec["secondary_metrics"] = merged
    return spec


code_bank: List[str] = []
summary_lines: List[str] = []


@contextmanager
def _track_node(orch: GlobalOrchestrator,
                node_id: Optional[str],
                parent_node_id: Optional[str],
                *,
                kind: str,
                task: str):
    """
    ЕДИНСТВЕННОЕ место создания/старта/завершения узлов.
    """
    nid = orch.tree_ensure_node(node_id, parent_node_id, kind=kind, task=task)
    orch.tree_start(nid)
    try:
        yield nid
        orch.tree_finish(nid, status="done")
    except Exception as e:
        orch.tree_set_failed(nid, error=str(e))
        raise


# -------------------- checker-driven post validation --------------------

def _extract_fixed_aggregate_report(stdout: str) -> Optional[str]:
    """Parse revised report from fix_answer-generated script stdout."""
    import re

    if not (stdout or "").strip():
        return None
    m = re.search(
        r"AGGREGATE_REPORT_BEGIN\s*\r?\n(.*?)\r?\nAGGREGATE_REPORT_END",
        stdout,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    body = (m.group(1) or "").strip()
    return body or None


_ARTIFACT_EXTS = (
    r"parquet|csv|json|npy|npz|pkl|pickle|joblib|pt|pth|bin|h5|hdf5|"
    r"feather|arrow|tsv|txt|zip|tar|gz|yaml|yml"
)
_PATH_TOKEN = rf"['\"`]?([^\s'\"`,\)\]<>]+\.(?:{_ARTIFACT_EXTS}))['\"`]?"

_ARTIFACT_SAVE_PATTERNS = [
    # "saved to <path>", "saved at <path>", "saved ... to <path>"
    # Allow up to ~60 chars between 'save' and the preposition so phrases like
    # "saved the engineered features to <path>" match too.
    re.compile(rf"(?i)\bsave[ds]?\b[^\n]{{0,60}}?\s(?:to|at|as|into|in)\s+{_PATH_TOKEN}"),
    # "writing <path>", "wrote <path>"
    re.compile(rf"(?i)\b(?:writing|wrote)\s+{_PATH_TOKEN}"),
    # "dumped to <path>"
    re.compile(rf"(?i)\b(?:dump(?:ed)?|exported?)\b[^\n]{{0,60}}?\s(?:to|at|into|as)\s+{_PATH_TOKEN}"),
    # "-> <path>", "→ <path>"
    re.compile(rf"(?:->|→)\s*{_PATH_TOKEN}"),
    # "output: <path>", "artifact: <path>"
    re.compile(rf"(?i)\b(?:output|artifact|file|path|result|checkpoint)\s*[:=]\s*{_PATH_TOKEN}"),
]


def _verify_claimed_artifacts(orch: GlobalOrchestrator, text_blobs: List[str]) -> Dict[str, List[str]]:
    """
    Scan the provided text blobs (stdout tails / project log) for claimed
    artifact saves and check which of those files actually exist on disk.

    Returns:
        {
            "verified": [<relative paths that exist>],
            "missing":  [<relative paths claimed but not found>],
        }

    Motivation: the aggregate agent has been observed to copy filenames from
    the task description as if they were saved, producing bogus summaries
    that blocked downstream preflight. Passing verified/missing evidence to
    the agent lets us hard-require file-path provenance.
    """
    root = Path(orch.project_root).resolve()
    seen: Dict[str, bool] = {}
    found_order: List[str] = []
    for blob in text_blobs:
        if not blob:
            continue
        for pat in _ARTIFACT_SAVE_PATTERNS:
            for m in pat.finditer(blob):
                raw = (m.group(1) or "").strip().strip(".,;:")
                if not raw:
                    continue
                # Normalise slashes and skip obvious URLs.
                if "://" in raw:
                    continue
                norm = raw.replace("\\", "/")
                if norm in seen:
                    continue
                seen[norm] = True
                found_order.append(norm)

    verified: List[str] = []
    missing: List[str] = []
    for claim in found_order:
        candidate = Path(claim)
        if not candidate.is_absolute():
            candidate = root / claim
        try:
            resolved = candidate.resolve()
        except Exception:
            missing.append(claim)
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            # Path escapes project_root — skip silently; not our business.
            continue
        if resolved.exists() and resolved.is_file():
            try:
                rel = resolved.relative_to(root).as_posix()
            except ValueError:
                rel = str(resolved)
            verified.append(rel)
        else:
            missing.append(claim)
    # De-duplicate while preserving order.
    verified = list(dict.fromkeys(verified))
    missing = list(dict.fromkeys(missing))
    return {"verified": verified, "missing": missing}


def check_and_fix_answer(
        orch: GlobalOrchestrator,
        llm_fast,
        task: str,
        initial_answer: str,
        spec: Dict[str, Any],
        node_id: str,
        max_retries: int = 2,
        only_root: bool = True,
        quiet: bool = True,
        code_summary: str = "",
        metrics_json: str = "",
        stdout_tail: str = "",
        stderr_tail: str = "",
        improvement_summary: str = "",
):
    if only_root and task.strip().lower().startswith("sub-task"):
        return initial_answer

    orc = getattr(orch.cfg, "orchestration", None)
    if isinstance(orc, dict):
        fail_threshold = float(orc.get("check_fail_threshold", 0.25))
    else:
        fail_threshold = float(getattr(orc, "check_fail_threshold", 0.25))

    checker_code = checker_code_agent(
        llm_fast,
        task=task,
        spec=spec,
        code_summary=code_summary,
        final_answer=initial_answer,
        metrics_json=metrics_json,
        stdout_tail=stdout_tail[-4000:],
        stderr_tail=stderr_tail[-4000:],
        improvement_summary=improvement_summary[-4000:],
    )
    checker_name = f"check_{uuid.uuid4().hex[:6]}.py"
    _ok_checker, _checker_issues = _audit_generated_code_policy(checker_code)
    if not _ok_checker:
        orch.log("checker_blocked_by_policy", {"issues": _checker_issues})
        summary = {
            "total": 1,
            "failed": 1,
            "fail_names": [f"checker_policy:{';'.join(_checker_issues)}"],
            "pass_names": [],
        }
        return initial_answer

    checker_holder: Dict[str, str] = {"code": checker_code, "name": checker_name}

    def run_checker() -> Dict[str, Any]:
        res = orch.run_python_code(
            checker_holder["code"],
            filename=checker_holder["name"],
            timeout=min(orch.cfg.runtime.checker_timeout_cap_sec, orch.cfg.runtime.code_timeout_sec)
        )
        orch.log("checker_run", {
            "exit": res.get("exit_code", None),
            "stderr_tail": shorten_string_middle(res.get("errors", ""), 400)
        })
        return _parse_check_summary(res.get("output", "") or "")

    summary = run_checker()
    total = int(summary.get("total", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    fail_rate = (failed / total) if total else 0.0

    if not quiet:
        print(Fore.CYAN + f"[CHECK] total={total} failed={failed} rate={fail_rate:.1%} threshold={fail_threshold:.1%}")

    if total == 0 or fail_rate <= fail_threshold:
        if not quiet:
            print(Fore.GREEN + "[CHECK] порог выдержан — фиксы не требуются")
        return initial_answer

    current_answer = initial_answer
    attempt = 0
    while attempt < max_retries and fail_rate > fail_threshold:
        attempt += 1
        fail_report = "FAILED CHECKS:\n" + "\n".join(f"- {n}" for n in (summary.get("fail_names") or []))

        fix_code = fix_answer(llm_fast, task, current_answer, fail_report, spec)
        _ok_fix, _fix_issues = _audit_generated_code_policy(fix_code)
        if not _ok_fix:
            orch.log("fix_blocked_by_policy", {"issues": _fix_issues})
            break
        exec_res = orch.run_python_code(
            fix_code,
            filename=f"fix_{uuid.uuid4().hex[:6]}.py",
            timeout=min(orch.cfg.runtime.checker_timeout_cap_sec, orch.cfg.runtime.code_timeout_sec)
        )
        orch.log("fix_attempt", {
            "exit": exec_res.get("exit_code"),
            "stderr_tail": shorten_string_middle(exec_res.get("errors", ""), 400),
            "fail_rate_before": fail_rate
        })

        fixed_report = _extract_fixed_aggregate_report(exec_res.get("output") or "")
        if fixed_report:
            current_answer = fixed_report

        refreshed = checker_code_agent(
            llm_fast,
            task=task,
            spec=spec,
            code_summary=code_summary,
            final_answer=current_answer,
            metrics_json=metrics_json,
            stdout_tail=stdout_tail[-4000:],
            stderr_tail=stderr_tail[-4000:],
            improvement_summary=improvement_summary[-4000:],
        )
        _ok_ref, _ref_issues = _audit_generated_code_policy(refreshed)
        if not _ok_ref:
            orch.log("checker_refresh_blocked_by_policy", {"issues": _ref_issues})
            break
        checker_holder["code"] = refreshed
        checker_holder["name"] = f"check_{uuid.uuid4().hex[:6]}.py"

        summary = run_checker()
        total = int(summary.get("total", 0) or 0)
        failed = int(summary.get("failed", 0) or 0)
        fail_rate = (failed / total) if total else 0.0
        if not quiet:
            print(Fore.YELLOW + f"[CHECK] retry#{attempt}: total={total} failed={failed} rate={fail_rate:.1%}")

    if fail_rate > fail_threshold and not quiet:
        print(Fore.RED + f"[CHECK] не удалось опустить FAIL-rate ниже порога ({fail_rate:.1%} > {fail_threshold:.1%})")

    return current_answer


# -------------------- main pipeline --------------------
def improvement_generate_and_execute(
        orch: GlobalOrchestrator,
        llm_fast,
        code_llm,
        frozen_spec: Dict[str, Any],
        task: str,
        iter_idx: int,
        task_idx: int,
        part_name: str,
        max_iter: int = 0,
        previous_answers: str = "",
        node_id: Optional[str] = None,
        parent_node_id: Optional[str] = None,
        mcp_tools: List[Any] = None,
) -> Dict[str, Any]:
    with _track_node(orch, node_id, parent_node_id, kind="improve_leaf", task=task) as leaf_id:
        incident_node_id = str(leaf_id) if leaf_id else f"adhoc_{hashlib.md5((task or '').encode('utf-8')).hexdigest()[:10]}"
        # Modality-aware schema + artifact tools for improver coder
        try:
            _schema_snapshot = build_schema_snapshot(orch, frozen_spec) or ""
        except Exception:
            _schema_snapshot = ""
        try:
            _art_tools = build_structured_artifact_tools(orch) or []
        except Exception:
            _art_tools = []
        if mcp_tools is None:
            mcp_tools = []
        _existing_names = {getattr(t, "name", "") for t in mcp_tools}
        for _t in _art_tools:
            if getattr(_t, "name", "") not in _existing_names:
                mcp_tools.append(_t)
        # Comment translated to English.
        art_dir = Path(orch.cfg.paths.artifacts_dir)
        iter_dir = Path(orch.project_root) / art_dir / "improve" / f"iter_{iter_idx:02d}"
        _ensure_dir(iter_dir / "logs")
        _ensure_dir(iter_dir / "scripts")
        _ensure_dir(iter_dir / "submissions")

        # Comment translated to English.
        safe_name = _slug(part_name)[:50]
        scr_name = f"task_{iter_idx:02d}_{task_idx:02d}_{safe_name}.py"
        scr_rel = f"{orch.cfg.paths.scripts_dir}/{scr_name}"

        # Comment translated to English.
        improve_guidelines = (
            "IMPROVE_MODE RULES:\n"
            "1. **DATA INTEGRITY**: Load the FULL dataset from `spec.data`. NEVER use `nrows=...`, `.sample(...)`, or `head(...)` for training data loading. We need the full data.\n"
            "2. **MONOLITH PREFERENCE**: In competition mode, prefer a single unified code artifact for training/validation/prediction. Avoid fragile chains of multiple scripts unless the logic is extremely large (>1000 lines).\n"
            "3. **ARTIFACT SPECIFICATION**: Before using artifacts (e.g. .pkl, .csv), check `project_context.md` for their ACTUAL structure (columns, dict keys). Do NOT guess filenames or column names. If a file exists, use its verified name from context.\n"
            "4. **PATHS**: Keep input/output paths strictly from `spec` or `project_context.md`.\n"
            "5. **VALIDATION**: Preserve the CV protocol. If GroupKFold is used, ensure groups are respected.\n"
            "6. **METRICS**: You MUST calculate and print the metric defined in `spec['primary_metric']` as JSON to stdout.\n"
        )

        guidance_ctx = improve_guidelines + "\n" + (previous_answers or "")

        # Comment translated to English.
        verifier = FormalVerifier(llm=llm_fast)
        router = ErrorRouter(
            llm_fast,
            os_name=platform.system() or "Windows",
            repeat_to_lead=2,
            google_api_key=getattr(getattr(orch.cfg, "google", object()), "api_key", None),
            google_cse_id=getattr(getattr(orch.cfg, "google", object()), "cse_id", None),
            orch=orch,
        )

        # Comment translated to English.
        answer_code = ""
        max_gen_retries = 3
        for attempt in range(max_gen_retries):
            print(Fore.BLUE + f"[IMPROVER] Generating code for Task {task_idx} (Attempt {attempt + 1})...")
            answer_code = perform_task_python_v2(
                code_llm, task, frozen_spec,
                previous_code=code_bank[-1] if code_bank else "",
                context=guidance_ctx + f"Target Metric: {frozen_spec['primary_metric']['name']}",
                tools=mcp_tools,
                orch=orch,
                schema_snapshot=_schema_snapshot,
            )

            clean_code = answer_code.replace("```python", "").replace("```", "").strip()

            # Comment translated to English.
            safety_check = verifier.verify_code_safety(clean_code, frozen_spec)
            _policy_ok, _policy_issues = _audit_generated_code_policy(clean_code)
            if safety_check.valid and _policy_ok:
                break
            else:
                reason = safety_check.violation_reason if not safety_check.valid else "; ".join(_policy_issues)
                print(Fore.RED + f"[VERIFIER] Code Unsafe: {reason}")
                guidance_ctx += f"\n[SECURITY_FIX] Previous code rejected: {reason}. Fix it."

        try:
            _enforce_stack_guardrails(answer_code, orch)
        except Exception as e:
            print(Fore.RED + f"[GUARDRAIL] Violation: {e}")

        # Comment translated to English.
        last_stdout, last_errors = "", ""
        retry = 0
        metrics = {}
        _seen_improver_err_sigs: Dict[str, int] = {}  # error dedup within this task

        while True:
            preflight = react_preexec_auditor_agent(
                llm_fast=llm_fast,
                orch=orch,
                task=task,
                spec=frozen_spec,
                code_text=answer_code,
                context="Improver pre-execution audit. Inspect task_plan and artifacts before allowing run.",
            )
            allow_run = bool(preflight.get("allow_run", True))
            planning_only = bool(preflight.get("planning_only", False))
            preflight_issues = [str(x) for x in (preflight.get("issues") or [])]
            if planning_only:
                preflight_issues.append("planning_only_task_detected")

            # ── Verify "Missing dependency" claims programmatically ──
            if preflight_issues:
                preflight_issues = _verify_dependency_claims(orch, preflight_issues)
                if not preflight_issues:
                    allow_run = True

            if not allow_run or preflight_issues:
                print(Fore.YELLOW + f"[IMPROVER/PREFLIGHT][AGENT] blocked: {preflight_issues}")
                answer_code = finetune_code_v2(
                    code_llm,
                    task,
                    answer_code,
                    frozen_spec,
                    error=(
                        "Agentic preflight blocked execution. Fix all issues: "
                        + "; ".join(preflight_issues)
                        + ". Follow required_fixes and evidence from preflight, then regenerate executable code."
                    ),
                    tools=mcp_tools,
                )
                retry += 1
                if retry > max(2, int(getattr(orch.cfg.runtime, "generation_retry_limit", 6) or 6)):
                    print(Fore.RED + "[IMPROVER/PREFLIGHT] Retry limit exceeded for path-policy fixes.")
                    break
                continue

            # Comment translated to English.
            orch.write_file(scr_rel, answer_code.replace("```", "").replace("python", ""))
            _persist_last_code_artifact(orch, answer_code.replace("```", "").replace("python", ""))
            try:
                # Comment translated to English.
                _copy_text_file(orch, Path(orch.project_root) / scr_rel, iter_dir / "scripts" / scr_name)
            except Exception:
                pass

            print(Fore.MAGENTA + f"[IMPROVER] RUN {scr_name}")

            # Predict execution time
            prediction = execution_predictor_agent(llm_fast, answer_code, frozen_spec)

            # Per-task budget is the SOFT TARGET passed to the bash agent.
            # The bash agent will expand it up to `hard_cap` (run-wide remaining)
            # when the predictor says the step needs more — see bash_agent.run.
            task_node = orch.tree_node(leaf_id) if leaf_id else {}
            task_budget = task_node.get("time_budget_sec", orch.cfg.runtime.default_task_budget_sec)

            predicted_time = prediction.get('expected_time_sec', orch.cfg.runtime.prediction_fallback_sec)

            # Recalculate remaining time for logging only — the actual hard
            # cap is computed inside orchestrator.run_python_file from the
            # global deadline, so we don't pre-clamp here.
            current_remaining = getattr(
                orch, "global_deadline_sec", orch.cfg.orchestration.total_budget_sec
            ) - orch.effective_elapsed_sec()

            floor = orch.cfg.runtime.min_exec_timeout_sec
            soft_target = max(int(floor), int(task_budget))

            print(
                Fore.CYAN
                + f"[IMPROVER/MONITOR] Predicted time: {predicted_time}s, "
                + f"Intensity: {prediction.get('resource_intensity')}. "
                + f"Soft target: {soft_target}s, hard_cap≈{int(max(0, current_remaining))}s, "
                + f"floor={floor}s (predictor may expand within hard_cap)"
            )

            res = orch.run_python_file(
                scr_rel,
                stream=True,
                spec=frozen_spec,
                prediction=prediction,
                timeout=soft_target,
            )
            last_stdout, last_errors = res.get("output", ""), res.get("errors", "")

            # Comment translated to English.
            try:
                orch.write_file(str(iter_dir / "logs" / f"{safe_name}_stdout.txt"), last_stdout[-100000:])
                orch.write_file(str(iter_dir / "logs" / f"{safe_name}_stderr.txt"), last_errors[-50000:])
            except Exception:
                pass

            # Comment translated to English.
            heuristic_ok = (last_errors == "" and len(last_stdout) > 2) or ("error" not in last_errors.lower())

            # Comment translated to English.
            metrics = parse_metrics_from_stdout(last_stdout) or {}

            art_summary = ""
            try:
                art_summary = update_project_context_after_execution(orch, task, last_stdout, last_errors, metrics, answer_code)
            except Exception as e:
                print(Fore.RED + f"Failed to update project context: {e}")

            # Comment translated to English.
            if metrics.get('type') == 'skipped':
                print(Fore.CYAN + f"[IMPROVER] Task skipped by model: {metrics.get('reason', 'No reason')}")
                preflight_after = react_preexec_auditor_agent(
                    llm_fast=llm_fast,
                    orch=orch,
                    task=task,
                    spec=frozen_spec,
                    code_text=answer_code,
                    context="Detect planning-only output after skipped METRICS_JSON.",
                )
                if bool(preflight_after.get("planning_only", False)):
                    print(Fore.YELLOW + "[IMPROVER] planning-only task detected; forcing executable recovery task.")
                    answer_code = perform_task_python_v2(
                        code_llm,
                        "Inspect artifacts/checkpoints and produce canonical submission.csv + METRICS_JSON for current iteration",
                        frozen_spec,
                        previous_code=answer_code,
                        context=(
                            guidance_ctx
                            + "\nAvoid planning-only output. Must execute code and create concrete artifacts."
                        ),
                        tools=mcp_tools,
                        orch=orch,
                        schema_snapshot=_schema_snapshot,
                    )
                    retry += 1
                    if retry <= max(2, int(getattr(orch.cfg.runtime, "metric_validation_retry_limit", 2) or 2)):
                        continue
                # Comment translated to English.
                if heuristic_ok:
                    break

            # Comment translated to English.
            llm_ok = False
            if heuristic_ok and metrics:
                llm_ok = True
            else:
                llm_ok = evaluate_run_ok_with_retry(
                    llm_fast=llm_fast,
                    stdout=last_stdout,
                    stderr=last_errors,
                    spec=frozen_spec,
                    code_text=answer_code,
                    additional_context=f"Metric {frozen_spec['primary_metric']['name']} is REQUIRED."
                )

            if llm_ok:
                break

            # ── Error-signature dedup (improver) ──────────────────────────────
            if last_errors:
                _ierr_sig = (last_errors or "").strip()[:200]
                _seen_improver_err_sigs[_ierr_sig] = _seen_improver_err_sigs.get(_ierr_sig, 0) + 1
                if _seen_improver_err_sigs[_ierr_sig] >= 3:
                    print(Fore.RED + f"[IMPROVER/TRIAGE] Same error seen {_seen_improver_err_sigs[_ierr_sig]}x — forcing lead escalation.")
                    plan = router.route(last_errors, last_stdout, frozen_spec, answer_code)
                    plan["route"] = "lead"
                    plan["reason"] = (
                        f"Error repeated {_seen_improver_err_sigs[_ierr_sig]} times. "
                        + plan.get("reason", "")
                    )
                    route = "lead"
                else:
                    plan = router.route(last_errors, last_stdout, frozen_spec, answer_code)
                    route = plan.get("route", "coding")
            else:
                plan = router.route(last_errors, last_stdout, frozen_spec, answer_code)
                route = plan.get("route", "coding")
            attempts_hist = []
            try:
                attempts_hist = (orch.state_get(incident_node_id) or {}).get("attempts", []) or []
            except Exception:
                attempts_hist = []
            if route == "lead":
                managed_plan = lead_incident_manager_agent(
                    llm_fast,
                    task=task,
                    spec=frozen_spec,
                    triage_plan=plan,
                    attempts=attempts_hist,
                    stderr_tail=last_errors,
                    stdout_tail=last_stdout,
                    code_head=answer_code,
                )
                if isinstance(managed_plan, dict) and managed_plan.get("route"):
                    plan = managed_plan
                    route = plan.get("route", route)
            print(Fore.YELLOW + f"[IMPROVER/TRIAGE] attempts={len(attempts_hist)} route={route} reason={plan.get('reason', '')}")
            orch.state_append_attempt(incident_node_id, {
                "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "phase": "triage",
                "route": route,
                "reason": str(plan.get("reason", ""))[:500],
            })

            if route == "install":
                pkgs = plan.get("packages", [])
                if pkgs:
                    print(Fore.CYAN + f"Installing: {pkgs}")
                    install_res = orch.pip_install(pkgs)
                    orch.state_append_attempt(incident_node_id, {
                        "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        "phase": "action",
                        "route": "install",
                        "packages": pkgs,
                        "exit_code": install_res.get("exit_code", 1),
                        "stderr_tail": str(install_res.get("stderr", ""))[-1000:],
                    })
                    if install_res.get("exit_code", 1) != 0:
                        last_errors = f"{install_res.get('stderr', '')}\n{install_res.get('stdout', '')}"
                        print(Fore.RED + "[IMPROVER/TRIAGE] install failed; re-triaging based on pip output.")
                        # Track per-package failures so router can stop retrying hopeless packages.
                        if hasattr(router, "_install_fail_counts"):
                            for _pkg in pkgs:
                                _pk = (_pkg or "").lower().strip()
                                if _pk:
                                    router._install_fail_counts[_pk] = router._install_fail_counts.get(_pk, 0) + 1
                        retry += 1
                        if retry > orch.cfg.runtime.router_retry_limit:
                            print(Fore.RED + "[IMPROVER] Retry limit exceeded after install failures.")
                            break
                        continue
                else:
                    route = "coding"  # Fallback

            if route == "spec_update":
                # Comment translated to English.
                guidance = "SPEC IS FROZEN. Adapt CODE to data/environment. Do NOT change spec."
                answer_code = perform_task_python_v2(
                    code_llm, task, frozen_spec,
                    previous_code=code_bank[-1] if code_bank else "",
                    context=improve_guidelines + "\n" + guidance + "\nPREV_ERROR:\n" + last_errors[:1200],
                    tools=mcp_tools,
                    orch=orch,
                    schema_snapshot=_schema_snapshot,
                )

            elif route == "bash":
                for cmd in _sanitize_triage_bash_cmds(plan.get("bash_cmds", [])):
                    print(Fore.CYAN + f"[BASH] {cmd}")
                    orch.bash.run(cmd, timeout=orch.cfg.runtime.bash_timeout_sec, stream=True)
                    orch.state_append_attempt(incident_node_id, {
                        "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        "phase": "action",
                        "route": "bash",
                        "cmd": str(cmd)[:500],
                    })
                # Comment translated to English.
                answer_code = finetune_code_v2(
                    code_llm, task, answer_code, frozen_spec,
                    error="Bash commands executed. Now fix the code if needed.\n" + last_errors
                , tools=mcp_tools)

            elif route == "lead":
                note = plan.get("notes", "")
                answer_code = perform_task_python_v2(
                    code_llm, task, frozen_spec,
                    previous_code=code_bank[-1] if code_bank else "",
                    context=improve_guidelines + "\nLEAD_ADVICE: " + note,
                    tools=mcp_tools,
                    orch=orch,
                    schema_snapshot=_schema_snapshot,
                )

            else:  # coding
                answer_code = finetune_code_v2(
                    code_llm, task, answer_code, frozen_spec,
                    error=improve_guidelines + "\n" + (last_errors or ""),
                    tools=mcp_tools
                )

            _ok_new, _issues_new = _audit_generated_code_policy(answer_code)
            if not _ok_new:
                print(Fore.RED + f"[IMPROVER/POLICY] Generated code blocked: {_issues_new}")
                answer_code = finetune_code_v2(
                    code_llm, task, answer_code, frozen_spec,
                    error=(
                        "Code policy violation detected: "
                        + "; ".join(_issues_new)
                        + ". Remove destructive operations and load spec dynamically from artifacts/spec.json."
                    ),
                    tools=mcp_tools
                )

            retry += 1
            if retry > orch.cfg.runtime.router_retry_limit:  # Comment translated to English.
                print(Fore.RED + "[IMPROVER] Retry limit exceeded for this task.")
                break

        # Comment translated to English.
        try:
            orch.write_file(str(iter_dir / f"metrics_{task_idx:02d}.json"),
                            json.dumps(metrics or {"missing": True}, ensure_ascii=False, indent=2))
        except Exception:
            pass

        submission_rel = _detect_and_store_submissions(orch, frozen_spec, iter_dir=iter_dir)

        try:
            clean_code = answer_code.replace("```", "").replace("python", "")
            part_tag = f"improve_iter_{iter_idx:02d}_{safe_name}"

            if metrics and metrics.get('type') != 'skipped':
                _update_best_from_candidate(
                    orch,
                    candidate_metrics=metrics,
                    code_text=clean_code,
                    tag=part_tag,
                    enforce_validation=True,
                    submission_path=submission_rel,
                    spec=frozen_spec,
                )
        except Exception as e:
            print(Fore.YELLOW + f"[IMPROVER] couldn't update metrics/best: {e}")

        try:
            orch.tree_finish(leaf_id, status="done", meta={
                "iter_idx": iter_idx,
                "task_idx": task_idx,
                "metrics": metrics or {},
                "script": scr_rel,
                "artifact_summary": art_summary
            })
        except Exception:
            pass

        return {
            "stdout": last_stdout,
            "stderr": last_errors,
            "exit_code": 0 if ("error" not in (last_errors or "").lower()) else 1,
            "code": answer_code,
            "metrics": metrics or {},
        }


def _normalize_improve_plan_tasks(raw: List[Any], default_budget: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    db = max(60, int(default_budget))
    for item in raw or []:
        if isinstance(item, dict):
            t = str(item.get("task", item)).strip()
            try:
                tb = int(item.get("time_budget_sec", db) or db)
            except Exception:
                tb = db
            out.append({"task": t, "time_budget_sec": max(30, tb)})
        else:
            s = str(item).strip()
            if s:
                out.append({"task": s, "time_budget_sec": db})
    return out


def _dedupe_improve_plan_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for x in tasks:
        k = (x.get("task") or "").lower().strip()[:280]
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _improve_level1_prerun_context(
    task: str,
    previous_answers: str,
    summary_lines: List[str],
    cur_best: Dict[str, Any],
    frozen_spec: Dict[str, Any],
    iter_i: int,
    iters: int,
) -> str:
    """
    Rich context for top-level (depth==0) replanning before the first task of an improver iteration runs.
    """
    parts: List[str] = []
    parts.append(
        f"IMPROVER_ITERATION: {iter_i} / {iters} — no tasks in this iteration have finished yet."
    )
    parts.append(f"OVERALL_GOAL:\n{shorten_string_middle(task or '', 3500)}")
    pm = frozen_spec.get("primary_metric") if isinstance(frozen_spec, dict) else {}
    parts.append(
        "PRIMARY_METRIC_CONTRACT:\n"
        + shorten_string_middle(json.dumps(pm or {}, ensure_ascii=False), 2500)
    )
    parts.append(
        "CURRENT_BEST_METRICS_JSON:\n"
        + shorten_string_middle(json.dumps(cur_best or {}, ensure_ascii=False), 3500)
    )
    if previous_answers and str(previous_answers).strip():
        parts.append(
            "PRIOR_PIPELINE_AND_MAIN_BRANCH_NOTES:\n"
            + shorten_string_middle(str(previous_answers).strip(), 4000)
        )
    tail = "\n".join(summary_lines[-24:])
    parts.append("RECENT_IMPROVER_AND_RUN_SUMMARY:\n" + (tail.strip() or "(none)"))
    return "\n\n".join(parts)


def _collect_main_pipeline_artifacts(orch: GlobalOrchestrator, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Read all .md + key .json artifacts from main pipeline for improver context."""
    art_dir = Path(orch.project_root) / orch.cfg.paths.artifacts_dir
    artifacts: Dict[str, Any] = {}

    # Read all .md files from artifacts dir
    try:
        for md_file in art_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                artifacts[md_file.name] = content[:4000]
            except Exception:
                pass
    except Exception:
        pass

    # Read best metrics
    for mp in (art_dir / "best" / "metrics.json", art_dir / "last" / "metrics.json"):
        try:
            if mp.exists():
                artifacts["best_metrics"] = json.loads(mp.read_text(encoding="utf-8"))
                break
        except Exception:
            pass

    # Read best code
    for cp in (art_dir / "best" / "code.py", art_dir / "last" / "code.py"):
        try:
            if cp.exists():
                artifacts["best_code"] = cp.read_text(encoding="utf-8")[:8000]
                break
        except Exception:
            pass

    # Data schema from spec
    csv_summaries = (spec.get("data") or {}).get("meta", {}).get("csv_summaries", {})
    if isinstance(csv_summaries, dict) and csv_summaries:
        artifacts["data_schema"] = {
            name: {"columns": info.get("columns", []), "dtypes": info.get("dtypes", {})}
            for name, info in csv_summaries.items()
            if isinstance(info, dict)
        }

    # Version history (score progression)
    versions_dir = art_dir / "versions"
    try:
        if versions_dir.exists():
            scores = []
            for vdir in sorted(versions_dir.iterdir()):
                if vdir.is_dir():
                    vm = vdir / "metrics.json"
                    if vm.exists():
                        try:
                            vdata = json.loads(vm.read_text(encoding="utf-8"))
                            scores.append({"version": vdir.name, "primary": vdata.get("primary")})
                        except Exception:
                            pass
            if scores:
                artifacts["version_history"] = scores[-10:]
    except Exception:
        pass

    return artifacts


def _collect_and_enrich_artifacts(
    orch: GlobalOrchestrator,
    spec: Dict[str, Any],
    llm_strong,
    task: str,
    previous_iteration_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Two-phase artifact collection:
    1. Basic scan via _collect_main_pipeline_artifacts (fast, no LLM)
    2. ReAct agent deep analysis (reads files itself, 5 iterations, Kaggle Master level)
    Returns merged enriched artifacts dict.
    """
    # Phase 1: basic scan
    initial_scan = _collect_main_pipeline_artifacts(orch, spec)
    print(Fore.CYAN + f"[ARTIFACTS] Basic scan: {len(initial_scan)} artifacts collected")

    # Phase 2: ReAct deep analysis
    try:
        enriched = react_artifacts_collector_agent(
            llm_strong,
            orch,
            initial_scan=initial_scan,
            spec=spec,
            task=task,
            previous_iteration_context=previous_iteration_context,
            max_steps=5,
        )
        if enriched and isinstance(enriched, dict):
            _extra_keys = [k for k in enriched if k not in initial_scan]
            print(Fore.CYAN + f"[ARTIFACTS] ReAct enrichment added {len(_extra_keys)} new keys: {_extra_keys[:5]}")
            return enriched
    except Exception as e:
        print(Fore.YELLOW + f"[ARTIFACTS] ReAct enrichment failed ({e}), using basic scan")

    return initial_scan


def improvement_pipeline(
        orch: GlobalOrchestrator,
        llm_strong,
        llm_fast,
        code_llm,
        task: str,
        spec: Dict[str, Any],
        previous_answers: str = "",
        node_id: str | None = None,
        parent_node_id: str | None = None,
        resume: bool = True,
        max_iters_override: int | None = None,
        mcp_tools: Optional[List[Any]] = None,
        depth: int = 0,
        improve_start_time: float | None = None,
        main_pipeline_artifacts: Dict[str, Any] | None = None,
) -> tuple[str, Dict[str, Any]]:
    if improve_start_time is None:
        improve_start_time = time.time()

    # Normalize spec to JSON-safe primitives so improver paths do not crash on Path-like values.
    try:
        spec = _json_roundtrip_safe(spec or {})
    except Exception:
        spec = {}

    # Build modality-aware artifacts index (images/audio/text/tabular/pickle probes)
    try:
        _idx_ref = write_artifacts_index(orch)
        _idx: Dict[str, Any] = {}
        if isinstance(_idx_ref, dict):
            _idx = _idx_ref
        elif isinstance(_idx_ref, Path) and _idx_ref.exists():
            try:
                _idx = json.loads(_idx_ref.read_text(encoding="utf-8"))
            except Exception:
                _idx = {}
        if main_pipeline_artifacts is None:
            main_pipeline_artifacts = {}
        main_pipeline_artifacts["artifacts_index"] = _idx
        print(Fore.CYAN + f"[IMPROVER] artifacts_index written ({len(_idx.get('files', {}))} files indexed)")
    except Exception as _e:
        print(Fore.YELLOW + f"[IMPROVER] write_artifacts_index failed: {_e}")

    # FIX: Use direct attribute access instead of .get() for dataclass
    improve_budget = float(orch.cfg.orchestration.improve_budget_sec)

    max_depth = int(getattr(orch.cfg.orchestration, "max_tree_depth", 5))
    max_width = int(getattr(orch.cfg.orchestration, "max_tree_width", 4))
    max_tasks_per_iter = int(getattr(orch.cfg.orchestration, "max_improve_tasks_per_iter", 7))
    plan_cap = max(1, min(max_tasks_per_iter, max_width))
    default_tb = int(getattr(orch.cfg.runtime, "default_task_budget_sec", 1800))
    # At or beyond configured depth: no deeper nested improve nodes (prevents runaway trees).
    force_leaf_only = depth >= max_depth
    if force_leaf_only:
        print(Fore.YELLOW + f"[IMPROVER] Depth {depth} >= max_tree_depth {max_depth}. Nested improves disabled (leaf-only).")

    if resume and not node_id:
        next_id = orch.tree_pick_next_node(prefer_kind="improve")
        if next_id:
            node = orch.tree_node(next_id)
            node_id = next_id
            parent_node_id = node.get("parent_node_id")

    with _track_node(orch, node_id, parent_node_id, kind="improve", task=task) as this_node_id:
        global summary_lines

        orc = getattr(orch.cfg, "orchestration", None)
        if isinstance(orc, dict):
            iters = int(orc.get("optimize_iters", 4))
            rel_thr = float(orc.get("min_metric_improvement_rel", 0.05))
        else:
            iters = int(getattr(orc, "optimize_iters", 4))
            rel_thr = float(getattr(orc, "min_metric_improvement_rel", 0.05))

        if max_iters_override is not None:
            iters = int(max(1, max_iters_override))

        frozen_spec = _json_roundtrip_safe(spec)
        improve_root = Path(orch.project_root) / orch.cfg.paths.artifacts_dir / "improve"
        _ensure_dir(improve_root)

        # Inject main pipeline's best code into code_bank for improver context
        if main_pipeline_artifacts and main_pipeline_artifacts.get("best_code"):
            best_code = main_pipeline_artifacts["best_code"]
            if best_code.strip():
                code_bank.insert(0, best_code)
                print(Fore.CYAN + f"[IMPROVER] Injected main pipeline best code ({len(best_code)} chars) into code_bank")

        try:
            orch.write_file(str(improve_root / "spec_frozen.json"),
                            json.dumps(frozen_spec, ensure_ascii=False, indent=2))
        except Exception:
            pass

        # Improve resume metadata for Improver Head / debugging (does not replace tree.json resume).
        try:
            prev_head = {}
            prp = improve_root / "pipeline_resume.json"
            if resume and prp.exists():
                prev_head = json.loads(prp.read_text(encoding="utf-8"))
        except Exception:
            prev_head = {}

        root = Path(orch.project_root)
        art_dir = root / orch.cfg.paths.artifacts_dir
        best_path = art_dir / "best" / "metrics.json"
        last_path = art_dir / "last" / "metrics.json"
        try:
            cur_best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else \
                (json.loads(last_path.read_text(encoding="utf-8")) if last_path.exists() else {})
        except Exception:
            cur_best = {}

        summary_lines.append(f"[IMPROVER] start: iters={iters}, rel_thr={rel_thr}")

        verifier = FormalVerifier(llm=llm_fast)
        safety_spec = VerificationSpec(
            max_steps=getattr(orch.cfg.orchestration, "improve_verifier_max_steps", 10),
            required_preconditions={},
            forbidden_states=["data_corruption", "metric_degradation"],
            resource_limits={"complexity": 5.0, "risk": 3.0}
        )

        _min_split_imp = int(getattr(orch.cfg.orchestration, "min_remaining_sec_to_split", 600))
        _rem_improve_node = max(0, int(improve_budget - (time.time() - improve_start_time)))
        if node_id is None:
            task_double = "True"
        else:
            task_double = task_complexity_check(
                llm_fast,
                task,
                previous_answers or "Metric improvement context",
                tree_depth=depth,
                remaining_total_sec=_rem_improve_node,
                min_split_sec=_min_split_imp,
                tree_max_depth=max_depth,
            )

        if force_leaf_only:
            task_double = "False"

        if "false" in str(task_double).lower():
            # Single task mode
            res = improvement_generate_and_execute(
                orch=orch,
                llm_fast=llm_fast,
                code_llm=code_llm,
                frozen_spec=frozen_spec,
                task=f"Improve: {task}",
                iter_idx=0,
                task_idx=1,
                part_name="single_opt",
                previous_answers=previous_answers,
                max_iter=0,
                node_id=None,
                parent_node_id=this_node_id,
            )
            try:
                new_best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else \
                    (json.loads(last_path.read_text(encoding="utf-8")) if last_path.exists() else {})
            except Exception:
                new_best = {}
            if _rel_better(new_best, cur_best, rel_thr=rel_thr):
                summary_lines.append(
                    f"[IMPROVER] direct: improved {cur_best.get('primary')} -> {new_best.get('primary')}")
                cur_best = new_best or cur_best
            else:
                summary_lines.append(
                    f"[IMPROVER] direct: no relative improvement (kept {cur_best.get('primary')}, got {new_best.get('primary')})")
            try:
                _finalize_single_submission_by_all_metrics_llm(
                    orch, llm_fast, frozen_spec, task=task, code_llm=code_llm, mcp_tools=mcp_tools
                )
                gate = run_final_output_gate(orch, frozen_spec, task_txt_root=Path(orch.project_root))
                summary_lines.append(f"[IMPROVER][FINAL_GATE] ok={gate.get('ok')} errors={gate.get('errors')}")
            except Exception as e:
                summary_lines.append(f"[IMPROVER][FINAL_GATE] failed: {e}")
            return "\n".join(summary_lines), cur_best or {}

        # Iterative mode
        for i in range(1, max(1, iters) + 1):
            rem_improve = improve_budget - (time.time() - improve_start_time)
            if rem_improve <= 30:
                summary_lines.append(
                    f"[IMPROVER] improve_budget_sec exhausted (~{int(rem_improve)}s left); stopping iterations and finalizing."
                )
                break

            iter_dir = improve_root / f"iter_{i:02d}"
            _ensure_dir(iter_dir)

            # --- Child filtering logic (preserved from previous fix) ---
            all_children = orch.tree_children_ordered(this_node_id)
            current_children = []
            iter_prefix = f"Iter {i}: "
            legacy_prefix = "Improve: "

            if i == 1:
                current_children = [
                    c for c in all_children
                    if c['task'].startswith(iter_prefix) or
                       (c['task'].startswith(legacy_prefix) and not c['task'].startswith("Iter "))
                ]
            else:
                current_children = [c for c in all_children if c['task'].startswith(iter_prefix)]

            if current_children:
                print(
                    Fore.YELLOW + f"[IMPROVER/RESUME] Resuming iteration {i} with {len(current_children)} existing tasks.")
                ordered_tasks = []
                for c in current_children:
                    t_name = c['task']
                    if t_name.startswith(iter_prefix):
                        stripped = t_name[len(iter_prefix):]
                    elif i == 1 and t_name.startswith(legacy_prefix):
                        stripped = t_name[len(legacy_prefix):]
                    else:
                        stripped = t_name
                    cid0 = c.get("node_id")
                    meta_tb = default_tb
                    try:
                        if cid0:
                            meta_tb = int((orch.tree_node(cid0).get("meta") or {}).get("time_budget_sec", default_tb))
                    except Exception:
                        meta_tb = default_tb
                    ordered_tasks.append({"task": stripped, "time_budget_sec": max(30, meta_tb)})

                child_ids = [c['node_id'] for c in current_children]
                kinds = [c['kind'] for c in current_children]
                if force_leaf_only:
                    kinds = ["improve_leaf"] * len(kinds)
            else:
                # Generation logic
                max_gen_retries = 3
                ordered_tasks = []

                # --- ARTIFACT REVIEW STEP ---
                artifacts_snapshot = ""
                try:
                    files = [f.name for f in art_dir.glob("*") if f.is_file()]
                    artifacts_snapshot = "Files: " + ", ".join(files)
                    if (art_dir / "best").exists():
                        best_files = [f.name for f in (art_dir / "best").glob("*")]
                        artifacts_snapshot += "\nBest Version Files: " + ", ".join(best_files)
                except Exception:
                    artifacts_snapshot = "Could not read artifacts directory."

                print(Fore.CYAN + f"[IMPROVER] Reviewing artifacts before iteration {i}...")
                latest_code = code_bank[-1] if code_bank else ""
                review_doc = review_artifacts_agent(
                    llm_strong, task, cur_best or {},
                    latest_code,
                    artifacts_snapshot
                )
                print(Fore.MAGENTA + f"--- Technical Review ---\n{review_doc}\n-----------------------")

                # --- ReAct Meta-Planner step (guides deep task decomposition) ---
                # If it returns a safe non-empty deep plan, we skip the legacy task generation path.
                used_meta_planner = False
                try:
                    current_improve_remaining = improve_budget - (time.time() - improve_start_time)
                    meta_pct = float(getattr(orch.cfg.orchestration, "meta_planner_time_pct", 0.25))
                    meta_attempts = int(getattr(orch.cfg.orchestration, "meta_planner_max_attempts", 3))
                    meta_attempts = max(1, meta_attempts)
                    meta_budget_sec = int(max(0, current_improve_remaining) * max(0.0, meta_pct))
                    meta_budget_sec = min(meta_budget_sec, max(0, int(current_improve_remaining)))
                    # Keep budget <= available remaining time slice.
                    if meta_budget_sec <= 0:
                        meta_budget_sec = max(0, int(current_improve_remaining))
                    if current_improve_remaining >= 60:
                        meta_budget_sec = max(60, meta_budget_sec)

                    recent_tail = shorten_string_middle("\n".join(summary_lines[-8:]) or "", 6000)
                    graph_hint = shorten_string_middle(orch.format_task_graph_to_string(), 3500)

                    # Build rich artifacts_hint with actual .md content from main pipeline
                    md_content_block = ""
                    if main_pipeline_artifacts:
                        for fname, content in main_pipeline_artifacts.items():
                            if isinstance(content, str) and fname.endswith(".md"):
                                md_content_block += f"\n--- {fname} ---\n{content[:2000]}\n"
                        if main_pipeline_artifacts.get("data_schema"):
                            md_content_block += f"\n--- data_schema ---\n{json.dumps(main_pipeline_artifacts['data_schema'], ensure_ascii=False)[:2000]}\n"
                        if main_pipeline_artifacts.get("version_history"):
                            md_content_block += f"\n--- version_history ---\n{json.dumps(main_pipeline_artifacts['version_history'], ensure_ascii=False)[:1500]}\n"
                        if main_pipeline_artifacts.get("best_metrics"):
                            md_content_block += f"\n--- best_metrics ---\n{json.dumps(main_pipeline_artifacts['best_metrics'], ensure_ascii=False)[:1000]}\n"
                        if main_pipeline_artifacts.get("artifacts_index"):
                            _idx = main_pipeline_artifacts["artifacts_index"]
                            # Compact: file names + per-file kind/shape/columns/class_counts
                            _idx_files = _idx.get("files", {}) if isinstance(_idx, dict) else {}
                            md_content_block += f"\n--- artifacts_index ({len(_idx_files)} files) ---\n{json.dumps(_idx_files, ensure_ascii=False)[:3500]}\n"

                    artifacts_hint = (
                        f"Project root: {orch.project_root}\n"
                        f"Artifacts dir: {art_dir}\n"
                        f"Prefer reading (if exists): {art_dir/'aggregate_summary.md'}, "
                        f"{art_dir/'versions'/'ledger.md'}, {art_dir/'final'/'submission_validation.json'}, "
                        f"{art_dir/'improve'/'pipeline_resume.json'}, and the latest improve/iter_* metrics.\n"
                        f"\nMAIN PIPELINE ARTIFACTS CONTENT:\n{md_content_block}\n"
                    )

                    # Knowledge Curator BEFORE: tailored context for the improver meta-planner.
                    _cur_mp_ctx = _curator_before(orch, llm_fast, llm_strong, role="meta_planner", task_hint=task or "")
                    if _cur_mp_ctx:
                        artifacts_hint = (artifacts_hint or "") + "\n" + _cur_mp_ctx
                    for meta_try in range(meta_attempts):
                        meta_obj = react_improver_meta_planner_agent(
                            llm_strong,
                            orch,
                            task=task,
                            spec=frozen_spec,
                            metrics_summary=cur_best or {},
                            recent_summaries=recent_tail + ("\n\nTechnical review:\n" + shorten_string_middle(review_doc, 3500)),
                            depth=depth,
                            max_depth=max_depth,
                            remaining_improve_sec=meta_budget_sec,
                            artifacts_hint=artifacts_hint,
                            graph_hint=graph_hint,
                            attempt_idx=meta_try,
                        )

                        hl = meta_obj.get("high_level_plan") or []
                        deep_raw: list[dict[str, Any]] = []
                        for h in hl if isinstance(hl, list) else []:
                            deep_tasks = h.get("deep_tasks") or []
                            if not isinstance(deep_tasks, list):
                                continue
                            for dt in deep_tasks:
                                if not isinstance(dt, dict):
                                    continue
                                ttxt = str(dt.get("task", "") or "").strip()
                                if not ttxt:
                                    continue
                                acs = dt.get("acceptance_checks") or []
                                if isinstance(acs, list) and acs:
                                    short_acs = [str(x).strip() for x in acs[:3] if str(x).strip()]
                                    if short_acs:
                                        ttxt = ttxt + "\nAcceptance checks:\n" + "\n".join([f"- {x}" for x in short_acs])
                                deep_raw.append(
                                    {
                                        "task": ttxt,
                                        "time_budget_sec": int(dt.get("time_budget_sec") or default_tb),
                                    }
                                )

                        normalized = _normalize_improve_plan_tasks(deep_raw, default_tb)
                        normalized = _dedupe_improve_plan_tasks(normalized)[:plan_cap]
                        if not normalized:
                            continue

                        task_strs = [t["task"] for t in normalized]
                        print(Fore.CYAN + f"[META-PLANNER] Verifying safety for meta deep plan (try {meta_try + 1}/{meta_attempts})...")
                        ver_result = verifier.verify_plan(
                            task_strs,
                            safety_spec,
                            context=f"Improvement iteration {i}. Derived from ReAct meta-planner. Current Best: {cur_best}",
                        )
                        if ver_result.valid:
                            ordered_tasks = normalized
                            used_meta_planner = True
                            report_md = str(meta_obj.get("report_markdown") or "")
                            if report_md.strip():
                                try:
                                    orch.write_file(str(iter_dir / "meta_planner_report.md"), report_md)
                                    orch.write_file(str(art_dir / "meta_planner_report.md"), report_md)
                                except Exception:
                                    pass
                            break

                except Exception as e:
                    print(Fore.YELLOW + f"[META-PLANNER] Meta-planner failed/disabled: {e}")

                for gen_attempt in range(max_gen_retries):
                    if ordered_tasks:
                        break
                    try:
                        imp_tasks = improvement_tasks_generation(
                            llm_strong,
                            task=task,
                            spec=frozen_spec,
                            code_bank=[review_doc, latest_code],
                            metrics=cur_best or {},
                            max_tasks=plan_cap,
                            constraints_block=format_spec_constraints_block(frozen_spec),
                        )
                    except Exception as e:
                        print(Fore.RED + f"[IMPROVER] tasks generation failed: {e}")
                        imp_tasks = []

                    normalized = _normalize_improve_plan_tasks(imp_tasks, default_tb)
                    normalized = _dedupe_improve_plan_tasks(normalized)[:plan_cap]
                    order_list = [t["task"] for t in normalized]
                    try:
                        ordered_raw = order_tasks_with_retry(
                            llm_strong, task, order_list, frozen_spec, max_retries=3,
                            overall_time_limit_sec=max(0, int(current_improve_remaining)),
                            constraints_block=format_spec_constraints_block(frozen_spec),
                        )
                        ordered_norm = _normalize_improve_plan_tasks(ordered_raw, default_tb)
                        ordered_norm = _dedupe_improve_plan_tasks(ordered_norm)[:plan_cap]
                        candidates = ordered_norm if ordered_norm else normalized
                    except Exception:
                        candidates = normalized

                    cand_norm = _dedupe_improve_plan_tasks(
                        _normalize_improve_plan_tasks(candidates, default_tb)
                    )[:plan_cap]
                    task_strs = [t["task"] for t in cand_norm]

                    print(Fore.CYAN + f"[VERIFIER] Checking plan safety (Attempt {gen_attempt + 1})...")
                    ver_result = verifier.verify_plan(
                        task_strs,
                        safety_spec,
                        context=f"Improvement iteration {i}. Current Best: {cur_best}"
                    )

                    if ver_result.valid:
                        print(Fore.GREEN + "[VERIFIER] Plan Verified: SAFE")
                        ordered_tasks = cand_norm
                        break
                    else:
                        print(Fore.RED + f"[VERIFIER] Plan Rejected: {ver_result.violation_reason}")
                        task += f"\n[PLANNING CONSTRAINT] Previous plan rejected. Reason: {ver_result.violation_reason}. Avoid this pattern."

                if not ordered_tasks:
                    print(Fore.RED + "[IMPROVER] Could not generate verified plan. Using fallback.")
                    try:
                        ordered_tasks = _dedupe_improve_plan_tasks(
                            _normalize_improve_plan_tasks(locals().get("candidates", []), default_tb)
                        )[:plan_cap]
                    except Exception:
                        ordered_tasks = []

                kinds: list[str] = []
                _rim_loop = max(0, int(improve_budget - (time.time() - improve_start_time)))
                for tsk in ordered_tasks:
                    t_str = tsk.get("task", "") if isinstance(tsk, dict) else str(tsk)
                    if force_leaf_only:
                        kinds.append("improve_leaf")
                        continue
                    sub_double = task_complexity_check(
                        llm_fast,
                        f"Improve: {t_str}",
                        previous_answers or "Metric improvement subtask",
                        tree_depth=depth + 1,
                        remaining_total_sec=_rim_loop,
                        min_split_sec=_min_split_imp,
                        tree_max_depth=max_depth,
                    )
                    kinds.append("improve_leaf" if "false" in str(sub_double).lower() else "improve")

                creation_prefix = legacy_prefix if i == 1 else iter_prefix
                child_ids = orch.tree_init_children_with_kinds(this_node_id,
                                                               [
                                                                   f"{creation_prefix}{t.get('task') if isinstance(t, dict) else t}"
                                                                   for t in ordered_tasks],
                                                               kinds=kinds)
                try:
                    for idx_c, cid_c in enumerate(child_ids):
                        tb = ordered_tasks[idx_c].get("time_budget_sec", default_tb) if idx_c < len(ordered_tasks) else default_tb
                        orch.tree_update_meta(cid_c, {"time_budget_sec": int(tb)})
                except Exception:
                    pass

            try:
                orch.write_file(str(iter_dir / "tasks.json"),
                                json.dumps({"tasks": ordered_tasks}, ensure_ascii=False, indent=2))
            except Exception:
                pass

            # --- Improved Logging: Show Plan ---
            print(Fore.MAGENTA + "=" * 60)
            print(Fore.MAGENTA + f"[IMPROVER] Iteration {i}/{iters} Plan:")
            for idx, t_obj in enumerate(ordered_tasks, 1):
                t_str = t_obj.get("task", "") if isinstance(t_obj, dict) else str(t_obj)
                print(Fore.MAGENTA + f"  {idx}. {t_str}")
            print(Fore.MAGENTA + "=" * 60)

            # --- EXECUTION LOOP WITH REPLANNING ---
            j = 0
            while j < len(ordered_tasks):
                rem_improve = improve_budget - (time.time() - improve_start_time)
                if rem_improve <= 0:
                    print(Fore.YELLOW + "[IMPROVER] improve_budget_sec exhausted; stopping remaining tasks in this iteration.")
                    summary_lines.append(
                        f"[IMPROVER] Time budget hit before task {j + 1}/{len(ordered_tasks)} (iter {i})."
                    )
                    break

                t_obj = ordered_tasks[j]
                t_label = t_obj.get("task", "") if isinstance(t_obj, dict) else str(t_obj)
                cid = child_ids[j]
                kind = kinds[j]

                child_status = orch.tree_node_status(cid)
                if child_status == 'done':
                    print(Fore.GREEN + f"  [DONE] Task {j + 1}/{len(ordered_tasks)}: {t_label}")
                    j += 1
                    continue

                # --- IMPROVEMENT REPLANNING PHASE (runs on resume too) ---
                # Top-level improver (depth==0): optional careful replan *before* the first task of the
                # iteration (full queue + goal + metric + prior conclusions). Deeper nests keep tail-only replan.
                _rem_for_replan = improve_budget - (time.time() - improve_start_time)
                replan_pre_first = (
                    depth == 0
                    and j == 0
                    and len(ordered_tasks) > 1
                    and _rem_for_replan >= 180
                )
                replan_after_progress = j > 0
                if replan_pre_first or replan_after_progress:
                    pre_run = bool(replan_pre_first)
                    if pre_run:
                        print(
                            Fore.MAGENTA
                            + "[IMPROVER] Level-1 pre-run replan: sequencing queue (goal, metrics, prior notes)..."
                        )
                        remaining = list(ordered_tasks)
                        completed_summary = _improve_level1_prerun_context(
                            task,
                            previous_answers,
                            summary_lines,
                            cur_best or {},
                            frozen_spec,
                            i,
                            iters,
                        )
                    else:
                        print(Fore.MAGENTA + f"[IMPROVER] Evaluating remaining tasks for pruning/adjustment...")
                        remaining = ordered_tasks[j:]
                        completed_summary = "\n".join(summary_lines[-8:])

                    head_recent = completed_summary
                    if pre_run:
                        head_recent = "LEVEL-1 PRE-FIRST-TASK (iteration not started):\n" + head_recent
                    if isinstance(prev_head, dict) and prev_head.get("improver_head"):
                        try:
                            head_recent = (
                                head_recent
                                + "\n\nPREV_DECISION_MEMORY (last head):\n"
                                + json.dumps(prev_head.get("improver_head"), ensure_ascii=False)[:2500]
                            )
                        except Exception:
                            pass

                    current_improve_remaining = improve_budget - (time.time() - improve_start_time)
                    graph_hint = shorten_string_middle(orch.format_task_graph_to_string(), 3500)
                    # Knowledge Curator BEFORE: supply role-tailored brief for the improver head.
                    _cur_ih_ctx = _curator_before(orch, llm_fast, llm_strong, role="improver_head", task_hint=task or "")
                    if _cur_ih_ctx:
                        head_recent = head_recent + "\n\n" + _cur_ih_ctx
                    head = improver_head_agent(
                        llm_fast,
                        task=task,
                        spec=frozen_spec,
                        metrics_summary=json.dumps(cur_best or {}, ensure_ascii=False),
                        recent_summaries=head_recent,
                        depth=depth,
                        max_depth=max_depth,
                        remaining_improve_sec=max(0, int(current_improve_remaining)),
                        graph_hint=graph_hint,
                        main_pipeline_artifacts=main_pipeline_artifacts,
                    )
                    hn = (
                        f"verdict={head.get('verdict')} stuck={head.get('stuck')} trend={head.get('metric_trend')}\n"
                        f"reasoning={head.get('reasoning', '')}\n"
                        f"notes_for_replanner={head.get('notes_for_replanner', '')}"
                    )
                    try:
                        orch.write_file(
                            str(improve_root / "pipeline_resume.json"),
                            json.dumps({
                                "improve_node_id": this_node_id,
                                "iteration": i,
                                "task_index": j,
                                "pre_run_replan": pre_run,
                                "improver_head": head,
                                "prev": prev_head,
                            }, ensure_ascii=False, indent=2),
                        )
                    except Exception:
                        pass
                    prev_head = {
                        "improver_head": head,
                        "iteration": i,
                        "task_index": j,
                    }

                    _im_tail_cap = max(0, plan_cap - j)
                    _log_body = completed_summary
                    if pre_run:
                        _log_body = (
                            "[MODE=LEVEL1_PRE_ITERATION_REPLAN]\n"
                            "Full queue in REMAINING_TASKS; none started this iteration.\n"
                            "Reorder/merge for sequential experiments; stay within HARD_CAP.\n\n"
                            + completed_summary
                        )
                    _log_cap = 4500 if pre_run else 2000
                    # Knowledge Curator BEFORE: inject canonical lessons/pruned for improvement replanner.
                    _cur_ir_ctx = _curator_before(orch, llm_fast, llm_strong, role="replanner", task_hint=task or "")
                    _log_body_final = shorten_string_middle(_log_body, _log_cap)
                    if _cur_ir_ctx:
                        _log_body_final = _log_body_final + "\n\n" + _cur_ir_ctx
                    replan_res = improvement_replanning_agent(
                        llm_strong, task,
                        _log_body_final,
                        remaining,
                        depth=i, max_depth=iters,
                        remaining_time=int(current_improve_remaining),
                        head_notes=hn,
                        max_tail_tasks=_im_tail_cap,
                        extra_budget_sec=int(spec.get("_extra_budget_sec", 0)) if spec else 0,
                    )
                    if str(head.get("verdict", "")).lower() == "finalize":
                        replan_final = [{
                            "task": "Generate final submission.csv for this iteration and save to iteration artifacts folder.",
                            "time_budget_sec": max(120, min(600, int(current_improve_remaining))),
                        }]
                        replan_res = {
                            "reasoning": "Improver Head requested finalize; collapsing queue to submission task.",
                            "updated_remaining_tasks": replan_final,
                        }
                    updated_remaining = _normalize_improve_plan_tasks(
                        replan_res.get("updated_remaining_tasks", remaining),
                        default_tb,
                    )
                    if _im_tail_cap:
                        updated_remaining = updated_remaining[:_im_tail_cap]
                    else:
                        updated_remaining = []
                    if not updated_remaining:
                        updated_remaining = _normalize_improve_plan_tasks(remaining, default_tb)
                        if _im_tail_cap:
                            updated_remaining = updated_remaining[:_im_tail_cap]

                    # Detect if plan changed
                    plan_changed = False
                    if len(updated_remaining) != len(remaining):
                        plan_changed = True
                    else:
                        for u, r in zip(updated_remaining, remaining):
                            ut = u.get("task", u) if isinstance(u, dict) else str(u)
                            rt = r.get("task", r) if isinstance(r, dict) else str(r)
                            if str(ut) != str(rt):
                                plan_changed = True
                                break

                    if plan_changed:
                        print(
                            Fore.YELLOW + f"[IMPROVER] Plan modified! Reason: {replan_res.get('reasoning', 'Aggressive optimization')}")

                        # Abandon the old nodes so they don't stay pending forever
                        old_cids_to_abandon = child_ids[j:]
                        try:
                            skipped_log_path = orch.dir_paths["artifacts"] / "skipped_tasks.log"
                            with open(skipped_log_path, "a", encoding="utf-8") as f:
                                for old_cid in old_cids_to_abandon:
                                    if orch.tree_node_status(old_cid) not in ("done", "failed"):
                                        task_text = str(orch.tree_node(old_cid).get("task", ""))
                                        f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] IMPROVER PRUNED: {task_text}\n")
                                        orch.tree_log_event(
                                            "PRUNED",
                                            task_text,
                                            node_id=old_cid,
                                            parent_node_id=this_node_id,
                                            reason="improvement_replanning",
                                        )
                                        orch.tree_remove_node(old_cid)
                        except Exception as e:
                            print(f"Failed to log skipped tasks: {e}")

                        ordered_tasks = ordered_tasks[:j] + updated_remaining

                        # Re-init children in tree for modified branch
                        creation_prefix = legacy_prefix if i == 1 else iter_prefix
                        new_labels = [
                            f"{creation_prefix}{u.get('task') if isinstance(u, dict) else u}"
                            for u in updated_remaining
                        ]
                        new_cids = orch.tree_init_children_with_kinds(this_node_id, new_labels,
                                                                      kinds=["improve_leaf"] * len(new_labels))
                        try:
                            for ncid, lbl in zip(new_cids, new_labels):
                                orch.tree_log_event(
                                    "ADDED",
                                    lbl,
                                    node_id=ncid,
                                    parent_node_id=this_node_id,
                                    reason="improvement_replanning",
                                )
                        except Exception:
                            pass
                        child_ids = child_ids[:j] + new_cids
                        kinds = kinds[:j] + ["improve_leaf"] * len(new_labels)

                        if j >= len(ordered_tasks):
                            break

                        # Update local pointers
                        t_obj = ordered_tasks[j]
                        t_label = t_obj.get("task", "") if isinstance(t_obj, dict) else str(t_obj)
                        cid = child_ids[j]
                        kind = kinds[j]

                print(Fore.CYAN + f"  [RUNNING] Task {j + 1}/{len(ordered_tasks)}: {t_label}")
                node_task = orch.tree_node(cid).get("task", "")

                if kind == "improve_leaf":
                    res = improvement_generate_and_execute(
                        orch=orch,
                        llm_fast=llm_fast,
                        code_llm=code_llm,
                        frozen_spec=frozen_spec,
                        task=node_task,
                        iter_idx=i,
                        task_idx=j + 1,
                        part_name=t_label,
                        previous_answers=previous_answers,
                        max_iter=len(ordered_tasks),
                        node_id=cid,
                        parent_node_id=this_node_id,
                        mcp_tools=mcp_tools
                    )
                    code_txt = (res.get("code") or "").replace("```", "").replace("python", "")
                    if code_txt.strip():
                        code_bank.append(code_txt)

                    if res.get("metrics"):
                        summary_lines.append(f"Task {j + 1}: Success. Metrics: {res['metrics']}")
                    else:
                        summary_lines.append(f"Task {j + 1}: Success (no metrics).")
                else:
                    if force_leaf_only:
                        res = improvement_generate_and_execute(
                            orch=orch,
                            llm_fast=llm_fast,
                            code_llm=code_llm,
                            frozen_spec=frozen_spec,
                            task=node_task,
                            iter_idx=i,
                            task_idx=j + 1,
                            part_name=t_label,
                            previous_answers=previous_answers,
                            max_iter=len(ordered_tasks),
                            node_id=cid,
                            parent_node_id=this_node_id,
                            mcp_tools=mcp_tools,
                        )
                        code_txt = (res.get("code") or "").replace("```", "").replace("python", "")
                        if code_txt.strip():
                            code_bank.append(code_txt)
                        summary_lines.append(f"Sub-pipeline {j + 1} flattened to leaf (depth cap).")
                    else:
                        _, sub_best = improvement_pipeline(
                            orch=orch,
                            llm_strong=llm_strong,
                            llm_fast=llm_fast,
                            code_llm=code_llm,
                            task=node_task,
                            spec=frozen_spec,
                            previous_answers=previous_answers,
                            node_id=cid,
                            parent_node_id=this_node_id,
                            resume=resume,
                            max_iters_override=1,
                            mcp_tools=mcp_tools,
                            depth=depth + 1,
                            improve_start_time=improve_start_time,
                        )
                        summary_lines.append(f"Sub-pipeline {j + 1} finished.")
                j += 1

            if improve_budget - (time.time() - improve_start_time) <= 0:
                summary_lines.append("[IMPROVER] improve_budget_sec exhausted; exiting Improve loop.")
                break

            try:
                new_best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
                if not new_best and last_path.exists():
                    new_best = json.loads(last_path.read_text(encoding="utf-8"))
            except Exception:
                new_best = {}

            improved = _rel_better(new_best, cur_best, rel_thr=rel_thr)
            if improved:
                summary_lines.append(
                    f"[IMPROVER] iter {i:02d}: improved {cur_best.get('primary')} -> {new_best.get('primary')}")
                cur_best = new_best or cur_best
            else:
                summary_lines.append(
                    f"[IMPROVER] iter {i:02d}: no relative improvement (kept {cur_best.get('primary')}, got {new_best.get('primary')})")

            # Build inter-iteration context for next iteration's ReAct artifacts collector
            prev_ctx = spec.get("_inter_iteration_context") or {"what_worked": [], "what_failed": [], "next_to_try": [], "key_insights": []}
            iter_tasks_desc = [t.get("task", "") if isinstance(t, dict) else str(t) for t in ordered_tasks[:5]]
            if improved:
                prev_ctx["what_worked"].append(f"iter {i}: {iter_tasks_desc} -> improved to {new_best.get('primary')}")
            else:
                prev_ctx["what_failed"].append(f"iter {i}: {iter_tasks_desc} -> no improvement (got {new_best.get('primary')})")
            # Carry forward improvement suggestions from ReAct collector if available
            if main_pipeline_artifacts and main_pipeline_artifacts.get("improvement_suggestions"):
                suggs = main_pipeline_artifacts["improvement_suggestions"]
                prev_ctx["next_to_try"] = [s.get("idea", "") for s in suggs[:5] if isinstance(s, dict)]
            if main_pipeline_artifacts and main_pipeline_artifacts.get("inter_iteration_context"):
                ric = main_pipeline_artifacts["inter_iteration_context"]
                if isinstance(ric, dict):
                    for k in ("key_insights",):
                        if k in ric and isinstance(ric[k], list):
                            prev_ctx[k] = (prev_ctx.get(k) or []) + ric[k]
                            prev_ctx[k] = prev_ctx[k][-10:]  # keep last 10
            spec["_inter_iteration_context"] = prev_ctx

            # Re-enrich artifacts for next iteration with updated context
            if i < iters:
                try:
                    main_pipeline_artifacts = _collect_and_enrich_artifacts(
                        orch, spec, llm_strong, task,
                        previous_iteration_context=prev_ctx,
                    )
                    if main_pipeline_artifacts:
                        spec["_main_pipeline_artifacts"] = main_pipeline_artifacts
                        print(Fore.CYAN + f"[IMPROVER] Re-enriched artifacts for iter {i+1} with inter-iteration context")
                except Exception as e:
                    print(Fore.YELLOW + f"[IMPROVER] Re-enrichment failed: {e}")

        try:
            if hasattr(orch, "tree_update_meta"):
                orch.tree_update_meta(this_node_id, {"best": cur_best or {}})
        except Exception:
            pass

        try:
            _finalize_single_submission_by_all_metrics_llm(
                orch, llm_fast, frozen_spec, task=task
            )
            gate = run_final_output_gate(orch, frozen_spec, task_txt_root=Path(orch.project_root))
            summary_lines.append(f"[IMPROVER][FINAL_GATE] ok={gate.get('ok')} errors={gate.get('errors')}")
        except Exception as e:
            summary_lines.append(f"[IMPROVER][FINAL_GATE] failed: {e}")

        return "\n".join(summary_lines), cur_best or {}


def _should_try_metrics_llm_recovery(stdout: Optional[str], task: str) -> bool:
    """Avoid LLM cost on trivial errors; try recovery when metrics may be buried in logs."""
    if not stdout or len(stdout.strip()) < 80:
        return False
    low = stdout.lower()
    if "metrics_json" in low:
        return True
    train_signals = (
        "epoch", "val_loss", "val_auc", "roc_auc", "train_loss", "accuracy",
        "f1", "metric", "best val", "validation",
    )
    if any(s in low for s in train_signals):
        return True
    tlow = (task or "").lower()
    if any(s in tlow for s in ("train", "fit", "model", "epoch", "tune", "cv", "cross", "classifier", "forecast")):
        return True
    if "modulenotfounderror" in low and not any(s in low for s in train_signals):
        return False
    return False


def _build_artifacts_snapshot(orch: GlobalOrchestrator, *, max_checkpoints: int = 25) -> str:
    """Compact listing of existing model checkpoints and metric files for code-agent context."""
    try:
        art = (orch.project_root / orch.dir_paths["artifacts"]).resolve()
    except Exception:
        return ""
    if not art.is_dir():
        return ""
    exts = {".pth", ".pt", ".ckpt", ".pkl", ".joblib", ".safetensors"}
    timed: List[tuple[float, Path]] = []
    try:
        for p in art.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                timed.append((p.stat().st_mtime, p))
            except OSError:
                continue
    except OSError:
        return ""
    timed.sort(key=lambda x: x[0], reverse=True)
    rels: List[str] = []
    for _, p in timed[:max_checkpoints]:
        try:
            rels.append(str(p.relative_to(art)).replace("\\", "/"))
        except ValueError:
            rels.append(p.name)
    metric_hints: List[str] = []
    # Canonical order: best/metrics.json is authoritative; last/metrics.json is fallback.
    # best_metrics.json is a secondary optimizer-only write — check last so it doesn't shadow canonical.
    for rel in ("best/metrics.json", "last/metrics.json", "best_metrics.json", "metrics.json"):
        mp = art / rel
        if not mp.is_file():
            continue
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                metric_hints.append(
                    f"  {rel}: type={data.get('type', '?')} primary={data.get('primary', '?')}"
                )
            else:
                metric_hints.append(f"  {rel}: (non-dict json)")
        except Exception:
            metric_hints.append(f"  {rel}: (unreadable)")
    # Key canonical artifact paths for agent orientation (existence-checked)
    canon_hints: List[str] = []
    for rel in ("best/code.py", "last/code.py", "best/submission.csv"):
        if (art / rel).is_file():
            canon_hints.append(f"  artifacts/{rel}")
    if not rels and not metric_hints and not canon_hints:
        return ""
    lines = [
        "ARTIFACTS_SNAPSHOT (reuse checkpoints when sub-task allows; respect pretrained rules):",
        "  Canonical paths: artifacts/best/metrics.json, artifacts/last/metrics.json, "
        "artifacts/best/code.py, artifacts/last/code.py — see task_plan.md §2.4 for full existence table.",
    ]
    if rels:
        lines.append("Checkpoints / model files (most recent first):")
        lines.extend(f"  - {r}" for r in rels)
    if metric_hints:
        lines.append("Metric files (first = best available):")
        lines.extend(metric_hints)
    if canon_hints:
        lines.append("Canonical code/submission artifacts present:")
        lines.extend(canon_hints)
    return "\n".join(lines) + "\n\n"


def generate_code_and_execute(
        orch: GlobalOrchestrator,
        llm_fast,
        code_llm,
        spec: Optional[Dict[str, Any]],
        task: str,
        previous_answers="No Previous Answers",
        allow_spec_update: bool = True,
        improve_mode: bool = False,
        iter_idx: int = 0,
        max_iter: int = 0,
        leaf_id: Any = None,
        mcp_tools: Optional[List[Any]] = None
):
    # Resume paths can occasionally pass an empty spec; normalize to safe defaults.
    spec = spec or {
        "primary_metric": {"name": "metric", "maximize": True},
        "secondary_metrics": [],
        "data": {},
        "submission": {"columns": []},
    }
    spec.setdefault("primary_metric", {"name": "metric", "maximize": True})
    spec.setdefault("secondary_metrics", [])
    spec.setdefault("data", {})
    spec.setdefault("submission", {"columns": []})

    verifier = FormalVerifier(llm=llm_fast)
    incident_node_id = str(leaf_id) if leaf_id else f"adhoc_{hashlib.md5((task or '').encode('utf-8')).hexdigest()[:10]}"

    # Modality-aware schema cheat-sheet + artifact-introspection tools for the coder
    try:
        _schema_snapshot = build_schema_snapshot(orch, spec) or ""
    except Exception as _e:
        print(Fore.YELLOW + f"[SCHEMA] build_schema_snapshot failed: {_e}")
        _schema_snapshot = ""
    try:
        _artifact_tools = build_structured_artifact_tools(orch) or []
    except Exception as _e:
        print(Fore.YELLOW + f"[TOOLS] build_structured_artifact_tools failed: {_e}")
        _artifact_tools = []
    if mcp_tools is None:
        mcp_tools = []
    _existing_names = {getattr(t, "name", "") for t in mcp_tools}
    for _t in _artifact_tools:
        if getattr(_t, "name", "") not in _existing_names:
            mcp_tools.append(_t)

    # Load project context if it exists
    project_context = ""
    try:
        context_path = orch.project_root / orch.dir_paths["artifacts"] / "project_context.md"
        if context_path.exists():
            ctx_txt = context_path.read_text(encoding='utf-8')
            # Keep context compact for code generation prompts.
            ctx_lines = [ln.rstrip() for ln in ctx_txt.splitlines() if ln.strip()]
            ctx_txt = "\n".join(ctx_lines[-30:])
            project_context = f"\n\nPROJECT CONTEXT (COMPACT):\n{ctx_txt}"
    except Exception:
        pass

    artifacts_snapshot = _build_artifacts_snapshot(orch)

    router = ErrorRouter(
        llm_fast,
        os_name=platform.system() or "Windows",
        repeat_to_lead=2,
        google_api_key=getattr(getattr(orch.cfg, "google", object()), "api_key", None),
        google_cse_id=getattr(getattr(orch.cfg, "google", object()), "cse_id", None),
        orch=orch,
    )

    if len(previous_answers) > 10000:
        previous_answers = shorten_string_middle(previous_answers, 10000)

    progress = min((iter_idx + 1) / max_iter, 1.0) if max_iter > 0 else 0.0

    ml_evaluation_context_note = (
        "EVALUATION & DATA SPLITS (guidance — agents decide implementation)\n"
        "- **You choose** validation (holdout, k-fold, LOO, …). Use `spec.validation` as guidance + time, data size, modality (heavy vision vs cheap tabular).\n"
        "- Respect **group / time** constraints when the task or spec requires them.\n"
        "- **Inference-only** files: often no real labels — metrics from labeled data only; use inference rows for predictions/submission.\n"
        "- **Optional stdout** (helps the chain): `DEBUG: DATA_ROLE_SUMMARY`, `DEBUG: VALIDATION_DECISION` / `VALIDATION_PROTOCOL_SUMMARY` when you set or run a scheme.\n\n"
    )

    improve_guidelines = (
        "IMPROVE_MODE\n"
        "- MONOLITH PREFERENCE: Prefer a single robust script for end-to-end logic. Avoid split dependencies unless logic is extremely complex.\n"
        "- ARTIFACT AWARENESS: Read `project_context.md` to identify actual column names and pkl/dict structures. Never guess paths or keys.\n"
        "- Do NOT rebuild from scratch if existing code already handles data loading/validation correctly.\n"
        "- Obey `spec.constraints` (no internet, no external weights if forbidden).\n"
        "- Print `METRICS_JSON` with `primary` metric matching `spec`.\n"
        "- Keep data paths STRICTLY from spec.data.*; DO NOT change I/O, filenames, or submission schema.\n"
    )

    remaining_time = getattr(
        orch, "global_deadline_sec", orch.cfg.orchestration.total_budget_sec
    ) - orch.effective_elapsed_sec()
    if remaining_time <= 0:
        raise TimeoutError("HARD DEADLINE EXCEEDED. Aborting task execution.")

    time_context = f"\n[CRITICAL DEADLINE] You have exactly {int(remaining_time)} seconds left. Prioritize fast, direct solutions. DO NOT use extensive tuning if time is low.\n"
    context_prefix = ml_evaluation_context_note
    context_prefix += format_spec_constraints_block(spec)
    context_prefix += (improve_guidelines + "\n") if improve_mode else ""
    context_prefix += time_context
    context_prefix += project_context  # Add project context to all prompts
    if artifacts_snapshot:
        context_prefix += artifacts_snapshot
    # Knowledge Curator: synchronous BEFORE supervisor — routes role-tailored MD context to coder.
    _curator_ctx = _curator_before(orch, llm_fast, code_llm, role="coder", task_hint=task or "")
    if _curator_ctx:
        context_prefix += "\n" + _curator_ctx + "\n"

    answer_code = ""
    max_gen_retries = 8

    for attempt in range(max_gen_retries):
        remaining_time = getattr(orch, "global_deadline_sec", orch.cfg.orchestration.total_budget_sec) - orch.effective_elapsed_sec()
        if remaining_time <= 0:
            raise TimeoutError("HARD DEADLINE EXCEEDED during code generation retries.")
        print(Fore.BLUE + f"GENERATING CODE (Attempt {attempt + 1}/{max_gen_retries}) | TIME LEFT: {int(remaining_time)}s")
        try:
            answer_code = perform_task_python_v2(
                code_llm,
                task,
                spec,
                previous_code=code_bank[-1] if code_bank else "",
                context=context_prefix + previous_answers,
                tools=mcp_tools,
                orch=orch,
                schema_snapshot=_schema_snapshot,
            )
        except Exception as _llm_err:
            print(Fore.RED + f"[CODE_GEN] LLM call failed after retries: {type(_llm_err).__name__}: {_llm_err}")
            if attempt < max_gen_retries - 1:
                previous_answers += f"\n[LLM_ERROR] Provider returned: {type(_llm_err).__name__}. Retrying with fresh attempt."
                continue
            answer_code = code_bank[-1] if code_bank else ""

        clean_code = answer_code.replace("```python", "").replace("```", "").strip()

        print(Fore.CYAN + "[VERIFIER] Auditing code safety...")
        safety_check = verifier.verify_code_safety(clean_code, spec)
        policy_ok, policy_issues = _audit_generated_code_policy(clean_code)

        if safety_check.valid and policy_ok:
            print(Fore.GREEN + "[VERIFIER] Code Audit: SAFE")
            break
        else:
            reason = safety_check.violation_reason if not safety_check.valid else "; ".join(policy_issues)
            print(Fore.RED + f"[VERIFIER] Code Audit: UNSAFE - {reason}")
            previous_answers += f"\n[SECURITY_AUDIT_FAIL] Your previous code was rejected. Reason: {reason}. Fix this violation."
            if attempt == max_gen_retries - 1:
                print(Fore.RED + "[VERIFIER] Max retries reached. Proceeding with caution (or could raise Error).")

    _enforce_stack_guardrails(answer_code, orch)

    is_ok = False
    last_stdout, last_errors = "", ""
    retry_count = 0
    metrics: Dict[str, Any] = {}
    spec_update_consecutive_count = 0  # Guard against spec_update-only loops
    # Initialize heuristic_ok so the post-loop metric-verification block doesn't
    # crash with UnboundLocalError when the while-loop exits via a cap break
    # before any code execution actually ran (e.g. every attempt blocked by preflight).
    heuristic_ok = False

    # Hard cap: prevents infinite "triage-only" loops when the model keeps failing.
    triage_iter_count = 0
    max_triaeg_iters = int(getattr(orch.cfg.runtime, "generation_retry_limit", 20))
    max_triaeg_iters = max(3, max_triaeg_iters)

    # Error-signature deduplication: track how many times each unique error appears.
    # If the same error repeats >= 3 times, force lead-agent escalation instead of
    # routing the triage agent to the same broken fix strategy again.
    _seen_error_sigs: Dict[str, int] = {}

    while not is_ok:
        preflight = react_preexec_auditor_agent(
            llm_fast=llm_fast,
            orch=orch,
            task=task,
            spec=spec,
            code_text=answer_code,
            context="Main pre-exec audit. Inspect task_plan tree and artifact filesystem before run.",
        )
        allow_run = bool(preflight.get("allow_run", True))
        preflight_issues = [str(x) for x in (preflight.get("issues") or [])]
        if bool(preflight.get("planning_only", False)):
            preflight_issues.append("planning_only_task_detected")

        # ── Verify "Missing dependency" claims programmatically ──
        # The LLM often guesses packages are missing without running import checks.
        # We verify each claim and remove false positives.
        if preflight_issues:
            preflight_issues = _verify_dependency_claims(orch, preflight_issues)
            if not preflight_issues:
                allow_run = True

        skipped_execution = False
        if not allow_run or preflight_issues:
            print(Fore.YELLOW + f"[PREFLIGHT][AGENT] blocked: {preflight_issues}")
            last_stdout = ""
            last_errors = "AGENTIC PRE-EXECUTION AUDIT BLOCKED: " + "; ".join(preflight_issues)
            if preflight.get("required_fixes"):
                last_errors += "\nREQUIRED_FIXES: " + str(preflight.get("required_fixes"))
            skipped_execution = True
        else:
            triage_iter_count += 1
            if triage_iter_count > max_triaeg_iters:
                print(
                    Fore.RED
                    + f"[TRIAGE] Hard cap reached ({triage_iter_count-1}/{max_triaeg_iters}). Exiting triage loop."
                )
                orch.state_append_attempt(incident_node_id, {
                    "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    "phase": "triage_cap",
                    "route": "loop_cap",
                    "reason": f"Exceeded max triage iterations: {max_triaeg_iters}",
                    "stderr_tail": (last_errors or "")[-1000:],
                    "stdout_tail": (last_stdout or "")[-1000:],
                })
                break

            print(Fore.BLUE + "RUNNING CODE")
            if type(answer_code) is list:
                answer_code = '\n'.join(answer_code)

            # Predict execution time
            prediction = execution_predictor_agent(llm_fast, answer_code, spec)
            print(
                Fore.CYAN + f"[MONITOR] Predicted time: {prediction.get('expected_time_sec')}s, Intensity: {prediction.get('resource_intensity')}")

            result = orch.code_executor(
                answer_code.replace("```", "").replace("python", ""),
                spec=spec,
                prediction=prediction
            )
            print(Fore.BLUE + "FINISHED RUNNING CODE")

            last_stdout, last_errors = result["output"], result["errors"]
            heuristic_ok = (result["errors"] == "" and len(result["output"]) > 2) or (
                    "error" not in result["errors"].lower())

            # Hard rule: if the script explicitly signaled metrics as "skipped",
            # treat this execution as success for EDA/Load-style tasks.
            # This prevents triage from endlessly trying to compute metrics that should not exist.
            metrics_now = parse_metrics_from_stdout(last_stdout) or {}
            if isinstance(metrics_now, dict) and metrics_now.get("type") == "skipped":
                metrics = metrics_now
                is_ok = True
                orch.state_append_attempt(incident_node_id, {
                    "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    "phase": "run",
                    "result": "ok",
                    "metrics_type": "skipped",
                })
                break

            llm_ok = evaluate_run_ok_with_retry(
                llm_fast=llm_fast,
                stdout=last_stdout,
                stderr=last_errors,
                spec=spec,
                code_text=answer_code,
                additional_context=(
                    f"METRIC: {spec['primary_metric']['name']} SHOULD BE CALCULATED! + as addition {', '.join(spec['secondary_metrics'])}"
                    if progress > 0.2 else ""
                ),
            )

            is_ok = bool(heuristic_ok and llm_ok)
            if is_ok:
                orch.state_append_attempt(incident_node_id, {
                    "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    "phase": "run",
                    "result": "ok",
                    "heuristic_ok": heuristic_ok,
                    "llm_ok": llm_ok,
                })
                break

        # Triage Phase (Handles both pre-flight blocks and runtime errors)
        if skipped_execution:
            # Ensure we count the block as an attempt for cap protection
            triage_iter_count += 1
            if triage_iter_count > max_triaeg_iters:
                print(Fore.RED + f"[PREFLIGHT] Hard cap reached on blocks. Exiting.")
                break

        # ── Error-signature dedup ──────────────────────────────────────────────
        # Fingerprint the current error (first 200 chars, stripped). If the exact
        # same error has appeared 3+ times we are stuck in a loop: force escalation
        # to the lead agent, which has broader context and can try a different strategy.
        if last_errors:
            _err_sig = (last_errors or "").strip()[:200]
            _seen_error_sigs[_err_sig] = _seen_error_sigs.get(_err_sig, 0) + 1
            if _seen_error_sigs[_err_sig] >= 3:
                print(Fore.RED + f"[TRIAGE] Same error seen {_seen_error_sigs[_err_sig]}x — forcing lead escalation to break loop.")
                plan = router.route(last_errors, last_stdout, spec, answer_code)
                plan["route"] = "lead"
                plan["reason"] = (
                    f"Error repeated {_seen_error_sigs[_err_sig]} times without resolution. "
                    + plan.get("reason", "")
                )
                route = "lead"
            else:
                plan = router.route(last_errors, last_stdout, spec, answer_code)
                route = plan.get("route", "coding")
        else:
            plan = router.route(last_errors, last_stdout, spec, answer_code)
            route = plan.get("route", "coding")

        # Let lead incident manager re-route based on accumulated attempts and context.
        attempts_hist = []
        try:
            attempts_hist = (orch.state_get(incident_node_id) or {}).get("attempts", []) or []
        except Exception:
            attempts_hist = []

        route_thrashing = False
        if len(attempts_hist) >= 3:
            recent_routes = [str(a.get("route", "")) for a in attempts_hist[-3:] if isinstance(a, dict)]
            route_thrashing = len(set([r for r in recent_routes if r])) >= 2

        if route == "lead" or route_thrashing:
            managed_plan = lead_incident_manager_agent(
                llm_fast,
                task=task,
                spec=spec,
                triage_plan=plan,
                attempts=attempts_hist,
                stderr_tail=last_errors,
                stdout_tail=last_stdout,
                code_head=answer_code,
            )
            if isinstance(managed_plan, dict) and managed_plan.get("route"):
                plan = managed_plan
                route = plan.get("route", route)

        print(Fore.YELLOW + f"[TRIAGE] attempts={len(attempts_hist)} route={route} reason={plan.get('reason', '')}")

        orch.state_append_attempt(incident_node_id, {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "phase": "triage",
            "route": route,
            "reason": str(plan.get("reason", ""))[:500],
            "stderr_tail": (last_errors or "")[-1000:],
            "stdout_tail": (last_stdout or "")[-1000:],
        })

        # Guardrail against long "coding-only" loops (observed as many-hours attempts with no progress).
        # If we keep routing to coding for too long, refresh spec.data/meta via spec_update.
        attempts_after = len(attempts_hist) + 1
        if (
            route == "coding"
            and attempts_after >= orch.cfg.runtime.router_retry_limit
            and len(attempts_hist) >= 3
        ):
            recent_routes = [str(a.get("route", "")) for a in attempts_hist[-3:] if isinstance(a, dict)]
            recent_routes.append(route)
            if all(r == "coding" for r in recent_routes[-4:]):
                route = "spec_update"
                plan["route"] = "spec_update"
                plan.setdefault("spec_patch", {})
                plan["reason"] = (plan.get("reason", "") + " | forced spec_update after repeated coding attempts").strip()
                print(Fore.RED + "[TRIAGE] Forced spec_update to break coding-only loop.")

        # Inverse guardrail: prevent long "spec_update-only" loops.
        if route == "spec_update":
            spec_update_consecutive_count += 1
            if spec_update_consecutive_count >= 3:
                route = "coding"
                plan["route"] = "coding"
                plan["reason"] = (plan.get("reason", "") + " | forced coding after 3 consecutive spec_updates").strip()
                print(Fore.RED + "[TRIAGE] Forced coding to break spec_update-only loop.")
                spec_update_consecutive_count = 0
        else:
            spec_update_consecutive_count = 0

        if route == "install":
            pkgs = plan.get("packages", []) or []
            extra = plan.get("pip_extra", "") or ""
            if pkgs:
                print(Fore.CYAN + f"Installing into .venv: {pkgs}" + (f" [extra: {extra}]" if extra else ""))
                install_res = orch.pip_install(pkgs, extra=extra)
                orch.state_append_attempt(incident_node_id, {
                    "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    "phase": "action",
                    "route": "install",
                    "packages": pkgs,
                    "pip_extra": extra,
                    "exit_code": install_res.get("exit_code", 1),
                    "stderr_tail": str(install_res.get("stderr", ""))[-1000:],
                })
                if install_res.get("exit_code", 1) != 0:
                    last_errors = f"{install_res.get('stderr', '')}\n{install_res.get('stdout', '')}"
                    print(Fore.RED + "[TRIAGE] install failed; re-triaging based on pip output.")
                    # Track per-package failures so router can stop retrying hopeless packages.
                    if hasattr(router, "_install_fail_counts"):
                        for _pkg in pkgs:
                            _pk = (_pkg or "").lower().strip()
                            if _pk:
                                router._install_fail_counts[_pk] = router._install_fail_counts.get(_pk, 0) + 1
                    retry_count += 1
                    if retry_count > orch.cfg.runtime.router_retry_limit:
                        print(Fore.RED + "[TRIAGE] Retry limit exceeded after install failures.")
                        break
                    continue
            else:
                # Prefer coding retry over lead escalation when install route is empty.
                # This avoids wasting cycles in lead for dependency-resolution cases.
                print(Fore.RED + "Router requested install but provided no packages; falling back to coding.")
                route = "coding"

        elif route == "spec_update":
            # Refresh spec only in this single place (triage), then let code adapt to it.
            guidance: str
            if allow_spec_update:
                try:
                    spec_patch = plan.get("spec_patch") or {}
                    if isinstance(spec_patch, dict) and spec_patch:
                        spec = _deep_merge(spec, spec_patch)

                    # Re-probe dataset structure + rebuild data meta from filesystem facts.
                    spec = probe_dataset_with_bash(orch, spec)
                    max_samples = getattr(getattr(orch.cfg, "data_check", object()), "max_samples_per_dir", 200)
                    spec = build_data_meta(orch, llm_fast, spec, task, max_samples_per_dir=max_samples)
                    spec = clean_specs(spec)

                    spec_path = f"{orch.cfg.paths.artifacts_dir}/spec.json"
                    orch.write_file(spec_path, json.dumps(spec, ensure_ascii=False, indent=2))
                    guidance = "SPEC WAS REFRESHED. Update the CODE to follow the refreshed spec.data paths and schema. Keep I/O and submission schema unchanged."
                    print(Fore.CYAN + "[TRIAGE] spec.json refreshed via spec_update.")
                except Exception as e:
                    guidance = f"SPEC UPDATE FAILED ({e}). Update the CODE to handle the current spec.data/schema safely. Do NOT change spec.json."
            else:
                guidance = (
                    "SPEC IS FROZEN. Update the CODE instead: "
                    "derive k folds from data: k = min(spec_k, max(2, n_unique_groups(group_by))). "
                    "Do NOT change spec; keep submission schema as-is."
                )

            answer_code = perform_task_python_v2(
                code_llm,
                task,
                spec,
                previous_code=code_bank[-1] if code_bank else "",
                context=(improve_guidelines + "\n" if improve_mode else "") + guidance + "\nPREV_ERROR:\n" + (last_errors or "")[:1200],
                tools=mcp_tools,
                orch=orch,
                schema_snapshot=_schema_snapshot,
            )
            continue

        elif route == "coding":
            answer_code = finetune_code_v2(
                code_llm, task, answer_code, spec,
                error=((improve_guidelines + "\n") if improve_mode else "") + last_errors +
                      f"\nMETRIC TARGET: {spec['primary_metric']['name']}",
                tools=mcp_tools
            )

        elif route == "bash":
            # If the error is about a missing artifact/input file, auto-prepend
            # inspection commands so the Lead Agent sees the REAL filesystem
            # state (not just the triage's guess). Observed hallucination
            # pattern: aggregate_answers claims a file was saved, preflight
            # blocks on it forever, triage produces `mkdir -p artifacts` which
            # tells the Lead nothing. Prepend discovery commands to fix.
            raw_bash_cmds = _sanitize_triage_bash_cmds(plan.get("bash_cmds", []))
            _err_lower = (last_errors or "").lower()
            _missing_hint = any(
                kw in _err_lower
                for kw in (
                    "missing required input artifact",
                    "no such file",
                    "filenotfounderror",
                    "file not found",
                )
            )
            if _missing_hint:
                _is_win = (getattr(orch.bash, "os", "") == "Windows")
                _art = orch.cfg.paths.artifacts_dir
                if _is_win:
                    _discovery = [
                        f"Get-ChildItem -Force -LiteralPath '{_art}' | Format-Table Mode, Length, Name -AutoSize",
                        f"Get-ChildItem -Recurse -Filter *.parquet -LiteralPath '.' -ErrorAction SilentlyContinue | Select-Object -First 50 FullName, Length",
                    ]
                else:
                    _discovery = [
                        f"ls -la {_art}",
                        "find . -name '*.parquet' -not -path './.*' 2>/dev/null | head -50",
                    ]
                # Prepend, avoiding duplicates of exact strings already in raw_bash_cmds.
                _existing = set(raw_bash_cmds)
                bash_cmds_final = [c for c in _discovery if c not in _existing] + raw_bash_cmds
                print(Fore.CYAN + f"[BASH] auto-prepended {len(bash_cmds_final) - len(raw_bash_cmds)} discovery cmd(s) for missing-artifact error")
            else:
                bash_cmds_final = raw_bash_cmds

            answer_bash = {}
            bash_transcript: List[str] = []
            for cmd in bash_cmds_final:
                print(Fore.CYAN + f"[BASH] {cmd}")
                answer_bash = orch.bash.run(cmd, timeout=orch.cfg.runtime.bash_timeout_sec, stream=True)
                # Accumulate a short transcript so the Lead Agent sees output
                # from ALL discovery commands, not just the last one.
                _stdout_chunk = (answer_bash.get("stdout") or "")[:1200]
                _stderr_chunk = (answer_bash.get("stderr") or "")[:600]
                bash_transcript.append(
                    f"$ {cmd}\n{_stdout_chunk}"
                    + (f"\n[stderr] {_stderr_chunk}" if _stderr_chunk.strip() else "")
                )
                orch.state_append_attempt(incident_node_id, {
                    "ts": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    "phase": "action",
                    "route": "bash",
                    "cmd": str(cmd)[:500],
                })

            print("1. Consulting Lead Agent for suggestions...")
            current_code = code_bank[-1] if code_bank else ""
            _bash_blob = "\n\n".join(bash_transcript) if bash_transcript else json.dumps(answer_bash)
            suggestions = lead_agent_propose_changes(
                llm=code_llm,
                lead_reason=plan.get('reason', ''),
                task=task,
                spec=spec,
                code=current_code,
                last_stdout=f"Bash discovery output:\n{_bash_blob}\n\nOriginal execution error:\n{(last_errors or '')[:2000]}",
                orch=orch,
            )
            print(f"LEAD AGENT SUGGESTIONS:\n{suggestions}")

            answer_text = f"LEAD-DRIVEN FIX ANALYSIS:\n{suggestions}"

            print("2. Engaging Implementation Agent to write code...")
            answer_code = implement_changes_agent(
                llm=code_llm,
                suggestions=suggestions,
                original_code=current_code,
                task=task,
                spec=spec
            )

        elif route == "lead":
            current_code = code_bank[-1] if code_bank else ""
            note = plan.get("notes", "")

            print("1. Consulting Lead Agent for suggestions...")
            suggestions = lead_agent_propose_changes(
                llm=code_llm,
                lead_reason=plan.get('reason', '') + f"\n Notes: {note}",
                task=task,
                spec=spec,
                code=current_code,
                last_stdout=last_stdout,
                orch=orch,
            )
            print(f"LEAD AGENT SUGGESTIONS:\n{suggestions}")

            answer_text = f"LEAD-DRIVEN FIX ANALYSIS:\n{suggestions}"

            print("2. Engaging Implementation Agent to write code...")
            answer_code = implement_changes_agent(
                llm=code_llm,
                suggestions=suggestions,
                original_code=current_code,
                task=task,
                spec=spec
            )
            print("--- CORRECTION COMPLETE ---")

        else:
            answer_code = finetune_code_v2(
                code_llm, task, answer_code, spec,
                error=((improve_guidelines + "\n") if improve_mode else "") + last_errors +
                      f"\nMETRIC TARGET: {spec['primary_metric']['name']}",
                tools=mcp_tools
            )

        p_ok, p_issues = _audit_generated_code_policy(answer_code)
        if not p_ok:
            print(Fore.RED + f"[POLICY] Generated code blocked: {p_issues}")
            answer_code = finetune_code_v2(
                code_llm, task, answer_code, spec,
                error=(
                    "Policy violation in generated code: "
                    + "; ".join(p_issues)
                    + ". Remove destructive operations and do NOT hardcode spec; read artifacts/spec.json dynamically."
                ),
                tools=mcp_tools
            )

    try:
        _enforce_stack_guardrails(answer_code, orch)
    except Exception as e:
        print(Fore.RED + f"Guardrail violation: {e}")

    # --- METRIC VERIFICATION BLOCK ---
    # Runs once after the while-loop has exited (either because is_ok became
    # True or because the triage cap broke the loop). Any ``is_ok`` / retry
    # mutations below are post-loop bookkeeping only — do NOT try to re-use
    # loop-control keywords here.
    metrics = parse_metrics_from_stdout(last_stdout) or {}

    # 1. Check if model signaled "skipped" (Valid for EDA/Load)
    if metrics.get('type') == 'skipped':
        print(Fore.CYAN + "[INFO] Task signaled NO METRICS needed (type='skipped'). Proceeding.")

    # 2. If metrics not found in stdout, try LLM recovery on tail (training logs often bury METRICS_JSON)
    if not metrics and _should_try_metrics_llm_recovery(last_stdout, task):
        try:
            print(Fore.YELLOW + "[METRICS] No parseable METRICS_JSON — trying LLM recovery from stdout tail...")
            raw = metrics_recover_from_stdout(llm_fast, last_stdout, spec, subtask=task)
            recovered = validate_recovered_metrics(raw, spec)
            if recovered:
                metrics = recovered
                print(Fore.GREEN + f"[METRICS] LLM recovery succeeded: type={metrics.get('type')}")
        except Exception as ex:
            print(Fore.YELLOW + f"[METRICS] LLM recovery failed: {ex}")

    # 3. If still missing, run lightweight verification script
    if not metrics:
        print(Fore.YELLOW + "No METRICS_JSON found. Generating verification script...")
        verif_ctx = (
            "Look into ./artifacts for checkpoints/preds. \n"
            f"Current task: {task}\n"
            f"CODE_BANK: {answer_code}\n\nMETRIC TARGET: {spec['primary_metric']['name']}"
        )
        # FIX: Use repr() for spec to avoid syntax errors in generated code
        verif_code = verification_code_gen(code_llm, spec, context=verif_ctx)
        _v_ok, _v_issues = _audit_generated_code_policy(verif_code)
        if not _v_ok:
            print(Fore.YELLOW + f"[METRICS] verifier code blocked by policy: {_v_issues}; invoking agentic verifier repair")
            verif_code = _agentic_repair_verifier_code(
                code_llm=code_llm,
                spec=spec,
                initial_code=verif_code,
                context=(
                    f"Current task: {task}\n"
                    "Need verifier that reads existing artifacts and emits METRICS_JSON without spec hardcoding."
                ),
                mcp_tools=mcp_tools,
            )
        verif_res = orch.code_executor(verif_code, 'metric_check.py')
        metrics = parse_metrics_from_stdout(verif_res["output"])

    if metrics is None:
        metrics = {}

    art_summary = ""
    try:
        art_summary = update_project_context_after_execution(orch, task, last_stdout, last_errors, metrics, answer_code)
    except Exception as e:
        print(Fore.RED + f"Failed to update project context: {e}")
    # Knowledge Curator AFTER: record experiment result in experiments_ledger + lessons.
    _curator_after(
        orch, llm_fast, code_llm,
        role="coder", task_hint=task or "",
        trigger="after",
        payload={
            "metrics": metrics or {},
            "stdout_tail": (last_stdout or "")[-1500:],
            "errors_tail": (last_errors or "")[-1500:],
        },
    )

    # 4. Validate metrics if they are supposed to exist.
    # Double check "skipped" again in case verification script returned it.
    if metrics.get('type') == 'skipped':
        print(Fore.CYAN + "[INFO] Verification confirmed NO METRICS needed.")

    _pm_name = str((spec.get('primary_metric') or {}).get('name', '')).lower()
    _got_name = str(metrics.get('name', '')).lower()
    if (metrics.get('type') != 'skipped') and ((not metrics) or (_got_name != _pm_name)):
        print(Fore.RED + f"[FAIL] Metric validation failed. Found: {metrics.keys()}")
        print(Fore.YELLOW + "[FAIL] Proceeding without metrics (post-loop block — no retries possible here).")

    # --- FINALIZATION ---
    clean_code = answer_code.replace("```", "").replace("python", "")
    _persist_last_code_artifact(orch, clean_code)

    # Persist metrics artifact for external checkers (even when type='skipped').
    try:
        root = Path(orch.project_root)
        final_dir = root / orch.cfg.paths.artifacts_dir / "final"
        _ensure_dir(final_dir)
        orch.write_file(
            str(final_dir / "metrics.json"),
            json.dumps(metrics or {}, ensure_ascii=False, indent=2),
        )
    except Exception as e:
        print(Fore.YELLOW + f"[WARN] could not write final metrics.json: {e}")

    if metrics and metrics.get('type') != 'skipped':
        print(Fore.GREEN + f"METRICS => {metrics}")
        try:
            _update_best_from_candidate(
                orch,
                candidate_metrics=metrics,
                code_text=clean_code,
                tag=("main_run" if not improve_mode else "improve_inline"),
                enforce_validation=bool(improve_mode),
                spec=spec,
            )
        except Exception as e:
            print(Fore.YELLOW + f"[WARN] could not persist metrics/version: {e}")
    elif metrics.get('type') == 'skipped':
        print(Fore.GREEN + "METRICS => Skipped (as expected).")
    else:
        print(Fore.RED + "WARNING: metrics still missing after verification attempt.")

    if len(last_stdout) > orch.cfg.runtime.execution_output_shorten_threshold:
        last_stdout = shorten_string_middle(last_stdout, orch.cfg.runtime.execution_output_shorten_target)

    orch.log("final_code_result", {"task": shorten_string_middle(task, 200), "preview": last_stdout[:500]})
    if leaf_id:
        orch.tree_finish(leaf_id, status="done", meta={
            "iter_idx": iter_idx,
            "task": last_stdout,
            "script_file": f"{orch.cfg.paths.scripts_dir}/gen_code.py",
            "metrics": metrics or {},
            "artifact_summary": art_summary
        })
    return str(last_stdout), answer_code


def _is_eda_task(task_text: str) -> bool:
    t = (task_text or "").lower()
    return any(k in t for k in ["initial data analysis", "initial data", "eda", "data analysis", "profil", "profiling", "data profiling"])


def _is_feature_task(task_text: str) -> bool:
    t = (task_text or "").lower()
    return any(k in t for k in ["feature engineering", "preprocessing", "preprocess", "feature extraction", "scaling", "encoding"])


def _enforce_eda_feature_time_ratio(ordered_tasks: list[Any], target_ratio: float = 0.7) -> list[Any]:
    """
    Rebalances time_budget_sec so that EDA + Feature Engineering consume at least `target_ratio`
    of total time budgets (when time budgets are present).
    """
    if not ordered_tasks or not all(isinstance(t, dict) for t in ordered_tasks):
        return ordered_tasks

    time_budgets = []
    for t in ordered_tasks:
        tb = t.get("time_budget_sec", None)
        try:
            tb_val = int(tb) if tb is not None else 0
        except Exception:
            tb_val = 0
        time_budgets.append(max(0, tb_val))

    total = sum(time_budgets)
    if total <= 0:
        return ordered_tasks

    eda_sum = 0
    other_sum = 0
    for t, tb in zip(ordered_tasks, time_budgets):
        tt = t.get("task", "") if isinstance(t, dict) else ""
        if _is_eda_task(tt) or _is_feature_task(tt):
            eda_sum += tb
        else:
            other_sum += tb

    if eda_sum / total >= target_ratio:
        return ordered_tasks
    if other_sum <= 0:
        return ordered_tasks

    target_eda = int(round(target_ratio * total))
    target_other = max(0, total - target_eda)

    eda_factor = (target_eda / eda_sum) if eda_sum > 0 else 1.0
    other_factor = (target_other / other_sum) if other_sum > 0 else 1.0

    # Keep time budgets stable (>=60s) while rebalancing.
    min_each = 60
    new_tasks: list[Any] = []
    for t, tb in zip(ordered_tasks, time_budgets):
        tt = t.get("task", "") if isinstance(t, dict) else ""
        if _is_eda_task(tt) or _is_feature_task(tt):
            new_tb = max(min_each, int(round(tb * eda_factor)))
        else:
            new_tb = max(min_each, int(round(tb * other_factor)))
        nt = dict(t)
        nt["time_budget_sec"] = new_tb
        new_tasks.append(nt)

    return new_tasks


def _select_root_tasks_with_priority(ordered_tasks: list[Any], max_tasks: int) -> list[Any]:
    """
    Selects a subset of root tasks when width constraints are active.
    Priority:
    1) EDA tasks
    2) Feature engineering/preprocessing tasks
    3) Final evaluation/submission tasks
    4) Fill with remaining tasks in original order
    """
    if max_tasks <= 0 or not ordered_tasks:
        return ordered_tasks
    if len(ordered_tasks) <= max_tasks:
        return ordered_tasks

    # If tasks are not dicts with 'task', just truncate.
    if not all(isinstance(t, dict) for t in ordered_tasks):
        return ordered_tasks[:max_tasks]

    def is_final(tt: str) -> bool:
        t = (tt or "").lower()
        return ("final" in t) and any(k in t for k in ["submission", "evaluation", "confusion_matrix", "confusion", "submit"])

    selected: list[Any] = []
    used_idx: set[int] = set()

    def add_matching(pred):
        nonlocal selected, used_idx
        for i, t in enumerate(ordered_tasks):
            if i in used_idx:
                continue
            tt = t.get("task", "")
            if pred(tt):
                selected.append(t)
                used_idx.add(i)
                if len(selected) >= max_tasks:
                    return

    add_matching(lambda tt: _is_eda_task(tt))
    add_matching(lambda tt: _is_feature_task(tt))
    add_matching(is_final)

    if len(selected) < max_tasks:
        for i, t in enumerate(ordered_tasks):
            if i in used_idx:
                continue
            selected.append(t)
            used_idx.add(i)
            if len(selected) >= max_tasks:
                break

    return selected


def main_pipeline(
        orch: GlobalOrchestrator,
        llm_strong,
        llm_fast,
        code_llm,
        task: str,
        previous_answers="No Previous Answers",
        main_task_context="This is the main task",
        node_id=None,
        parent_node_id=None,
        spec=None,
        resume: bool = False,
        allow_spawn_improvement: bool = True,
        freeze_spec: bool = False,
        improve_mode: bool = False,
        mcp_tools: Optional[List[Any]] = None,
        depth: int = 0,
        max_allowed_depth: int = 99,
        max_allowed_width: int = 99
):
    # Comment translated to English.
    if resume and node_id is None:
        print(Fore.YELLOW + "[RESUME] Locating root, then deepest failed (else deepest pending/running)...")
        existing_root = orch.tree_find_most_recent_root()
        if existing_root:
            orch.tree_sanitize_running_tasks()
            deep_id = orch.tree_deepest_resume_target(existing_root)
            if deep_id:
                dn = orch.tree_node(deep_id)
                node_id = deep_id
                parent_node_id = dn.get("parent_node_id")
                nt = (dn.get("task") or "").strip()
                if nt:
                    main_task_context = task
                    task = nt
                try:
                    d = int(dn.get("depth") or 0)
                except (TypeError, ValueError):
                    d = 0
                st = str(dn.get("status") or "")
                print(
                    Fore.GREEN
                    + f"[RESUME] Deepest actionable node `{deep_id}` depth={d} status={st} "
                    f"parent={parent_node_id!r} — continuing this branch (task text from node)."
                )
            else:
                print(
                    Fore.GREEN
                    + f"[RESUME] No failed/pending/running under root `{existing_root}` — subtree complete."
                )
                return (
                    "[RESUME] Nothing to execute: no failed/pending/running nodes under root.",
                    None,
                )
        else:
            print(Fore.YELLOW + "[RESUME] No existing root found. Starting fresh.")

    effective_depth = _pipeline_effective_depth(orch, depth, node_id)
    max_depth_cfg = getattr(orch.cfg.orchestration, "max_tree_depth", 5)
    force_simple = (effective_depth >= max_allowed_depth) or (effective_depth >= max_depth_cfg)
    max_width_cfg = getattr(orch.cfg.orchestration, "max_tree_width", None)
    # Width: root (tree depth 0) stays wide; depth 1 = full max_tree_width; deeper = roughly half (min 2).
    if isinstance(max_width_cfg, int) and max_width_cfg > 0:
        if effective_depth == 0:
            pass
        elif effective_depth == 1:
            max_allowed_width = min(max_allowed_width, max_width_cfg)
        else:
            deep_cap = max(2, (max_width_cfg + 1) // 2)
            max_allowed_width = min(max_allowed_width, deep_cap)

    if effective_depth != depth and node_id:
        print(
            Fore.CYAN
            + f"[DEPTH] pipeline depth param={depth} tree_depth={effective_depth} — limits use tree_depth."
        )

    if force_simple and not (effective_depth >= max_depth_cfg):
        print(Fore.YELLOW + f"[MAIN] Max allowed depth {max_allowed_depth} reached. Forcing simple execution.")
    elif force_simple:
        print(Fore.YELLOW + f"[MAIN] Max config depth {max_depth_cfg} reached (tree_depth={effective_depth}). Forcing direct execution.")

    # Comment translated to English.
    verifier = FormalVerifier(llm=llm_fast)
    safety_spec = VerificationSpec(
        max_steps=getattr(orch.cfg.orchestration, "main_verifier_max_steps", 15),
        required_preconditions={},
        forbidden_states=["infinite_loop", "data_loss"],
        resource_limits={"complexity": 8.0}
    )

    # Comment translated to English.
    # FIX: Root node should always be 'main', not 'improve'
    node_kind = "main" if parent_node_id is None else "main"  # Default to main, could be improve later
    if improve_mode:
        node_kind = "improve"

    # Before _track_node: detect failed/running leaf (no child nodes). After tree_start, status becomes "running".
    resume_leaf_retry = False
    if resume and node_id and parent_node_id is not None:
        try:
            _ps = orch.tree_node_status(node_id)
            _ch = orch.tree_children_ordered(node_id)
            if _ps in ("failed", "running") and len(_ch) == 0:
                resume_leaf_retry = True
        except Exception:
            pass

    with _track_node(orch, node_id, parent_node_id, kind=node_kind, task=task) as main_node_id:
        try:
            cur_node = orch.tree_node(main_node_id) or {}
        except Exception:
            cur_node = {}
        is_root = (cur_node.get("parent_node_id") is None)
        is_fresh_root = is_root and (node_id is None)

        print(
            Fore.CYAN
            + f"[SPEC] context: is_fresh_root={is_fresh_root} is_root={is_root} node_id={node_id!r} resume={resume}"
        )

        if node_id is not None and spec is None:
            try:
                sfile = Path(orch.project_root) / orch.cfg.paths.artifacts_dir / "spec.json"
                if sfile.exists():
                    spec = json.loads(sfile.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Nested main / resume edge cases: always try artifacts/spec.json if spec still missing.
        if spec is None or not isinstance(spec, dict):
            try:
                sfile = Path(orch.project_root) / orch.cfg.paths.artifacts_dir / "spec.json"
                if sfile.exists():
                    spec = json.loads(sfile.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Comment translated to English.
        if is_fresh_root:
            print(
                Fore.MAGENTA
                + "[SPEC] Fresh root — running spec pipeline: bootstrap → (LLM spec if needed) → datapath → probe → data_meta → write spec.json"
            )
            bootstrap_gpu_stack(orch, orch.cfg, llm_fast)
            # Initialize git-anchored artifacts sandbox at the earliest point.
            # Idempotent + silent on no-git environments.
            try:
                _base = ensure_artifacts_repo(orch)
                if _base is not None:
                    print(Fore.GREEN + f"[ARTIFACT-GIT] artifacts repo ready at {_base}")
            except Exception as _e:
                print(Fore.YELLOW + f"[ARTIFACT-GIT] init skipped: {_e}")
            if spec is None:
                print(Fore.MAGENTA + "[SPEC] Step 1/4: LLM problem_spec_from_text (no usable spec.json yet).")
                try:
                    spec = problem_spec_from_text(llm_strong, task, tools=mcp_tools)
                except Exception as e:
                    print(
                        Fore.RED
                        + f"[SPEC] problem_spec_from_text failed after retries: {e}. Using default_spec_skeleton(); edit spec before production."
                    )
                    spec = default_spec_skeleton()
                    spec["constraints"] = dict(spec.get("constraints") or {})
                    spec["constraints"]["notes"] = (spec["constraints"].get("notes") or "") + f" [SPEC_FALLBACK_LLM_ERROR: {e}]"
                spec["hardware"] = {
                    "require_cuda": getattr(orch.cfg.hardware, "require_cuda", False),
                    "fail_if_no_cuda": getattr(orch.cfg.hardware, "fail_if_no_cuda", False),
                    "cuda_devices": getattr(orch.cfg.hardware, "cuda_devices", ""),
                }
                print(Fore.MAGENTA + "[SPEC] Steps 2–4: datapath_agent → probe_dataset_with_bash → build_data_meta")
                tree = snapshot_data_tree(str(orch.project_root), data_dirname=orch.cfg.paths.data_dir)
                dp_json = datapath_agent(llm_fast, task, tree, os_name=platform.system() or "Windows")
                dp = extract_json(dp_json) or {}
                if dp.get("data"):
                    proposed_data = dp.get("data") or {}
                    # Fast consistency check: prevent invented paths (e.g., labels_csv).
                    try:
                        chk = datapath_consistency_check_agent(llm_fast, proposed_data, tree)
                        if isinstance(chk, dict) and isinstance(chk.get("data"), dict):
                            proposed_data = chk.get("data") or proposed_data
                    except Exception:
                        pass
                    spec["data"] = proposed_data
                spec = probe_dataset_with_bash(orch, spec)
                max_samples = getattr(getattr(orch.cfg, "data_check", object()), "max_samples_per_dir", 200)
                spec = build_data_meta(orch, llm_fast, spec, task, max_samples_per_dir=max_samples)
                spec = _reconcile_submission_columns_from_sample(orch, spec)
                try:
                    spec = attach_hardware_to_spec(
                        orch, spec, limit_files=orch.cfg.runtime.attach_hardware_limit_files
                    )  # type: ignore
                except Exception:
                    pass
            else:
                print(
                    Fore.YELLOW
                    + "[SPEC] Step 1/4: SKIPPED LLM spec — spec already loaded from artifacts/spec.json. "
                    "Delete artifacts/spec.json to force full LLM spec + datapath. "
                    "Running probe/meta refresh only (below)."
                )
                if not freeze_spec:
                    try:
                        spec = probe_dataset_with_bash(orch, spec)
                    except Exception:
                        pass
                    if not (spec or {}).get("data", {}).get("meta"):
                        max_samples = getattr(getattr(orch.cfg, "data_check", object()), "max_samples_per_dir", 200)
                        spec = build_data_meta(orch, llm_fast, spec, task, max_samples_per_dir=max_samples)
                    spec = _reconcile_submission_columns_from_sample(orch, spec)

            spec_path = f"{orch.cfg.paths.artifacts_dir}/spec.json"
            spec = clean_specs(spec)
            spec = merge_default_secondary_metrics(spec)
            spec = _reconcile_submission_columns_from_sample(orch, spec)
            spec['project_root'] = str(orch.project_root)
            print(Fore.CYAN + f"[SPEC] submission.columns={((spec.get('submission') or {}).get('columns') or [])}")
            orch.write_file(spec_path, json.dumps(spec, ensure_ascii=False, indent=2))
            print(Fore.CYAN + f"[SPEC]\n{json.dumps(spec, indent=2)}")

            if resume:
                root = Path(orch.project_root)
                art_dir = root / orch.cfg.paths.artifacts_dir
                best_code_file = art_dir / "best" / "code.py"
                last_code_file = art_dir / "last" / "code.py"
                resumed = ""
                if best_code_file.exists():
                    try:
                        resumed = best_code_file.read_text(encoding="utf-8")
                        print(Fore.YELLOW + f"[RESUME] Using code from artifacts/best/code.py")
                    except Exception as e:
                        print(Fore.RED + f"[RESUME] Failed to read best code: {e}")
                elif last_code_file.exists():
                    try:
                        resumed = last_code_file.read_text(encoding="utf-8")
                        print(Fore.YELLOW + f"[RESUME] Using code from artifacts/last/code.py")
                    except Exception as e:
                        print(Fore.RED + f"[RESUME] Failed to read last code: {e}")

                if resumed.strip():
                    code_bank.append(resumed.replace("```", "").replace("python", ""))
                    previous_answers = (previous_answers or "") + "\n\n[RESUMED_CODE_SNIPPET]\n" + resumed[:2000]

                try:
                    best_metrics_file = art_dir / "best" / "metrics.json"
                    last_metrics_file = art_dir / "last" / "metrics.json"
                    mp = None
                    if best_metrics_file.exists():
                        mp = best_metrics_file
                    elif last_metrics_file.exists():
                        mp = last_metrics_file

                    if mp and mp.exists():
                        mets = json.loads(mp.read_text(encoding="utf-8"))
                        print(Fore.YELLOW + f"[RESUME] Current metrics from {mp.relative_to(root)}: {mets}")
                except Exception:
                    pass

            if improve_mode and freeze_spec:
                frozen_path = f"{orch.cfg.paths.artifacts_dir}/spec_frozen.json"
                try:
                    orch.write_file(frozen_path, json.dumps(spec, ensure_ascii=False, indent=2))
                    print(Fore.YELLOW + f"[SPEC] frozen copy saved to {frozen_path}")
                except Exception:
                    pass

        else:
            # Resume tree or nested main_pipeline: the fresh-root spec pipeline (bootstrap, LLM spec, write spec.json) is skipped.
            print(
                Fore.YELLOW
                + f"[SPEC] Fresh-root spec pipeline SKIPPED — not a fresh root (node_id={node_id!r}). "
                "Typical on --resume with an existing tree: go straight to saved plan / children."
            )
            if spec is None or not isinstance(spec, dict):
                try:
                    sfile = Path(orch.project_root) / orch.cfg.paths.artifacts_dir / "spec.json"
                    if sfile.exists():
                        spec = json.loads(sfile.read_text(encoding="utf-8"))
                        print(Fore.GREEN + f"[SPEC] Loaded {sfile} for non-fresh root.")
                    else:
                        print(
                            Fore.RED
                            + "[SPEC] WARNING: no spec in memory and no artifacts/spec.json — downstream may be weak."
                        )
                except Exception as e:
                    print(Fore.RED + f"[SPEC] Failed to load spec.json: {e}")

        # Minimal spec dict for all downstream paths (meta-planner, verifier, nested main).
        if not isinstance(spec, dict):
            spec = {}
        spec.setdefault("primary_metric", {"name": "metric", "maximize": True})
        spec.setdefault("secondary_metrics", [])
        spec.setdefault("data", {})
        spec.setdefault("submission", {"columns": []})

        # Knowledge Curator BOOTSTRAP: seed canonical MD files from the finalized spec (once per run).
        if not getattr(orch, "_curator_bootstrapped", False):
            try:
                _curator_after(
                    orch, llm_fast, llm_strong,
                    role="planner", task_hint=task or "",
                    trigger="bootstrap",
                    payload={"spec_excerpt": {k: spec.get(k) for k in ("primary_metric", "submission", "data")}},
                )
            finally:
                orch._curator_bootstrapped = True

        _rem_g_main = _global_remaining_sec(orch)
        _min_split_main = int(getattr(orch.cfg.orchestration, "min_remaining_sec_to_split", 600))
        low_time_leaf = _rem_g_main < _min_split_main
        _total_b_main = int(orch.cfg.orchestration.total_budget_sec)

        # Comment translated to English.
        if improve_mode:
            out = generate_code_and_execute(
                orch, llm_fast, code_llm, spec, task,
                previous_answers=previous_answers,
                allow_spec_update=not freeze_spec,
                improve_mode=True,
                leaf_id=main_node_id,
                mcp_tools=mcp_tools
            )
            return out

        # Comment translated to English.
        # Comment translated to English.
        children = orch.tree_children_ordered(main_node_id)

        # Comment translated to English.
        # Comment translated to English.
        subtask_children = [c for c in children if c.get('kind') != 'improve']
        improve_children = [c for c in children if c.get('kind') == 'improve']

        # --resume: a failed/interrupted leaf has no rows in tree_init_children; do not regenerate the plan.
        resume_leaf_executed = False
        if resume_leaf_retry and not improve_mode and not subtask_children and not is_root:
            print(
                Fore.YELLOW
                + f"[RESUME] Retrying failed/interrupted leaf {main_node_id} (execute once; no subtasks on disk)."
            )
            task_out, task_code = generate_code_and_execute(
                orch, llm_fast, code_llm, spec, task, previous_answers,
                allow_spec_update=True, improve_mode=False, max_iter=0, leaf_id=main_node_id,
                mcp_tools=mcp_tools
            )
            if task_code:
                code_bank.append(task_code.replace("```", "").replace("python", ""))
            full_answer = task_out or ""
            resume_leaf_executed = True
            ordered_tasks = []
            child_ids = []

        if not resume_leaf_executed:
            if force_simple:
                task_double = "False"
            elif node_id is None:
                task_double = "False" if low_time_leaf else "True"
                task = "ROOT NODE: " + task
            elif subtask_children:
                print(
                    Fore.YELLOW + f"[RESUME] Node {node_id} has {len(subtask_children)} subtasks. Forcing traversal (skipping complexity check).")
                task_double = "True"
            else:
                # Force splitting at first level (children of root) to avoid ambiguity in downstream re-planning.
                try:
                    task_double = task_complexity_check(
                        llm_fast,
                        task,
                        main_task_context,
                        previous_answers,
                        tree_depth=effective_depth,
                        remaining_total_sec=_rem_g_main,
                        min_split_sec=_min_split_main,
                        tree_max_depth=max_depth_cfg,
                    )
                except TypeError:
                    # Backward-compatible fallback for stale runtime/imports with old signature.
                    task_double = task_complexity_check(
                        llm_fast,
                        task,
                        main_task_context,
                        previous_answers,
                    )

            full_answer = ""
            ordered_tasks = []
            child_ids = []

            if "false" in str(task_double).lower() and (not is_root or low_time_leaf):
                print(Fore.CYAN + "[MAIN] Task classified as SIMPLE. Executing directly (skipping subtasks).")
                task_out, task_code = generate_code_and_execute(
                    orch, llm_fast, code_llm, spec, task, previous_answers,
                    allow_spec_update=True, improve_mode=False, max_iter=0, leaf_id=main_node_id,
                    mcp_tools=mcp_tools
                )
                if task_code:
                    code_bank.append(task_code.replace("```", "").replace("python", ""))
                full_answer = task_out
                time.sleep(0.1)
            else:
                # Comment translated to English.
                # Comment translated to English.
                if subtask_children:
                    print(
                        Fore.YELLOW + f"[RESUME] Resuming branch node {main_node_id} with {len(subtask_children)} existing children.")
                    ordered_tasks = [c['task'].replace("Sub-Task ", "", 1) for c in subtask_children]
                    child_ids = [c['node_id'] for c in subtask_children]
                else:
                    # NEW: Use Meta-Planner for the root node to create a high-level plan
                    if is_root:
                        print(Fore.MAGENTA + "[META-PLANNER] Creating high-level project skeleton...")
                        ordered_tasks = meta_planner_agent(
                            llm_strong,
                            task,
                            spec,
                            max_stages=max_allowed_width,
                            remaining_sec=_rem_g_main,
                            total_budget_sec=_total_b_main,
                            constraints_block=format_spec_constraints_block(spec),
                        )
                        print(Fore.MAGENTA + f"[META-PLANNER] Plan created with {len(ordered_tasks)} stages.")
                        # Enforce: most budget on EDA + Feature engineering
                        ordered_tasks = _enforce_eda_feature_time_ratio(ordered_tasks, target_ratio=0.7)
                    else:
                        sub_tasks = None
                        max_plan_retries = 3
                        ordered_tasks = []

                        for plan_attempt in range(max_plan_retries):
                            try:
                                tasks_history = orch.format_task_graph_to_string()
                                width_prompt = (
                                    f"You can generate a maximum of {max_allowed_width} sub-tasks. "
                                    f"REMAINING_TOTAL_TIME_SEC={_rem_g_main} MIN_SPLIT_SEC={_min_split_main}."
                                )
                                # Knowledge Curator BEFORE: inject canonical brief/schema/lessons for planner.
                                _cur_plan_ctx = _curator_before(orch, llm_fast, llm_strong, role="planner", task_hint=task or "")
                                if _cur_plan_ctx:
                                    width_prompt += "\n" + _cur_plan_ctx
                                sub_tasks = generate_tasks_with_retry(
                                    llm_strong, task, spec, previous_answers,
                                    tasks_history,
                                    max_retries=3,
                                    extra_context=width_prompt,
                                    remaining_total_sec=_rem_g_main,
                                    total_budget_sec=_total_b_main,
                                    min_split_sec=_min_split_main,
                                    constraints_block=format_spec_constraints_block(spec),
                                )
                                # Convert sub_tasks to strings if they are dictionaries
                                sub_tasks_strings = []
                                for item in sub_tasks:
                                    if isinstance(item, dict) and "task" in item:
                                        sub_tasks_strings.append(item["task"])
                                    elif isinstance(item, dict):
                                        sub_tasks_strings.append(str(item))
                                    else:
                                        sub_tasks_strings.append(str(item))
                                candidates = order_tasks_with_retry(
                                    llm_fast, task, sub_tasks_strings, spec, max_retries=3,
                                    overall_time_limit_sec=_rem_g_main,
                                    constraints_block=format_spec_constraints_block(spec),
                                )
                                if not isinstance(candidates, list):
                                    candidates = (sub_tasks if isinstance(sub_tasks, list) else []) or []
                            except YAMLParseError as e:
                                print(Fore.RED + f"[TASKS] YAML failure: {e}")
                                candidates = []

                            if not candidates:
                                break

                            print(Fore.CYAN + f"[VERIFIER] Checking Main Plan safety (Attempt {plan_attempt + 1})...")
                            ver_result = verifier.verify_plan(
                                candidates,
                                safety_spec,
                                context=f"Main Task Planning. Spec: {(spec or {}).get('primary_metric')}",
                            )

                            if ver_result.valid:
                                print(Fore.GREEN + "[VERIFIER] Main Plan Verified: SAFE")
                                ordered_tasks = candidates
                                break
                            else:
                                print(Fore.RED + f"[VERIFIER] Main Plan Rejected: {ver_result.violation_reason}")
                                previous_answers += f"\n[PLANNING_ERROR] Previous plan rejected: {ver_result.violation_reason}. Fix dependencies."

                    if not ordered_tasks:
                        ordered_tasks = candidates if 'candidates' in locals() else []

                    if len(ordered_tasks) > max_allowed_width:
                        print(
                            Fore.YELLOW
                            + f"[WIDTH_LIMIT] Truncating {len(ordered_tasks)} tasks to {max_allowed_width} "
                            "(priority: EDA, features, final stages)."
                        )
                        ordered_tasks = _select_root_tasks_with_priority(ordered_tasks, max_allowed_width)


                    def _clean_t(t):
                        import re, ast
                        # 1. If it's already a dict, just take the task
                        if isinstance(t, dict):
                            return t.get('task', str(t))

                        # 2. If it's a string, clean it
                        s = str(t).strip()
                        # Remove "Sub-Task " prefix to avoid doubling
                        s = re.sub(r"^(Sub-Task\s*)+", "", s, flags=re.IGNORECASE).strip()

                        # 3. If it looks like a dictionary string, try to parse it
                        if s.startswith('{'):
                            try:
                                # literal_eval handles both "{'a':1}" and '{"a":1}'
                                data = ast.literal_eval(s)
                                if isinstance(data, dict):
                                    return data.get('task', s)
                            except Exception:
                                # Regex fallback if parsing fails
                                m = re.search(r"['\"]task['\"]:\s*['\"](.*?)['\"]", s)
                                if m: return m.group(1)

                        return s

                    child_labels = [f"Sub-Task {_clean_t(t)}" for t in ordered_tasks]
                    child_ids = orch.tree_init_children_with_kinds(main_node_id, child_labels,
                                                                   kinds=["main"] * len(ordered_tasks))
                    # Persist time_budget_sec into node meta for later timeout/policy logic.
                    try:
                        for idx, t_obj in enumerate(ordered_tasks):
                            if isinstance(t_obj, dict) and "time_budget_sec" in t_obj:
                                orch.tree_update_meta(child_ids[idx], {"time_budget_sec": t_obj.get("time_budget_sec")})
                    except Exception:
                        pass

        # Comment translated to English.
        if isinstance(ordered_tasks, dict):
            mt_context = "Main Task Information: " + task + "\n" + str(ordered_tasks)
        else:
            # FIX: Handle list of dicts for ordered_tasks
            task_strings = [t.get('task', str(t)) if isinstance(t, dict) else str(t) for t in ordered_tasks]
            mt_context = "Main Task Information: " + task + "\n" + "\n".join(task_strings)

        i = 0
        # Anti-thrashing guardrails for tail replanning:
        # allow multiple replans, but only with execution progress between them.
        _replan_calls = 0
        _replan_nochange_streak = 0
        _replan_disabled = False
        _last_replan_at_i = -1
        try:
            _replan_max_calls = int(getattr(orch.cfg.orchestration, "replan_max_calls", 3) or 3)
        except Exception:
            _replan_max_calls = 3
        try:
            _replan_cooldown_steps = int(getattr(orch.cfg.orchestration, "replan_cooldown_steps", 1) or 1)
        except Exception:
            _replan_cooldown_steps = 1
        try:
            _replan_min_remaining_sec = int(
                getattr(orch.cfg.orchestration, "min_remaining_sec_to_split", 600) or 600
            )
        except Exception:
            _replan_min_remaining_sec = 600
        while i < len(ordered_tasks):
            current_task_obj = ordered_tasks[i]
            current = current_task_obj.get("task", "") if isinstance(current_task_obj, dict) else str(current_task_obj)
            current_time_budget = current_task_obj.get("time_budget_sec", None) if isinstance(current_task_obj,
                                                                                              dict) else None

            cid = child_ids[i]
            child_status = orch.tree_node_status(cid)
            try:
                _child_tree_depth = int((orch.tree_node(cid) or {}).get("depth") or 0)
            except (TypeError, ValueError):
                _child_tree_depth = depth + 1

            # Comment translated to English.
            if child_status == 'done':
                print(
                    Fore.GREEN
                    + f"Sub-Task [{i + 1}/{len(ordered_tasks)}] tree_depth={_child_tree_depth} pipeline_level={depth + 1} {current} is already done. Loading context..."
                )
                ctx = orch.tree_get_node_context(cid)
                if ctx and ctx.get("code"):
                    code_bank.append(ctx["code"].replace("```", "").replace("python", ""))
                if ctx and ctx.get("output"):
                    full_answer = (full_answer or "") + ctx["output"] + "\n"
                i += 1
                continue

            # --- REPLANNING PHASE ---
            # FIX: Disable replanning if the previous step failed, to focus on fixing it.
            prev_child_failed = (i > 0 and orch.tree_node_status(child_ids[i - 1]) == 'failed')
            # Tail replan: after at least one subtask finished (i>0), optionally shrink/reorder remaining tasks.
            # Previously gated with `not is_root` and `effective_depth > 1`, which disabled replanning for the
            # root meta-plan and all depth-1 branches — so long root-stage lists never hit the replanner.
            _recent_text = (full_answer or "")[-4000:].lower()
            _rate_limited_recent = (
                "429" in _recent_text
                or "rate limit" in _recent_text
                or "too many requests" in _recent_text
            )
            _remaining_now = max(0, int(_global_remaining_sec(orch)))
            _progress_since_last_replan = (i - _last_replan_at_i) >= max(1, _replan_cooldown_steps)
            _replan_ok = (
                i > 0
                and not prev_child_failed
                and max_allowed_width > 1
                and not resume
                and not _replan_disabled
                and _replan_calls < max(0, _replan_max_calls)
                and _replan_nochange_streak < 2
                and _progress_since_last_replan
                and not _rate_limited_recent
                and _remaining_now >= max(120, _replan_min_remaining_sec)
            )
            if _replan_ok:
                print(Fore.MAGENTA + f"[REPLANNING] Checking if remaining tasks need adjustment...")
                remaining_tasks = ordered_tasks[i:]
                # safely slice full_answer for context
                context_for_replan = shorten_string_middle(
                    full_answer or "No previous outputs",
                    orch.cfg.runtime.replan_context_chars,
                )

                # Calculate remaining time
                current_remaining = _remaining_now
                _tail_cap = max(0, max_allowed_width - i)
                # Avoid HARD_CAP=0 when the tail is non-empty (width/index edge cases).
                if _tail_cap == 0 and remaining_tasks:
                    _tail_cap = len(remaining_tasks)
                # Relax the cap when the spec has accumulated extra_budget to burn.
                _eb = int(spec.get("_extra_budget_sec", 0)) if spec else 0
                if _eb >= 600 and current_remaining > 0 and (_eb / max(1, current_remaining)) >= 0.2:
                    _tail_cap = _tail_cap + max(2, int(_eb // 1800))
                    print(Fore.CYAN + f"[REPLAN] Budget relaxation active: tail_cap raised to {_tail_cap} (extra_budget={_eb}s)")
                _replan_calls += 1
                _last_replan_at_i = i
                # Knowledge Curator BEFORE: give replanner a snapshot of current brief/lessons/pruned.
                _cur_rp_ctx = _curator_before(orch, llm_fast, llm_strong, role="replanner", task_hint=task or "")
                if _cur_rp_ctx:
                    context_for_replan = (context_for_replan or "") + "\n" + _cur_rp_ctx
                try:
                    replanning_result = replanning_agent(
                        llm_strong,
                        task,
                        context_for_replan,
                        remaining_tasks,
                        remaining_time=current_remaining,
                        max_tail_tasks=_tail_cap,
                        extra_budget_sec=int(spec.get("_extra_budget_sec", 0)) if spec else 0,
                    )
                except Exception as e:
                    print(Fore.YELLOW + f"[REPLANNING] failed ({e}); disabling further replanning in this branch.")
                    _replan_disabled = True
                    replanning_result = {
                        "updated_remaining_tasks": remaining_tasks,
                        "reasoning": f"fallback_after_replan_error: {e}",
                    }
                if replanning_result.get("escalate_to_parent"):
                    print(Fore.YELLOW + "[REPLANNING] escalate_to_parent=true: stopping this branch; parent continues.")
                    old_cids_to_abandon = child_ids[i:]
                    try:
                        skipped_log_path = orch.dir_paths["artifacts"] / "skipped_tasks.log"
                        with open(skipped_log_path, "a", encoding="utf-8") as f:
                            for old_cid in old_cids_to_abandon:
                                if orch.tree_node_status(old_cid) not in ("done", "failed"):
                                    task_text = str(orch.tree_node(old_cid).get("task", ""))
                                    f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] REPLAN ESCALATE: {task_text}\n")
                                    orch.tree_log_event(
                                        "PRUNED",
                                        task_text,
                                        node_id=old_cid,
                                        parent_node_id=main_node_id,
                                        reason="main_replan_escalate",
                                    )
                                    orch.tree_remove_node(old_cid)
                    except Exception as e:
                        print(f"Failed to log skipped tasks: {e}")
                    break

                _db_tb = int(getattr(orch.cfg.runtime, "default_task_budget_sec", 1800))
                updated_remaining = _normalize_plan_tail_entries(
                    replanning_result.get("updated_remaining_tasks", remaining_tasks),
                    _db_tb,
                )
                if _tail_cap:
                    updated_remaining = updated_remaining[:_tail_cap]
                else:
                    updated_remaining = []

                # Re-check remaining time AFTER LLM thinking consumed real seconds.
                # LLM replanning itself can take 10-60s; task budgets must fit reality.
                _post_replan_remaining = _global_remaining_sec(orch)
                _planned_total = sum(
                    int(t.get("time_budget_sec", _db_tb)) if isinstance(t, dict) else _db_tb
                    for t in updated_remaining
                )
                if _planned_total > _post_replan_remaining > 0:
                    _budget_scale = _post_replan_remaining / _planned_total
                    for _t in updated_remaining:
                        if isinstance(_t, dict) and "time_budget_sec" in _t:
                            _t["time_budget_sec"] = max(30, int(_t["time_budget_sec"] * _budget_scale))
                    print(
                        Fore.YELLOW
                        + f"[REPLANNING] Budget rescaled after LLM think-time: "
                        + f"planned={_planned_total}s > actual_remaining={_post_replan_remaining}s "
                        + f"(scale={_budget_scale:.2f})"
                    )

                # Submission-terminal guard: ensure the replanned tail still ends in
                # a submission-shaped task. Without this, a replan that prunes the
                # final predict/submission node leaves the DAG with no successor —
                # observed in run 2026-05-08T02-10-14 (wids-datathon-2020) where the
                # predict node was PRUNED via main_replanning with nothing added.
                def _is_submission_task(_t: Any) -> bool:
                    _txt = _t.get("task", "") if isinstance(_t, dict) else str(_t)
                    return bool(re.search(r"submission|submit", _txt or "", re.I))

                _has_submission = any(_is_submission_task(_t) for _t in updated_remaining)
                _orig_had_submission = any(_is_submission_task(_t) for _t in remaining_tasks)
                if not _has_submission and (_post_replan_remaining > 0 or _orig_had_submission):
                    _inject_budget = max(120, min(600, int(max(0, _post_replan_remaining))))
                    _inject_task = {
                        "task": (
                            "Generate the final submission file (submission.py / submission.csv) "
                            "from the best available trained artifact (versions/index.json or "
                            "artifacts/last/code.py) and save to the canonical submission path."
                        ),
                        "time_budget_sec": _inject_budget,
                    }
                    if updated_remaining:
                        updated_remaining.append(_inject_task)
                        print(
                            Fore.YELLOW
                            + "[REPLANNING] Submission-terminal guard: injected missing "
                            + f"submission task (budget={_inject_budget}s)."
                        )
                    else:
                        updated_remaining = [_inject_task]
                        print(
                            Fore.YELLOW
                            + "[REPLANNING] Submission-terminal guard: replanner cleared queue; "
                            + f"forcing submission collapse (budget={_inject_budget}s)."
                        )

                # Check for changes
                plan_changed = False
                rem_norm = _normalize_plan_tail_entries(remaining_tasks, _db_tb)
                if len(updated_remaining) != len(rem_norm):
                    plan_changed = True
                else:
                    for u, r in zip(updated_remaining, rem_norm):
                        str_u = str(u.get("task", u)) if isinstance(u, dict) else str(u)
                        str_r = str(r.get("task", r)) if isinstance(r, dict) else str(r)
                        if str_u != str_r:
                            plan_changed = True
                            break

                if plan_changed:
                    _replan_nochange_streak = 0
                    msg = Fore.YELLOW + f"[REPLANNING] Plan updated! Reason: {replanning_result.get('reasoning', 'No reason given')}"
                    try:
                        print(msg)
                    except UnicodeEncodeError:
                        import sys
                        enc = sys.stdout.encoding or "utf-8"
                        safe_msg = msg.encode(enc, errors="ignore").decode(enc, errors="ignore")
                        print(safe_msg)

                    # Abandon the old nodes so they don't stay pending forever
                    old_cids_to_abandon = child_ids[i:]
                    try:
                        skipped_log_path = orch.dir_paths["artifacts"] / "skipped_tasks.log"
                        with open(skipped_log_path, "a", encoding="utf-8") as f:
                            for old_cid in old_cids_to_abandon:
                                if orch.tree_node_status(old_cid) not in ("done", "failed"):
                                    task_text = str(orch.tree_node(old_cid).get("task", ""))
                                    f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] REPLAN PRUNED: {task_text}\n")
                                    orch.tree_log_event(
                                        "PRUNED",
                                        task_text,
                                        node_id=old_cid,
                                        parent_node_id=main_node_id,
                                        reason="main_replanning",
                                    )
                                    orch.tree_remove_node(old_cid)
                    except Exception as e:
                        print(f"Failed to log skipped tasks: {e}")

                    ordered_tasks = ordered_tasks[:i] + updated_remaining
                    if len(ordered_tasks) > max_allowed_width:
                        ordered_tasks = ordered_tasks[:max_allowed_width]

                    new_child_ids = orch.tree_init_children_with_kinds(
                        main_node_id,
                        [f"Sub-Task {t.get('task') if isinstance(t, dict) else t}" for t in ordered_tasks[i:]],
                        kinds=["main"] * len(ordered_tasks[i:])
                    )
                    try:
                        new_labels = [f"Sub-Task {t.get('task') if isinstance(t, dict) else t}" for t in ordered_tasks[i:]]
                        for ncid, lbl in zip(new_child_ids, new_labels):
                            orch.tree_log_event(
                                "ADDED",
                                lbl,
                                node_id=ncid,
                                parent_node_id=main_node_id,
                                reason="main_replanning",
                            )
                    except Exception:
                        pass
                    child_ids = child_ids[:i] + new_child_ids

                    # If replanning removed tasks, 'i' might now be out of bounds.
                    if i >= len(ordered_tasks):
                        print(Fore.YELLOW + "[REPLANNING] Re-planning removed the current task. Exiting loop.")
                        break

                    # Update current pointers after replanning for the current loop iteration
                    current_task_obj = ordered_tasks[i]
                    current = current_task_obj.get("task", "") if isinstance(current_task_obj, dict) else str(
                        current_task_obj)
                    current_time_budget = current_task_obj.get("time_budget_sec", None) if isinstance(current_task_obj,
                                                                                                      dict) else None
                    cid = child_ids[i]
                else:
                    _replan_nochange_streak += 1
                    print(Fore.GREEN + "[REPLANNING] Plan is still optimal. Proceeding.")
                    # Do not spend additional cycles on repeated no-op replanning.
                    if _replan_nochange_streak >= 2:
                        _replan_disabled = True
                        print(Fore.YELLOW + "[REPLANNING] Disabled for this branch after no-op result.")

            elif (
                i > 0
                and not prev_child_failed
                and max_allowed_width > 1
                and resume
                and len(ordered_tasks) > 2
                and i == 1
            ):
                print(
                    Fore.YELLOW
                    + "[REPLANNING] Skipped: resume=True (tail replanning disabled for stable replay)."
                )

            print(
                f"Sub-Task [{i + 1}/{len(ordered_tasks)}] tree_depth={_child_tree_depth} pipeline_level={depth + 1} {current} (status: {child_status})"
            )
            # Pass actual remaining time and effective budget to agents via spec
            actual_remaining = _global_remaining_sec(orch)
            if current_time_budget:
                effective_budget = min(int(current_time_budget), int(actual_remaining)) if actual_remaining > 0 else int(current_time_budget)
                print(Fore.CYAN + f"   -> Task Budget: {effective_budget}s (planned={current_time_budget}s, global_remaining={int(actual_remaining)}s)")
            else:
                effective_budget = int(actual_remaining) if actual_remaining > 0 else 0
                print(Fore.CYAN + f"   -> Task Budget: {effective_budget}s (from global remaining)")
            if spec is not None:
                spec["_current_task_budget_sec"] = effective_budget
                spec["_global_remaining_sec"] = int(actual_remaining)

            _task_start_time = time.time()
            # Git-anchored pre-snapshot of artifacts_dir. Ground-truth diffing
            # against this sha at the end of the subtask replaces regex-based
            # "did the agent actually save this?" heuristics.
            try:
                _pre_sha = snapshot_artifacts(orch, message=f"pre-subtask {cid}")
                orch._last_artifacts_snapshot_sha = _pre_sha  # type: ignore[attr-defined]
            except Exception as _e:
                print(Fore.YELLOW + f"[ARTIFACT-GIT] pre-snapshot skipped: {_e}")
            try:
                temp_answer, code_answer = main_pipeline(
                    orch,
                    llm_strong,
                    llm_fast,
                    code_llm,
                    task=f"Sub-Task {current}",
                    previous_answers=full_answer,
                    main_task_context=mt_context,
                    node_id=cid,
                    parent_node_id=main_node_id,
                    spec=spec,
                    resume=resume,
                    allow_spawn_improvement=False,
                    freeze_spec=freeze_spec,
                    improve_mode=False,
                    mcp_tools=mcp_tools,
                    depth=depth + 1,
                    max_allowed_depth=max_allowed_depth - 1,
                    max_allowed_width=(99 if depth == 0 else max(1, max_allowed_width - 1))
                )
            except (TimeoutError, Exception) as e:
                # Do not abort the entire root pipeline on hard deadline or provider crash.
                # Instead: stop executing remaining subtasks and allow root finalization/improver to run.
                msg = Fore.YELLOW + f"[DEADLINE/ERROR] {type(e).__name__}: {e} Stopping remaining tasks in this branch."
                try:
                    print(msg)
                except UnicodeEncodeError:
                    import sys
                    enc = sys.stdout.encoding or "utf-8"
                    safe_msg = msg.encode(enc, errors="ignore").decode(enc, errors="ignore")
                    print(safe_msg)
                try:
                    orch.tree_log_event(
                        "SKIPPED",
                        f"Sub-Task {current}",
                        node_id=cid,
                        parent_node_id=main_node_id,
                        reason="hard_deadline_exceeded",
                    )
                except Exception:
                    pass
                break
            if code_answer is not None:
                code_bank.append(code_answer.replace("```", "").replace("python", ""))
            time.sleep(0.2)

            if isinstance(temp_answer, list):
                temp_answer = "\n".join(map(str, temp_answer))
            elif not isinstance(temp_answer, str):
                temp_answer = str(temp_answer) if temp_answer is not None else ""

            full_answer = (full_answer or "") + temp_answer + "\n"

            # Track budget savings from early-finishing tasks
            if effective_budget and effective_budget > 0:
                actual_task_elapsed = time.time() - _task_start_time
                saved_sec = max(0, int(effective_budget - actual_task_elapsed))
                if saved_sec > 30 and spec is not None:
                    extra_budget = spec.get("_extra_budget_sec", 0) + saved_sec
                    spec["_extra_budget_sec"] = extra_budget
                    print(Fore.CYAN + f"[BUDGET] Task saved {saved_sec}s. Extra budget pool: {extra_budget}s")

            i += 1

        # Comment translated to English.
        if is_root:
            try:
                root = Path(orch.project_root)
                art_dir = root / orch.cfg.paths.artifacts_dir
                last_m = {}
                last_c = ""
                mpath = art_dir / "last" / "metrics.json"
                cpath = art_dir / "last" / "code.py"
                if mpath.exists():
                    try:
                        last_m = json.loads(mpath.read_text(encoding="utf-8"))
                    except Exception:
                        last_m = {}
                if cpath.exists():
                    try:
                        last_c = cpath.read_text(encoding="utf-8")
                    except Exception:
                        last_c = ""

                ok, norm, _ = _validate_and_normalize_metrics(last_m)
                if not ok:
                    if _global_remaining_sec(orch) <= 0:
                        print(Fore.YELLOW + "[ROOT] metrics recovery skipped: no global time remaining")
                        ok = False
                        norm = {}
                    else:
                        print(
                            Fore.YELLOW + "[ROOT] last metrics invalid or missing — running verification to recover METRICS_JSON")
                        verif_code = verification_code_gen(code_llm, spec,
                                                           context="Look into ./artifacts for checkpoints/preds.")
                        _v_ok, _v_issues = _audit_generated_code_policy(verif_code)
                        if not _v_ok:
                            print(Fore.YELLOW + f"[ROOT] verifier code blocked by policy: {_v_issues}; invoking agentic verifier repair")
                            verif_code = _agentic_repair_verifier_code(
                                code_llm=code_llm,
                                spec=spec,
                                initial_code=verif_code,
                                context=(
                                    "Root post-subtask metric recovery. "
                                    "Verifier must use filesystem evidence and dynamic spec loading."
                                ),
                                mcp_tools=mcp_tools,
                            )
                        vfile = f"{orch.cfg.paths.scripts_dir}/verify_root_{uuid.uuid4().hex[:6]}.py"
                        orch.write_file(vfile, verif_code)
                        vres = orch.run_python_file(vfile, stream=True)
                        recov = parse_metrics_from_stdout(vres.get("output", ""))
                        ok, norm, _ = _validate_and_normalize_metrics(recov)
                        if ok:
                            _update_best_from_candidate(
                                orch,
                                candidate_metrics=norm,
                                code_text=last_c,
                                tag="main_post_subtasks",
                                spec=spec,
                            )
                else:
                    _update_best_from_candidate(
                        orch,
                        candidate_metrics=norm,
                        code_text=last_c,
                        tag="main_post_subtasks",
                        spec=spec,
                    )
            except Exception as e:
                print(Fore.YELLOW + f"[ROOT] metrics post-check failed: {e}")

        # Comment translated to English.
        if is_root:
            try:
                art_dir = Path(orch.project_root) / orch.cfg.paths.artifacts_dir
                base_code = ""
                best_code_path = art_dir / "best" / "code.py"
                last_code_path = art_dir / "last" / "code.py"
                if best_code_path.exists():
                    base_code = best_code_path.read_text(encoding="utf-8")
                elif last_code_path.exists():
                    base_code = last_code_path.read_text(encoding="utf-8")
                elif code_bank:
                    base_code = code_bank[-1]

                base_metrics = {}
                best_metrics_path = art_dir / "best" / "metrics.json"
                last_metrics_path = art_dir / "last" / "metrics.json"
                if best_metrics_path.exists():
                    base_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
                elif last_metrics_path.exists():
                    base_metrics = json.loads(last_metrics_path.read_text(encoding="utf-8"))

                # FIX: Do not run improver for demo/skipped tasks
                if base_metrics.get("type") == "skipped":
                    print(
                        Fore.YELLOW + "[IMPROVER] Skipping improvement pipeline because last run was a demo/skipped task.")
                    allow_spawn_improvement = False

                orc = getattr(orch.cfg, "orchestration", None)
                if isinstance(orc, dict):
                    iters = int(orc.get("optimize_iters", 4))
                    rel_thr = float(orc.get("min_metric_improvement_rel", 0.05))
                else:
                    iters = int(getattr(orc, "optimize_iters", 4))
                    rel_thr = float(getattr(orc, "min_metric_improvement_rel", 0.05))

                # Policy: if improve budget is zero, skip improver entirely (no improve node creation).
                try:
                    imp_budget_sec = int(getattr(orch.cfg.orchestration, "improve_budget_sec", 0) or 0)
                except Exception:
                    imp_budget_sec = 0
                if imp_budget_sec <= 0:
                    print(
                        Fore.YELLOW
                        + f"[IMPROVER] improve_budget_sec={imp_budget_sec}; skipping improvement pipeline."
                    )
                    allow_spawn_improvement = False

                if base_code and iters > 0:
                    if allow_spawn_improvement:
                        print(Fore.MAGENTA + "[IMPROVER] launching dedicated improvement_pipeline")

                        # Comment translated to English.
                        resume_imp_id = improve_children[0]['node_id'] if improve_children else None
                        if resume_imp_id:
                            print(Fore.YELLOW + f"[RESUME] Resuming existing improvement node: {resume_imp_id}")

                        # Collect and deeply analyze main pipeline artifacts (ReAct agent, 5 steps)
                        _main_artifacts = _collect_and_enrich_artifacts(
                            orch, spec, llm_strong, task,
                            previous_iteration_context=spec.get("_inter_iteration_context"),
                        )
                        if _main_artifacts:
                            spec["_main_pipeline_artifacts"] = _main_artifacts
                            _ma_keys = [k for k in _main_artifacts if isinstance(_main_artifacts[k], str)]
                            print(Fore.CYAN + f"[IMPROVER] Enriched artifacts collected: {_ma_keys}")

                        imp_summary, imp_best = improvement_pipeline(
                            orch=orch,
                            llm_strong=llm_strong,
                            llm_fast=llm_fast,
                            code_llm=code_llm,
                            task=task,
                            spec=spec,
                            previous_answers=previous_answers,
                            node_id=resume_imp_id,  # Comment translated to English.
                            parent_node_id=main_node_id,
                            resume=True,
                            mcp_tools=mcp_tools,
                            depth=depth + 1,
                            main_pipeline_artifacts=_main_artifacts,
                        )
                        if imp_summary:
                            try:
                                summary_lines_text = imp_summary if isinstance(imp_summary, str) else "\n".join(
                                    imp_summary)
                                orch.log("improver_summary", {"text": summary_lines_text[-4000:]})
                                try:
                                    full_answer = "\n[IMPROVEMENT SUMMARY]\n" + summary_lines_text + "\n"
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        if imp_best and isinstance(imp_best, dict) and "primary" in imp_best:
                            print(Fore.CYAN + f"[IMPROVER] best primary now: {imp_best.get('primary')}")
                    else:
                        print(Fore.MAGENTA + f"[OPT] start (iters={iters}, rel_thr={rel_thr})")
                        _, best_code, best_metrics = optimize_metrics(
                            orch, llm_strong, llm_fast, code_llm,
                            spec=spec,
                            base_code=base_code,
                            base_metrics=base_metrics,
                            max_iters=iters,
                            min_improvement_rel=rel_thr,
                        )
                        if best_code and best_metrics:
                            print(Fore.MAGENTA + "[OPT] done]")
            except Exception as e:
                import traceback
                print(Fore.RED + f"[IMPROVER] error in improvement pipeline:")
                traceback.print_exc()
                print(Fore.YELLOW + f"[IMPROVER] skipped due to error: {e}")

            try:
                final_sub = _finalize_single_submission_by_all_metrics_llm(
                    orch, llm_fast, spec or {}, task=task, code_llm=code_llm, mcp_tools=mcp_tools
                )
                if final_sub:
                    print(Fore.GREEN + f"[FINAL] single submission ready at {final_sub}")
                else:
                    print(Fore.YELLOW + "[FINAL] finalize could not resolve submission; running output gate anyway")
            except Exception as e:
                print(Fore.YELLOW + f"[FINAL] finalize submission failed: {e}")

            gate = run_final_output_gate(orch, spec or {}, task_txt_root=Path(orch.project_root))
            if not gate.get("ok"):
                # Re-run the selector with more ReAct rounds — it can add/fix submission code
                # using the existing code generation pipeline.  Do NOT generate a blind recovery
                # script (that tends to produce sample-copy or constant predictions).
                print(Fore.YELLOW + "[FINAL] Output gate failed. Re-running selector with extended repair...")
                print(Fore.YELLOW + "Gate errors: " + "; ".join(gate.get("errors") or []))
                try:
                    _cfg_orc = getattr(orch.cfg, "orchestration", object())
                    _prev_rounds = int(getattr(_cfg_orc, "react_max_rounds", 3) or 3)
                    _cfg_orc.react_max_rounds = max(_prev_rounds, 5)
                except Exception:
                    pass
                try:
                    final_sub2 = _finalize_single_submission_by_all_metrics_llm(
                        orch, llm_fast, spec or {}, task=task, code_llm=code_llm, mcp_tools=mcp_tools
                    )
                    if final_sub2:
                        print(Fore.CYAN + f"[FINAL] Selector retry produced: {final_sub2}")
                except Exception as e:
                    print(Fore.RED + f"[FINAL] Selector retry failed: {e}")
                gate = run_final_output_gate(orch, spec or {}, task_txt_root=Path(orch.project_root))
            if not gate.get("ok"):
                gate_errs = "; ".join(gate.get("errors") or [])
                print(Fore.RED + f"[FINAL] No valid submission produced. Errors: {gate_errs}")
                raise RuntimeError("[FINAL] output gate failed: " + gate_errs)
            print(Fore.GREEN + "[FINAL] output gate passed (canonical submission + validation)")

        # Comment translated to English.
        # FIX: Use project log for context instead of scattered variables
        # Comment translated to English.
        # Comment translated to English.
        project_log_content = orch.get_project_log_content()
        log_context = project_log_content or ""
        log_context_lower = log_context.lower().strip()
        if (not log_context_lower) or ("empty" in log_context_lower) or (len(log_context_lower) < 200):
            # Comment translated to English.
            exec_tail = (full_answer or "")
            exec_tail = (
                exec_tail[-orch.cfg.runtime.aggregate_tail_chars:]
                if len(exec_tail) > orch.cfg.runtime.aggregate_tail_chars
                else exec_tail
            )
            code_tail = code_bank[-1] if code_bank else ""
            code_tail = (
                code_tail[-orch.cfg.runtime.aggregate_tail_chars:]
                if len(code_tail) > orch.cfg.runtime.aggregate_tail_chars
                else code_tail
            )
            log_context = (
                f"{project_log_content}\n\n"
                f"[EXECUTION OUTPUT TAIL]\n{exec_tail}\n\n"
                f"[LAST CODE TAIL]\n{code_tail}\n"
            )
        # Artifact provenance: scan stdout/project log for claimed saves, keep
        # only the ones that actually exist on disk. Prevents the aggregate
        # agent from hallucinating artifacts copied out of the task description.
        _verify_blobs = [full_answer or "", project_log_content or ""]
        _artifact_check = _verify_claimed_artifacts(orch, _verify_blobs)
        _verified_paths = _artifact_check.get("verified") or []
        _missing_paths = _artifact_check.get("missing") or []

        # Ground truth via git-anchored snapshot diff of artifacts_dir. This
        # bypasses regex-guessing entirely: files that appear in git diff were
        # physically written to disk during the subtask. Prefer these paths.
        # Safe to call even without git — returns empty structures in that case.
        _pre_sha = getattr(orch, "_last_artifacts_snapshot_sha", None)
        _post_sha = snapshot_artifacts(orch, message=f"post-subtask {getattr(orch, 'current_node_id', '')}")
        orch._last_artifacts_snapshot_sha = _post_sha  # type: ignore[attr-defined]
        _git_diff = artifacts_diff_since(orch, _pre_sha) if _pre_sha else {"added": [], "modified": [], "deleted": []}
        _git_written = [
            f"{orch.cfg.paths.artifacts_dir}/{p}"
            for p in (_git_diff.get("added", []) + _git_diff.get("modified", []))
        ]
        # Merge git ground truth into verified paths (primary) while keeping
        # regex-claimed for backwards compat on writes that happened outside
        # artifacts_dir (we still want to know about them).
        if _git_written:
            _seen = set(_verified_paths)
            for p in _git_written:
                if p not in _seen:
                    _verified_paths.append(p)
                    _seen.add(p)
            print(Fore.GREEN + f"[ARTIFACT-GIT] post-snapshot={(_post_sha or '')[:8]} "
                               f"added={len(_git_diff.get('added', []))} "
                               f"modified={len(_git_diff.get('modified', []))} "
                               f"deleted={len(_git_diff.get('deleted', []))}")
        if _verified_paths or _missing_paths:
            print(
                Fore.CYAN
                + f"[ARTIFACT] verified={len(_verified_paths)} missing={len(_missing_paths)}"
            )
            if _missing_paths:
                print(Fore.YELLOW + f"[ARTIFACT] claimed but not on disk: {_missing_paths[:5]}")
        answer = aggregate_answers(
            llm_fast,
            task,
            log_context,
            spec,
            verified_artifacts=_verified_paths,
            claimed_but_missing=_missing_paths,
        )

        # --- FIX: Ensure answer is a string ---
        if isinstance(answer, list):
            answer = "\n".join(map(str, answer))
        elif not isinstance(answer, str):
            answer = str(answer)

        if is_root and llm_fast is not None:
            try:
                art_dir = Path(orch.project_root) / orch.cfg.paths.artifacts_dir
                code_summary_chk = ""
                for p in (
                    art_dir / "final" / "best_code.py",
                    art_dir / "best" / "code.py",
                    art_dir / "last" / "code.py",
                ):
                    try:
                        if p.exists():
                            code_summary_chk = p.read_text(encoding="utf-8", errors="ignore")
                            break
                    except Exception:
                        pass
                if not code_summary_chk and code_bank:
                    code_summary_chk = code_bank[-1]
                code_summary_chk = shorten_string_middle(code_summary_chk or "", 2500)
                metrics_json_chk = ""
                for mp in (art_dir / "best" / "metrics.json", art_dir / "last" / "metrics.json"):
                    try:
                        if mp.exists():
                            metrics_json_chk = mp.read_text(encoding="utf-8", errors="ignore")
                            break
                    except Exception:
                        pass
                metrics_json_chk = shorten_string_middle(metrics_json_chk or "", 4000)
                stdout_tail_chk = shorten_string_middle(full_answer or "", 4000)
                nid_chk = str(main_node_id) if main_node_id else "root"
                answer = check_and_fix_answer(
                    orch,
                    llm_fast,
                    task,
                    answer,
                    spec or {},
                    node_id=nid_chk,
                    code_summary=code_summary_chk,
                    metrics_json=metrics_json_chk,
                    stdout_tail=stdout_tail_chk,
                    stderr_tail="",
                    improvement_summary="",
                )
            except Exception as e:
                orch.log("check_and_fix_answer_skipped", {"error": str(e)})
                print(Fore.YELLOW + f"[CHECK] check_and_fix_answer skipped: {e}")

        if is_root and llm_fast is not None:
            try:
                rubric = checks_generation(llm_fast, task, spec or {})
                rubric_txt = shorten_string_middle(str(rubric), 2500)
                rubric_ok = check_answer(
                    llm_fast, task, answer, rubric_txt, spec or {}
                )
                orch.log(
                    "aggregate_rubric_check",
                    {"verdict": rubric_ok, "rubric_head": rubric_txt[:400]},
                )
                if rubric_ok != "True":
                    print(
                        Fore.YELLOW
                        + "[AGGREGATE] LLM rubric check did not pass on final summary (see orchestrator log)."
                    )
            except Exception as e:
                orch.log("aggregate_rubric_check_skipped", {"error": str(e)})

        # Safe print for Windows consoles with legacy encodings (e.g. cp1251)
        try:
            print("AGGREGATE ANSWER\n", answer)
        except UnicodeEncodeError:
            import sys
            enc = sys.stdout.encoding or "utf-8"
            safe_answer = answer.encode(enc, errors="ignore").decode(enc, errors="ignore")
            print("AGGREGATE ANSWER\n", safe_answer)

        # Persist aggregate summary as an artifact for task_plan.md
        try:
            agg_rel = f"{orch.cfg.paths.artifacts_dir}/aggregate_summary.md"
            orch.write_file(agg_rel, answer)
            # Refresh rich task_plan.md to include the latest summary
            try:
                orch._update_markdown_plan()  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        if is_root and llm_fast is not None:
            try:
                sec_entry = log_update_agent(
                    llm_fast,
                    task,
                    "AGGREGATE_SUMMARY",
                    shorten_string_middle(answer, 3500),
                )
                if (sec_entry or "").strip():
                    head = (task or "").splitlines()[0][:240] if task else "Aggregate"
                    orch.log_to_project_log(
                        f"[Secretary] {head}",
                        depth,
                        sec_entry.strip(),
                        "done",
                    )
            except Exception as e:
                orch.log("log_update_agent_skipped", {"error": str(e)})
            try:
                agg_rel = f"{orch.cfg.paths.artifacts_dir}/aggregate_summary.md"
                preview = shorten_string_middle(answer, 8000)
                arch_note = artifact_reviewer_agent(llm_fast, agg_rel, preview)
                if (arch_note or "").strip():
                    orch.write_file(
                        f"{orch.cfg.paths.artifacts_dir}/aggregate_summary_review.md",
                        arch_note.strip(),
                    )
                    orch.log(
                        "artifact_reviewer_aggregate",
                        {"chars": len(arch_note.strip())},
                    )
            except Exception as e:
                orch.log("artifact_reviewer_skipped", {"error": str(e)})

        # AUTO-DISCOVERY: Automatically update spec based on EDA findings
        # Check if this was an EDA task and if target column was discovered
        if "Initial Data Analysis" in task:
            print(Fore.CYAN + "[AUTO-DISCOVERY] Processing EDA task for automatic spec updates...")
            # Look for multiple possible patterns that might appear in logs
            target_patterns = [
                "DEBUG: Target Distribution in Train Dataset:",
                "DEBUG: Target Distribution in Train Data:",
                "DEBUG: Train Target Distribution:"
            ]
            
            pattern_found = False
            for pattern in target_patterns:
                if pattern in project_log_content:
                    pattern_found = True
                    print(Fore.CYAN + f"[AUTO-DISCOVERY] Found target distribution pattern: {pattern}")
                    break
            
            if pattern_found:
                print(Fore.CYAN + "[AUTO-DISCOVERY] Found target distribution in logs, attempting to extract target column...")
                # Extract target column name from the log
                import re
                # Look for the actual target column name in the log output
                # The log shows something like "target\n0    2501\n1    2499"
                target_match = re.search(r"DEBUG: Train Target Distribution:\s*(\w+)", project_log_content)
                if target_match:
                    target_col_name = target_match.group(1)
                    print(Fore.CYAN + f"[AUTO-DISCOVERY] Discovered target column: {target_col_name}")
                    # Update spec with discovered target column in both locations
                    if "data" not in spec:
                        spec["data"] = {}
                    if "meta" not in spec:
                        spec["meta"] = {}
                    spec["data"]["target_column"] = target_col_name
                    spec["meta"]["target_column"] = target_col_name
                    # Save updated spec
                    try:
                        spec_path = f"{orch.cfg.paths.artifacts_dir}/spec.json"
                        orch.write_file(spec_path, json.dumps(spec, ensure_ascii=False, indent=2))
                        print(Fore.YELLOW + f"[AUTO-DISCOVERY] Updated spec with target column: {target_col_name}")
                    except Exception as e:
                        print(Fore.RED + f"[AUTO-DISCOVERY] Failed to update spec: {e}")
                else:
                    # Fallback to direct inspection if regex fails
                    target_col_name = "target"  # We know this from the CSV structure
                    print(Fore.CYAN + f"[AUTO-DISCOVERY] Using fallback target column: {target_col_name}")
                    if "data" not in spec:
                        spec["data"] = {}
                    if "meta" not in spec:
                        spec["meta"] = {}
                    spec["data"]["target_column"] = target_col_name
                    spec["meta"]["target_column"] = target_col_name
                    try:
                        spec_path = f"{orch.cfg.paths.artifacts_dir}/spec.json"
                        orch.write_file(spec_path, json.dumps(spec, ensure_ascii=False, indent=2))
                        print(Fore.YELLOW + f"[AUTO-DISCOVERY] Updated spec with target column: {target_col_name}")
                    except Exception as e:
                        print(Fore.RED + f"[AUTO-DISCOVERY] Failed to update spec: {e}")
            else:
                print(Fore.CYAN + "[AUTO-DISCOVERY] No target distribution found in logs")

        # Return final aggregate answer for the caller
        return answer, None


def update_project_context_after_execution(orch: GlobalOrchestrator, task: str, stdout: str, stderr: str, metrics: Dict[str, Any], code: str) -> str:
    """
    Append to `artifacts/project_context.md` with a compact, LLM-friendly block (rolling tail cap).
    Surfaces metric contract, shapes, outcomes; strips TensorFlow/absl noise from stderr.
    Returns technical artifact specs discovered.
    """
    artifact_summary = ""
    try:
        art_dir = orch.project_root / orch.dir_paths["artifacts"]
        context_path = art_dir / "project_context.md"

        # Primary metric contract (from spec when possible).
        primary_name = "accuracy"
        primary_maximize = True
        secondary_hint = (
            "many keys OK — primary is leaderboard; extras = diagnostics (per-class P/R/F1, "
            "confusion path, segment errors, … per spec.secondary_metrics)"
        )
        try:
            spec_path = art_dir / "spec.json"
            if spec_path.exists():
                spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
                pm = spec_obj.get("primary_metric") or {}
                primary_name = pm.get("name", primary_name)
                primary_maximize = bool(pm.get("maximize", primary_maximize))
                sm = spec_obj.get("secondary_metrics") or []
                if isinstance(sm, list) and sm:
                    secondary_hint = ", ".join(str(x) for x in sm[:20])
        except Exception:
            pass

        lines: List[str] = []
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Keep full task text — truncating mid-string (shorten_string_middle) loses
        # the acceptance criteria that live at the tail of task descriptions.
        full_task = (task or "").replace("\n", " ").strip()
        lines.append(f"## Update [{ts}]")
        lines.append(f"**Task**: {full_task}")
        lines.append("")
        lines.append("**METRICS_JSON (contract)**")
        lines.append(f"- type: `calculated` | `skipped`")
        lines.append(f"- name: `{primary_name}` (must match spec.primary_metric.name)")
        lines.append(f"- primary: float; maximize: {primary_maximize}")
        lines.append(
            f"- extras: optional dict (often large); choose metrics that explain *where* to improve; "
            f"spec.secondary_metrics: {secondary_hint}"
        )
        lines.append("")

        # Minimal data hints for debugging splits/shapes.
        for needle in ("Train Data Shape:", "Data Shape:", "Target Distribution:", "Test Data Shape:"):
            found = next((ln.strip() for ln in stdout.splitlines() if needle in ln), "")
            if found:
                lines.append(found)
                break

        # Metric outcome summary (if it matched parse contract).
        if isinstance(metrics, dict) and metrics:
            mtype = metrics.get("type", "")
            if mtype == "calculated":
                pval = metrics.get("primary", metrics.get("primary_score", None))
                if pval is not None:
                    try:
                        lines.append(f"METRICS: {primary_name}={float(pval)}")
                    except Exception:
                        lines.append(f"METRICS: {primary_name}={pval}")
            elif mtype:
                # If LLM printed a final-like payload (e.g. project_complete), highlight it.
                if mtype not in ("calculated", "skipped"):
                    keys = list(metrics.keys())
                    lines.append(f"ATTENTION: invalid METRICS_JSON type='{mtype}' keys={keys[:6]}...")

        # Recent error signatures (only the fix hints; drop TF/absl chatter).
        error_hints: List[str] = []
        stderr_use = stderr or ""
        if stderr_use:
            skip_sub = ("absl::", "oneDNN", "I0000", "cuda_", "tensorflow/core")
            stderr_use = "\n".join(
                ln for ln in stderr_use.splitlines()
                if ln.strip() and not any(s in ln for s in skip_sub)
            )
            if "transformers_" in stderr_use:
                error_hints.append("ERR: sklearn ColumnTransformer transformers_ accessed before fit -> inspect after fit_transform")
            if "joblib.loads" in stderr_use:
                error_hints.append("ERR: joblib.loads() doesn't exist -> use deepcopy or joblib.load/dump")
            if "not JSON serializable" in stderr_use or "numpy" in stderr_use:
                error_hints.append("ERR: numpy scalar JSON serialization -> convert numpy types or use NumpyEncoder")
            if "Metric validation failed" in stdout or "Invalid metrics" in stdout:
                error_hints.append("ERR: METRICS_JSON contract mismatch -> ensure type='calculated', name matches spec, use 'primary'")

        if error_hints:
            lines.append("")
            lines.append("**Errors / hints**:")
            for hint in error_hints[:8]:
                lines.append(f"- {hint}")

        # --- NEW: Automated Artifact Documentation ---
        artifact_notes: List[str] = []
        try:
            # Files changed in the last 15 minutes
            now = time.time()
            for f in art_dir.glob("*"):
                if f.is_file() and (now - f.stat().st_mtime) < 900:
                    if f.name in ("project_context.md", "PROJECT_LOG.md", "spec.json", "last_run.log"):
                        continue
                    
                    preview = ""
                    if f.suffix.lower() == ".csv":
                        try:
                            with open(f, 'r', encoding='utf-8', errors='ignore') as cf:
                                header = cf.readline().strip()
                                preview = f"CSV with columns: {header}"
                        except: pass
                    elif f.suffix.lower() in (".pkl", ".pickle"):
                        probe_code = f"import pickle, pandas as pd; \nwith open(r'{f}', 'rb') as pf: \n  obj = pickle.load(pf)\nif isinstance(obj, pd.DataFrame): print(f'DataFrame: shape={{obj.shape}}, columns={{list(obj.columns)}}')\nelif isinstance(obj, dict): \n  print(f'Dict: keys={{list(obj.keys())}}')\n  for k,v in list(obj.items())[:3]: print(f'  - {{k}}: {{type(v).__name__}}')\nelse: print(f'Type: {{type(obj).__name__}}')"
                        res = orch.run_python_code(probe_code, filename="probe_art.py", timeout=20)
                        preview = res.get("output", "").strip()
                    
                    if preview:
                        _llm = getattr(orch, "llm_fast", None) or getattr(orch, "llm", None)
                        if _llm is not None:
                            note = artifact_reviewer_agent(_llm, str(f.name), preview)
                            if note:
                                artifact_notes.append(f"- **{f.name}**: {note.strip()}")
        except Exception as e:
            print(Fore.RED + f"[CONTEXT] Artifact probe failed: {e}")

        if artifact_notes:
            lines.append("")
            lines.append("**Artifact Specs (verified)**:")
            lines.extend(artifact_notes)
            artifact_summary = "\n".join(artifact_notes)

        block = "\n".join(lines).strip() + "\n"
        prev = ""
        if context_path.exists():
            try:
                prev = context_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                prev = ""
        merged = (prev.strip() + "\n\n" + block).strip() + "\n"
        # Rolling cap BY SECTIONS, not by bytes — never cut mid-update.
        # Keep the file header (everything before first "## Update") + the last N update sections.
        # This preserves acceptance criteria and data passports while bounding size.
        MAX_SECTIONS = 40
        try:
            parts = re.split(r"(?m)^(## Update \[)", merged)
            if len(parts) >= 3:
                header = parts[0]
                # Re-glue each "## Update [" marker with its body
                sections = ["## Update [" + parts[i + 1] for i in range(1, len(parts) - 1, 2)]
                if len(sections) > MAX_SECTIONS:
                    sections = sections[-MAX_SECTIONS:]
                merged = (header.rstrip() + "\n\n" + "\n".join(sections)).strip() + "\n"
        except Exception:
            pass

        context_path.write_text(merged, encoding="utf-8")
        print(Fore.GREEN + f"[CONTEXT] Appended project_context.md (section-capped, max {MAX_SECTIONS} updates)")
    except Exception as e:
        print(Fore.RED + f"[CONTEXT] Failed to update project context: {e}")
    
    return artifact_summary
