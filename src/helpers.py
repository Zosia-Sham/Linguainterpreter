from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
import csv
import math
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from colorama import Fore
import os
from .utils import _ensure_dir, _slug, _now_tag, _validate_and_normalize_metrics
from .orchestrator import GlobalOrchestrator
from .validators import detect_mixed_stacks
from .llm_utils import invoke_with_tools


def _copy_text_file(orch: GlobalOrchestrator, src: Path, dst: Path) -> None:
    try:
        text = src.read_text(encoding="utf-8", errors="ignore")
        _ensure_dir(dst.parent)
        orch.write_file(str(dst), text)
    except Exception:
        pass


def canonical_submission_path(orch: GlobalOrchestrator) -> Path:
    """Configured canonical submission file: project_root / submission_dir / submission_filename."""
    root = Path(orch.project_root)
    return root / orch.cfg.paths.submission_dir / orch.cfg.paths.submission_filename


def _react_repair_submission_candidate(
    orch: GlobalOrchestrator,
    llm_fast: Any,
    spec: Dict[str, Any],
    *,
    code_rel: str,
    code_llm: Any = None,
    mcp_tools: Optional[List[Any]] = None,
    candidate_tag: str = "",
    max_rounds: int = 3,
    deadline_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Explicit Python ReAct loop for final submission repair.

    Each round:
      1. validate_final_submission → if ok, return immediately
      2. run current candidate code → validate again → if ok, return
      3. patch code via finetune_code_v2 with full history context → continue

    Key properties:
    - max_rounds is enforced in Python, not delegated to invoke_with_tools
    - Attempt history is accumulated and passed to every patch so the LLM
      doesn't repeat the same failed strategy
    - Full context injected: task_plan.md, spec, sample_submission header,
      secondary metrics, data paths, validation errors, previous attempts
    - Proportional time logging: each round knows how many rounds remain
    """
    try:
        from src.prompts_agents import finetune_code_v2  # type: ignore
    except Exception as e:
        return {"ok": False, "reason": f"finetune_import_failed: {e}"}

    root = Path(orch.project_root)
    final_dir = root / orch.cfg.paths.artifacts_dir / "final"
    _ensure_dir(final_dir)
    current_code_rel = str(code_rel or "").strip()
    patch_count = 0
    attempt_history: List[Dict[str, Any]] = []
    last_errors: List[str] = []

    def _remaining_sec() -> float:
        if deadline_sec is not None:
            return max(0.0, deadline_sec - orch.effective_elapsed_sec())
        return float("inf")

    def _load_repair_context() -> str:
        """Collect task_plan.md, sample_submission header, data tree, secondary metrics."""
        parts: List[str] = []
        try:
            tp = root / "task_plan.md"
            if tp.exists():
                txt = tp.read_text(encoding="utf-8", errors="ignore")
                parts.append(f"=== task_plan.md (tail 3000 chars) ===\n{txt[-3000:]}")
        except Exception:
            pass
        try:
            sample = _resolve_sample_submission_path(orch, spec or {})
            if sample and sample.exists():
                with sample.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    head_lines = [f.readline() for _ in range(6)]
                parts.append(f"=== sample_submission.csv (first 5 rows) ===\n{''.join(head_lines)}")
        except Exception:
            pass
        try:
            sub_meta = spec.get("submission", {}) or {}
            cols = sub_meta.get("columns") or []
            delim = sub_meta.get("delimiter", ",")
            parts.append(f"=== submission contract ===\ncolumns={cols}\ndelimiter={repr(delim)}")
        except Exception:
            pass
        try:
            sec = spec.get("secondary_metrics") or []
            pm = (spec.get("primary_metric") or {})
            parts.append(
                f"=== metrics contract ===\n"
                f"primary={pm.get('name')} maximize={pm.get('maximize')}\n"
                f"secondary={sec}"
            )
        except Exception:
            pass
        try:
            data_info = spec.get("data", {}) or {}
            parts.append(f"=== data paths (from spec) ===\n{json.dumps(data_info, ensure_ascii=False)[:800]}")
        except Exception:
            pass
        return "\n\n".join(parts)

    def _run_code() -> Dict[str, Any]:
        """Run current code file, return dict with exit_code, stdout_tail, stderr_tail."""
        if not current_code_rel:
            return {"exit_code": -1, "stdout": "", "stderr": "empty current_code_rel"}
        p = root / current_code_rel
        if not p.exists():
            return {"exit_code": -1, "stdout": "", "stderr": f"code not found: {current_code_rel}"}
        try:
            rr = orch.run_python_file(current_code_rel, stream=True, spec=spec or {})
            return {
                "exit_code": rr.get("exit_code", 1),
                "stdout": (rr.get("output", "") or "")[:6000],
                "stderr": (rr.get("errors", "") or "")[:2500],
            }
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": f"run_exception: {e}"}

    repair_ctx = _load_repair_context()
    expected_cols = []
    try:
        expected_cols = [str(c).strip() for c in (spec.get("submission", {}).get("columns") or [])]
    except Exception:
        pass

    for round_num in range(max(1, max_rounds)):
        remaining = _remaining_sec()
        rounds_left = max_rounds - round_num
        per_round_est = (remaining / rounds_left) if rounds_left > 0 else 0
        print(
            Fore.CYAN
            + f"[REPAIR] round {round_num + 1}/{max_rounds} | tag={candidate_tag} "
            + f"| remaining={remaining:.0f}s | est_per_round={per_round_est:.0f}s"
        )

        # ── Step 1: validate current state ──────────────────────────────────
        val = validate_final_submission(orch, spec or {})
        last_errors = val.get("errors") or []
        if val.get("ok"):
            print(Fore.GREEN + f"[REPAIR] validated OK at round {round_num + 1}")
            return {
                "ok": True,
                "used_code_rel": current_code_rel,
                "summary": f"ok at round {round_num + 1}",
                "attempts": attempt_history,
            }
        attempt_history.append({
            "round": round_num + 1, "action": "validate",
            "ok": False, "errors": last_errors[:10],
        })

        # ── Step 2: run current code, validate again ─────────────────────────
        print(Fore.CYAN + f"[REPAIR] running code: {current_code_rel}")
        run_res = _run_code()
        attempt_history.append({
            "round": round_num + 1, "action": "run_code",
            "exit_code": run_res["exit_code"],
            "stderr_tail": run_res["stderr"][-400:],
        })
        val2 = validate_final_submission(orch, spec or {})
        last_errors = val2.get("errors") or []
        if val2.get("ok"):
            print(Fore.GREEN + f"[REPAIR] validated OK after run at round {round_num + 1}")
            return {
                "ok": True,
                "used_code_rel": current_code_rel,
                "summary": f"ok after run at round {round_num + 1}",
                "attempts": attempt_history,
            }
        attempt_history.append({
            "round": round_num + 1, "action": "validate_after_run",
            "ok": False, "errors": last_errors[:10],
        })

        # ── Step 3: patch code ───────────────────────────────────────────────
        # Don't patch on the last round (no point — we won't run it again)
        if round_num >= max_rounds - 1:
            print(Fore.YELLOW + f"[REPAIR] last round reached without fix — giving up this candidate")
            break

        if code_llm is None:
            print(Fore.YELLOW + "[REPAIR] no code_llm — cannot patch")
            break

        src_abs = root / current_code_rel if current_code_rel else None
        if not src_abs or not src_abs.exists():
            print(Fore.YELLOW + f"[REPAIR] source code not found: {current_code_rel}")
            break
        try:
            src_code = src_abs.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(Fore.YELLOW + f"[REPAIR] cannot read source: {e}")
            break

        # Build history summary so LLM doesn't repeat failed approaches
        history_txt = "\n".join(
            f"  Round {a['round']} [{a['action']}]: "
            + (f"errors={a.get('errors', [])}" if "errors" in a else f"exit={a.get('exit_code')} stderr={a.get('stderr_tail', '')[:200]}")
            for a in attempt_history
        )

        err_txt = (
            f"=== Final submission repair — round {round_num + 1}/{max_rounds} ===\n"
            f"Candidate tag: {candidate_tag}\n"
            f"Expected submission columns: {expected_cols}\n"
            f"Current validation errors: {last_errors}\n"
            f"Last run: exit={run_res['exit_code']} stderr_tail={run_res['stderr'][-600:]}\n\n"
            f"=== ATTEMPT HISTORY (do NOT repeat failed strategies) ===\n{history_txt}\n\n"
            f"=== PROJECT CONTEXT ===\n{repair_ctx}\n\n"
            "REQUIREMENTS:\n"
            "- Load spec from artifacts/spec.json; read submission path, columns, delimiter from spec.submission\n"
            "- Also save a copy as submission/submission.csv (standard bench location)\n"
            "- Predictions must be non-constant and finite\n"
            "- Do NOT overwrite submission.csv with metrics/reports tables\n"
            "- If training is needed and checkpoints exist in artifacts/best/ or artifacts/last/, reuse them\n"
        )

        print(Fore.CYAN + f"[REPAIR] patching code (attempt {patch_count + 1})")
        try:
            patched = finetune_code_v2(
                code_llm,
                task="Fix final submission generation",
                code=src_code,
                spec=spec or {},
                error=err_txt,
                tools=mcp_tools,
            )
            patch_count += 1
            patched_rel = str(
                (final_dir / f"best_code_react_fix_{patch_count}.py").relative_to(root)
            )
            orch.write_file(patched_rel, patched)
            current_code_rel = patched_rel
            attempt_history.append({
                "round": round_num + 1, "action": "patch",
                "new_code_rel": patched_rel, "patch_num": patch_count,
            })
            print(Fore.CYAN + f"[REPAIR] patched → {patched_rel}")
        except Exception as e:
            attempt_history.append({"round": round_num + 1, "action": "patch_failed", "error": str(e)})
            print(Fore.YELLOW + f"[REPAIR] patch failed: {e}")
            break

    return {
        "ok": False,
        "used_code_rel": current_code_rel,
        "summary": f"exhausted {max_rounds} rounds without valid submission",
        "last_errors": last_errors,
        "attempts": attempt_history,
    }


def _find_valid_submissions(orch: GlobalOrchestrator, spec: Dict[str, Any]) -> List[Tuple[Path, float]]:
    """
    Находит только файлы с каноническим именем submission.csv и возвращает [(path, mtime)].
    Критерий валидности — совпадение колонок со SPEC (если spec.submission.columns есть).
    Сканируем в пределах проекта: ./, ./artifacts, ./artifacts/submissions
    """
    root = Path(orch.project_root)
    candidates_dirs = [
        root,
        root / orch.cfg.paths.artifacts_dir,
        root / orch.cfg.paths.artifacts_dir / "submissions",
    ]

    expected_cols = []
    try:
        expected_cols = list(spec.get("submission", {}).get("columns") or [])
    except Exception:
        expected_cols = []

    out: List[Tuple[Path, float]] = []
    for d in candidates_dirs:
        if not d.exists():
            continue
        for p in d.rglob("submission.csv"):
            # Comment translated to English.
            if "versions" in p.parts and p.name != "submission.csv":
                continue
            try:
                with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue
                    if expected_cols:
                        # Comment translated to English.
                        if [h.strip() for h in header] != [c.strip() for c in expected_cols]:
                            continue
                out.append((p, p.stat().st_mtime))
            except Exception:
                continue

    # Comment translated to English.
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _read_csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None) or []
        return [str(h).strip() for h in header]
    except Exception:
        return []


def _submission_header_matches(path: Path, expected_cols: List[str]) -> bool:
    if not expected_cols:
        return True
    header = _read_csv_header(path)
    return header == [str(c).strip() for c in expected_cols]


def _load_expected_submission_cols(orch: GlobalOrchestrator) -> List[str]:
    """Best-effort expected submission columns from artifacts/spec.json."""
    root = Path(orch.project_root)
    spec_path = root / orch.cfg.paths.artifacts_dir / "spec.json"
    if not spec_path.exists():
        return []
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        cols = (spec.get("submission", {}) or {}).get("columns", []) if isinstance(spec, dict) else []
        return [str(c).strip() for c in (cols or [])]
    except Exception:
        return []


def _resolve_sample_submission_path(orch: GlobalOrchestrator, spec: Dict[str, Any]) -> Optional[Path]:
    root = Path(orch.project_root)
    sample_path_raw = ""
    try:
        data_spec = spec.get("data") if isinstance(spec, dict) else {}
        if isinstance(data_spec, dict):
            sample_path_raw = str(
                data_spec.get("sample_submission_csv")
                or data_spec.get("sample_submission")
                or ""
            ).strip()
    except Exception:
        sample_path_raw = ""
    if sample_path_raw:
        p = Path(sample_path_raw)
        p = p if p.is_absolute() else (root / p)
        if p.exists() and p.is_file():
            return p
    fallback = root / "data" / "sample_submission.csv"
    if fallback.exists() and fallback.is_file():
        return fallback
    return None


def _csv_row_count(path: Path, *, max_rows: Optional[int] = None) -> int:
    cnt = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for _ in reader:
            cnt += 1
            if max_rows is not None and cnt >= int(max_rows):
                break
    return cnt


def _validate_candidate_submission_file(
    orch: GlobalOrchestrator,
    submission_path: str,
    *,
    expected_cols: Optional[List[str]] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Validate a candidate submission path (existence, non-empty, header).
    Returns (ok, reason).
    """
    if not submission_path:
        return False, "empty submission_path"
    root = Path(orch.project_root)
    sp = root / submission_path
    if not sp.exists():
        return False, f"file does not exist: {submission_path}"
    try:
        if sp.stat().st_size <= 0:
            return False, f"file is empty: {submission_path}"
    except Exception:
        return False, f"cannot read file metadata: {submission_path}"

    cols = expected_cols if expected_cols is not None else _load_expected_submission_cols(orch)
    if cols and not _submission_header_matches(sp, cols):
        return False, "header mismatch vs spec.submission.columns"
    # Guardrail against partial submissions: compare row count with sample_submission when available.
    try:
        sample_path = _resolve_sample_submission_path(orch, spec or {})
        if sample_path is not None:
            sample_n = _csv_row_count(sample_path)
            sub_n = _csv_row_count(sp)
            if sample_n > 0 and sub_n != sample_n:
                return False, f"row count mismatch vs sample_submission: expected {sample_n}, got {sub_n}"
    except Exception:
        pass
    return True, "ok"


def _rewrite_csv_keep_columns(src: Path, dst: Path, cols_in_order: List[str]) -> bool:
    """
    Best-effort repair: rewrite CSV keeping only specified columns in the given order.
    Returns True if rewrite succeeded and dst exists and is non-empty.
    """
    if not cols_in_order:
        return False
    try:
        with src.open("r", encoding="utf-8", errors="ignore", newline="") as fin:
            r = csv.reader(fin)
            header = next(r, None) or []
            header = [str(h).strip() for h in header]
            idx = {h: i for i, h in enumerate(header)}
            if any(c not in idx for c in cols_in_order):
                return False
            _ensure_dir(dst.parent)
            with dst.open("w", encoding="utf-8", newline="") as fout:
                w = csv.writer(fout)
                w.writerow(cols_in_order)
                for row in r:
                    if not row:
                        continue
                    out_row = []
                    for c in cols_in_order:
                        j = idx[c]
                        out_row.append(row[j] if j < len(row) else "")
                    w.writerow(out_row)
        try:
            return dst.exists() and dst.stat().st_size > 0
        except Exception:
            return False
    except Exception:
        return False


def _detect_and_store_submissions(
    orch: GlobalOrchestrator,
    spec: Dict[str, Any],
    iter_dir: Optional[Path] = None
) -> str:
    """
    Ищет валидный сабмишен и, если указан iter_dir, копирует его в iter_dir/submissions/.
    Возвращает ОТНОСИТЕЛЬНЫЙ путь до выбранного сабмишена (от корня проекта) или "".
    """
    root = Path(orch.project_root)
    found = _find_valid_submissions(orch, spec)
    if not found:
        return ""

    best_path, _ = found[0]
    rel = str(best_path.relative_to(root))

    if iter_dir:
        try:
            dst = iter_dir / "submissions" / best_path.name
            _copy_text_file(orch, best_path, dst)
        except Exception:
            pass

    return rel


def _append_ledger_row(orch: GlobalOrchestrator, row: Dict[str, Any]) -> None:
    """
    Ведём 'таблицу' в artifacts/versions/ledger.csv:
    ts,tag,primary,maximize,metrics_path,code_path,submission_path
    """
    root = Path(orch.project_root)
    ver = root / orch.cfg.paths.artifacts_dir / "versions"
    _ensure_dir(ver)
    ledger = ver / "ledger.csv"
    header = "ts,tag,primary,maximize,metrics_path,code_path,submission_path\n"
    line = ",".join([
        str(row.get("ts", "")),
        row.get("tag", "").replace(",", " "),
        str(row.get("primary", "")),
        str(row.get("maximize", True)),
        row.get("metrics_path", "").replace(",", " "),
        row.get("code_path", "").replace(",", " "),
        row.get("submission_path", "").replace(",", " "),
    ]) + "\n"
    try:
        if not ledger.exists():
            orch.write_file(str(ledger), header + line)
        else:
            # Comment translated to English.
            cur = ledger.read_text(encoding="utf-8", errors="ignore")
            orch.write_file(str(ledger), cur + line)
    except Exception:
        pass


def _record_metrics_version(
    orch: GlobalOrchestrator,
    metrics: Dict[str, Any],
    code_text: str,
    tag: str,
    submission_path: str = "",
) -> Dict[str, Any]:
    """
    Сохраняет снапшот в:
      artifacts/versions/<ts>_<tag>/{metrics.json, code.py, submission.csv?}
    и дописывает:
      artifacts/versions/index.json  (JSON список объектов)
      artifacts/versions/ledger.csv  (CSV только по primary)
    Возвращает запись-элемент индекса.
    """
    root = Path(orch.project_root)
    art = root / orch.cfg.paths.artifacts_dir
    ver = art / "versions"
    _ensure_dir(ver)
    ts = _now_tag()
    entry_dir = ver / f"{ts}_{_slug(tag)}"
    _ensure_dir(entry_dir)

    # Comment translated to English.
    metrics_path = entry_dir / "metrics.json"
    code_path = entry_dir / "code.py"
    try:
        orch.write_file(str(metrics_path), json.dumps(metrics, ensure_ascii=False, indent=2))
    except Exception:
        pass
    try:
        if code_text:
            orch.write_file(str(code_path), code_text)
    except Exception:
        pass

    # Comment translated to English.
    submission_rel = ""
    submission_name = ""
    submission_reason = ""
    if submission_path:
        try:
            sp = root / submission_path
            if sp.exists():
                submission_name = sp.name
                dst = entry_dir / submission_name
                _copy_text_file(orch, sp, dst)
                submission_rel = str(dst.relative_to(root))
                submission_reason = "ok"
            else:
                submission_reason = "submission path does not exist"
        except Exception:
            submission_rel = ""
            submission_reason = "submission copy failed"
    else:
        submission_reason = "missing or invalid submission for this candidate"

    # Comment translated to English.
    idx_path = ver / "index.json"
    entry = {
        "ts": ts,
        "tag": tag,
        "paths": {
            "metrics": str(metrics_path.relative_to(root)),
            "code": str(code_path.relative_to(root)) if code_text else "",
            "submission": submission_rel,
        },
        "submission_name": submission_name,
        "submission_valid": bool(submission_rel),
        "submission_reason": submission_reason,
        "name": metrics.get("name", "primary"),
        "primary": metrics.get("primary"),
        "maximize": metrics.get("maximize", True),
    }
    try:
        if idx_path.exists():
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
            if not isinstance(idx, list):
                idx = []
        else:
            idx = []
        idx.append(entry)
        orch.write_file(str(idx_path), json.dumps(idx, ensure_ascii=False, indent=2))
    except Exception:
        pass

    try:
        orch._refresh_versions_table()  # type: ignore[attr-defined]
    except Exception:
        pass

    # Comment translated to English.
    try:
        _append_ledger_row(orch, {
            "ts": ts,
            "tag": tag,
            "primary": entry.get("primary"),
            "maximize": entry.get("maximize"),
            "metrics_path": entry["paths"]["metrics"],
            "code_path": entry["paths"]["code"],
            "submission_path": entry["paths"]["submission"],
        })
    except Exception:
        pass

    return entry


def _metric_improved(new: Dict[str, Any], old: Dict[str, Any]) -> bool:
    try:
        maximize = bool(new.get("maximize", old.get("maximize", True)))
        nv = float(new["primary"])
        ov = float(old.get("primary", float("-inf") if maximize else float("+inf")))
    except Exception:
        return False
    return (nv > ov) if maximize else (nv < ov)


def _update_best_from_candidate(
    orch: GlobalOrchestrator,
    candidate_metrics: Dict[str, Any],
    code_text: str,
    tag: str,
    enforce_validation: bool = False,   # Comment translated to English.
    submission_path: str = "",
    spec: Optional[Dict[str, Any]] = None,
) -> None:
    """
    - Валидирует метрики.
    - Сохраняет артефакты (код, метрики, сабмишен) в 3 места:
      1. `artifacts/last/` (перезаписывается): для быстрого доступа к результатам последнего запуска.
      2. `artifacts/versions/<ts>_<tag>/` (неизменяемый): для полной истории всех запусков.
      3. `artifacts/best/` (перезаписывается): ТОЛЬКО если метрика кандидата лучше текущей лучшей.
    """
    # Comment translated to English.
    # Comment translated to English.
    ok, norm_metrics, _ = _validate_and_normalize_metrics(candidate_metrics)
    if not ok:
        # Comment translated to English.
        # Comment translated to English.
        print(Fore.YELLOW + f"[WARN] Invalid metrics received for tag '{tag}'. Skipping artifact update.")
        return
    if norm_metrics.get("type") == "skipped":
        # Ledger / versions / best are only for calculated runs with a numeric primary.
        return

    root = Path(orch.project_root)
    art_dir = root / orch.cfg.paths.artifacts_dir
    expected_cols = _load_expected_submission_cols(orch)
    sub_ok, sub_reason = _validate_candidate_submission_file(
        orch,
        submission_path or "",
        expected_cols=expected_cols,
        spec=spec if isinstance(spec, dict) else None,
    )
    if not sub_ok:
        print(Fore.YELLOW + f"[SUBMISSION] Candidate `{tag}` has invalid/missing submission: {sub_reason}")

    # Comment translated to English.
    # Comment translated to English.
    last_dir = art_dir / "last"
    _ensure_dir(last_dir)
    try:
        orch.write_file(str(last_dir / "metrics.json"), json.dumps(norm_metrics, ensure_ascii=False, indent=2))
        if code_text:
            orch.write_file(str(last_dir / "code.py"), code_text)
        # Comment translated to English.
        if sub_ok and submission_path:
            src_sub = root / submission_path
            if src_sub.exists():
                _copy_text_file(orch, src_sub, last_dir / "submission.csv")
    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed to update 'last' artifacts: {e}")

    # Comment translated to English.
    version_entry = _record_metrics_version(
        orch,
        norm_metrics,
        code_text,
        tag=tag,
        submission_path=(submission_path if sub_ok else ""),
    )

    # Comment translated to English.
    # Comment translated to English.
    best_dir = art_dir / "best"
    best_metrics_path = best_dir / "metrics.json"
    current_best_metrics = {}
    try:
        if best_metrics_path.exists():
            current_best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    except Exception:
        # Comment translated to English.
        current_best_metrics = {}

    if (not current_best_metrics or _metric_improved(norm_metrics, current_best_metrics)) and sub_ok:
        print(Fore.GREEN + f"[BEST] New best metric found: {norm_metrics.get('primary')}. Updating artifacts/best/.")
        _ensure_dir(best_dir)
        try:
            orch.write_file(str(best_dir / "metrics.json"), json.dumps(norm_metrics, ensure_ascii=False, indent=2))
            if code_text:
                orch.write_file(str(best_dir / "code.py"), code_text)

            # Comment translated to English.
            best_submission_rel_path = version_entry.get("paths", {}).get("submission")
            if best_submission_rel_path:
                src_sub = root / best_submission_rel_path
                if src_sub.exists():
                    _copy_text_file(orch, src_sub, best_dir / src_sub.name)
        except Exception as e:
            print(Fore.RED + f"[ERROR] Failed to update 'best' artifacts: {e}")
    elif not sub_ok and (not current_best_metrics or _metric_improved(norm_metrics, current_best_metrics)):
        print(
            Fore.YELLOW
            + f"[BEST] Skipped best update for `{tag}`: no valid submission bound to this metrics/code ({sub_reason})."
        )


def _finalize_single_submission(orch: GlobalOrchestrator) -> Optional[str]:
    """
    Pick the best entry from versions/index.json and pin final artifacts.
    Returns canonical submission path relative to project root or None.
    """
    root = Path(orch.project_root)
    ver = root / orch.cfg.paths.artifacts_dir / "versions"
    idx_path = ver / "index.json"
    idx: List[Dict[str, Any]] = []
    if idx_path.exists():
        try:
            raw_idx = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(raw_idx, list):
                idx = raw_idx
        except Exception:
            idx = []

    # Comment translated to English.
    def _score(e):
        p = e.get("primary", None)
        mx = bool(e.get("maximize", True))
        try:
            v = float(p)
        except Exception:
            return float("-inf")
        return v if mx else -v

    sub_rel = ""
    code_rel = ""
    met_rel = ""
    if idx:
        best = max(idx, key=_score)
        paths = best.get("paths", {}) or {}
        sub_rel = paths.get("submission", "")
        code_rel = paths.get("code", "")
        met_rel = paths.get("metrics", "")

    final_dir = root / orch.cfg.paths.artifacts_dir / "final"
    _ensure_dir(final_dir)

    cfg_canonical = canonical_submission_path(orch)
    legacy_root_csv = root / "submission.csv"

    canonical_name = str(getattr(orch.cfg.paths, "submission_filename", "submission.csv") or "submission.csv")

    def _copy_final_to_canonical(src_name: str) -> None:
        src = final_dir / src_name
        if not src.exists():
            return
        _ensure_dir(cfg_canonical.parent)
        _copy_text_file(orch, src, cfg_canonical)
        # Legacy: some runners expect project_root/submission.csv
        try:
            if cfg_canonical.resolve() != legacy_root_csv.resolve() and canonical_name.lower().endswith(".csv"):
                _copy_text_file(orch, src, legacy_root_csv)
        except Exception:
            if canonical_name.lower().endswith(".csv"):
                _copy_text_file(orch, src, legacy_root_csv)

    # Comment translated to English.
    try:
        if code_rel:
            _copy_text_file(orch, root / code_rel, final_dir / "best_code.py")
        if met_rel:
            _copy_text_file(orch, root / met_rel, final_dir / "metrics.json")
        if sub_rel and (root / sub_rel).exists():
            src = root / sub_rel
            _copy_text_file(orch, src, final_dir / src.name)
            _copy_final_to_canonical(src.name)
            if cfg_canonical.exists():
                return str(cfg_canonical.relative_to(root))
    except Exception:
        pass

    # If index entry has no submission path, still try to materialize canonical from known safe locations.
    try:
        final_sub = final_dir / "submission.csv"
        if final_sub.exists() and final_sub.stat().st_size > 0:
            _copy_final_to_canonical("submission.csv")
            if cfg_canonical.exists() and cfg_canonical.stat().st_size > 0:
                return str(cfg_canonical.relative_to(root))
    except Exception:
        pass

    # Comment translated to English.
    # Prefer only submissions that match the configured spec header (if available).
    # Fallback to empty spec keeps legacy behavior.
    try:
        sp = root / orch.cfg.paths.artifacts_dir / "spec.json"
        spec_obj = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    except Exception:
        spec_obj = {}
    found = _find_valid_submissions(orch, spec_obj if isinstance(spec_obj, dict) else {})
    if found:
        src = found[0][0]
        _copy_text_file(orch, src, final_dir / src.name)
        _copy_final_to_canonical(src.name)
        if cfg_canonical.exists():
            return str(cfg_canonical.relative_to(root))

    return None


def _finalize_single_submission_by_all_metrics_llm(
    orch: GlobalOrchestrator,
    llm_fast,
    spec: Dict[str, Any],
    *,
    task: str = "",
    max_candidates: int = 12,
    code_llm: Any = None,
    mcp_tools: Optional[List[Any]] = None,
) -> Optional[str]:
    """
    Choose final submission using ALL available metrics history (stability-biased),
    via an LLM selector agent. Falls back to primary-only selection on any failure.
    """
    # Best-effort: load spec.json if missing/empty, so we have maximize/name semantics.
    try:
        root = Path(orch.project_root)
        if not spec or not isinstance(spec, dict) or "primary_metric" not in spec:
            spec_path = root / orch.cfg.paths.artifacts_dir / "spec.json"
            if spec_path.exists():
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    try:
        from src.prompts_agents import final_metric_selector_agent
    except Exception:
        # If for some reason agent can't be imported, keep legacy behavior.
        return _finalize_single_submission(orch)

    root = Path(orch.project_root)
    ver = root / orch.cfg.paths.artifacts_dir / "versions"
    idx_path = ver / "index.json"

    if not idx_path.exists():
        # No versions recorded at all — pipeline never produced a submission.
        # Report honestly rather than silently trying to salvage nothing.
        has_any_sub = any(
            (root / orch.cfg.paths.artifacts_dir / d / "submission.csv").exists()
            for d in ("best", "last", "final")
        )
        if not has_any_sub:
            print(Fore.YELLOW + "[FINALIZE] No versions index and no submission artifacts — pipeline did not reach submission stage.")
            orch.log("finalize_no_submission_produced", {"reason": "no_versions_index_no_submission_artifacts"})
            return None
        return _finalize_single_submission(orch)

    idx: List[Dict[str, Any]] = []
    try:
        raw_idx = json.loads(idx_path.read_text(encoding="utf-8"))
        if isinstance(raw_idx, list):
            idx = raw_idx
    except Exception:
        idx = []

    # Fallback: if index.json is empty/corrupt, reconstruct from ledger.csv + version dirs
    if not idx:
        ledger_path = ver / "ledger.csv"
        if ledger_path.exists():
            print(Fore.YELLOW + "[FINALIZE] index.json empty but ledger.csv exists — reconstructing index")
            try:
                with ledger_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = str(row.get("ts", "")).strip()
                        tag = str(row.get("tag", "")).strip()
                        m_path = str(row.get("metrics_path", "")).strip()
                        c_path = str(row.get("code_path", "")).strip()
                        s_path = str(row.get("submission_path", "")).strip()
                        if not m_path:
                            continue
                        try:
                            primary = float(row.get("primary", ""))
                        except Exception:
                            continue
                        maximize = str(row.get("maximize", "True")).strip().lower() in ("true", "1", "yes")
                        idx.append({
                            "ts": ts,
                            "tag": tag,
                            "paths": {"metrics": m_path, "code": c_path, "submission": s_path},
                            "primary": primary,
                            "maximize": maximize,
                        })
            except Exception as e:
                print(Fore.RED + f"[FINALIZE] ledger reconstruction failed: {e}")

    if not idx:
        print(Fore.RED + "[FINALIZE] No version records found (index.json + ledger.csv both empty).")
        return _finalize_single_submission(orch)

    # Build candidates from versions history, preferring calculated metrics that also have a submission.
    candidates: List[Dict[str, Any]] = []
    expected_cols: List[str] = []
    try:
        expected_cols = [str(c).strip() for c in (spec.get("submission", {}).get("columns") or [])]
    except Exception:
        expected_cols = []
    for e in idx:
        paths = e.get("paths", {}) or {}
        metrics_rel = paths.get("metrics", "") or ""
        submission_rel = paths.get("submission", "") or ""
        code_rel = paths.get("code", "") or ""
        if not metrics_rel or (not submission_rel and not code_rel):
            continue

        metrics_abs = root / metrics_rel
        if not metrics_abs.exists():
            continue

        try:
            m = json.loads(metrics_abs.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(m, dict):
            continue
        if m.get("type") != "calculated":
            continue
        if m.get("primary") is None:
            continue

        # Hard guardrail: never consider submissions that don't match expected header.
        # Skip header check if there's no submission file yet (will be regenerated via ReAct).
        if expected_cols and submission_rel:
            sub_abs = root / submission_rel
            if not (sub_abs.exists() and _submission_header_matches(sub_abs, expected_cols)):
                continue

        # Truncate/trim extras for prompt size.
        extras = m.get("extras")
        if isinstance(extras, dict):
            extras_items = list(extras.items())[:24]
            trimmed_extras: Dict[str, Any] = {}
            for k, v in extras_items:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    sv = v
                else:
                    sv = str(v)
                if isinstance(sv, str) and len(sv) > 400:
                    sv = sv[:400] + "..."
                trimmed_extras[str(k)] = sv
            m = {**m, "extras": trimmed_extras}

        candidates.append(
            {
                "tag": e.get("tag", ""),
                "ts": e.get("ts", ""),
                "metrics": m,
                "submission_rel": submission_rel,
                "code_rel": code_rel,
                "metrics_rel": metrics_rel,
            }
        )

    # --- Guardrail: filter out candidates whose submission values look like logits ---
    # This is best-effort and intentionally lightweight (prefix probing only).
    filtered_out_semantics = 0
    semantics_mode = "unknown"
    try:
        sample_abs = _resolve_sample_submission_path(orch, spec or {})
        if sample_abs is None:
            sample_abs = root / "data" / "sample_submission.csv"

        eps = 1e-6
        max_probe = 2000
        sample_vals: List[float] = []
        if sample_abs.exists() and sample_abs.is_file():
            with sample_abs.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                sample_header = [str(h).strip() for h in (next(reader, None) or [])]
                pred_idx = 1 if len(sample_header) > 1 else 0
                for row_idx, row in enumerate(reader):
                    if row_idx >= max_probe:
                        break
                    if not row or len(row) <= pred_idx:
                        continue
                    try:
                        sample_vals.append(float(str(row[pred_idx]).strip()))
                    except Exception:
                        continue

        if sample_vals:
            all_in_01 = all((math.isfinite(x) and -eps <= x <= 1.0 + eps) for x in sample_vals)
            all_int_like = all((math.isfinite(x) and abs(x - round(x)) <= eps) for x in sample_vals)
            unique_ints = sorted({int(round(x)) for x in sample_vals if math.isfinite(x)})
            if all_in_01:
                semantics_mode = "probabilities"
            elif all_int_like and len(unique_ints) <= 50:
                semantics_mode = "class_labels"
                allowed_classes = set(unique_ints)
            else:
                semantics_mode = "numeric_unknown"
                allowed_classes = set(unique_ints)
        else:
            print(Fore.YELLOW + f"[FINALIZE] sample_submission has no probeable numeric prediction values: {sample_abs}")

        if semantics_mode in ("probabilities", "class_labels") and candidates:
            def _candidate_semantics_ok(sub_abs: Path) -> bool:
                probe_n = 5000
                with sub_abs.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.reader(f)
                    sub_header = [str(h).strip() for h in (next(reader, None) or [])]
                    pred_idx = 1 if len(sub_header) > 1 else 0
                    for row_idx, row in enumerate(reader):
                        if row_idx >= probe_n:
                            break
                        if not row or len(row) <= pred_idx:
                            return False
                        try:
                            pv = float(str(row[pred_idx]).strip())
                        except Exception:
                            return False
                        if not math.isfinite(pv):
                            return False
                        if semantics_mode == "probabilities":
                            if pv < -eps or pv > 1.0 + eps:
                                return False
                        else:
                            iv = int(round(pv))
                            if abs(pv - iv) > eps or iv not in allowed_classes:
                                return False
                return True

            filtered: List[Dict[str, Any]] = []
            code_only: List[Dict[str, Any]] = []  # candidates with code but no valid submission
            for c in candidates:
                sub_rel = c.get("submission_rel") or ""
                if not sub_rel:
                    # No submission but has code → can be repaired, keep separately
                    if c.get("code_rel"):
                        code_only.append(c)
                    else:
                        filtered_out_semantics += 1
                    continue
                sub_abs = root / sub_rel
                if sub_abs.exists() and _candidate_semantics_ok(sub_abs):
                    filtered.append(c)
                else:
                    filtered_out_semantics += 1
                    # Keep as code_only if it has code so repair can regenerate submission
                    if c.get("code_rel"):
                        code_only.append(c)
            if filtered:
                candidates = filtered
            elif code_only:
                # Semantic filter eliminated all submissions, but we have code — let repair try
                print(Fore.YELLOW + f"[FINALIZE] semantics filter removed all {filtered_out_semantics} submissions; "
                      f"falling back to {len(code_only)} code-only candidates for repair")
                candidates = code_only
    except Exception:
        # best-effort only; do not fail selection
        pass

    # --- Leakage detection: WARNING only — do NOT discard candidates ---
    # ROC_AUC=1.0 + kappa=1.0 on validation suggests data leakage in the metric computation,
    # but the submission file may still contain valid diverse predictions.
    # The submission-quality checks (diversity + sample-identity) are the real gatekeeper.
    # We flag leaky candidates here so the selector agent can factor it in, but we keep them.
    try:
        _LEAKAGE_THRESHOLD = 0.9999
        for c in candidates:
            extras = (c.get("metrics") or {}).get("extras") or {}
            leakage_signals = []
            for key in ("roc_auc", "cohen_kappa", "f1_weighted", "f1_macro"):
                val = extras.get(key)
                if val is not None:
                    try:
                        if float(val) >= _LEAKAGE_THRESHOLD:
                            leakage_signals.append(key)
                    except Exception:
                        pass
            if len(leakage_signals) >= 2:
                c["_leakage_warning"] = leakage_signals
                print(Fore.YELLOW + f"[FINALIZE] Leakage WARNING (validation metrics only) for '{c.get('tag')}': "
                      f"{leakage_signals} all near 1.0. Submission file quality will decide.")
    except Exception:
        pass

    # Keep a reasonable set: try higher primary first, then fall back to order.
    def _cand_primary(c: Dict[str, Any]) -> float:
        try:
            return float(((c.get("metrics") or {}).get("primary")))
        except Exception:
            return float("-inf")

    if candidates:
        # Respect maximize from each candidate; default maximize=True.
        def _cand_score_for_sort(c: Dict[str, Any]) -> float:
            p = _cand_primary(c)
            mx = bool(((c.get("metrics") or {}).get("maximize", True)))
            return p if mx else -p

        candidates.sort(key=_cand_score_for_sort, reverse=True)
        candidates = candidates[:max_candidates]

    if not candidates:
        print(Fore.RED + "[FINALIZE] No versioned candidates found. Pipeline did not produce model metrics.")
        return _finalize_single_submission(orch)

    choice = final_metric_selector_agent(
        llm_fast,
        task=task,
        spec=spec or {},
        candidates=candidates,
    )
    chosen_idx = choice.get("chosen_candidate_idx")
    if not isinstance(chosen_idx, int) or chosen_idx < 0 or chosen_idx >= len(candidates):
        # LLM selector returned nonsense — fall back to best primary
        chosen_idx = 0

    final_dir = root / orch.cfg.paths.artifacts_dir / "final"
    _ensure_dir(final_dir)
    canonical = canonical_submission_path(orch)
    legacy_root_csv = root / "submission.csv"

    # Try selector choice first, then fallback to the rest of top candidates.
    ordered_indices = [chosen_idx] + [i for i in range(len(candidates)) if i != chosen_idx]
    finalize_attempts: List[Dict[str, Any]] = []
    selected_idx = None
    # Accumulate error history across ALL candidates so LLM agents don't repeat failures
    cross_candidate_error_history: List[str] = []

    try:
        from src.prompts_agents import perform_task_python_v2, finetune_code_v2
    except Exception:
        perform_task_python_v2 = None  # type: ignore[assignment]
        finetune_code_v2 = None  # type: ignore[assignment]

    _cfg_orc = getattr(orch.cfg, "orchestration", object())
    _max_gen_attempts = int(getattr(_cfg_orc, "final_submission_max_attempts", 20) or 20)

    for cand_idx in ordered_indices:
        chosen = candidates[cand_idx]
        code_rel = chosen.get("code_rel", "") or ""
        metrics_rel = chosen.get("metrics_rel", "") or ""
        submission_rel = chosen.get("submission_rel", "") or ""
        tag = str(chosen.get("tag", "") or "")
        cand_metrics = chosen.get("metrics") or {}

        print(Fore.CYAN + f"\n[FINALIZE] === Candidate {cand_idx}: '{tag}' | "
              f"primary={cand_metrics.get('primary')} ===")

        # Copy code + metrics into artifacts/final/
        try:
            if code_rel:
                _copy_text_file(orch, root / code_rel, final_dir / "best_code.py")
            if metrics_rel:
                _copy_text_file(orch, root / metrics_rel, final_dir / "metrics.json")
            if submission_rel and (root / submission_rel).exists():
                src = root / submission_rel
                _copy_text_file(orch, src, final_dir / src.name)
                _copy_text_file(orch, src, canonical)
                if not legacy_root_csv.exists() or canonical.resolve() != legacy_root_csv.resolve():
                    _copy_text_file(orch, src, legacy_root_csv)
        except Exception as e:
            finalize_attempts.append({"idx": cand_idx, "tag": tag, "ok": False, "reason": f"copy_failed: {e}"})
            cross_candidate_error_history.append(f"Candidate '{tag}': copy failed: {e}")
            continue

        # ─── Validate submission as-is ───────────────────────────────────────
        val0 = validate_final_submission(orch, spec or {})
        if val0.get("ok"):
            print(Fore.GREEN + f"[FINALIZE] Candidate '{tag}' submission valid as-is!")
            finalize_attempts.append({"idx": cand_idx, "tag": tag, "ok": True, "pre_errors": []})
            selected_idx = cand_idx
            break

        # ─── Submission missing or invalid: agent-driven generation loop ─────
        # Read the model code — we'll use it as context for generating submission code
        model_code = ""
        if code_rel and (root / code_rel).exists():
            try:
                model_code = (root / code_rel).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

        if not model_code:
            # No model code → nothing to generate from
            finalize_attempts.append({
                "idx": cand_idx, "tag": tag, "ok": False,
                "reason": "no_model_code", "pre_errors": val0.get("errors", []),
            })
            cross_candidate_error_history.append(f"Candidate '{tag}': no model code available")
            continue

        if perform_task_python_v2 is None and finetune_code_v2 is None:
            finalize_attempts.append({
                "idx": cand_idx, "tag": tag, "ok": False,
                "reason": "no_code_gen_agents", "pre_errors": val0.get("errors", []),
            })
            continue

        val_errors = val0.get("errors") or []
        submission_is_missing = any("missing" in str(e).lower() for e in val_errors)
        attempt_errors: List[str] = list(cross_candidate_error_history)  # inherit from prior candidates
        gen_ok = False

        print(Fore.YELLOW + f"[FINALIZE] Submission {'missing' if submission_is_missing else 'invalid'}. "
              f"Errors: {val_errors}")
        print(Fore.CYAN + f"[FINALIZE] Starting agent-driven submission generation (max {_max_gen_attempts} attempts)...")

        # Build spec/submission contract info for agents
        sub_contract = ""
        try:
            sub_cols = (spec.get("submission") or {}).get("columns") or []
            sub_delim = (spec.get("submission") or {}).get("delimiter", ",")
            pm = spec.get("primary_metric") or {}
            sample_path = _resolve_sample_submission_path(orch, spec or {})
            sample_head = ""
            if sample_path and sample_path.exists():
                with sample_path.open("r", encoding="utf-8", errors="ignore") as f:
                    sample_head = "".join(f.readline() for _ in range(5))
            sub_contract = (
                f"SUBMISSION CONTRACT:\n"
                f"  columns: {sub_cols}\n"
                f"  delimiter: {repr(sub_delim)}\n"
                f"  primary_metric: {pm.get('name')} (maximize={pm.get('maximize')})\n"
                f"SAMPLE SUBMISSION (first rows):\n{sample_head}\n"
            )
        except Exception:
            pass

        for gen_attempt in range(1, _max_gen_attempts + 1):
            print(Fore.CYAN + f"[FINALIZE] Attempt {gen_attempt}/{_max_gen_attempts} for candidate '{tag}'")

            history_block = ""
            if attempt_errors:
                history_block = (
                    "\n=== PREVIOUS ATTEMPT ERRORS (do NOT repeat these) ===\n"
                    + "\n".join(f"  - {e}" for e in attempt_errors[-15:])
                    + "\n=== END PREVIOUS ERRORS ===\n"
                )

            if submission_is_missing or gen_attempt == 1:
                # ── Generate fresh submission code from scratch ──────────────
                gen_task = (
                    "Generate a Python script that produces the final submission.csv file. "
                    "The model training code is provided below — extract the trained model / predictions "
                    "and format them into the required submission format. "
                    "If the model is not saved as a checkpoint, re-run training (use cached data if available). "
                    "Read submission path and columns from artifacts/spec.json. "
                    "Print METRICS_JSON if validation metrics are computable."
                )
                gen_context = (
                    f"{sub_contract}\n"
                    f"MODEL CODE (this code produced the training/metrics — reuse it):\n"
                    f"```python\n{model_code[-8000:]}\n```\n"
                    f"CURRENT VALIDATION ERRORS: {val_errors}\n"
                    f"{history_block}"
                    f"IMPORTANT:\n"
                    f"- Load spec dynamically from artifacts/spec.json; never hardcode paths or columns\n"
                    f"- Read submission path, columns, delimiter from spec.submission\n"
                    f"- Also save a copy as submission/submission.csv (standard bench location)\n"
                    f"- Predictions must be diverse (not constant, not identical to sample_submission)\n"
                )
                try:
                    gen_code = perform_task_python_v2(
                        code_llm or llm_fast,
                        gen_task,
                        spec or {},
                        previous_code=model_code[-6000:],
                        context=gen_context,
                        tools=mcp_tools,
                        orch=orch,
                    )
                except Exception as e:
                    attempt_errors.append(f"Attempt {gen_attempt}: code generation failed: {e}")
                    continue
            else:
                # ── Patch existing submission code (finetune) ────────────────
                err_txt = (
                    f"Attempt {gen_attempt}: fix the submission generation script.\n"
                    f"Current errors: {val_errors}\n"
                    f"{sub_contract}\n"
                    f"{history_block}"
                    f"REQUIREMENTS:\n"
                    f"- Load spec from artifacts/spec.json; read paths and columns from spec.submission\n"
                    f"- Predictions must be diverse (not constant/all same value)\n"
                    f"- Must NOT be identical to sample_submission\n"
                    f"- Also save a copy as submission/submission.csv (standard bench location)\n"
                )
                prev_code = model_code if gen_attempt <= 2 else gen_code
                try:
                    gen_code = finetune_code_v2(
                        code_llm or llm_fast,
                        task="Fix submission generation",
                        code=prev_code,
                        spec=spec or {},
                        error=err_txt,
                        tools=mcp_tools,
                    )
                except Exception as e:
                    attempt_errors.append(f"Attempt {gen_attempt}: finetune failed: {e}")
                    continue

            gen_code = str(gen_code or "").replace("```python", "").replace("```", "").strip()
            if not gen_code:
                attempt_errors.append(f"Attempt {gen_attempt}: empty code generated")
                continue

            # Save and run the generated submission code
            gen_script_rel = str(
                (final_dir / f"submission_gen_attempt_{gen_attempt}.py").relative_to(root)
            )
            try:
                orch.write_file(gen_script_rel, gen_code)
            except Exception as e:
                attempt_errors.append(f"Attempt {gen_attempt}: cannot write script: {e}")
                continue

            print(Fore.CYAN + f"[FINALIZE] Running {gen_script_rel}...")
            try:
                run_res = orch.run_python_file(gen_script_rel, stream=True, spec=spec or {})
                exit_code = run_res.get("exit_code", 1)
                stdout = (run_res.get("output") or "")[:3000]
                stderr = (run_res.get("errors") or "")[:1500]
            except Exception as e:
                attempt_errors.append(f"Attempt {gen_attempt}: execution exception: {e}")
                continue

            if exit_code != 0:
                attempt_errors.append(
                    f"Attempt {gen_attempt}: script failed (exit={exit_code}): "
                    f"stderr={stderr[:500]}"
                )
                submission_is_missing = False  # file may have been partially created
                continue

            # Materialize submission to all expected locations before validating
            try:
                materialize_project_root_submission_csv(orch)
            except Exception:
                pass

            # Validate the new submission
            val_new = validate_final_submission(orch, spec or {})
            val_errors = val_new.get("errors") or []
            if val_new.get("ok"):
                print(Fore.GREEN + f"[FINALIZE] Submission generated successfully on attempt {gen_attempt}!")
                gen_ok = True
                break
            else:
                attempt_errors.append(
                    f"Attempt {gen_attempt}: submission generated but validation failed: {val_errors}"
                )
                submission_is_missing = False
                print(Fore.YELLOW + f"[FINALIZE] Attempt {gen_attempt} validation failed: {val_errors}")

        finalize_attempts.append({
            "idx": cand_idx, "tag": tag, "ok": gen_ok,
            "pre_errors": val0.get("errors", []),
            "gen_attempts": len(attempt_errors),
            "last_errors": val_errors if not gen_ok else [],
        })
        cross_candidate_error_history.extend(attempt_errors)

        if gen_ok:
            selected_idx = cand_idx
            break

    if selected_idx is None:
        print(Fore.RED + "[FINALIZE] All candidates exhausted. No valid submission produced.")
        print(Fore.RED + f"[FINALIZE] Error history ({len(cross_candidate_error_history)} entries):")
        for eh in cross_candidate_error_history[-10:]:
            print(Fore.RED + f"  {eh}")
        return None

    chosen = candidates[selected_idx]

    # Persist selector trace.
    try:
        report = {
            "ok": True,
            "chosen_idx": selected_idx,
            "chosen_tag": (chosen.get("tag") or choice.get("chosen_tag", "") or ""),
            "chosen_source": ("improve" if "improve" in str(chosen.get("tag", "")).lower() else "main_or_other"),
            "chosen_primary": (chosen.get("metrics") or {}).get("primary"),
            "chosen_metrics": chosen.get("metrics") or {},
            "reasoning": choice.get("reasoning", ""),
            "finalize_attempts": finalize_attempts,
            "semantics_guard": {
                "enabled": True,
                "sample_mode": semantics_mode,
                "filtered_out_invalid_semantics": filtered_out_semantics,
            },
        }
        orch.write_file(
            str(final_dir / "llm_metric_selection_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        # Also print reasoning for the "final logs" requirement.
        print("\n[FINAL][LLM_METRIC_AUDIT] Selected candidate:")
        print(f"- chosen_tag: {report.get('chosen_tag')}")
        print(f"- chosen_primary: {report.get('chosen_primary')}")
        print(f"- chosen_source: {report.get('chosen_source')}")
        print(f"- reasoning:\n{report.get('reasoning')}")
    except Exception:
        pass

    return str(canonical.relative_to(root)) if canonical.exists() else None


def validate_final_submission(orch: GlobalOrchestrator, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strictly validate final canonical submission location and basic format.
    Primary path: paths.submission_dir / paths.submission_filename; fallback: project_root/submission.csv.
    Always writes artifacts/final/submission_validation.json.
    """
    root = Path(orch.project_root)
    primary = canonical_submission_path(orch)
    legacy = root / "submission.csv"
    canonical = primary if primary.exists() else legacy
    final_dir = root / orch.cfg.paths.artifacts_dir / "final"
    _ensure_dir(final_dir)

    errors: List[str] = []
    warnings: List[str] = []
    if not primary.exists() and not legacy.exists():
        errors.append(
            f"Canonical submission missing: expected {primary} or legacy {legacy}"
        )
    elif not canonical.exists():
        errors.append("Canonical submission path does not exist")
    else:
        try:
            if canonical.stat().st_size == 0:
                errors.append("Canonical submission file is empty")
        except Exception:
            errors.append("Unable to read canonical submission metadata")

    expected_cols: List[str] = []
    try:
        expected_cols = [str(c).strip() for c in (spec.get("submission", {}).get("columns") or [])]
    except Exception:
        expected_cols = []

    if canonical.exists():
        submission_header: List[str] = []
        try:
            with canonical.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None) or []
            submission_header = [str(h).strip() for h in header]
            if not submission_header:
                errors.append("Canonical submission.csv has no header")
            if expected_cols and submission_header != expected_cols:
                # Attempt an automatic repair if the file contains expected columns (but in wrong order / with extras).
                tmp = canonical.parent / (canonical.stem + "._fixed.csv")
                repaired = _rewrite_csv_keep_columns(canonical, tmp, expected_cols)
                if repaired:
                    try:
                        _copy_text_file(orch, tmp, canonical)
                        # Legacy mirror
                        legacy = root / "submission.csv"
                        if canonical.exists() and legacy.resolve() != canonical.resolve():
                            _copy_text_file(orch, canonical, legacy)
                        submission_header = expected_cols
                    except Exception:
                        pass
                if submission_header != expected_cols:
                    errors.append(f"Header mismatch: expected {expected_cols}, got {submission_header}")
        except Exception as e:
            errors.append(f"Failed to parse canonical submission.csv: {e}")

    # --- Diversity check: reject constant predictions in target columns ---
    if canonical.exists():
        try:
            with canonical.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                header = [str(h).strip() for h in (next(reader, None) or [])]
                if header:
                    id_idx = 0 if header and header[0].lower() == "id" else -1
                    target_idxs = [i for i in range(len(header)) if i != id_idx]
                    col_min: Dict[int, float] = {}
                    col_max: Dict[int, float] = {}
                    col_unique: Dict[int, set] = {i: set() for i in target_idxs}
                    max_probe = 20000
                    for ridx, row in enumerate(reader):
                        if ridx >= max_probe:
                            break
                        for i in target_idxs:
                            if i >= len(row):
                                continue
                            try:
                                v = float(str(row[i]).strip())
                            except Exception:
                                continue
                            if not math.isfinite(v):
                                continue
                            col_min[i] = v if i not in col_min else min(col_min[i], v)
                            col_max[i] = v if i not in col_max else max(col_max[i], v)
                            if len(col_unique[i]) < 64:
                                col_unique[i].add(round(v, 12))
                    eps_div = 1e-12
                    for i in target_idxs:
                        cmin = col_min.get(i)
                        cmax = col_max.get(i)
                        if cmin is None or cmax is None:
                            continue
                        uniq = len(col_unique.get(i) or set())
                        if abs(cmax - cmin) <= eps_div or uniq <= 1:
                            cname = header[i] if i < len(header) else f"col_{i}"
                            errors.append(f"submission_diversity_{cname}: constant predictions detected")
        except Exception:
            # Best-effort; validation still continues on other checks.
            pass

    # --- Semantic submission checks vs sample_submission (anti-logits / correct class type) ---
    # These checks are best-effort: if sample_submission is missing or unreadable, they are skipped.
    sample_checked = False
    inferred_pred_mode: str = "unknown"
    if canonical.exists():
        try:
            sample_path = _resolve_sample_submission_path(orch, spec or {})
            if sample_path is None:
                sample_path = root / "data" / "sample_submission.csv"

            if sample_path.exists() and sample_path.is_file():
                sample_checked = True

                # Infer prediction mode from sample_submission values (only on a prefix for speed).
                max_probe = 2000
                sample_pred_vals: List[float] = []
                with sample_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.reader(f)
                    sample_header = [str(h).strip() for h in (next(reader, None) or [])]
                    pred_idx = 1 if len(sample_header) > 1 else 0
                    for row_idx, row in enumerate(reader):
                        if row_idx >= max_probe:
                            break
                        if not row or len(row) <= pred_idx:
                            continue
                        try:
                            sample_pred_vals.append(float(str(row[pred_idx]).strip()))
                        except Exception:
                            continue

                if not sample_pred_vals:
                    warnings.append(
                        "sample_submission probing yielded no numeric prediction values; semantic check skipped"
                    )

                if sample_pred_vals:
                    eps = 1e-6
                    all_in_01 = all((x is not None and math.isfinite(x) and -eps <= x <= 1.0 + eps) for x in sample_pred_vals)
                    all_int_like = all((abs(x - round(x)) <= eps) for x in sample_pred_vals)
                    unique_ints = sorted({int(round(x)) for x in sample_pred_vals if math.isfinite(x)})
                    if all_in_01:
                        inferred_pred_mode = "probabilities"
                    elif all_int_like and len(unique_ints) <= 50:
                        inferred_pred_mode = "class_labels"
                    else:
                        inferred_pred_mode = "numeric_unknown"

                    # Probe submission predictions (prefix) for invalid ranges/types.
                    max_probe_sub = 5000
                    invalid_found = False
                    pred_min = None
                    pred_max = None
                    seen_unique = set()

                    with canonical.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                        reader = csv.reader(f)
                        sub_header = [str(h).strip() for h in (next(reader, None) or [])]
                        sub_pred_idx = 1 if len(sub_header) > 1 else 0
                        for row_idx, row in enumerate(reader):
                            if row_idx >= max_probe_sub:
                                break
                            if not row or len(row) <= sub_pred_idx:
                                continue
                            try:
                                pv = float(str(row[sub_pred_idx]).strip())
                            except Exception:
                                invalid_found = True
                                break
                            if not math.isfinite(pv):
                                invalid_found = True
                                break

                            pred_min = pv if pred_min is None else min(pred_min, pv)
                            pred_max = pv if pred_max is None else max(pred_max, pv)
                            if len(seen_unique) < 1000:
                                # Helps detect logits (many unique floats) early.
                                seen_unique.add(pv)

                            if inferred_pred_mode == "probabilities":
                                if pv < -eps or pv > 1.0 + eps:
                                    invalid_found = True
                                    break
                            elif inferred_pred_mode == "class_labels":
                                iv = int(round(pv))
                                if abs(pv - iv) > eps or iv not in set(unique_ints):
                                    invalid_found = True
                                    break

                    if invalid_found:
                        if inferred_pred_mode == "probabilities":
                            errors.append(
                                f"Anti-logits check failed: predicted probability values not confined to [0,1]. "
                                f"Observed range approx: min={pred_min}, max={pred_max} (sample-based inference)."
                            )
                        elif inferred_pred_mode == "class_labels":
                            errors.append(
                                "Anti-logits/classes check failed: submission values do not match sample_submission discrete class type."
                            )
                        else:
                            errors.append("Submission semantic type check vs sample_submission failed.")

        except Exception as e:
            # Best-effort only: do not hard-fail finalization on semantic-check plumbing errors.
            pass

    # --- Sample-submission identity check: reject if submission is essentially just a copy ---
    if canonical.exists():
        try:
            sample_path_id = _resolve_sample_submission_path(orch, spec or {})
            if sample_path_id is None:
                sample_path_id = root / "data" / "sample_submission.csv"
            if sample_path_id and sample_path_id.exists() and sample_path_id.is_file():
                max_probe_id = 10000
                sample_vals_id: List[float] = []
                sub_vals_id: List[float] = []
                with sample_path_id.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    rdr = csv.reader(f)
                    sh = [str(h).strip() for h in (next(rdr, None) or [])]
                    pred_col = 1 if len(sh) > 1 else 0
                    for ri, row in enumerate(rdr):
                        if ri >= max_probe_id:
                            break
                        if row and len(row) > pred_col:
                            try:
                                sample_vals_id.append(float(str(row[pred_col]).strip()))
                            except Exception:
                                pass
                with canonical.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                    rdr = csv.reader(f)
                    sh2 = [str(h).strip() for h in (next(rdr, None) or [])]
                    pred_col2 = 1 if len(sh2) > 1 else 0
                    for ri, row in enumerate(rdr):
                        if ri >= max_probe_id:
                            break
                        if row and len(row) > pred_col2:
                            try:
                                sub_vals_id.append(float(str(row[pred_col2]).strip()))
                            except Exception:
                                pass
                if sample_vals_id and sub_vals_id:
                    n = min(len(sample_vals_id), len(sub_vals_id))
                    if n >= 10:
                        matches = sum(1 for a, b in zip(sample_vals_id[:n], sub_vals_id[:n]) if abs(a - b) < 1e-9)
                        if matches / n >= 0.99:
                            errors.append(
                                "not_identical_to_sample: submission predictions are identical to sample_submission "
                                "(real model predictions required)"
                            )
        except Exception:
            pass

    # Normalize errors: guard against empty/whitespace-only entries which would
    # incorrectly flip ok=False while producing a blank exception message.
    errors = [str(e).strip() for e in errors if str(e).strip()]

    report = {
        "ok": len(errors) == 0,
        "canonical_path": str(canonical),
        "file_name": canonical.name,
        "expected_columns": expected_cols,
        "sample_submission_checked": sample_checked,
        "inferred_pred_mode": inferred_pred_mode,
        "warnings": warnings,
        "errors": errors,
    }
    try:
        orch.write_file(
            str(final_dir / "submission_validation.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    except Exception:
        pass
    return report


@dataclass
class SubmissionCheckResult:
    """Result of a deterministic submission validation check."""
    ok: bool
    errors: List[str] = dataclass_field(default_factory=list)
    warnings: List[str] = dataclass_field(default_factory=list)
    header: List[str] = dataclass_field(default_factory=list)
    row_count: int = 0
    inferred_pred_mode: str = "unknown"


def quick_submission_check(
    submission_path: Path,
    *,
    expected_columns: Optional[List[str]] = None,
    sample_submission_path: Optional[Path] = None,
    max_probe_rows: int = 20000,
) -> SubmissionCheckResult:
    """
    Fast, deterministic submission validator.  No LLM, no orchestrator, no file writes.

    Checks (all independent — every check runs regardless of prior errors):
      1. File existence and non-emptiness
      2. Header / column match (if expected_columns provided)
      3. Row count vs sample_submission (if sample path provided)
      4. Prediction diversity (constant predictions)
      5. Non-finite values (NaN / Inf)
      6. Semantic range check vs sample_submission (anti-logits / class labels)

    Returns SubmissionCheckResult with .ok, .errors, .warnings.
    """
    submission_path = Path(submission_path)
    errors: List[str] = []
    warnings: List[str] = []
    header: List[str] = []
    row_count: int = 0
    inferred_pred_mode: str = "unknown"

    expected_cols = [str(c).strip() for c in (expected_columns or [])]

    if not submission_path.exists():
        errors.append(f"Submission file does not exist: {submission_path}")
        return SubmissionCheckResult(
            ok=False, errors=errors, warnings=warnings,
            header=header, row_count=row_count, inferred_pred_mode=inferred_pred_mode,
        )

    try:
        fsize = submission_path.stat().st_size
    except Exception:
        fsize = 0

    if fsize == 0:
        errors.append(f"Submission file is empty: {submission_path}")
        return SubmissionCheckResult(
            ok=False, errors=errors, warnings=warnings,
            header=header, row_count=row_count, inferred_pred_mode=inferred_pred_mode,
        )

    try:
        with submission_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            raw_header = next(reader, None) or []
        header = [str(h).strip() for h in raw_header]
    except Exception as e:
        errors.append(f"Failed to parse submission CSV: {e}")
        return SubmissionCheckResult(
            ok=False, errors=errors, warnings=warnings,
            header=header, row_count=row_count, inferred_pred_mode=inferred_pred_mode,
        )

    if not header:
        errors.append("Submission CSV has no header row")
        return SubmissionCheckResult(
            ok=False, errors=errors, warnings=warnings,
            header=header, row_count=row_count, inferred_pred_mode=inferred_pred_mode,
        )

    if expected_cols and header != expected_cols:
        errors.append(
            f"Column mismatch: expected {expected_cols}, got {header}"
        )

    id_col_names = {"id", "index", "idx"}
    id_idx = 0 if (header and header[0].lower() in id_col_names) else -1
    target_idxs = [i for i in range(len(header)) if i != id_idx]
    pred_idx = target_idxs[0] if target_idxs else (1 if len(header) > 1 else 0)

    col_min: Dict[int, float] = {}
    col_max: Dict[int, float] = {}
    col_unique: Dict[int, set] = {i: set() for i in target_idxs}
    has_non_finite: bool = False
    non_finite_example: str = ""
    all_pred_values: List[float] = []

    try:
        with submission_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None) 
            for ridx, row in enumerate(reader):
                row_count += 1
                if ridx >= max_probe_rows:
                    continue 

                for i in target_idxs:
                    if i >= len(row):
                        continue
                    raw_val = str(row[i]).strip()
                    try:
                        v = float(raw_val)
                    except (ValueError, TypeError):
                        continue

                    if not math.isfinite(v):
                        if not has_non_finite:
                            has_non_finite = True
                            col_name = header[i] if i < len(header) else f"col_{i}"
                            non_finite_example = (
                                f"row {ridx + 1}, col '{col_name}', value='{raw_val}'"
                            )
                        continue

                    col_min[i] = v if i not in col_min else min(col_min[i], v)
                    col_max[i] = v if i not in col_max else max(col_max[i], v)
                    if len(col_unique[i]) < 64:
                        col_unique[i].add(round(v, 12))

                    if i == pred_idx and len(all_pred_values) < max_probe_rows:
                        all_pred_values.append(v)
    except Exception as e:
        errors.append(f"Failed to read submission data rows: {e}")

    if has_non_finite:
        errors.append(
            f"Non-finite prediction values detected (NaN/Inf): {non_finite_example}"
        )

    # Diversity check 
    eps_div = 1e-12
    for i in target_idxs:
        cmin = col_min.get(i)
        cmax = col_max.get(i)
        if cmin is None or cmax is None:
            continue
        uniq = len(col_unique.get(i) or set())
        if abs(cmax - cmin) <= eps_div or uniq <= 1:
            cname = header[i] if i < len(header) else f"col_{i}"
            errors.append(
                f"Constant predictions in column '{cname}': "
                f"all values = {cmin} (unique count={uniq})"
            )

    # Row count vs sample 
    sample_path = Path(sample_submission_path) if sample_submission_path else None

    if sample_path is not None and sample_path.exists() and sample_path.is_file():
        try:
            sample_row_count = _csv_row_count(sample_path)
            if sample_row_count > 0 and row_count != sample_row_count:
                errors.append(
                    f"Row count mismatch: submission has {row_count} rows, "
                    f"sample_submission has {sample_row_count} rows"
                )
        except Exception:
            warnings.append("Could not read sample_submission for row count comparison")
    if (
        sample_path is not None
        and sample_path.exists()
        and sample_path.is_file()
        and all_pred_values
    ):
        try:
            eps = 1e-6
            sample_pred_vals: List[float] = []
            with sample_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                sample_header = [str(h).strip() for h in (next(reader, None) or [])]
                s_pred_idx = 1 if len(sample_header) > 1 else 0
                for row_idx, row in enumerate(reader):
                    if row_idx >= max_probe_rows:
                        break
                    if not row or len(row) <= s_pred_idx:
                        continue
                    try:
                        sample_pred_vals.append(float(str(row[s_pred_idx]).strip()))
                    except (ValueError, TypeError):
                        continue

            if sample_pred_vals:
                all_in_01 = all(
                    math.isfinite(x) and -eps <= x <= 1.0 + eps
                    for x in sample_pred_vals
                )
                all_int_like = all(
                    math.isfinite(x) and abs(x - round(x)) <= eps
                    for x in sample_pred_vals
                )
                unique_ints = sorted({
                    int(round(x)) for x in sample_pred_vals if math.isfinite(x)
                })

                if all_in_01:
                    inferred_pred_mode = "probabilities"
                elif all_int_like and len(unique_ints) <= 50:
                    inferred_pred_mode = "class_labels"
                else:
                    inferred_pred_mode = "numeric_unknown"

                # Check submission values against inferred mode
                if inferred_pred_mode == "probabilities":
                    bad_vals = [
                        v for v in all_pred_values
                        if v < -eps or v > 1.0 + eps
                    ]
                    if bad_vals:
                        pred_min = min(all_pred_values) if all_pred_values else None
                        pred_max = max(all_pred_values) if all_pred_values else None
                        errors.append(
                            f"Anti-logits check failed: submission values not in [0,1] range. "
                            f"Observed min={pred_min}, max={pred_max}. "
                            f"Sample submission indicates probability mode."
                        )

                elif inferred_pred_mode == "class_labels":
                    allowed_classes = set(unique_ints)
                    bad_vals = [
                        v for v in all_pred_values
                        if abs(v - round(v)) > eps or int(round(v)) not in allowed_classes
                    ]
                    if bad_vals:
                        errors.append(
                            f"Discrete class label check failed: submission values "
                            f"do not match sample_submission class set {sorted(allowed_classes)}. "
                            f"Found {len(bad_vals)} invalid values."
                        )
        except Exception:
            warnings.append("Semantic range check vs sample_submission failed (best-effort)")

    errors = [str(e).strip() for e in errors if str(e).strip()]
    warnings = [str(w).strip() for w in warnings if str(w).strip()]

    return SubmissionCheckResult(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        header=header,
        row_count=row_count,
        inferred_pred_mode=inferred_pred_mode,
    )

# -------------------- helpers --------------------

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def compare_with_previous_code(new_code: str, previous_code: str) -> bool:
    if not new_code or not previous_code:
        return False
    return new_code.strip() == previous_code.strip()

def _enforce_stack_guardrails(code: str, orch: GlobalOrchestrator) -> None:
    # Comment translated to English.
    try:
        orc_cfg = orch.cfg.orchestration
        enforce = bool(getattr(orc_cfg, "enforce_single_stack", True))
        allow_ens = bool(getattr(orc_cfg, "allow_ensembles", True))
    except Exception:
        enforce, allow_ens = True, True
    if enforce and detect_mixed_stacks(code) and not allow_ens:
        raise RuntimeError("Guardrail: mixed unrelated stacks detected in a single stage.")


def snapshot_data_tree(
    project_root: str,
    data_dirname: str = "data",
    max_files: int = 200,
    *,
    exclude_logs: bool = True,
) -> str:
    """
    Shallow tree listing for prompts. By default skips `logs/`, `__pycache__`, `.git` and `*.log`
    so task_plan / agents are not flooded with install noise.
    """
    root = os.path.join(project_root, data_dirname)

    lines = []
    if not os.path.exists(root):
        return f"(no {data_dirname}/ found at {root})"

    skip_dir_names = {"logs", "__pycache__", ".git", "node_modules", ".pytest_cache"}

    max_files_per_dir = 5

    for dp, dn, fn in os.walk(root):
        if exclude_logs:
            dn[:] = [d for d in dn if d.lower() not in skip_dir_names]
        full_path = os.path.abspath(dp)
        lines.append(f"[DIR] {full_path}")

        use_files = sorted(fn)
        if exclude_logs:
            use_files = [f for f in fn if not f.lower().endswith(".log")]
            use_files = sorted(use_files)

        shown = use_files[:min(max_files_per_dir, max_files)]
        for f in shown:
            lines.append(f"  [FILE] {os.path.join(full_path, f)}")

        hidden_count = max(0, len(use_files) - len(shown))
        if hidden_count > 0:
            lines.append(f"  ... (+{hidden_count} more files)")

        if len(lines) > 1200:
            lines.append("... (trimmed)")
            break

    return "\n".join(lines)


def materialize_project_root_submission_csv(orch: GlobalOrchestrator) -> Optional[Path]:
    """
    Bench-friendly guarantee: always have a non-empty ./submission.csv at project root when any
    candidate exists. Does not raise; safe for finally:/deadline exits.
    """
    root = Path(orch.project_root)
    legacy = root / "submission.csv"
    bench_dir = root / "submission"
    bench_path = bench_dir / "submission.csv"
    try:
        ensure_canonical_submission_copy(orch)
    except Exception:
        pass
    dest = canonical_submission_path(orch)
    try:
        # 1) Если канонический сабмит есть и не пустой — зеркалим его в оба bench-friendly места.
        if dest.exists() and dest.stat().st_size > 0:
            if dest.resolve() != legacy.resolve():
                _copy_text_file(orch, dest, legacy)
            try:
                _ensure_dir(bench_dir)
                if bench_path.resolve() != dest.resolve():
                    _copy_text_file(orch, dest, bench_path)
            except Exception:
                pass
            return legacy if legacy.exists() and legacy.stat().st_size > 0 else dest
    except Exception:
        pass

    # 2) Если корневой ./submission.csv есть — считаем его источником истины и дублируем в bench/ и канонический путь.
    try:
        if legacy.exists() and legacy.stat().st_size > 0:
            try:
                _ensure_dir(bench_dir)
                if bench_path.resolve() != legacy.resolve():
                    _copy_text_file(orch, legacy, bench_path)
            except Exception:
                pass
            if dest.resolve() != legacy.resolve():
                _copy_text_file(orch, legacy, dest)
            return legacy
    except Exception:
        pass

    # 3) Если существует только bench-путь ./submission/submission.csv — поднимаем его в корень и канон.
    try:
        if bench_path.exists() and bench_path.stat().st_size > 0:
            _copy_text_file(orch, bench_path, legacy)
            if dest.resolve() != legacy.resolve():
                _copy_text_file(orch, bench_path, dest)
            return legacy if legacy.exists() and legacy.stat().st_size > 0 else dest
    except Exception:
        pass
    try:
        spec_obj: Dict[str, Any] = {}
        sp = root / orch.cfg.paths.artifacts_dir / "spec.json"
        if sp.exists():
            try:
                loaded = json.loads(sp.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    spec_obj = loaded
            except Exception:
                spec_obj = {}
        found = _find_valid_submissions(orch, spec_obj)
        if found:
            _copy_text_file(orch, found[0][0], legacy)
    except Exception:
        pass
    return legacy if legacy.exists() and legacy.stat().st_size > 0 else None


def ensure_canonical_submission_copy(orch: GlobalOrchestrator) -> Optional[Path]:
    """
    Ensure submission exists at paths.submission_dir/submission_filename.
    Copies from artifacts/final, versions, or first valid CSV. Returns path if ok.
    """
    root = Path(orch.project_root)
    dest = canonical_submission_path(orch)
    _ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    final_sub = root / orch.cfg.paths.artifacts_dir / "final" / "submission.csv"
    if final_sub.exists():
        _copy_text_file(orch, final_sub, dest)
        if dest.exists():
            return dest
    rel = _finalize_single_submission(orch)
    if rel:
        p = root / rel
        if p.exists():
            _copy_text_file(orch, p, dest)
    return dest if dest.exists() else None


def task_txt_requires_code_artifact(project_root: Path) -> bool:
    """Heuristic: competition/task text asks for code submission."""
    p = project_root / "task.txt"
    if not p.exists():
        return False
    try:
        t = p.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    keys = ("submit code", "code submission", "upload code", "notebook", ".py", "source code", "github")
    return any(k in t for k in keys)


def run_final_output_gate(
    orch: GlobalOrchestrator,
    spec: Dict[str, Any],
    *,
    task_txt_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Final guarantee: canonical submission path exists; optional code artifact check from task.txt.
    Writes artifacts/final/output_gate_report.json.
    """
    root = Path(orch.project_root)
    final_dir = root / orch.cfg.paths.artifacts_dir / "final"
    _ensure_dir(final_dir)
    errors: List[str] = []
    warnings: List[str] = []

    dest = canonical_submission_path(orch)
    legacy = root / "submission.csv"
    bench_dir = root / "submission"
    bench_path = bench_dir / "submission.csv"

    if not dest.exists() or dest.stat().st_size == 0:
        # ReAct-style recovery: попробуй собрать сабмит из всех известных мест, а не просто сдаться.
        ensure_canonical_submission_copy(orch)
        # Если после этого канонический путь всё ещё пуст — попробуй подтянуть из ./submission.csv и ./submission/submission.csv.
        if (not dest.exists() or dest.stat().st_size == 0):
            try:
                # 1) Корневой ./submission.csv как источник.
                if legacy.exists() and legacy.stat().st_size > 0:
                    _copy_text_file(orch, legacy, dest)
                # 2) Bench-путь ./submission/submission.csv как источник.
                elif bench_path.exists() and bench_path.stat().st_size > 0:
                    _copy_text_file(orch, bench_path, dest)
                    # Поднимем его в корень тоже, чтобы всё было согласовано.
                    _copy_text_file(orch, bench_path, legacy)
            except Exception:
                pass
    if not dest.exists() or dest.stat().st_size == 0:
        errors.append(f"Missing or empty canonical submission: {dest}")
    else:
        # Legacy / bench copies for runners and external checkers.
        legacy = root / "submission.csv"
        try:
            if dest.resolve() != legacy.resolve():
                _copy_text_file(orch, dest, legacy)
            try:
                _ensure_dir(bench_dir)
                if bench_path.resolve() != dest.resolve():
                    _copy_text_file(orch, dest, bench_path)
            except Exception:
                pass
        except Exception:
            pass

    code_ok = True
    tr = task_txt_root or root
    if task_txt_requires_code_artifact(tr):
        best_py = root / orch.cfg.paths.artifacts_dir / "best" / "code.py"
        last_py = root / orch.cfg.paths.artifacts_dir / "last" / "code.py"
        if not best_py.exists() and not last_py.exists():
            code_ok = False
            errors.append("task.txt implies code submission but no artifacts/best|last/code.py")
        elif best_py.exists() and best_py.stat().st_size < 50:
            warnings.append("artifacts/best/code.py is very small")

    val = validate_final_submission(orch, spec or {})
    if not val.get("ok"):
        errors.extend(val.get("errors") or [])

    # Normalize errors again after merging (defensive).
    errors = [str(e).strip() for e in errors if str(e).strip()]

    try:
        orch._refresh_versions_table()  # type: ignore[attr-defined]
    except Exception:
        pass

    report = {
        "ok": len(errors) == 0,
        "canonical_submission": str(dest),
        "errors": errors,
        "warnings": warnings,
        "code_artifact_ok": code_ok,
        "validation": val,
    }
    try:
        orch.write_file(
            str(final_dir / "output_gate_report.json"),
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    except Exception:
        pass
    return report
