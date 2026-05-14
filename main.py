from __future__ import annotations
import argparse
import atexit
import os
import sys
from pathlib import Path
import json
import time
import shutil

# --- ДОБАВИТЬ ЭТОТ БЛОК ДЛЯ WINDOWS ---
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')
# --------------------------------------

try:
    from colorama import init as colorama_init, Fore
    colorama_init(autoreset=True)
except Exception:
    class Fore:
        RED=""; GREEN=""; BLUE=""; YELLOW=""; CYAN=""

from src.config import AppConfig, load_dotenv_from_cwd
from src.orchestrator import GlobalOrchestrator
from src.pipeline import main_pipeline
from src.llm_utils import set_timing_orchestrator, load_llm_pricing


def validate_config_or_exit(cfg: AppConfig) -> None:
    errs: list[str] = []
    warns: list[str] = []
    if cfg.orchestration.max_tree_depth <= 0:
        errs.append("orchestration.max_tree_depth must be > 0")
    if cfg.orchestration.max_tree_width <= 0:
        errs.append("orchestration.max_tree_width must be > 0")
    if cfg.orchestration.improve_budget_min > cfg.orchestration.total_budget_min:
        warns.append("orchestration.improve_budget_min is greater than total_budget_min")
    if cfg.runtime.min_exec_timeout_sec < 1:
        errs.append("runtime.min_exec_timeout_sec must be >= 1")
    if cfg.runtime.metric_validation_retry_limit < 0:
        errs.append("runtime.metric_validation_retry_limit must be >= 0")
    if cfg.runtime.router_retry_limit < 1:
        errs.append("runtime.router_retry_limit must be >= 1")
    if cfg.runtime.generation_retry_limit < 1:
        errs.append("runtime.generation_retry_limit must be >= 1")
    if cfg.data_check.max_samples_per_dir < 1:
        errs.append("data_check.max_samples_per_dir must be >= 1")
    if cfg.data_check.probe_timeout_sec < 1:
        errs.append("data_check.probe_timeout_sec must be >= 1")

    for w in warns:
        print(Fore.YELLOW + f"[CONFIG WARNING] {w}")
    if errs:
        for e in errs:
            print(Fore.RED + f"[CONFIG ERROR] {e}")
        sys.exit(2)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    # FIX: Default to task.txt, making it the standard. No more demo mode.
    ap.add_argument("--task_file", type=str, default="task.txt", help="Path to a text file with the task")
    ap.add_argument("--resume", action="store_true", help="Resume from artifacts (spec/code/metrics) if present")
    ap.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable tee of stdout/stderr to logs/run_*.log and logs/last_run.log (console only)",
    )
    return ap.parse_args()


class _TeeTextStream:
    """Write to multiple text streams (e.g. console + log files)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    # Some libraries test for isatty on stdout
    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

def read_task(path: str) -> str:
    # FIX: Remove the demo fallback. If the file doesn't exist, it's a fatal error.
    task_path = Path(path)
    if not task_path.exists():
        print(Fore.RED + f"FATAL: Task file not found at '{path}'. Please create it or specify a different file with --task_file.")
        sys.exit(1)
    return task_path.read_text(encoding="utf-8")


def _prepend_project_venv_to_syspath(project_root: Path, cfg: AppConfig) -> None:
    """Project venv paths (same as GlobalOrchestrator.project.env_dir) without constructing the orchestrator."""
    env_dir = project_root / cfg.paths.venv_dir
    if os.name == "nt":
        candidates = [env_dir / "Lib/site-packages"]
    else:
        candidates = [env_dir / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"]
    for sp in candidates:
        if sp.exists() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
            print(f"[bootstrap] Added venv site-packages to sys.path: {sp}")
    vpy = env_dir / "Scripts" / "python.exe" if os.name == "nt" else env_dir / "bin" / "python"
    if not vpy.exists():
        return
    try:
        import subprocess
        out = subprocess.check_output(
            [vpy.as_posix(), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
        if out and out not in sys.path:
            sys.path.insert(0, out)
            print(f"[bootstrap] Added introspected site-packages to sys.path: {out}")
    except Exception as e:
        print(Fore.YELLOW + f"[bootstrap] Failed to introspect venv site-packages: {e}")


def main():
    start_time = time.time()
    print(Fore.CYAN + "--- Application Starting ---")

    load_dotenv_from_cwd()
    args = parse_args()
    print(f"Loading configuration from: {args.config}")
    cfg = AppConfig.from_yaml(args.config)
    cfg.apply_env()
    validate_config_or_exit(cfg)
    _cfg_path = Path(args.config).resolve()
    load_llm_pricing(_cfg_path.parent / cfg.llm.model_pricing_file)
    print("Configuration loaded and environment variables applied.")

    cfg_project_root = str(getattr(cfg.runtime, "project_root", "") or "").strip()
    if cfg_project_root:
        project_root = Path(cfg_project_root).expanduser()
        if not project_root.is_absolute():
            project_root = (Path.cwd() / project_root).resolve()
        else:
            project_root = project_root.resolve()
        print(f"Using runtime.project_root as project_root: {project_root}")
    else:
        # Human-friendly fallback: cwd/<project_name> (default: cwd/ml_project).
        project_name = str(getattr(cfg.runtime, "project_name", "ml_project") or "ml_project").strip() or "ml_project"
        project_root = (Path.cwd() / project_name).resolve()
        print(f"Using runtime.project_name fallback as project_root: {project_root}")

    log_handles: list = []
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    if not args.no_log_file:
        try:
            logs_dir = project_root / cfg.paths.logs_dir
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path_run = logs_dir / f"run_{ts}.log"
            path_last = logs_dir / "last_run.log"
            f_run = open(path_run, "w", encoding="utf-8", newline="", buffering=1)
            f_last = open(path_last, "w", encoding="utf-8", newline="", buffering=1)
            log_handles.extend([f_run, f_last])
            sys.stdout = _TeeTextStream(orig_stdout, f_run, f_last)
            sys.stderr = _TeeTextStream(orig_stderr, f_run, f_last)
            print(Fore.CYAN + f"[LOG] Full run log: {path_run} | copy: {path_last}")

            def _cleanup_run_logging() -> None:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                for h in log_handles:
                    try:
                        h.close()
                    except Exception:
                        pass

            atexit.register(_cleanup_run_logging)
        except OSError as e:
            print(Fore.YELLOW + f"[LOG] Could not open log files (continuing console-only): {e}")
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            log_handles.clear()

    # Config is already parsed above; orchestrator does not read YAML — it receives cfg.
    # LLM + factory ping before GlobalOrchestrator so venv/pip bootstrap does not run first.
    print("\n" + Fore.CYAN + "--- Initializing LLMs ---")
    print(f"Preferred LLM backend: {cfg.llm.prefer}")
    _prepend_project_venv_to_syspath(project_root, cfg)
    from src.llm_factory import build_llms
    try:
        llm_strong, llm_fast, code_llm = build_llms(cfg)
    except RuntimeError as e:
        print(Fore.RED + f"[LLM] {e}")
        sys.exit(2)
    print(Fore.GREEN + "LLMs initialized successfully.")
    print(f"  - Strong model type: {type(llm_strong).__name__}")
    print(f"  - Fast model type:   {type(llm_fast).__name__}")
    print(f"  - Code model type:   {type(code_llm).__name__}")

    print("\n" + Fore.CYAN + "--- Initializing Orchestrator ---")
    orch = GlobalOrchestrator(cfg, project_root=project_root, monitor_llm=llm_fast)
    orch.global_start_time = start_time
    orch.global_deadline_sec = cfg.orchestration.total_budget_sec
    set_timing_orchestrator(orch)
    print("GlobalOrchestrator initialized.")

    # Ultra-low budget guard:
    # If we're essentially out of time for the main pipeline, skip it, but ALWAYS run
    # the final submission auditor/fixer without a global-deadline cap.
    skip_main_pipeline = False
    try:
        remaining = max(0, int(orch.global_deadline_sec - orch.effective_elapsed_sec()))
    except Exception:
        remaining = 0
    if remaining <= 60:
        skip_main_pipeline = True
        print(
            Fore.YELLOW
            + f"[BUDGET] Only {remaining}s left for main pipeline. Skipping main pipeline; finalizer will run without deadline cap."
        )

    # Preinstall (torch, etc.) after LLM is known-good; CUDA repair can use llm_fast.
    try:
        from src.bootstrap import bootstrap_gpu_stack
        bootstrap_gpu_stack(orch, cfg, llm_fast)
    except Exception as e:
        print(Fore.YELLOW + f"[bootstrap] Preinstall step failed: {e}")

    # If user didn't pass --resume but there is already a previous solution/artifacts,
    # reset the run (tree + outputs) to avoid duplicated first-level tasks.
    if not args.resume:
        art_dir = orch.project_root / cfg.paths.artifacts_dir
        has_solution_outputs = any(
            (art_dir / p).exists()
            for p in ["submission.csv", "confusion_matrix.png", "final_model.pkl", "metrics.json", "data_profile.json"]
        )
        has_existing_tree = (art_dir / "tree.json").exists() or (art_dir / "tasks_tree.json").exists()

        if has_solution_outputs or has_existing_tree:
            print(Fore.YELLOW + "[RESET] Clearing prior run artifacts (no --resume).")

            # Reset the task graph
            for rel in ["tree.json", "tasks_tree.json"]:
                p = art_dir / rel
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

            # Reset node state
            try:
                state_dir = art_dir / "state"
                if state_dir.exists():
                    shutil.rmtree(state_dir, ignore_errors=True)
            except Exception:
                pass

            # Reset previous context/logs for cleaner "Current Direction"
            for rel in ["project_context.md", "PROJECT_LOG.md", "task_plan.md"]:
                if rel == "task_plan.md":
                    p = orch.project_root / rel
                elif rel == "PROJECT_LOG.md":
                    p = orch.project_root / rel
                else:
                    p = art_dir / rel
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

            # Reset primary solution outputs
            for rel in ["submission.csv", "confusion_matrix.png", "final_model.pkl", "metrics.json", "data_profile.json"]:
                p = art_dir / rel
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
    
    # Initialize MCP
    if cfg.mcp.enabled:
        print("\n" + Fore.CYAN + "--- Initializing MCP ---")
        # Lazy import to avoid hard dependency when MCP disabled.
        # If mcp package is missing in current venv, repair once and retry import.
        try:
            from src.mcp_client import MCPManager
        except ModuleNotFoundError as e:
            if "mcp" in str(e).lower():
                print(Fore.YELLOW + "[MCP] Python package 'mcp' is missing. Installing into current venv...")
                inst = orch.pip_install(["mcp"], stream=True)
                if int(inst.get("exit_code", 1)) != 0:
                    raise
                from src.mcp_client import MCPManager
            else:
                raise
        mcp_mgr = MCPManager(cfg.mcp)
        mcp_mgr.start()
        mcp_tools = mcp_mgr.get_tools()
        print("MCP Manager started and tools are ready.")
    else:
        print("\n" + Fore.YELLOW + "MCP is disabled in the config.")
        mcp_mgr = None
        mcp_tools = []


    # Comment translated to English.
    spec_from_disk = None
    if args.resume:
        spec_path = orch.project_root / cfg.paths.artifacts_dir / "spec.json"
        if spec_path.exists():
            try:
                spec_from_disk = json.loads(spec_path.read_text(encoding="utf-8"))
                print(Fore.YELLOW + f"[RESUME] Loaded SPEC from {spec_path}")
            except Exception as e:
                print(Fore.RED + f"[RESUME] Failed to read spec.json: {e}")

    try:
        print("\n" + Fore.CYAN + "--- Starting Main Pipeline ---")
        init_done_time = time.time()
        print(f"Initialization took {init_done_time - start_time:.2f} seconds.")
        
        task = read_task(args.task_file)
        print(f"Task to be executed: {task[:200]}...")
        if not skip_main_pipeline:
            result = None
            max_resume_passes = 250
            pass_idx = 0
            while True:
                pass_idx += 1
                try:
                    result = main_pipeline(
                        orch, llm_strong, llm_fast, code_llm, task,
                        resume=args.resume,
                        spec=spec_from_disk,
                        allow_spawn_improvement=True,
                        mcp_tools=mcp_tools,
                    )
                except Exception as _top_err:
                    import traceback
                    tb = traceback.format_exc()
                    
                    try:
                        print(f"\033[33m[MAIN] Pipeline crashed: {type(_top_err).__name__}: {_top_err}\033[0m")
                        print(tb)
                    except UnicodeEncodeError:
                        print(f"[MAIN] Pipeline crashed (UnicodeEncodeError): {type(_top_err).__name__}")
                        print(tb.encode('utf-8', 'replace').decode('utf-8'))
                    
                    try:
                        print("\033[33m[MAIN] Attempting resume on next pass...\033[0m")
                    except UnicodeEncodeError:
                        print("[MAIN] Attempting resume on next pass...")
                    
                    if not args.resume:
                        break
                if not args.resume:
                    break
                root = orch.tree_find_most_recent_root()
                if not root or not orch.tree_subtree_has_unfinished_work(root):
                    if pass_idx > 1:
                        print(
                            Fore.GREEN
                            + f"[RESUME] Pass {pass_idx - 1}: no unfinished (failed/pending/running) nodes — done."
                        )
                    break
                if pass_idx >= max_resume_passes:
                    print(
                        Fore.YELLOW
                        + f"[RESUME] Stopping after {max_resume_passes} passes (unfinished nodes may remain)."
                    )
                    break
                print(
                    Fore.CYAN
                    + f"[RESUME] Pass {pass_idx}: unfinished work remains in tree — next pass (deepest failed, then pending)..."
                )
            print(Fore.GREEN + "\n=== PIPELINE RESULT ===")
            print(result)
    finally:
        # Finalizer should not be constrained by the main pipeline deadline.
        try:
            orch.ignore_global_deadline = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            from src.helpers import materialize_project_root_submission_csv
            p = materialize_project_root_submission_csv(orch)
            if p:
                print(Fore.GREEN + f"[FINAL] submission materialized at {p}")
        except Exception as ex:
            print(Fore.YELLOW + f"[FINAL] materialize project_root/submission.csv failed (non-fatal): {ex}")

        # Even if HARD DEADLINE aborted earlier, run a last-pass LLM auditor:
        # choose the final submission using ALL calculated metrics (main + improver),
        # and print reasoning at the very end.
        try:
            from src.helpers import _finalize_single_submission_by_all_metrics_llm
            if "llm_fast" in locals() and llm_fast is not None:
                task_for_audit = locals().get("task", "") or read_task(args.task_file)
                _finalize_single_submission_by_all_metrics_llm(
                    orch,
                    llm_fast,
                    spec_from_disk or {},
                    task=task_for_audit[:2000],
                    code_llm=locals().get("code_llm"),
                    mcp_tools=locals().get("mcp_tools") or [],
                )
        except Exception as ex:
            print(Fore.YELLOW + f"[FINAL] LLM metric audit failed (non-fatal): {ex}")

        # Final verifier run (checker script) — best-effort, even after deadline.
        try:
            if "llm_fast" in locals() and llm_fast is not None:
                import json as _json
                from src.prompts_agents import checker_code_agent
                from src.utils import shorten_string_middle

                root = Path(orch.project_root)
                spec_obj = spec_from_disk or {}
                try:
                    sp = root / cfg.paths.artifacts_dir / "spec.json"
                    if sp.exists():
                        spec_obj = _json.loads(sp.read_text(encoding="utf-8"))
                except Exception:
                    pass

                # Prefer the final selected code if present, else fall back to best/last.
                code_summary = ""
                for p in [
                    root / cfg.paths.artifacts_dir / "final" / "best_code.py",
                    root / cfg.paths.artifacts_dir / "best" / "code.py",
                    root / cfg.paths.artifacts_dir / "last" / "code.py",
                ]:
                    try:
                        if p.exists():
                            code_summary = p.read_text(encoding="utf-8", errors="ignore")
                            break
                    except Exception:
                        pass
                code_summary = shorten_string_middle(code_summary or "", 2500)

                task_for_check = locals().get("task", "") or read_task(args.task_file)
                checker_py = checker_code_agent(
                    llm_fast,
                    task=task_for_check[:4000],
                    spec=spec_obj if isinstance(spec_obj, dict) else {},
                    code_summary=code_summary,
                    final_answer="(final verifier run from main.py finally)",
                    metrics_json="",
                    stdout_tail="",
                    stderr_tail="",
                    improvement_summary="",
                )
                res = orch.run_python_code(
                    checker_py,
                    filename="final_verifier.py",
                    timeout=min(60, getattr(cfg.runtime, "checker_timeout_cap_sec", 60)),
                )
                out = (res.get("output") or "") + "\n" + (res.get("errors") or "")
                print("\n[FINAL][VERIFIER] Checker output (tail):")
                print(shorten_string_middle(out, 3000))
                try:
                    fdir = root / cfg.paths.artifacts_dir / "final"
                    fdir.mkdir(parents=True, exist_ok=True)
                    (fdir / "final_verifier_stdout.txt").write_text(out, encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        except Exception as ex:
            print(Fore.YELLOW + f"[FINAL] verifier checker failed (non-fatal): {ex}")

        if mcp_mgr:
            print("\n" + Fore.CYAN + "--- Shutting down MCP ---")
            mcp_mgr.stop()
            print("MCP Manager stopped.")

        end_time = time.time()
        effective_total = max(
            0.0, (end_time - start_time) - getattr(orch, "paused_llm_sleep_sec", 0.0)
        )
        print(f"\nTotal execution time: {effective_total:.2f} seconds.")
        print(Fore.CYAN + "--- Application Finished ---")


if __name__ == "__main__":
    sys.exit(main())