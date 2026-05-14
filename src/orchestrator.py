from __future__ import annotations

import collections
import datetime
import json
import os, time, uuid, platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
from urllib.request import urlretrieve
import re
import shlex

from colorama import Fore

from .bash_agent import BashAgent
from .utils import detect_os, shorten_string_middle, _slug, _now_tag, _ensure_dir
from .config import AppConfig


@dataclass
class Project:
    name: str
    root: Path
    env_dir: Path
    vpy: Optional[Path] = None
    vpip: Optional[Path] = None


class GlobalOrchestrator:
    def __init__(self, cfg: AppConfig, project_root: Optional[Path] = None, monitor_llm: Any = None):
        self.cfg = cfg
        self.monitor_llm = monitor_llm
        self.project_root = project_root or Path(cfg.runtime.project_name).resolve()
        self.paused_llm_sleep_sec: float = 0.0
        
        # FIX: Define directories and files separately
        self.dir_paths = {
            "root": self.project_root,
            "data": Path(cfg.paths.data_dir) if os.path.isabs(
                cfg.paths.data_dir) else self.project_root / cfg.paths.data_dir,
            "src": self.project_root / cfg.paths.src_dir,
            "artifacts": self.project_root / cfg.paths.artifacts_dir,
            "logs": self.project_root / cfg.paths.logs_dir,
            "tests": self.project_root / cfg.paths.tests_dir,
            "scripts": self.project_root / cfg.paths.scripts_dir,
        }
        self.project_log_path = self.project_root / "PROJECT_LOG.md"

        for p in self.dir_paths.values():
            p.mkdir(parents=True, exist_ok=True)

        self.bash = BashAgent(
            workdir=str(self.project_root),
            min_exec_timeout_sec=cfg.runtime.min_exec_timeout_sec,
            predictive_buffer_pct=cfg.runtime.predictive_buffer_pct,
        )
        self.project = Project(cfg.runtime.project_name, self.project_root, self.project_root / cfg.paths.venv_dir)
        
        # Always resolve venv paths based on platform
        if platform.system() == "Windows":
            self.project.vpy = self.project.env_dir / "Scripts" / "python.exe"
            self.project.vpip = self.project.env_dir / "Scripts" / "pip.exe"
        else:
            self.project.vpy = self.project.env_dir / "bin" / "python"
            self.project.vpip = self.project.env_dir / "bin" / "pip"

        self.created_files: List[Path] = []
        self.steps: List[Dict[str, Any]] = []
        
        if cfg.runtime.create_env:
            self._create_env()
        else:
            # If venv exists — use it; otherwise fall back to system python (no hard crash).
            if not self.project.vpy.exists():
                sys_py = detect_os()["python_exec"]
                print(f"[VENV] .venv not found at {self.project.env_dir}. Falling back to system python: {sys_py}")
                self.project.vpy = Path(sys_py)
                # Derive pip path relative to system python
                self.project.vpip = self.project.vpy.parent / (
                    "pip.exe" if platform.system() == "Windows" else "pip"
                )
            else:
                # venv exists but create_env=False — still ensure pip is healthy
                self._ensure_pip()

    def _create_env(self):
        base_py = os.getenv("BASE_PYTHON", detect_os()["python_exec"])
        self.project.env_dir.mkdir(parents=True, exist_ok=True)

        res = self.bash.run(
            f'{base_py} -m venv "{self.project.env_dir.as_posix()}"',
            stream=True, prefix="[VENV] ", tee_logfile=str(self.dir_paths["logs"] / "venv_create.log")
        )
        self.log("venv_create", {"exit": res["exit_code"], "stderr": res["stderr"], "log": res.get("log_path")})

        if res.get("exit_code", 0) != 0:
            # venv creation failed — fall back to system python so the run can continue
            sys_py = detect_os()["python_exec"]
            print(f"[VENV] WARNING: venv creation failed (exit={res.get('exit_code')}). "
                  f"Falling back to system python: {sys_py}")
            self.project.vpy = Path(sys_py)
            self.project.vpip = self.project.vpy.parent / (
                "pip.exe" if platform.system() == "Windows" else "pip"
            )
            return

        self._ensure_pip()

    def _ensure_pip(self):
        """Verify pip exists and proactively repair it if missing in the venv."""
        if not self.project.vpy or not self.project.vpy.exists():
            return

        up = self.bash.run(
            f'{self.project.vpy.as_posix()} -m pip install -U pip setuptools wheel',
            timeout=self.cfg.runtime.pip_timeout_sec,
            stream=True, prefix="[PIP] ", tee_logfile=str(self.dir_paths["logs"] / "pip_bootstrap.log")
        )
        # If venv has python but pip is missing, recover pip proactively.
        if up.get("exit_code", 1) != 0 and "No module named pip" in f"{up.get('stderr','')}\n{up.get('stdout','')}":
            self._repair_pip_in_venv(stream=True)
            up = self.bash.run(
                f'{self.project.vpy.as_posix()} -m pip install -U pip setuptools wheel',
                timeout=self.cfg.runtime.pip_timeout_sec,
                stream=True, prefix="[PIP] ", tee_logfile=str(self.dir_paths["logs"] / "pip_bootstrap.log")
            )
        self.log("venv_bootstrap", {"exit": up["exit_code"], "stderr": up["stderr"], "log": up.get("log_path")})

    def _repair_pip_in_venv(self, stream: bool = True) -> Dict[str, Any]:
        """
        Repair pip inside venv:
        1) try ensurepip
        2) fallback to get-pip.py
        """
        if not self.project.vpy:
            return {"stdout": "", "stderr": "No venv python to repair pip", "exit_code": 1}

        py = self.project.vpy.as_posix()
        ensure_cmd = f'{py} -m ensurepip --upgrade'
        ensure_log = (self.dir_paths["logs"] / f"pip_repair_ensurepip_{uuid.uuid4().hex[:6]}.log").as_posix()
        ensure_res = self.bash.run(
            ensure_cmd,
            timeout=self.cfg.runtime.pip_timeout_sec,
            stream=stream,
            tee_logfile=ensure_log if stream else None,
        )
        if ensure_res.get("exit_code", 1) == 0:
            self.log("pip_repair", {"method": "ensurepip", "exit": 0, "log": ensure_log})
            return ensure_res

        # ensurepip may be unavailable on stripped python builds; fallback to get-pip.py
        try:
            self.dir_paths["scripts"].mkdir(parents=True, exist_ok=True)
            get_pip_path = self.dir_paths["scripts"] / f"get-pip-{uuid.uuid4().hex[:8]}.py"
            urlretrieve("https://bootstrap.pypa.io/get-pip.py", str(get_pip_path))
            gp_cmd = f'{py} "{get_pip_path.as_posix()}"'
            gp_log = (self.dir_paths["logs"] / f"pip_repair_getpip_{uuid.uuid4().hex[:6]}.log").as_posix()
            gp_res = self.bash.run(
                gp_cmd,
                timeout=self.cfg.runtime.pip_timeout_sec,
                stream=stream,
                tee_logfile=gp_log if stream else None,
            )
            self.log("pip_repair", {"method": "get-pip.py", "exit": gp_res.get("exit_code", 1), "log": gp_log})
            try:
                get_pip_path.unlink(missing_ok=True)
            except Exception:
                pass
            return gp_res
        except Exception as e:
            self.log("pip_repair", {"method": "get-pip.py", "exit": 1, "stderr": str(e)})
            return {"stdout": "", "stderr": str(e), "exit_code": 1}

    def log(self, kind: str, payload: Dict[str, Any]):
        self.steps.append({"ts": time.time(), "kind": kind, "payload": payload})

    def log_to_project_log(self, task: str, depth: int, summary: str, status: str = "info"):
        """Appends a message to the PROJECT_LOG.md file."""
        log_path = self.project_log_path
        indent = "  " * depth
        icon = {"done": "✅", "failed": "❌", "running": "⏳", "info": "ℹ️", "skipped": "⏩"}.get(status, "📄")
        
        entry = f"{indent}- {icon} **{task.splitlines()[0]}** ({status.upper()})\n"
        if summary:
            summary_lines = summary.strip().splitlines()
            for line in summary_lines:
                entry += f"{indent}  - {line}\n"
        
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n---\n*Timestamp: {datetime.datetime.utcnow().isoformat()}Z*\n\n{entry}\n")

    def get_project_log_content(self) -> str:
        """Reads the entire content of the project log."""
        if self.project_log_path.exists():
            return self.project_log_path.read_text(encoding="utf-8")
        return "Project log is empty."

    def effective_elapsed_sec(self) -> float:
        start = getattr(self, "global_start_time", time.time())
        elapsed = time.time() - start
        paused = float(getattr(self, "paused_llm_sleep_sec", 0.0) or 0.0)
        return max(0.0, elapsed - paused)

    def sleep_with_pause_accounting(self, seconds: float) -> None:
        if seconds <= 0:
            return
        t0 = time.time()
        time.sleep(seconds)
        self.paused_llm_sleep_sec += max(0.0, time.time() - t0)

    def write_file(self, rel_path: str, content: str) -> Path:
        abs_path = self.project_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        self.created_files.append(abs_path)
        self.log("file_write", {"path": abs_path.as_posix()})
        return abs_path

    def run_python_file(self, rel_script_path: str, timeout: Optional[int] = None, stream: bool = False,
                        spec: Dict[str, Any] = None, prediction: Dict[str, Any] = None) -> Dict[str, Any]:
        abs_path = self.project_root / rel_script_path
        if not abs_path.exists():
            return {"output": "", "errors": f"No such file: {abs_path}", "exit_code": 1}
        py = self.project.vpy.as_posix() if self.project.vpy else self.bash.python_exec
        cmd = f'{py} "{abs_path.as_posix()}"'
        logf = (self.dir_paths["logs"] / f"run_{abs_path.stem}.log").as_posix()

        # Respect per-task budget (time_budget_sec) first, then global code timeout,
        # then global hard deadline.
        task_budget = None
        try:
            if spec and isinstance(spec, dict):
                task_budget = int(spec.get("_current_task_budget_sec") or 0) or None
        except Exception:
            task_budget = None

        desired = None
        if timeout is not None:
            try:
                desired = int(timeout)
            except Exception:
                desired = None
        if desired is None:
            # Prefer explicit task-level budget; otherwise fall back to the global
            # code timeout. Note: task_budget is NO LONGER capped by code_timeout_sec
            # so long-running tasks (e.g. image training) aren't artificially truncated.
            if task_budget is not None:
                desired = max(1, int(task_budget))
            else:
                desired = int(self.cfg.runtime.code_timeout_sec)
        # Compute the absolute ceiling (run-wide remaining budget) and pass it to
        # bash_agent.run as `hard_cap`. We deliberately DO NOT pre-clamp `desired`
        # here: bash_agent uses the predictor to expand the soft target above
        # `desired` when needed (heavy ML steps), then clamps by `hard_cap`.
        hard_cap_sec: Optional[int] = None
        if not bool(getattr(self, "ignore_global_deadline", False)):
            try:
                deadline = float(getattr(self, "global_deadline_sec", self.cfg.orchestration.total_budget_sec))
                remaining = max(0, int(deadline - self.effective_elapsed_sec()))
            except Exception:
                remaining = None
            if remaining is not None:
                hard_cap_sec = int(remaining)
        # Build watcher context extras: the ReAct watcher needs to know
        # WHAT the subprocess is supposed to be doing (task text) and WHAT
        # code is running, to decide whether observed behaviour matches intent.
        _task_text = ""
        try:
            if spec and isinstance(spec, dict):
                _task_text = str(spec.get("_current_task_text") or spec.get("task") or "")
        except Exception:
            _task_text = ""
        _code_excerpt = ""
        try:
            _code_excerpt = abs_path.read_text(encoding="utf-8", errors="replace")[:16000]
        except Exception:
            _code_excerpt = ""
        watcher_ctx_extras = {"task_text": _task_text, "code": _code_excerpt}

        res = self.bash.run(
            cmd,
            timeout=desired,
            stream=stream,
            tee_logfile=logf,
            monitor_llm=self.monitor_llm,
            spec=spec,
            prediction=prediction,
            watcher_ctx_extras=watcher_ctx_extras,
            hard_cap=hard_cap_sec,
        )
        self.log("python_exec", {"script": abs_path.as_posix(), "exit": res.get("exit_code")})
        return {"output": res.get("stdout", ""), "errors": res.get("stderr", ""), "exit_code": res.get("exit_code", 1)}

    def run_python_code(self, code: str, filename: str = "gen_code.py", timeout: Optional[int] = None,
                        stream: bool = False, spec: Dict[str, Any] = None, prediction: Dict[str, Any] = None) -> Dict[str, Any]:
        rel = os.path.join(self.cfg.paths.scripts_dir, filename)
        self.write_file(rel, code)
        return self.run_python_file(rel, timeout=timeout, stream=stream, spec=spec, prediction=prediction)

    def pip_install(self, packages: list[str], extra: str = "", stream: bool = True) -> Dict[str, Any]:
        if not self.project.vpy:
            return {"stdout": "", "stderr": "No venv, skip pip", "exit_code": 1}
        alias_map: Dict[str, str] = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "pil": "pillow",
            "yaml": "pyyaml",
            "dotenv": "python-dotenv",
            "bs4": "beautifulsoup4",
            "faiss": "faiss-cpu",
            "fitz": "pymupdf",
            "google.generativeai": "google-generativeai",
            "google_genai": "google-generativeai",
        }

        def _llm_package_replacements(
            requested: List[str],
            pip_error_text: str,
            max_items: int = 4,
        ) -> Dict[str, str]:
            """
            Ask LLM for package replacement hints when pip can't find distributions.
            Returns mapping old->new.
            """
            llm = self.monitor_llm
            if llm is None:
                return {}
            try:
                from langchain_core.prompts import ChatPromptTemplate
                from src.llm_utils import invoke_and_log
                from src.parsers import extract_json
                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a Python packaging expert. Given failed pip install packages and pip error output, "
                     "return ONLY JSON: {{\"replacements\": {{\"bad_pkg\": \"good_pkg\"}}, \"reason\": \"...\"}}. "
                     "Use only valid pip package names. Prefer minimal, high-confidence replacements. "
                     "If no confident replacement, return empty object."),
                    ("user",
                     "Requested packages:\n{requested}\n\nPip error:\n{err}\n\n"
                     f"Return at most {max_items} replacements.")
                ])
                res = invoke_and_log(
                    llm,
                    prompt,
                    {
                        "requested": json.dumps(requested, ensure_ascii=False),
                        "err": (pip_error_text or "")[-8000:],
                    },
                    agent_name="pip_package_resolver",
                )
                obj = extract_json(getattr(res, "content", "") or "") or {}
                if not isinstance(obj, dict):
                    return {}
                reps = obj.get("replacements", {})
                if not isinstance(reps, dict):
                    return {}
                clean: Dict[str, str] = {}
                for k, v in list(reps.items())[:max_items]:
                    ks = str(k).strip().lower()
                    vs = str(v).strip()
                    if ks and vs:
                        clean[ks] = vs
                return clean
            except Exception:
                return {}
        # Hard guardrail: never ask pip to install deprecated alias package.
        normalized = []
        for p in (packages or []):
            raw = str(p).strip()
            key = raw.lower()
            if key in alias_map:
                raw = alias_map[key]
            normalized.append(raw)
        # Deduplicate while preserving order.
        seen = set()
        packages = [p for p in normalized if p and not (p.lower() in seen or seen.add(p.lower()))]
        # Quote requirement tokens so shell doesn't treat constraints like "numpy<2.0" as redirections.
        quoted_pkgs = [shlex.quote(p) for p in packages]
        extra_tokens: list[str] = shlex.split(extra) if extra else []
        quoted_extra = " ".join(shlex.quote(t) for t in extra_tokens) if extra_tokens else ""
        cmd = f'{self.project.vpy.as_posix()} -m pip install -U ' + " ".join(quoted_pkgs) + (
            (" " + quoted_extra) if quoted_extra else "")
        log_path = (self.dir_paths["logs"] / f"pip_install_{uuid.uuid4().hex[:6]}.log").as_posix()
        res = self.bash.run(
            cmd,
            timeout=self.cfg.runtime.pip_timeout_sec,
            stream=stream,
            tee_logfile=log_path if stream else None,
            monitor_llm=self.monitor_llm
        )
        self.log("pip_install", {"cmd": cmd, "exit": res.get("exit_code"), "stderr": res.get("stderr"),
                                 "log": log_path if stream else ""})
        if res.get("exit_code", 1) != 0:
            txt = f"{res.get('stderr','')}\n{res.get('stdout','')}"
            if "No module named pip" in txt:
                repair = self._repair_pip_in_venv(stream=stream)
                self.log("pip_install_repair_attempt", {
                    "repair_exit": repair.get("exit_code", 1),
                    "repair_stderr": repair.get("stderr", "")
                })
                if repair.get("exit_code", 1) == 0:
                    # retry the original install once after successful repair
                    res = self.bash.run(
                        cmd,
                        timeout=self.cfg.runtime.pip_timeout_sec,
                        stream=stream,
                        tee_logfile=log_path if stream else None,
                        monitor_llm=self.monitor_llm
                    )
                    self.log("pip_install_retry_after_repair", {
                        "cmd": cmd,
                        "exit": res.get("exit_code"),
                        "stderr": res.get("stderr"),
                        "log": log_path if stream else ""
                    })
            else:
                # Generic self-heal for package alias/deprecation hints from pip output.
                # Example: "use 'scikit-learn' rather than 'sklearn' for pip commands."
                replacements: Dict[str, str] = {}
                for m in re.finditer(
                        r"use ['\"]([a-zA-Z0-9_\-\.]+)['\"] rather than ['\"]([a-zA-Z0-9_\-\.]+)['\"] for pip commands",
                        txt,
                        flags=re.IGNORECASE,
                ):
                    good = (m.group(1) or "").strip()
                    bad = (m.group(2) or "").strip()
                    if good and bad:
                        replacements[bad.lower()] = good

                if replacements:
                    fixed_pkgs = [replacements.get((p or "").lower(), p) for p in (packages or [])]
                    # dedupe preserving order
                    seen = set()
                    fixed_pkgs = [p for p in fixed_pkgs if p and not (p.lower() in seen or seen.add(p.lower()))]
                    if fixed_pkgs and fixed_pkgs != packages:
                        fixed_quoted_pkgs = [shlex.quote(p) for p in fixed_pkgs]
                        fixed_extra_tokens: list[str] = shlex.split(extra) if extra else []
                        fixed_quoted_extra = " ".join(shlex.quote(t) for t in fixed_extra_tokens) if fixed_extra_tokens else ""
                        fixed_cmd = f'{self.project.vpy.as_posix()} -m pip install -U ' + " ".join(fixed_quoted_pkgs) + (
                            (" " + fixed_quoted_extra) if fixed_quoted_extra else "")
                        res = self.bash.run(
                            fixed_cmd,
                            timeout=self.cfg.runtime.pip_timeout_sec,
                            stream=stream,
                            tee_logfile=log_path if stream else None,
                            monitor_llm=self.monitor_llm
                        )
                        self.log("pip_install_retry_after_alias_fix", {
                            "original_packages": packages,
                            "fixed_packages": fixed_pkgs,
                            "exit": res.get("exit_code"),
                            "stderr": res.get("stderr"),
                            "log": log_path if stream else ""
                        })
                # Additional self-heal: map common module aliases when pip cannot find a package.
                # Example errors:
                # - "Could not find a version that satisfies the requirement cv2"
                # - "ERROR: No matching distribution found for pil"
                missing = []
                for m in re.finditer(
                        r"(?:No matching distribution found for|Could not find a version that satisfies the requirement)\s+([a-zA-Z0-9_\-\.]+)",
                        txt,
                        flags=re.IGNORECASE,
                ):
                    pkg = (m.group(1) or "").strip().lower()
                    if pkg:
                        missing.append(pkg)
                unresolved = [p.lower() for p in (packages or []) if p.lower() not in alias_map]
                # If none parsed, still try alias map on requested packages.
                if not missing:
                    missing = [p.lower() for p in (packages or [])]
                mapped = [alias_map.get(x, "") for x in missing if alias_map.get(x)]
                if mapped:
                    fixed_pkgs = list(mapped) + unresolved
                    seen = set()
                    fixed_pkgs = [p for p in fixed_pkgs if p and not (p.lower() in seen or seen.add(p.lower()))]
                    if fixed_pkgs:
                        fixed_quoted_pkgs = [shlex.quote(p) for p in fixed_pkgs]
                        fixed_extra_tokens: list[str] = shlex.split(extra) if extra else []
                        fixed_quoted_extra = " ".join(shlex.quote(t) for t in fixed_extra_tokens) if fixed_extra_tokens else ""
                        fixed_cmd = f'{self.project.vpy.as_posix()} -m pip install -U ' + " ".join(fixed_quoted_pkgs) + (
                            (" " + fixed_quoted_extra) if fixed_quoted_extra else "")
                        res = self.bash.run(
                            fixed_cmd,
                            timeout=self.cfg.runtime.pip_timeout_sec,
                            stream=stream,
                            tee_logfile=log_path if stream else None,
                            monitor_llm=self.monitor_llm
                        )
                        self.log("pip_install_retry_after_missing_dist_fix", {
                            "original_packages": packages,
                            "missing_detected": missing,
                            "fixed_packages": fixed_pkgs,
                            "exit": res.get("exit_code"),
                            "stderr": res.get("stderr"),
                            "log": log_path if stream else ""
                        })
                # LLM-driven feedback loop: let model suggest better package names.
                if int(res.get("exit_code", 1)) != 0:
                    llm_repl = _llm_package_replacements(packages or [], txt)
                    if llm_repl:
                        llm_fixed = [llm_repl.get((p or "").lower(), p) for p in (packages or [])]
                        # dedupe preserving order
                        seen = set()
                        llm_fixed = [p for p in llm_fixed if p and not (p.lower() in seen or seen.add(p.lower()))]
                        if llm_fixed and llm_fixed != packages:
                            print(Fore.YELLOW + f"[PIP/LLM] package replacement suggested: {llm_repl}")
                            llm_fixed_quoted = [shlex.quote(p) for p in llm_fixed]
                            llm_extra_tokens: list[str] = shlex.split(extra) if extra else []
                            llm_quoted_extra = " ".join(shlex.quote(t) for t in llm_extra_tokens) if llm_extra_tokens else ""
                            llm_cmd = f'{self.project.vpy.as_posix()} -m pip install -U ' + " ".join(llm_fixed_quoted) + (
                                (" " + llm_quoted_extra) if llm_quoted_extra else "")
                            res = self.bash.run(
                                llm_cmd,
                                timeout=self.cfg.runtime.pip_timeout_sec,
                                stream=stream,
                                tee_logfile=log_path if stream else None,
                                monitor_llm=self.monitor_llm
                            )
                            self.log("pip_install_retry_after_llm_fix", {
                                "original_packages": packages,
                                "llm_replacements": llm_repl,
                                "fixed_packages": llm_fixed,
                                "exit": res.get("exit_code"),
                                "stderr": res.get("stderr"),
                                "log": log_path if stream else ""
                            })
        # Normalize noisy stream outcomes: treat as success if install succeeded.
        txt_final = f"{res.get('stderr','')}\n{res.get('stdout','')}"
        rc = res.get("exit_code", 1)
        try:
            rc = int(rc)
        except Exception:
            rc = 1
        if rc != 0 and "Successfully installed" in txt_final and "ERROR:" not in txt_final:
            rc = 0
        res["exit_code"] = rc
        return res

    def code_executor(self, code: str, file_name='gen_code.py', spec: Dict[str, Any] = None, prediction: Dict[str, Any] = None):
        self.write_file(f"{self.cfg.paths.scripts_dir}/{file_name}", code)
        res = self.run_python_file(f"{self.cfg.paths.scripts_dir}/{file_name}", stream=True, spec=spec, prediction=prediction)
        return {
            "output": shorten_string_middle(res.get("output", ""), 70000),
            "errors": shorten_string_middle(res.get("errors", ""), 10000),
            "exit_code": res.get("exit_code", 1),
        }

    def _state_path(self, node_id: str) -> Path:
        p = self.project_root / self.cfg.paths.artifacts_dir / "state"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"node_{node_id}.json"

    def _load_node_state(self, node_id: str) -> Dict[str, Any]:
        sp = self._state_path(node_id)
        if sp.exists():
            try:
                return json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def state_get(self, node_id: str) -> Dict[str, Any]:
        """Public accessor for per-node incident/runtime state."""
        return self._load_node_state(node_id)

    def state_set(self, node_id: str, state: Dict[str, Any]) -> None:
        """Public setter for per-node incident/runtime state."""
        self._save_node_state(node_id, state or {})

    def state_append_attempt(self, node_id: str, attempt: Dict[str, Any], max_items: int = 80) -> None:
        """Append one attempt record to node state with bounded history."""
        try:
            st = self._load_node_state(node_id)
            arr = st.get("attempts")
            if not isinstance(arr, list):
                arr = []
            arr.append(attempt or {})
            if len(arr) > max_items:
                arr = arr[-max_items:]
            st["attempts"] = arr
            self._save_node_state(node_id, st)
        except Exception:
            pass

    def format_task_graph_to_string(self) -> str:
        json_data = self._load_tree()
        nodes = json_data.get('nodes', {})
        roots = json_data.get('roots', [])
        output_lines: List[str] = []

        if not nodes or not roots:
            return "Task tree is empty."

        def build_hierarchy_recursive(node_id: str, prefix: str, lines_list: List[str], current_depth: int):
            node = nodes.get(node_id)
            if not node:
                return

            task_description = node.get('task', 'No description').replace('\n', ' ')
            status = node.get('status', 'pending').upper()
            status_marker = (
                "✅" if status == 'DONE'
                else "❌" if status == 'FAILED'
                else "⏳" if status == 'RUNNING'
                else "⏩" if status == 'SKIPPED'
                else "📄"
            )

            lines_list.append(f"{prefix}{status_marker} [Level {current_depth}] [{status}] {task_description}")

            children = node.get('children', [])
            for i, child_id in enumerate(children):
                new_prefix = prefix + ("    " if i == len(children) - 1 else "│   ")
                build_hierarchy_recursive(child_id, new_prefix, lines_list, current_depth + 1)

        output_lines.append("Current Task Execution Hierarchy:")
        # Only show the most recent root to avoid duplication from replanning
        most_recent_root = self.tree_find_most_recent_root()
        if most_recent_root:
            build_hierarchy_recursive(most_recent_root, "", output_lines, 0)
        else:
            # Fallback to showing all roots if no most recent can be determined
            for root_id in roots:
                build_hierarchy_recursive(root_id, "", output_lines, 0)

        # Append event feed so UI can display pruned/added/skipped transitions
        try:
            events_lines: List[str] = []
            events_path = self.dir_paths["artifacts"] / "task_graph_events.jsonl"
            if events_path.exists():
                rows = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                for ln in rows[-25:]:
                    try:
                        ev = json.loads(ln)
                        et = str(ev.get("event", "")).upper()
                        icon = "➕" if et == "ADDED" else "✂️" if et == "PRUNED" else "⏩" if et == "SKIPPED" else "ℹ️"
                        task = str(ev.get("task", ""))
                        ts = str(ev.get("ts", ""))
                        events_lines.append(f"{icon} [{et}] {task} ({ts})")
                    except Exception:
                        continue

            # Backward-compat for old PRUNED logs
            if not events_lines:
                legacy = self.dir_paths["artifacts"] / "skipped_tasks.log"
                if legacy.exists():
                    lns = [ln for ln in legacy.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    for ln in lns[-25:]:
                        if "PRUNED:" in ln:
                            task = ln.split("PRUNED:", 1)[1].strip()
                            events_lines.append(f"✂️ [PRUNED] {task}")

            if events_lines:
                output_lines.append("")
                output_lines.append("Graph Events (latest):")
                output_lines.extend(events_lines)
        except Exception:
            pass

        return "\n".join(output_lines)

    def tree_log_event(
            self,
            event: str,
            task: str,
            *,
            node_id: Optional[str] = None,
            parent_node_id: Optional[str] = None,
            reason: str = "",
    ) -> None:
        """Append graph lifecycle events (ADDED/PRUNED/SKIPPED) for UI visibility."""
        try:
            p = self.dir_paths["artifacts"] / "task_graph_events.jsonl"
            entry = {
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "event": str(event or "").upper(),
                "task": str(task or ""),
                "node_id": node_id or "",
                "parent_node_id": parent_node_id or "",
                "reason": reason or "",
            }
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _save_node_state(self, node_id: str, state: Dict[str, Any]) -> None:
        sp = self._state_path(node_id)
        try:
            state = dict(state or {})
            state["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            sp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _refresh_versions_table(self) -> Dict[str, Any]:
        root = Path(self.project_root)
        art = root / self.cfg.paths.artifacts_dir
        ver = art / "versions"
        idx_path = ver / "index.json"
        out_csv = ver / "ledger.csv"
        out_md = ver / "ledger.md"

        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else []
            if not isinstance(idx, list):
                idx = []
        except Exception:
            idx = []

        # If index never got writes (e.g. older runs), still show best/last + submission in ledger.
        if not idx:
            try:
                best_m = art / "best" / "metrics.json"
                last_m = art / "metrics_last.json"
                mpath = best_m if best_m.exists() else last_m
                if mpath.exists():
                    mdat = json.loads(mpath.read_text(encoding="utf-8"))
                    c_rel = ""
                    bp = art / "best" / "code.py"
                    lp = art / "last" / "code.py"
                    if bp.exists():
                        c_rel = str(bp.relative_to(root))
                    elif lp.exists():
                        c_rel = str(lp.relative_to(root))
                    sub_rel = ""
                    try:
                        csub = root / self.cfg.paths.submission_dir / self.cfg.paths.submission_filename
                        if csub.exists():
                            sub_rel = str(csub.relative_to(root))
                    except Exception:
                        sub_rel = ""
                    idx = [{
                        "ts": "bootstrap",
                        "tag": "best_or_last",
                        "paths": {
                            "metrics": str(mpath.relative_to(root)),
                            "code": c_rel,
                            "submission": sub_rel,
                        },
                        "name": mdat.get("name", "primary"),
                        "primary": mdat.get("primary"),
                        "maximize": mdat.get("maximize", True),
                    }]
            except Exception:
                pass

        def _score_key(e):
            try:
                val = float(e.get("primary"))
            except Exception:
                return (1, 0.0)
            maximize = bool(e.get("maximize", True))
            return (0, -val if maximize else +val)

        ranked = sorted(idx, key=_score_key)

        rel_csv = f"{self.cfg.paths.artifacts_dir}/versions/ledger.csv"
        rel_md = f"{self.cfg.paths.artifacts_dir}/versions/ledger.md"

        try:
            lines = ["rank,ts,tag,primary,maximize,metrics_path,code_path,submission_path"]
            for r, e in enumerate(ranked, 1):
                p = e.get("paths", {})
                lines.append(",".join([
                    str(r),
                    str(e.get("ts", "")),
                    str(e.get("tag", "")).replace(",", " "),
                    str(e.get("primary", "")),
                    str(e.get("maximize", True)),
                    str(p.get("metrics", "")).replace(",", " "),
                    str(p.get("code", "")).replace(",", " "),
                    str(p.get("submission", "")).replace(",", " "),
                ]))
            self.write_file(rel_csv, "\n".join(lines))
        except Exception:
            pass

        try:
            md = ["| rank | ts | tag | primary | maximize | metrics | code | submission |",
                  "|---:|---|---|---:|:---:|---|---|---|"]
            for r, e in enumerate(ranked, 1):
                p = e.get("paths", {})
                md.append(f"| {r} | {e.get('ts', '')} | {e.get('tag', '')} | {e.get('primary', '')} | "
                          f"{e.get('maximize', True)} | {p.get('metrics', '')} | {p.get('code', '')} | "
                          f"{p.get('submission', '')} |")
            self.write_file(rel_md, "\n".join(md))
        except Exception:
            pass

        best = ranked[0] if ranked else {}
        return {"index": ranked, "best": best, "csv": str(out_csv), "md": str(out_md)}

    def _pin_best_from_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if not entry:
            return {}
        root = Path(self.project_root)
        art = root / self.cfg.paths.artifacts_dir
        p = entry.get("paths", {})
        m_rel = p.get("metrics", "")
        c_rel = p.get("code", "")
        try:
            m_abs = root / m_rel if m_rel else None
            c_abs = root / c_rel if c_rel else None
            if m_abs and m_abs.exists():
                mets = json.loads(m_abs.read_text(encoding="utf-8"))
                self.write_file(str(art / "best_metrics.json"), json.dumps(mets, ensure_ascii=False, indent=2))
            if c_abs and c_abs.exists():
                code_txt = c_abs.read_text(encoding="utf-8")
                self.write_file(str(art / "best_code.py"), code_txt)
            return {
                "metrics": (json.loads((root / m_rel).read_text(encoding="utf-8")) if m_rel and (
                        root / m_rel).exists() else {}),
                "code_path": str(c_rel),
                "metrics_path": str(m_rel),
            }
        except Exception:
            pass

    def _save_metrics(self, metrics: Dict[str, Any], fname: str = "metrics_last.json") -> None:
        path = f"{self.cfg.paths.artifacts_dir}/{fname}"
        self.write_file(path, json.dumps(metrics, ensure_ascii=False, indent=2))

    def _record_metrics_version(self, metrics: Dict[str, Any], code_text: str, tag: str) -> Dict[
        str, Any]:
        """
        Delegate to helpers._record_metrics_version so index.json / ledger.csv / submission stay in sync.
        """
        from src.helpers import (
            _record_metrics_version as _helpers_record_metrics_version,
            ensure_canonical_submission_copy,
            _find_valid_submissions,
        )

        root = Path(self.project_root)
        sub_rel = ""
        try:
            dest = root / self.cfg.paths.submission_dir / self.cfg.paths.submission_filename
            # Best-effort materialization before binding submission to the version entry.
            if not (dest.exists() and dest.stat().st_size > 0):
                ensure_canonical_submission_copy(self)
            # If still missing, fallback to latest valid discovered submission (header-aware).
            if not (dest.exists() and dest.stat().st_size > 0):
                spec_obj: Dict[str, Any] = {}
                sp = root / self.cfg.paths.artifacts_dir / "spec.json"
                if sp.exists():
                    try:
                        loaded = json.loads(sp.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            spec_obj = loaded
                    except Exception:
                        spec_obj = {}
                found = _find_valid_submissions(self, spec_obj)
                if found:
                    try:
                        src = found[0][0]
                        self.write_file(str(dest.relative_to(root)), src.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        pass
            if dest.exists() and dest.stat().st_size > 0:
                sub_rel = str(dest.relative_to(root))
        except Exception:
            sub_rel = ""
        return _helpers_record_metrics_version(
            self, metrics, code_text, tag=tag, submission_path=sub_rel
        )

    def _metric_improved(self, new: Dict[str, Any], old: Dict[str, Any]) -> bool:
        try:
            nv = float(new.get("primary", 0))
            ov = float(old.get("primary", 0))
            maximize = bool(new.get("maximize", True))
            return (nv > ov) if maximize else (nv < ov)
        except (ValueError, TypeError):
            return False

    def _save_last_and_maybe_best(self, code_text: str, metrics: Dict[str, Any]) -> None:
        art = Path(self.cfg.paths.artifacts_dir)
        self.write_file(str(art / "last_code.py"), code_text)
        self.write_file(str(art / "metrics_last.json"), json.dumps(metrics, ensure_ascii=False, indent=2))
        best_m = {}
        best_path = Path(self.project_root) / art / "best_metrics.json"
        if best_path.exists():
            try:
                best_m = json.loads(best_path.read_text(encoding="utf-8"))
            except Exception:
                best_m = {}
        if not best_m or self._metric_improved(metrics, best_m):
            self.write_file(str(art / "best_metrics.json"), json.dumps(metrics, ensure_ascii=False, indent=2))
            self.write_file(str(art / "best_code.py"), code_text)

    def _read_best_or_last_metrics(self) -> dict:
        try:
            art = Path(self.project_root) / self.cfg.paths.artifacts_dir
            for p in (
                art / "best_metrics.json",       # written by optimizer.py
                art / "best" / "metrics.json",   # written by helpers._record_metrics_version
                art / "metrics_last.json",        # written by _save_last_and_maybe_best
                art / "last" / "metrics.json",   # written by pipeline / helpers
            ):
                if p.exists():
                    return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _tasks_state_relpath(self) -> str:
        return f"{self.cfg.paths.artifacts_dir}/tasks_tree.json"

    def _tree_relpath(self) -> str:
        return f"{self.cfg.paths.artifacts_dir}/tree.json"

    def _load_tree(self) -> Dict[str, Any]:
        p = self.dir_paths["artifacts"] / "tree.json"
        if not p.exists():
            # If the file doesn't exist, create it with a default structure
            default_tree = {
                "nodes": {},
                "roots": [],
                "last_active": "",
                "completed": False,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            self._save_tree(default_tree)
            return default_tree
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading tree.json: {e}. Re-initializing.")
            # If file is corrupted or empty, re-initialize
            default_tree = {
                "nodes": {},
                "roots": [],
                "last_active": "",
                "completed": False,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            self._save_tree(default_tree)
            return default_tree

    def _save_tree(self, tree: Dict[str, Any]) -> None:
        p = self.dir_paths["artifacts"] / "tree.json"
        try:
            tree["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            p.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
            self._update_markdown_plan()
        except (IOError, TypeError) as e:
            print(f"Critical error saving tree.json: {e}")

    def _update_markdown_plan(self) -> None:
        """
        Regenerate task_plan.md as the main project artifact:
        - Full tasks hierarchy and per-node overview (not truncated; tree.json remains canonical)
        - Artifacts layout and key files
        - Best/last metrics and recent experiments
        - Data/spec snapshot
        - Current direction / next steps (incl. latest aggregate summary if present)
        """
        try:
            from src.helpers import snapshot_data_tree
        except Exception:
            snapshot_data_tree = None  # type: ignore

        try:
            # Resolve paths
            root_dir = getattr(self, "project_root", Path("."))
            if isinstance(root_dir, str):
                root_dir = Path(root_dir)
            md_path = root_dir / "task_plan.md"
            tmp_path = root_dir / "task_plan.tmp"

            # -------- Tasks hierarchy & overview --------
            # Full graph text (no truncation): large trees still live in artifacts/tree.json as JSON.
            plan_text = self.format_task_graph_to_string()
            tree = self._load_tree()
            nodes: Dict[str, Any] = tree.get("nodes", {}) or {}

            active_node_id = str(tree.get("last_active") or "").strip()
            if active_node_id and active_node_id not in nodes:
                active_node_id = ""
            if not active_node_id and tree.get("roots"):
                # Use last root as a best-effort "current branch".
                try:
                    roots = list(tree.get("roots") or [])
                    active_node_id = str(roots[-1]) if roots else ""
                except Exception:
                    active_node_id = ""

            # Build an "active branch chain" (root -> ... -> active node).
            active_chain: list[str] = []
            if active_node_id and active_node_id in nodes:
                cur = active_node_id
                seen = set()
                while cur and cur in nodes and cur not in seen:
                    active_chain.append(cur)
                    seen.add(cur)
                    cur = nodes.get(cur, {}).get("parent_node_id")
                active_chain = list(reversed(active_chain))

            if not active_chain and nodes:
                # Fallback: show one node (most recent created_at).
                try:
                    nid = sorted(nodes.items(), key=lambda kv: kv[1].get("created_at", ""))[-1][0]
                    active_chain = [nid]
                except Exception:
                    active_chain = []

            # Single-line focus (avoid duplicating the hierarchy block above)
            focus_line = "No tasks recorded yet."
            if active_chain:
                tail = nodes.get(active_chain[-1], {}) or {}
                t_one = (tail.get("task") or "").strip().splitlines()[0] if (tail.get("task") or "").strip() else ""
                if len(t_one) > 120:
                    t_one = t_one[:117] + "..."
                focus_line = (
                    f"Active `{active_chain[-1]}` | kind={tail.get('kind', '')} | status={tail.get('status', '')} "
                    f"| depth~{len(active_chain) - 1} | title: {t_one}"
                )

            # -------- Artifacts & metrics summary --------
            art_dir = root_dir / self.cfg.paths.artifacts_dir
            artifacts_lines: list[str] = []

            def _flag(p: Path) -> str:
                return "✔" if p.exists() else "✖"

            spec_path = art_dir / "spec.json"
            spec_frozen_path = art_dir / "spec_frozen.json"
            best_metrics_path = art_dir / "best_metrics.json"
            last_metrics_path = art_dir / "metrics_last.json"
            versions_dir = art_dir / "versions"
            best_dir = art_dir / "best"
            last_dir = art_dir / "last"

            artifacts_lines.append(
                "**Purpose**: `spec.json` = problem+data paths; `best/`/`last/` = pinned code+metrics[+submission]; "
                "`versions/` = ranked experiment history; `improve/` = Improve loop state; `final/` = output gate reports."
            )
            tree_json = art_dir / "tree.json"
            events_jsonl = art_dir / "task_graph_events.jsonl"
            improve_resume = art_dir / "improve" / "pipeline_resume.json"
            artifacts_lines.append(f"{_flag(spec_path)} **spec.json** — task schema, data paths, metrics contract")
            artifacts_lines.append(f"{_flag(spec_frozen_path)} **spec_frozen.json** — Improve baseline spec snapshot")
            artifacts_lines.append(f"{_flag(tree_json)} **tree.json** — full task graph (single source of truth)")
            artifacts_lines.append(f"{_flag(events_jsonl)} **task_graph_events.jsonl** — PRUNED/replan/branch events")
            artifacts_lines.append(f"{_flag(improve_resume)} **improve/pipeline_resume.json** — Improver Head + iter cursor")
            artifacts_lines.append(f"{_flag(best_dir)} **best/** — best primary so far (metrics.json, code.py, submission.csv)")
            artifacts_lines.append(f"{_flag(last_dir)} **last/** — last successful run snapshot")
            artifacts_lines.append(f"{_flag(versions_dir)} **versions/** — index.json + ledger + per-run folders")

            # Use existing metrics helpers
            best_or_last = self._read_best_or_last_metrics()
            metrics_lines: list[str] = []
            best_submission_line = "N/A"
            if best_or_last:
                name = best_or_last.get("name", best_or_last.get("primary_metric", "")) or "primary"
                primary = best_or_last.get("primary") or best_or_last.get("primary_score")
                maximize = best_or_last.get("maximize", True)
                mtype = best_or_last.get("type", "calculated")
                metrics_lines.append("name | value | maximize | type")
                metrics_lines.append("---- | ----- | -------- | ----")
                metrics_lines.append(f"{name} | {primary} | {maximize} | {mtype}")
                ex = best_or_last.get("extras")
                if isinstance(ex, dict) and ex:
                    metrics_lines.append("")
                    metrics_lines.append("extras (diagnostics):")
                    for k, v in list(ex.items())[:24]:
                        metrics_lines.append(f"- {k}: {v}")
            else:
                metrics_lines.append("No validated metrics found yet.")

            try:
                best_sub = root_dir / self.cfg.paths.artifacts_dir / "best" / "submission.csv"
                canonical_sub = root_dir / self.cfg.paths.submission_dir / self.cfg.paths.submission_filename
                if canonical_sub.exists():
                    best_submission_line = str(canonical_sub)
                elif best_sub.exists():
                    best_submission_line = str(best_sub)
            except Exception:
                best_submission_line = "N/A"

            # Ensure versions ledger is refreshed to keep experiments table usable
            try:
                ver_info = self._refresh_versions_table()
            except Exception:
                ver_info = {}
            ledger_md_rel = ""
            try:
                ledger_md_rel = str(
                    (Path(ver_info.get("md", "")) if isinstance(ver_info, dict) else Path()).relative_to(root_dir)
                )
            except Exception:
                ledger_md_rel = ""

            # -------- Data & spec snapshot --------
            spec_snapshot_lines: list[str] = []
            spec_obj: Dict[str, Any] = {}
            try:
                if spec_path.exists():
                    spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
            except Exception:
                spec_obj = {}

            if spec_obj:
                primary_metric = spec_obj.get("primary_metric", {})
                modalities = spec_obj.get("modalities", [])
                validation = spec_obj.get("validation", {})
                data_info = spec_obj.get("data", {})

                spec_snapshot_lines.append(f"- **Modalities**: {modalities}")
                if primary_metric:
                    spec_snapshot_lines.append(
                        f"- **Primary metric**: {primary_metric.get('name')} (maximize={primary_metric.get('maximize')})"
                    )
                if validation:
                    spec_snapshot_lines.append(
                        f"- **Validation**: {validation.get('strategy')} (n_splits={validation.get('n_splits')})"
                    )
                if data_info:
                    root_hint = data_info.get("root_hint") or data_info.get("resolved_root") or ""
                    spec_snapshot_lines.append(
                        f"- **Data root**: {root_hint or data_info.get('resolved_root', '')}"
                    )
                    for key in ("train_csv", "labels_csv", "train_dir", "test_dir"):
                        if key in data_info:
                            spec_snapshot_lines.append(f"- **{key}**: {data_info[key]}")
            else:
                spec_snapshot_lines.append("Spec not initialized yet (spec.json missing or unreadable).")

            # -------- Project context & logs --------
            # Full content, no truncation — project_context.md is already section-capped at source
            # (see update_project_context_after_execution in pipeline.py). Cutting it again here
            # loses acceptance criteria and data passports that agents need.
            project_context_lines: list[str] = []
            try:
                ctx_path = self.dir_paths["artifacts"] / "project_context.md"
                if ctx_path.exists():
                    txt = ctx_path.read_text(encoding="utf-8")
                    project_context_lines.append(txt.strip())
                else:
                    project_context_lines.append("No project_context.md yet.")
            except Exception:
                project_context_lines.append("Failed to read project_context.md.")

            log_overview_lines: list[str] = [
                "Human-readable timeline: see `PROJECT_LOG.md` in project root (not duplicated here to save tokens)."
            ]

            # -------- Artifacts tree snapshot (optional, best-effort) --------
            artifacts_tree_text = "No artifacts tree available."
            data_tree_text = "No data tree available."
            if snapshot_data_tree is not None:
                try:
                    artifacts_tree_text = snapshot_data_tree(
                        str(root_dir),
                        data_dirname=self.cfg.paths.artifacts_dir,
                        max_files=45,
                        exclude_logs=True,
                    )
                    artifacts_tree_text = (
                        "(Log files and `logs/` dirs omitted from this listing; see `PROJECT_LOG.md` / `ml_project/logs/` on disk.)\n"
                        + artifacts_tree_text
                    )
                except Exception:
                    pass
                try:
                    data_tree_text = snapshot_data_tree(
                        str(root_dir), data_dirname=self.cfg.paths.data_dir, max_files=35
                    )
                except Exception:
                    pass

            # -------- Decision memory (state attempts) --------
            decision_memory_lines: list[str] = []
            graph_events_lines: list[str] = []
            try:
                if active_node_id:
                    st = self._load_node_state(active_node_id) or {}
                    attempts = st.get("attempts", []) if isinstance(st, dict) else []
                    attempts = attempts[-4:] if isinstance(attempts, list) else []

                    decision_memory_lines.append(f"Active node: `{active_node_id}`")
                    if active_node_id in nodes:
                        n = nodes.get(active_node_id, {}) or {}
                        status = n.get("status", "")
                        kind = n.get("kind", "")
                        # Full first line of the task (no mid-string truncation)
                        task_title = (n.get("task") or "").strip().splitlines()[0] if (n.get("task") or "").strip() else ""
                        decision_memory_lines.append(f"- kind={kind}, status={status}, title={task_title}")

                    if attempts:
                        decision_memory_lines.append("\nLast attempts (newest last):")
                        for a in attempts:
                            if not isinstance(a, dict):
                                continue
                            ph = a.get("phase", "")
                            r = a.get("route", "")
                            reason = str(a.get("reason", ""))
                            exit_code = a.get("exit_code", "")
                            decision_memory_lines.append(
                                f"- phase={ph} route={r} exit={exit_code} reason={reason}"
                            )
                    else:
                        decision_memory_lines.append("- No attempts recorded yet.")
                else:
                    decision_memory_lines.append("Active node is not set in tree.json.")
            except Exception:
                decision_memory_lines.append("Failed to build decision memory from state.")

            # -------- Graph events (latest) --------
            try:
                events_path = self.dir_paths["artifacts"] / "task_graph_events.jsonl"
                if events_path.exists():
                    lines = [ln for ln in events_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    tail = lines[-40:] if len(lines) > 40 else lines
                    for ln in tail:
                        try:
                            ev = json.loads(ln)
                            ts = str(ev.get("ts", "")).replace("T", " ").replace("Z", "").strip()
                            event = ev.get("event", "")
                            # Full task and reason — truncation hides the replan/prune rationale
                            task = str(ev.get("task", "") or "").replace("\n", " ")
                            nid = ev.get("node_id", "")
                            reason = str(ev.get("reason", "") or "").replace("\n", " ")
                            graph_events_lines.append(f"{ts} | {event} | node={nid} | {task} | {reason}")
                        except Exception:
                            continue
            except Exception:
                pass

            # -------- Current direction / next steps --------
            current_dir_lines: list[str] = []
            # 1) Try latest aggregate summary file from pipeline — full content, no truncation.
            # This summary is the improver's main context entry; cutting it hides stack + results.
            agg_path = art_dir / "aggregate_summary.md"
            if agg_path.exists():
                try:
                    agg_txt = agg_path.read_text(encoding="utf-8")
                    current_dir_lines.append(agg_txt.strip())
                except Exception:
                    pass

            # 2) Fallback: brief hint from metrics
            if not current_dir_lines:
                if best_or_last:
                    current_dir_lines.append(
                        f"Best known metric: {best_or_last.get('primary')} ({best_or_last.get('name', 'primary')})."
                    )
                else:
                    current_dir_lines.append("No aggregate summary yet; run main pipeline to populate.")

            # -------- Assemble markdown content --------
            parts: list[str] = []
            parts.append("# Project Execution Plan")
            parts.append(
                "\n_Snapshot for the next LLM step. Long timelines: `PROJECT_LOG.md` (repo root). "
                "Improve replanner + head leave traces in `artifacts/improve/pipeline_resume.json` and `task_graph_events.jsonl`._\n"
            )

            parts.append("\n## 1. Task hierarchy\n")
            parts.append("### Tree (source of truth: `artifacts/tree.json`)\n")
            parts.append("```text")
            parts.append(plan_text or "Task tree is empty.")
            parts.append("```")
            parts.append("\n### Where you are now\n")
            parts.append(focus_line)

            parts.append("\n## 2. Artifacts Structure\n")
            parts.append("### 2.1 Key artifact locations\n")
            parts.append("\n".join(f"- {line}" for line in artifacts_lines))

            parts.append("\n### 2.2 Artifacts file tree (snapshot)\n")
            parts.append("```text")
            parts.append(artifacts_tree_text)
            parts.append("```")
            parts.append("\n### 2.3 Data file tree (snapshot)\n")
            parts.append("```text")
            parts.append(data_tree_text)
            parts.append("```")

            # -------- Canonical paths table (agent reference) --------
            # Agents MUST use these paths — do not guess or hardcode alternatives.
            canon_lines: list[str] = []
            canon_lines.append("| path | exists | description |")
            canon_lines.append("| ---- | ------ | ----------- |")
            _art = self.cfg.paths.artifacts_dir
            _canon_files = [
                (f"{_art}/spec.json",                    "Task schema, data paths, metrics contract"),
                (f"{_art}/best/metrics.json",            "Best validated metrics (primary score)"),
                (f"{_art}/best/code.py",                 "Best solution code"),
                (f"{_art}/best/submission.csv",          "Best submission file"),
                (f"{_art}/last/metrics.json",            "Last run metrics"),
                (f"{_art}/last/code.py",                 "Last run code"),
                (f"{_art}/versions/index.json",          "Full ranked experiment history"),
                (f"{_art}/versions/ledger.md",           "Human-readable experiments ledger"),
            ]
            try:
                sub_canon = str(
                    Path(self.cfg.paths.submission_dir) / self.cfg.paths.submission_filename
                )
                _canon_files.append((sub_canon, "Canonical submission output (gate output)"))
            except Exception:
                pass
            for rel, desc in _canon_files:
                exists = "✔" if (root_dir / rel).exists() else "✖"
                canon_lines.append(f"| `{rel}` | {exists} | {desc} |")
            parts.append("\n### 2.4 Canonical artifact paths (agent reference)\n")
            parts.append(
                "> **Agents**: read your paths from this table — do **not** assume flat "
                "`best_metrics.json` or `metrics_last.json` (those are optimizer-only). "
                "Use `artifacts/best/metrics.json` and `artifacts/last/metrics.json`.\n"
            )
            parts.append("\n".join(canon_lines))

            parts.append("\n## 3. Metrics & Experiments\n")
            parts.append(f"- Current best metric: `{best_or_last.get('primary') if best_or_last else 'N/A'}`")
            parts.append(f"- Current best submission: `{best_submission_line}`")
            parts.append("\n".join(metrics_lines))
            if ledger_md_rel:
                parts.append(f"\n- Full experiments ledger: `{ledger_md_rel}`")

            parts.append("\n## 4. Data & Spec Snapshot\n")
            parts.append("\n".join(spec_snapshot_lines))

            parts.append("\n## 5. Context (LLM-oriented, compact)\n")
            parts.append("### 5.1 project_context.md (tail)\n")
            parts.append("```markdown")
            parts.append("\n".join(project_context_lines))
            parts.append("```")

            parts.append("\n### 5.2 Project log pointer\n")
            parts.append("\n".join(log_overview_lines))

            parts.append("\n### 5.3 Decision memory (active node)\n")
            parts.append("```markdown")
            parts.append("\n".join(decision_memory_lines))
            parts.append("```")

            if graph_events_lines:
                parts.append("\n### 5.4 Graph events (latest)\n")
                parts.append("```markdown")
                parts.append("\n".join(graph_events_lines))
                parts.append("```")

            parts.append("\n## 6. Current Direction & Next Steps\n")
            parts.append("\n".join(current_dir_lines))

            content = "\n".join(parts) + "\n"

            # Atomic write: tmp then replace
            try:
                tmp_path.write_text(content, encoding="utf-8")
                tmp_path.replace(md_path)
            except Exception:
                # Fallback to direct write
                md_path.write_text(content, encoding="utf-8")
        except Exception:
            # Do not let task graph persistence fail due to plan generation
            pass

    def tree_node(self, node_id: str) -> Dict[str, Any]:
        t = self._load_tree()
        return dict(t.get("nodes", {}).get(node_id) or {})

    def tree_node_status(self, node_id: str) -> str:
        n = self.tree_node(node_id)
        return str(n.get("status", "pending"))

    def tree_get_roots(self) -> List[str]:
        t = self._load_tree()
        return list(t.get("roots", []) or [])

    def tree_ensure_node(
            self,
            node_id: Optional[str],
            parent_node_id: Optional[str],
            *,
            kind: str,
            task: str = ""
    ) -> str:
        t = self._load_tree()
        nodes = t.setdefault("nodes", {})
        roots = t.setdefault("roots", [])
        now = datetime.datetime.utcnow().isoformat() + "Z"

        # Recover stale parent references after resume/replan.
        if parent_node_id is not None and parent_node_id not in nodes:
            fallback_root = self.tree_find_most_recent_root()
            parent_node_id = fallback_root if fallback_root in nodes else None

        # Calculate depth
        parent_depth = -1
        if parent_node_id and parent_node_id in nodes:
            parent_depth = nodes[parent_node_id].get("depth", 0)
        depth = parent_depth + 1

        if node_id and node_id in nodes:
            n = nodes[node_id]
            if kind:
                n["kind"] = kind
            if task:
                n["task"] = task
            n["depth"] = depth  # Update depth
            if parent_node_id is not None:
                if parent_node_id not in nodes:
                    parent_node_id = None
                if n.get("parent_node_id") != parent_node_id:
                    old_pid = n.get("parent_node_id")
                    if old_pid and old_pid in nodes:
                        try:
                            nodes[old_pid]["children"] = [c for c in nodes[old_pid].get("children", []) if c != node_id]
                        except Exception:
                            pass
                    n["parent_node_id"] = parent_node_id
                if node_id not in nodes[parent_node_id].setdefault("children", []):
                    nodes[parent_node_id]["children"].append(node_id)
                if node_id in roots:
                    roots.remove(node_id)
            else:
                n["parent_node_id"] = None
                if node_id not in roots:
                    roots.append(node_id)
            self._save_tree(t)
            return node_id

        if not node_id:
            node_id = f"node-{uuid.uuid4().hex[:10]}"

        if parent_node_id is not None and parent_node_id not in nodes:
            parent_node_id = None

        nodes[node_id] = {
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "children": [],
            "kind": kind,
            "task": task or "",
            "status": "pending",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "depth": depth,  # Add depth to new node
            "meta": {},
            "replan_count": 0,
        }

        if parent_node_id:
            if node_id not in nodes[parent_node_id].setdefault("children", []):
                nodes[parent_node_id]["children"].append(node_id)
        else:
            if node_id not in roots:
                roots.append(node_id)

        self._save_tree(t)
        return node_id
    
    def tree_increment_replan_count(self, node_id: str) -> int:
        """Increments the replan counter for a node and returns the new count."""
        t = self._load_tree()
        nodes = t.get("nodes", {})
        node = nodes.get(node_id)
        if not node:
            return 0
        
        current_count = node.get("replan_count", 0)
        new_count = current_count + 1
        node["replan_count"] = new_count
        self._save_tree(t)
        return new_count

    def tree_start(self, node_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        t = self._load_tree()
        n = t.get("nodes", {}).get(node_id)
        if not n:
            return
        n["status"] = "running"
        n["started_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        if meta:
            try:
                n["meta"] = {**(n.get("meta") or {}), **meta}
            except Exception:
                pass
        t["last_active"] = node_id
        self._save_tree(t)

    def tree_finish(self, node_id: str, status: str = "done", meta: Optional[Dict[str, Any]] = None) -> None:
        if status not in ("done", "failed", "skipped"): # Allow 'skipped'
            status = "done"
        t = self._load_tree()
        n = t.get("nodes", {}).get(node_id)
        if not n:
            return
        n["status"] = status
        n["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        if meta:
            try:
                # Update task description with technical artifact passport if provided
                art_sum = meta.get("artifact_summary", "")
                if art_sum and isinstance(art_sum, str):
                    orig_task = n.get("task", "")
                    if art_sum not in orig_task: # Avoid double append
                        n["task"] = f"{orig_task}\n\n[Technical Specs]:\n{art_sum}"
                
                n["meta"] = {**(n.get("meta") or {}), **meta}
            except Exception:
                pass
        try:
            all_done = all(nn.get("status") in ("done", "failed", "skipped") for nn in t.get("nodes", {}).values())
            t["completed"] = bool(all_done)
        except Exception:
            pass
        self._save_tree(t)
        if status == "skipped":
            try:
                self.tree_log_event(
                    "SKIPPED",
                    str(n.get("task", "")),
                    node_id=node_id,
                    parent_node_id=n.get("parent_node_id"),
                    reason=str((meta or {}).get("reason", "")),
                )
            except Exception:
                pass

    def tree_remove_node(self, node_id: str) -> None:
        """Safely removes a node from the tree and its parent's children list."""
        t = self._load_tree()
        nodes = t.get("nodes", {})
        if node_id not in nodes:
            return
        
        node = nodes[node_id]
        parent_id = node.get("parent_node_id")
        
        if parent_id and parent_id in nodes:
            parent = nodes[parent_id]
            if node_id in parent.get("children", []):
                parent["children"].remove(node_id)
        
        if node_id in t.get("roots", []):
            t["roots"].remove(node_id)
            
        del nodes[node_id]
        self._save_tree(t)

    def tree_set_failed(self, node_id: str, error: str = "", meta: Optional[Dict[str, Any]] = None) -> None:
        mm = {"error": shorten_string_middle(error, 1000)}
        if meta:
            mm.update(meta)
        self.tree_finish(node_id, status="failed", meta=mm)

    def tree_update_meta(self, node_id: str, meta: Dict[str, Any]) -> None:
        """Merge meta into an existing node (best-effort)."""
        if not node_id:
            return
        try:
            t = self._load_tree()
            n = t.get("nodes", {}).get(node_id)
            if not n:
                return
            cur = n.get("meta") or {}
            if not isinstance(cur, dict):
                cur = {}
            n["meta"] = {**cur, **(meta or {})}
            self._save_tree(t)
        except Exception:
            pass

    def tree_children_ordered(self, parent_id: str) -> list[dict]:
        t = self._load_tree()
        return _get_children_ordered(t.get("nodes", {}), parent_id)

    def tree_init_children_with_kinds(self,
                                      parent_id: str,
                                      tasks: list[str],
                                      kinds: list[str] | None = None) -> list[str]:
        assert isinstance(parent_id, str) and parent_id, "parent_id required"
        kinds = kinds or ["subtask"] * len(tasks)

        t = self._load_tree()
        nodes = t.setdefault("nodes", {})
        roots = t.setdefault("roots", [])

        if parent_id not in nodes:
            # Recover from stale/missing parent IDs by attaching to the most recent real root.
            fallback_root = self.tree_find_most_recent_root()
            if fallback_root and fallback_root in nodes and fallback_root != parent_id:
                parent_id = fallback_root
                parent_depth = nodes[parent_id].get("depth", 0)
            else:
                parent_depth = -1
                nodes[parent_id] = {
                    "node_id": parent_id,
                    "parent_node_id": None,
                    "children": [],
                    "kind": "main",
                    "task": "Recovered Root",
                    "status": "pending",
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "started_at": None,
                    "finished_at": None,
                    "depth": 0,
                    "meta": {},
                    "replan_count": 0,
                }
                if parent_id not in roots:
                    roots.append(parent_id)
        else:
            parent_depth = nodes[parent_id].get("depth", 0)

        parent = nodes[parent_id]
        existing_ids = list(parent.get("children", []))
        existing_by_task: dict[str, str] = {nodes[c]["task"]: c for c in existing_ids if c in nodes}

        out_ids: list[str] = []
        now = datetime.datetime.utcnow().isoformat() + "Z"
        child_depth = parent_depth + 1

        for idx, (task, kind) in enumerate(zip(tasks, kinds)):
            if task in existing_by_task:
                cid = existing_by_task[task]
                ch = nodes[cid]
                if ch.get("kind") != kind:
                    ch["kind"] = kind
                ch["order"] = idx
                ch["depth"] = child_depth
                ch["updated_at"] = now
                if cid not in parent["children"]:
                    parent["children"].append(cid)
                out_ids.append(cid)
                continue
            cid = f"node-{uuid.uuid4().hex[:10]}"
            nodes[cid] = {
                "node_id": cid,
                "parent_node_id": parent_id,
                "children": [],
                "kind": kind,
                "task": str(task),
                "status": "pending",
                "order": idx,
                "depth": child_depth,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "meta": {},
                "replan_count": 0,
            }
            parent.setdefault("children", []).append(cid)
            out_ids.append(cid)

        ch = _get_children_ordered(nodes, parent_id)
        parent["children"] = [n["node_id"] for n in ch]
        t["updated_at"] = now
        self._save_tree(t)
        return out_ids

    def tree_pick_next_node(self, prefer_kind: Optional[str] = None) -> Optional[str]:
        t = self._load_tree()
        nodes = t.get("nodes", {})
        if not nodes:
            return None

        root_id = self._tree_find_active_root() or self.tree_find_most_recent_root()
        if not root_id:
            return None

        queue = collections.deque([root_id])
        visited = {root_id}

        while queue:
            node_id = queue.popleft()
            node = nodes.get(node_id)

            if not node:
                continue

            if prefer_kind and node.get("kind") != prefer_kind:
                if node.get("status") == "done":
                    children = _get_children_ordered(nodes, node_id)
                    for child in children:
                        cid = child["node_id"]
                        if cid not in visited:
                            visited.add(cid)
                            queue.append(cid)
                continue

            status = node.get("status")

            if status in ("pending", "running"):
                return node_id

            # Interrupted runs mark nodes failed; resume should retry this node, not skip it.
            if status == "failed":
                return node_id

            if status == "done":
                children = _get_children_ordered(nodes, node_id)
                for child in children:
                    cid = child["node_id"]
                    if cid not in visited:
                        visited.add(cid)
                        queue.append(cid)

        return None

    def _tree_find_active_root(self) -> Optional[str]:
        t = self._load_tree()
        nodes = t.get("nodes", {})
        if not nodes:
            return None

        active_roots = []
        for root_id in t.get("roots", []):
            root_node = nodes.get(root_id)
            if root_node and root_node.get("status") not in ("done", "failed"):
                updated_at = root_node.get("finished_at") or root_node.get("started_at") or root_node.get(
                    "created_at") or ""
                active_roots.append((updated_at, root_id))

        if not active_roots:
            return None

        active_roots.sort(key=lambda x: x[0], reverse=True)
        return active_roots[0][1]

    def tree_find_most_recent_root(self) -> Optional[str]:
        t = self._load_tree()
        roots = t.get("roots", [])
        if not roots:
            return None
        nodes = t.get("nodes", {})

        def get_ts(nid):
            return nodes.get(nid, {}).get("created_at", "")

        roots_sorted = sorted(roots, key=get_ts, reverse=True)
        return roots_sorted[0]

    def tree_deepest_resume_target(self, root_id: str) -> Optional[str]:
        """
        For --resume: choose one node under root_id to continue from.
        Prefer the **deepest** `failed` node; if none, the deepest `pending` or `running`.
        Same-depth tie-break: latest finished_at / updated_at, then node_id (stable).
        """
        t = self._load_tree()
        nodes: Dict[str, Any] = t.get("nodes") or {}
        if not root_id or root_id not in nodes:
            return None

        failed_rows: list[tuple[int, str, str]] = []
        active_rows: list[tuple[int, str, str]] = []

        def walk(nid: str) -> None:
            n = nodes.get(nid)
            if not n:
                return
            try:
                depth = int(n.get("depth") or 0)
            except (TypeError, ValueError):
                depth = 0
            st = str(n.get("status") or "")
            ts = str(
                n.get("finished_at")
                or n.get("updated_at")
                or n.get("started_at")
                or n.get("created_at")
                or ""
            )
            if st == "failed":
                failed_rows.append((depth, ts, nid))
            elif st in ("pending", "running"):
                active_rows.append((depth, ts, nid))
            for ch in _get_children_ordered(nodes, nid):
                walk(ch["node_id"])

        walk(root_id)

        def pick(rows: list[tuple[int, str, str]]) -> Optional[str]:
            if not rows:
                return None
            rows.sort(key=lambda x: (-x[0], x[1], x[2]))
            return rows[0][2]

        got = pick(failed_rows)
        if got:
            return got
        return pick(active_rows)

    def tree_subtree_has_unfinished_work(self, root_id: str) -> bool:
        """True if any node under root_id is failed, pending, or running."""
        t = self._load_tree()
        nodes: Dict[str, Any] = t.get("nodes") or {}
        if not root_id or root_id not in nodes:
            return False

        def walk(nid: str) -> bool:
            n = nodes.get(nid)
            if not n:
                return False
            st = str(n.get("status") or "")
            if st in ("failed", "pending", "running"):
                return True
            for ch in _get_children_ordered(nodes, nid):
                if walk(ch["node_id"]):
                    return True
            return False

        return walk(root_id)

    def tree_sanitize_running_tasks(self) -> None:
        t = self._load_tree()
        nodes = t.get("nodes", {})
        changed = False
        for nid, n in nodes.items():
            if n.get("status") == "running":
                n["status"] = "failed"
                n["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
                meta = n.get("meta", {})
                meta["error"] = "Interrupted/Resumed from running state"
                n["meta"] = meta
                changed = True
        if changed:
            self._save_tree(t)

    def tree_get_node_context(self, node_id: str) -> Dict[str, str]:
        node = self.tree_node(node_id)
        meta = node.get("meta", {})

        context = {
            "code": "",
            "output": ""
        }

        script_file = meta.get("script_file")
        if script_file:
            try:
                p = Path(script_file)
                if not p.is_absolute():
                    p = self.project_root / p
                if p.exists():
                    context["code"] = p.read_text(encoding="utf-8")
            except Exception:
                pass

        if "task" in meta and isinstance(meta["task"], str) and len(meta["task"]) > 10:
            context["output"] = meta["task"]

        stdout_file = meta.get("stdout_file")
        if not context["output"] and stdout_file:
            try:
                p = Path(stdout_file)
                if not p.is_absolute():
                    p = self.project_root / p
                if p.exists():
                    full_text = p.read_text(encoding="utf-8")
                    context["output"] = full_text[-2000:]
            except Exception:
                pass

        return context
    def get_tree_metrics(self) -> Dict[str, int]:
        """Algorithmic calculation of tree metrics: depth, width, and total nodes."""
        tree = self._load_tree()
        nodes = tree.get("nodes", {})
        roots = tree.get("roots", [])
        if not nodes or not roots:
            return {"max_depth": 0, "max_width": 0, "total_nodes": 0}

        max_depth = 0
        
        # Calculate depth
        for root_id in roots:
            q = collections.deque([(root_id, 1)])
            visited = {root_id}
            while q:
                nid, depth = q.popleft()
                max_depth = max(max_depth, depth)
                node = nodes.get(nid, {})
                for child_id in node.get("children", []):
                    if child_id not in visited:
                        visited.add(child_id)
                        q.append((child_id, depth + 1))

        # Calculate width
        max_width = 0
        for root_id in roots:
            q = collections.deque([root_id])
            while q:
                level_size = len(q)
                max_width = max(max_width, level_size)
                for _ in range(level_size):
                    nid = q.popleft()
                    node = nodes.get(nid, {})
                    for child_id in node.get("children", []):
                        q.append(child_id)
        
        return {
            "max_depth": max_depth,
            "max_width": max_width,
            "total_nodes": len(nodes)
        }


def _safe_ts(s: str | None) -> str:
    return s or ""


def _status_done_like(st: str) -> bool:
    return st in ("done", "failed")


def _status_pending_like(st: str) -> bool:
    return st in ("pending", "running")


def _child_sort_key(n: dict) -> tuple:
    return (int(n.get("order", 10 ** 9)), _safe_ts(n.get("created_at")))


def _get_children_ordered(nodes: dict, parent_id: str) -> list[dict]:
    parent = nodes.get(parent_id, {})
    ids = list(parent.get("children", []))
    out = [nodes[c] for c in ids if c in nodes]
    out.sort(key=_child_sort_key)
    return out
