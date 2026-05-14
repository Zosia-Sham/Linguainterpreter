from __future__ import annotations
import os, subprocess, threading, time
from typing import Any, Dict, Optional, Callable, List
from pathlib import Path

from .utils import detect_os, shorten_string_middle
from .watcher_tools import WARNING_HEADER_RE, is_warning_header, WatcherCtx

_CHUNK_PRINT_LIMIT = 4000  # Comment translated to English.

def _safe_print(text: str, is_err: bool = False, printer: Optional[Callable[[str], None]] = None) -> None:
    """Печатает короткими кусками, безопасно для Win-консоли."""
    if printer is None:
        printer = print
    for i in range(0, len(text), _CHUNK_PRINT_LIMIT):
        try:
            printer(text[i:i+_CHUNK_PRINT_LIMIT], end="")
        except OSError:
            # Comment translated to English.
            try:
                s = text[i:i+_CHUNK_PRINT_LIMIT].encode("utf-8", "replace").decode("utf-8", "replace")
                printer(s, end="")
            except Exception:
                break

class BashAgent:
    def __init__(self, workdir: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                 min_exec_timeout_sec: int = 180, predictive_buffer_pct: int = 50):
        info = detect_os()
        self.os = info["os"]                 # "Windows" | "Linux" | "Darwin"
        self.shell = info["shell"]           # "powershell" | "bash"
        self.python_exec = info["python_exec"]
        self.workdir = workdir or os.getcwd()
        self._min_exec_timeout_sec = max(10, min_exec_timeout_sec)
        # Predictive buffer: pred_timeout = expected * (1 + pct/100). Default 50% → 1.5x.
        self._predictive_buffer_pct = max(0, int(predictive_buffer_pct))
        self.env = os.environ.copy()
        if env:
            self.env.update(env)

        # Comment translated to English.
        self.env.setdefault("PYTHONUTF8", "1")
        self.env.setdefault("PYTHONIOENCODING", "utf-8")
        self.env.setdefault("NO_COLOR", "1")
        if self.os == "Windows":
            # Comment translated to English.
            self.env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "1")

        self._logs_dir = Path(self.workdir) / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    def _wrap_command(self, command: str) -> List[str]:
        if self.shell == "powershell":
            # Comment translated to English.
            prelude = (
                "$ErrorActionPreference='Continue'; "
                "$enc=[System.Text.UTF8Encoding]::new(); "
                "$OutputEncoding=$enc; [Console]::OutputEncoding=$enc; "
            )
            return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", prelude + command]
        else:
            return ["bash", "-lc", command]

    def _reader_lines(self, pipe, sink_list: List[str], prefix: Optional[str],
                      tee_file, is_err: bool, printer: Optional[Callable[[str], None]],
                      warn_state: Optional[Dict[str, Any]] = None):
        """
        Стримим построчно. Python warning headers (``file.py:line: XWarning: …``)
        и их отступные continuation-строки выкидываем полностью — они не
        попадают ни в sink_list (то, что видит агент), ни в tee-лог, ни в
        консоль. Ведём счётчик подавленных строк и периодически печатаем
        одну итоговую строку.
        """
        # Shared mutable state between stdout/stderr readers (suppressed count).
        ws = warn_state if warn_state is not None else {
            "suppressed": 0, "last_report_ts": time.time(),
            "lock": threading.Lock(), "in_warning_block": False,
        }
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break

                # Warning filter: drop the header line AND its indented
                # continuation (traceback-style) lines that follow.
                stripped = line.rstrip("\n")
                is_header = is_warning_header(stripped)
                # A continuation line is non-empty, starts with whitespace,
                # and follows a header. Plain indented normal output isn't
                # normally interleaved one-per-line with warnings, so this
                # is safe in practice.
                is_continuation = False
                # NOTE: each reader has its own _in_warning_block via ws;
                # we keep a per-pipe flag to avoid stdout/stderr crosstalk.
                pipe_flag_key = "in_warning_block_err" if is_err else "in_warning_block_out"
                in_block = ws.get(pipe_flag_key, False)
                if is_header:
                    ws[pipe_flag_key] = True
                    is_continuation = False  # header itself
                elif in_block and stripped and (stripped[0].isspace()):
                    is_continuation = True
                else:
                    # Any non-indented, non-empty line closes the block.
                    if stripped and not stripped[0].isspace():
                        ws[pipe_flag_key] = False

                if is_header or is_continuation:
                    with ws["lock"]:
                        ws["suppressed"] += 1
                        now = time.time()
                        # One-line summary at most every 30s so users know
                        # warnings are being stripped and the process IS alive.
                        if now - ws["last_report_ts"] >= 30.0 and ws["suppressed"] > 0:
                            msg = f"[MONITOR] suppressed {ws['suppressed']} warning lines\n"
                            try:
                                _safe_print(msg, is_err=False, printer=printer)
                            except Exception:
                                pass
                            if tee_file:
                                try:
                                    tee_file.write(msg)
                                    tee_file.flush()
                                except Exception:
                                    pass
                            ws["last_report_ts"] = now
                    continue  # drop from sink / tee / stdout

                sink_list.append(line)
                if tee_file:
                    try:
                        tee_file.write(line)
                        tee_file.flush()
                    except Exception:
                        pass
                s = (prefix or "") + line
                _safe_print(s, is_err=is_err, printer=printer)
        except Exception as e:
            _safe_print(f"\n[stream-error] {e}\n", is_err=True, printer=printer)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def run(
        self,
        command: str,
        timeout: int = 3600,
        workdir: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        stream: bool = False,
        prefix: Optional[str] = None,
        tee_logfile: Optional[str] = None,
        printer: Optional[Callable[[str], None]] = print,
        monitor_llm: Any = None,
        spec: Dict[str, Any] = None,
        prediction: Dict[str, Any] = None,
        watcher_ctx_extras: Optional[Dict[str, Any]] = None,
        hard_cap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Если stream=False — буферно (capture_output).
        Если stream=True  — печать в реальном времени, плюс буферим stdout/stderr в память и файл.
        monitor_llm: optional LLM to watch for infinite loops/failures in real-time.
        prediction: optional dict from execution_predictor_agent.
        watcher_ctx_extras: optional {"task_text": str, "code": str} for the ReAct watcher.
        hard_cap: absolute ceiling in seconds (e.g. run-wide remaining budget).
                  Predictor and cascade target are clamped by this value.
        """
        if not isinstance(command, str) or not command.strip():
            return {"stdout": "", "stderr": "Empty command", "exit_code": 1}

        # --- Timeout policy ---------------------------------------------------
        # Two distinct knobs:
        #   * cascade_cap (=`timeout`): SOFT TARGET from the caller / per-task budget.
        #     The predictor may expand beyond it when it estimates the step needs more.
        #   * hard_cap: ABSOLUTE CEILING (run-wide remaining budget). Never exceeded.
        #
        # Target = max(cascade_cap, pred_timeout)   # honour predictor when bigger
        # Effective = max(1, min(hard_cap, max(floor, target)))
        #
        # Rationale: previously the formula was `min(cascade_cap, max(floor, pred))`,
        # which silently truncated heavy ML steps (Optuna, transformer training)
        # whose predictor estimate (1500–4200s) was capped down to the per-task
        # budget (e.g. 280s) and the subprocess died with TimeoutExpired.
        floor = self._min_exec_timeout_sec
        cascade_cap = None
        if timeout is not None:
            try:
                cascade_cap = int(timeout)
            except Exception:
                cascade_cap = None

        hcap_int: Optional[int] = None
        if hard_cap is not None:
            try:
                hcap_int = int(hard_cap)
            except Exception:
                hcap_int = None

        pred_timeout = 0
        pred_expected = None
        if prediction and "expected_time_sec" in prediction:
            try:
                pred_expected = int(prediction["expected_time_sec"])
                pred_timeout = int(pred_expected * (1 + self._predictive_buffer_pct / 100.0))
            except Exception:
                pred_timeout = 0

        # Build the soft target: take whichever is larger of cascade_cap and pred.
        if cascade_cap is not None and cascade_cap > 0 and pred_timeout > 0:
            target = max(cascade_cap, pred_timeout)
        elif pred_timeout > 0:
            target = pred_timeout
        elif cascade_cap is not None:
            target = cascade_cap
        else:
            target = floor

        # Floor first, then clamp by absolute hard_cap (if provided).
        floored = max(floor, target)
        if hcap_int is not None and hcap_int > 0:
            effective_timeout = max(1, min(hcap_int, floored))
        else:
            effective_timeout = floored

        if pred_expected is not None or hcap_int is not None:
            expanded = (
                pred_timeout > 0
                and cascade_cap is not None
                and pred_timeout > cascade_cap
                and effective_timeout > cascade_cap
            )
            tag = " (predictor expanded)" if expanded else ""
            print(
                f"[MONITOR] Predictive target: expected={pred_expected}s, "
                f"buffered={pred_timeout}s | cascade_cap={cascade_cap}s, "
                f"hard_cap={hcap_int}s, floor={floor}s "
                f"→ effective_timeout={effective_timeout}s{tag}"
            )

        env = self.env.copy()
        if extra_env:
            env.update(extra_env)

        # Optional: Clean up SOCKS proxy if it's breaking pip
        # We only want http/https proxies, but if SOCKS is missing dependencies, we drop proxy env
        if "socks" in env.get("HTTP_PROXY", "").lower() or "socks" in env.get("http_proxy", "").lower():
             env.pop("HTTP_PROXY", None)
             env.pop("http_proxy", None)
        if "socks" in env.get("HTTPS_PROXY", "").lower() or "socks" in env.get("https_proxy", "").lower():
             env.pop("HTTPS_PROXY", None)
             env.pop("https_proxy", None)

        cwd = workdir or self.workdir
        cmd = self._wrap_command(command)

        # Comment translated to English.
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = self._logs_dir / (Path(tee_logfile).name if tee_logfile else f"run-{ts}.log")
        log_fh = None
        try:
            log_fh = open(log_path, "w", encoding="utf-8", newline="")
        except Exception:
            log_fh = None

        if not stream:
            try:
                p = subprocess.run(
                    cmd, cwd=cwd, env=env,
                    capture_output=True, text=True,
                    timeout=effective_timeout,
                    encoding="utf-8", errors="replace",
                )
                if log_fh:
                    try:
                        if p.stdout:
                            log_fh.write(p.stdout)
                        if p.stderr:
                            log_fh.write("\n[stderr]\n")
                            log_fh.write(p.stderr)
                    except Exception:
                        pass
                    finally:
                        try: log_fh.close()
                        except Exception: pass
                return {
                    "stdout": shorten_string_middle(p.stdout or "", 70000),
                    "stderr": shorten_string_middle(p.stderr or "", 20000),
                    "exit_code": p.returncode,
                    "log_path": str(log_path),
                }
            except subprocess.TimeoutExpired as e:
                if log_fh:
                    try:
                        log_fh.write(f"\n[TimeoutExpired after {effective_timeout}s]\n")
                    except Exception:
                        pass
                    finally:
                        try: log_fh.close()
                        except Exception: pass
                return {"stdout": "", "stderr": f"TimeoutExpired: {e}", "exit_code": 124, "log_path": str(log_path)}
            except Exception as e:
                if log_fh:
                    try: log_fh.close()
                    except Exception: pass
                return {"stdout": "", "stderr": f"Exception: {e}", "exit_code": 1, "log_path": str(log_path)}

        # stream=True
        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            if log_fh:
                try: log_fh.close()
                except Exception: pass
            return {"stdout": "", "stderr": f"Exception: {e}", "exit_code": 1}

        out_buf: List[str] = []
        err_buf: List[str] = []
        # Shared warning-filter state so both readers count into one pool.
        warn_state: Dict[str, Any] = {
            "suppressed": 0, "last_report_ts": time.time(),
            "lock": threading.Lock(),
        }
        t_out = threading.Thread(target=self._reader_lines, args=(p.stdout, out_buf, prefix, log_fh, False, printer, warn_state), daemon=True)
        t_err = threading.Thread(target=self._reader_lines, args=(p.stderr, err_buf, prefix, log_fh, True, printer, warn_state), daemon=True)
        t_out.start(); t_err.start()

        # Monitoring thread
        stop_event = threading.Event()
        kill_reason = ""
        _monitor_start = time.time()

        def monitor_process():
            nonlocal kill_reason
            from .prompts_agents import execution_watcher_agent
            last_checked_len = 0
            last_progress_ts = time.time()
            # Idle timeout: long silence → treat as hang.
            idle_limit = max(60, min(effective_timeout, int(getattr(self, "env", {}).get("BASH_IDLE_LIMIT_SEC", effective_timeout) or effective_timeout)))

            # Build WatcherCtx once — its getters always see the live buffers.
            extras = watcher_ctx_extras or {}
            _task_text = str(extras.get("task_text", "") or "")[:8000]
            _code_text = str(extras.get("code", "") or "")[:16000]
            _pred = prediction or {}

            def _get_stdout():
                return "".join(out_buf)

            def _get_stderr():
                return "".join(err_buf)

            def _get_timing():
                el = time.time() - _monitor_start
                _extra = float((spec or {}).get("_extra_budget_sec", 0) or 0)
                return {
                    "elapsed_sec": round(el, 1),
                    "effective_timeout_sec": int(effective_timeout),
                    "cascade_cap_sec": int(cascade_cap) if cascade_cap is not None else None,
                    "floor_sec": int(floor),
                    "extra_budget_sec": _extra,
                    "suppressed_warning_lines": int(warn_state.get("suppressed", 0)),
                }

            ctx = WatcherCtx(
                get_stdout=_get_stdout,
                get_stderr=_get_stderr,
                task_text=_task_text,
                code=_code_text,
                prediction=_pred,
                get_timing=_get_timing,
                pid=getattr(p, "pid", None),
            )

            while not stop_event.is_set() and p.poll() is None:
                time.sleep(30)  # Check every 30 seconds
                elapsed_in_task = time.time() - _monitor_start
                current_out = "".join(out_buf)
                current_err = "".join(err_buf)

                if len(current_out) + len(current_err) == last_checked_len:
                    # No new output, but process is still running.
                    # Check for hard hangs / deadlocks based on silence duration.
                    if time.time() - last_progress_ts >= idle_limit:
                        kill_reason = f"no output for {int(idle_limit)}s (idle timeout)"
                        print(f"\n[MONITOR] Terminating process due to idle timeout: {kill_reason}")
                        try:
                            p.terminate()
                        except Exception:
                            try:
                                p.kill()
                            except Exception:
                                pass
                        break

                if len(current_out) + len(current_err) > last_checked_len:
                    last_progress_ts = time.time()

                if monitor_llm:
                    try:
                        res = execution_watcher_agent(monitor_llm, ctx)
                    except Exception as e:
                        print(f"\n[MONITOR] watcher error (ignored): {e}")
                        res = {"status": "normal", "action": "continue", "reason": f"watcher error: {e}"}
                    action = res.get("action", "continue")
                    status = res.get("status", "normal")
                    if action == "kill" or status == "overtime":
                        kill_reason = res.get("reason", f"Killed by watcher (status={status})")
                        print(f"\n[MONITOR] Terminating process: {kill_reason}")
                        try:
                            p.terminate()
                        except Exception:
                            pass
                        break
                    elif action == "warn":
                        print(f"\n[MONITOR] Watcher warning: {res.get('reason', '')} (elapsed={elapsed_in_task:.0f}s/{effective_timeout}s)")

                last_checked_len = len(current_out) + len(current_err)

        t_mon = None
        if monitor_llm:
            t_mon = threading.Thread(target=monitor_process, daemon=True)
            t_mon.start()

        try:
            rc = p.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired as e:
            try:
                p.kill()
            except Exception:
                pass
            rc = 124
            err_buf.append(f"TimeoutExpired: {e}\n")
            if log_fh:
                try:
                    log_fh.write(f"\n[TimeoutExpired after {effective_timeout}s]\n")
                except Exception:
                    pass
        finally:
            stop_event.set()
            if t_mon: t_mon.join(timeout=1.0)
            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)
            if kill_reason:
                err_buf.append(f"\nKILLED_BY_MONITOR: {kill_reason}\n")
            # Final suppressed-warnings summary so the run log shows the total.
            sup = int(warn_state.get("suppressed", 0))
            if sup > 0 and log_fh:
                try:
                    log_fh.write(f"\n[MONITOR] total suppressed warning lines: {sup}\n")
                except Exception:
                    pass
            if log_fh:
                try: log_fh.close()
                except Exception: pass

        return {
            "stdout": shorten_string_middle("".join(out_buf), 70000),
            "stderr": shorten_string_middle("".join(err_buf), 20000),
            "exit_code": rc,
            "log_path": str(log_path),
        }
