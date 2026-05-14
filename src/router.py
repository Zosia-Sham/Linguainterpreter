from __future__ import annotations
import os
import re
import json
import hashlib
from typing import Dict, Any, Optional, Callable, List

from .prompts_agents import error_triage_agent
from .parsers import extract_json
from .llm_utils import _invoke_with_retry, _log_to_jsonl, _log_token_usage, invoke_with_tools, log_agent_trace

try:
    from langchain_core.tools import StructuredTool
except Exception:
    StructuredTool = None

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatPromptTemplate = None

try:
    from langchain_google_community import GoogleSearchAPIWrapper
except Exception:
    GoogleSearchAPIWrapper = None


CUDA_PROBE_PY = r"""
import json
try:
  import torch
  info = {
    "torch_version": getattr(torch, "__version__", None),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device_name_0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
  }
except Exception as e:
  info = {"torch_version": None, "cuda_available": False, "error": str(e)}
print("BOOTSTRAP_CUDA_PROBE_JSON:", json.dumps(info))
"""


def build_pip_react_tools(orch: Any, search_func: Optional[Callable[..., str]] = None) -> List[Any]:
    """Real tools for install ReAct: pip in project venv + torch CUDA probe (+ optional Google)."""
    if not orch or StructuredTool is None:
        return []

    def pip_install(packages: str, pip_extra: str = "") -> str:
        """Install packages into the project venv via pip. `packages`: comma- or space-separated names."""
        raw = (packages or "").strip()
        pkgs = [p.strip() for p in re.split(r"[\s,]+", raw) if p.strip()]
        if not pkgs:
            return "error: empty packages list"
        extra = (pip_extra or "").strip()
        res = orch.pip_install(pkgs, extra=extra)
        ec = res.get("exit_code", 1)
        if ec is None:
            ec = 1
        err = (res.get("stderr") or "")[:4000]
        out = (res.get("stdout") or "")[:2000]
        return f"exit_code={ec}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

    def cuda_probe() -> str:
        """Run torch CUDA probe in the same venv as training scripts."""
        res = orch.run_python_code(CUDA_PROBE_PY, filename="probe_cuda.py", timeout=90)
        o = (res.get("output") or "") + "\n" + (res.get("errors") or "")
        return o[:8000]

    tools: List[Any] = [
        StructuredTool.from_function(
            pip_install,
            name="pip_install",
            description="Install Python packages with pip in the project virtualenv. "
            "Pass comma-separated names (e.g. 'torch,torchvision' or 'pandas'). "
            "Optional pip_extra: e.g. '--index-url https://download.pytorch.org/whl/cu124' for CUDA wheels.",
        ),
        StructuredTool.from_function(
            cuda_probe,
            name="cuda_probe",
            description="Check whether PyTorch sees CUDA in this venv (torch.cuda.is_available()). Call after installing torch.",
        ),
    ]
    if search_func:
        def google_search(query: str) -> str:
            try:
                return str(search_func(query))[:6000]
            except Exception as e:
                return f"search error: {e}"

        tools.append(
            StructuredTool.from_function(
                google_search,
                name="google_search",
                description="Search the web for pip package names, PyTorch CUDA wheel URLs, or error fixes.",
            )
        )
    return tools


def repair_torch_cuda_with_react(llm_fast: Any, orch: Any, cfg: Any) -> bool:
    """
    If PyTorch is CPU-only in venv but user configured CUDA index + torch packages, run a short ReAct loop
    to reinstall torch* with the right --index-url and verify cuda_probe.
    Returns True if cuda_probe reports cuda_available=True after attempts.
    """
    if not (llm_fast and orch and StructuredTool and ChatPromptTemplate):
        return False
    pre = getattr(cfg, "preinstall", None) or {}
    if not getattr(pre, "enable", True):
        return False
    index_url = (getattr(pre, "torch_cuda_index_url", "") or "").strip()
    pkgs_cfg = [str(p).strip() for p in (getattr(pre, "pkgs", []) or []) if str(p).strip()]
    torch_roots = ("torch", "torchvision", "torchaudio")
    needs_torch = any(any(p.lower().startswith(t) for t in torch_roots) for p in pkgs_cfg)
    if not index_url and not needs_torch:
        return False

    tools = build_pip_react_tools(orch, None)
    if not tools:
        return False

    extra = f"--index-url {index_url}" if index_url else ""
    torch_line = " ".join([p for p in pkgs_cfg if any(p.lower().startswith(t) for t in torch_roots)]) or "torch torchvision torchaudio"

    sys_prompt = (
        "You repair PyTorch GPU support in the project venv. The host may have an NVIDIA GPU (nvidia-smi) but "
        "torch.cuda.is_available() is False — usually a CPU-only wheel or wrong CUDA build.\n"
        "Tools: pip_install (real pip), cuda_probe (checks torch.cuda.is_available()).\n"
        "Strategy: call pip_install with CUDA torch packages and pip_extra if an index URL is given, then cuda_probe. "
        "You may retry pip_install once with adjusted packages if pip fails.\n"
        "Do not use bash. Stop when cuda_probe shows cuda_available true or after 2 pip attempts.\n"
        "Finally reply with ONE JSON object only: {\"ok\": true/false, \"reason\": \"...\"}."
    )
    user_prompt = (
        f"Configured torch_cuda_index_url: {index_url or '(none)'}\n"
        f"Suggested packages from config: {torch_line}\n"
        f"pip_extra to prefer: {extra or '(none)'}\n"
        "Call tools, then output JSON."
    )
    prompt = ChatPromptTemplate.from_messages([("system", "{sys_prompt}"), ("user", "{user_prompt}")])
    try:
        res = invoke_with_tools(
            llm_fast,
            prompt,
            {"sys_prompt": sys_prompt, "user_prompt": user_prompt},
            tools=tools,
            agent_name="bootstrap_cuda_react",
        )
        txt = getattr(res, "content", "") or ""
        log_agent_trace("bootstrap_cuda_react", "final", txt[:12000])
    except Exception as ex:
        log_agent_trace("bootstrap_cuda_react", "error", str(ex))
    probe_out = orch.run_python_code(CUDA_PROBE_PY, filename="probe_cuda.py", timeout=90)
    out = (probe_out.get("output") or "") + (probe_out.get("errors") or "")
    return bool(re.search(r'"cuda_available"\s*:\s*true', out, re.I))


class ErrorRouter:
    """
    LLM-роутер ошибок исполнения.

    Итоговый план:
    {
      "route": "install" | "bash" | "coding" | "lead" | "spec_update",
      "packages": ["..."],
      "bash_cmds": ["..."],
      "reason": "short text",
      "notes": "optional",
      "pip_extra": "--index-url https://...",
      "spec_patch": { ... }   # Comment translated to English.
    }
    """

    def __init__(
        self,
        llm_fast,
        os_name: str = "Windows",
        repeat_to_lead: int = 2,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        google_search: Optional[callable] = None,  # Comment translated to English.
        orch: Any = None,
    ):
        self.llm_fast = llm_fast
        self.os_name = os_name
        self._orch = orch

        # Comment translated to English.
        self._last_sig: str | None = None
        self._repeats: int = 0
        self._consec_spec_updates: int = 0
        self._last_spec_update_reason: str = ""
        self._repeat_to_lead: int = repeat_to_lead

        # Comment translated to English.
        self._last_route: str | None = None
        self._route_repeats: int = 0

        # Comment translated to English.
        self._last_patch_sig: str | None = None
        self._patch_repeats: int = 0

        # Per-package install failure tracking.
        # After _max_install_fails failures for the same package, stop retrying and route to coding.
        self._install_fail_counts: Dict[str, int] = {}
        self._max_install_fails: int = 3

        # Comment translated to English.
        self._search_func = google_search
        if self._search_func is None and GoogleSearchAPIWrapper is not None:
            api = google_api_key or os.getenv("GOOGLE_API_KEY", "")
            cse = google_cse_id or os.getenv("GOOGLE_CSE_ID", "") or os.getenv("GOOGLE_CSE_ID".lower(), "")
            if api and cse:
                os.environ.setdefault("GOOGLE_API_KEY", api)
                os.environ.setdefault("GOOGLE_CSE_ID", cse)
                try:
                    wrapper = GoogleSearchAPIWrapper()
                    self._search_func = wrapper.run
                except Exception:
                    self._search_func = None

    @staticmethod
    def _sig(text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _derive_query(stderr: str) -> str:
        """Примитивная эвристика для запроса в поиск, если агент не сработает."""
        s = (stderr or "").strip()
        if not s:
            return "python error"
        line = s.splitlines()[0][:200]
        # Comment translated to English.
        import re
        m = re.search(r"No module named ['\"]([a-zA-Z0-9_\.]+)['\"]", s)
        if m:
            mod = m.group(1)
            return f"pip install {mod}"
        return f"{line} pip install"

    @staticmethod
    def _normalize_package_name(name: str) -> str:
        n = (name or "").strip()
        if not n:
            return ""
        return n

    @staticmethod
    def _extract_missing_package(stderr: str, stdout: str) -> str:
        """
        Try to detect missing dependency and map it to a pip package name.
        Priority: explicit module errors -> common alias/runtime hints.
        """
        import re

        text = f"{stderr or ''}\n{stdout or ''}"

        # Canonical python missing-module forms
        m = re.search(r"No module named ['\"]([a-zA-Z0-9_\.]+)['\"]", text)
        if not m:
            m = re.search(r"ModuleNotFoundError:\s*No module named ['\"]([a-zA-Z0-9_\.]+)['\"]", text)
        if m:
            mod = (m.group(1) or "").strip()
            if mod:
                # usually top-level package is install target
                top = mod.split(".")[0]
                return ErrorRouter._normalize_package_name(top)

        lo = text.lower()
        # Common runtime hints that indicate missing pandas despite alias error
        if "pandas is not installed" in lo or "name 'pd' is not defined" in lo:
            return "pandas"

        return ""

    def _agent_hints(self, stderr: str, stdout: str, spec: Dict[str, Any], code_text: str) -> Dict[str, Any]:
        """
        ReAct with real tools: pip_install (venv), cuda_probe (torch.cuda), optional google_search.
        LangGraph not required; uses invoke_with_tools + StructuredTool.
        """
        if not self.llm_fast or not ChatPromptTemplate:
            return {}
        tools = build_pip_react_tools(self._orch, self._search_func)
        if not tools:
            return {}

        sys_instr = (
            "You are a build/run failure triage assistant with REAL tools.\n"
            "- pip_install: install packages in the project virtualenv (use for missing modules, torch CUDA wheels, etc.).\n"
            "- cuda_probe: run torch CUDA check in that venv.\n"
            "- google_search: optional web hints.\n"
            "If the log shows training on CPU but the machine has a GPU, call cuda_probe first; if cuda is false, "
            "use pip_install with torch torchvision torchaudio and pip_extra like --index-url https://download.pytorch.org/whl/cu124 "
            "when the user config suggests CUDA (do not guess URLs; prefer search or stderr hints).\n"
            "When finished, reply with ONLY ONE JSON object: "
            '{"packages": [], "pip_extra": "", "bash_cmds": [], "spec_patch": {}, "reason": ""}. '
            "List packages you already installed via pip_install in this session too. No prose outside JSON."
        )
        user_msg = (
            f"OS: {self.os_name}\n"
            f"STDERR:\n{stderr or 'N/A'}\n\n"
            f"STDOUT (tail):\n{(stdout or '')[-800:]}\n\n"
            f"SPEC.data:\n{json.dumps(spec.get('data', {}), ensure_ascii=False)}\n\n"
            f"SPEC.hardware (gpu):\n{json.dumps(spec.get('hardware', {}), ensure_ascii=False)[:2000]}\n\n"
            f"CODE (head):\n{(code_text or '')[:800]}\n"
        )
        prompt = ChatPromptTemplate.from_messages([("system", "{sys_prompt}"), ("user", "{user_prompt}")])
        try:
            res = invoke_with_tools(
                self.llm_fast,
                prompt,
                {"sys_prompt": sys_instr, "user_prompt": user_msg},
                tools=tools,
                agent_name="pip_install_react",
            )
            content = getattr(res, "content", None) or ""
            if not isinstance(content, str):
                return {}
            _log_to_jsonl("pip_install_react", user_msg[:2000], content[:8000])
            try:
                _log_token_usage("pip_install_react", res)
            except Exception:
                pass
            hints = extract_json(content) or {}
            if isinstance(hints, dict):
                hints.setdefault("packages", [])
                hints.setdefault("bash_cmds", [])
                hints.setdefault("pip_extra", "")
                hints.setdefault("spec_patch", {})
                hints.setdefault("reason", "")
                return hints
        except Exception as ex:
            log_agent_trace("pip_install_react", "error", str(ex))
        return {}

    def route(self, stderr: str, stdout: str, spec: Dict[str, Any], code_text: str) -> Dict[str, Any]:
        # Comment translated to English.
        sig = self._sig(stderr)
        if sig == self._last_sig:
            self._repeats += 1
        else:
            self._last_sig, self._repeats = sig, 0

        # Comment translated to English.
        resp = error_triage_agent(
            self.llm_fast, stderr=stderr, stdout=stdout, spec=spec, code_text=code_text, os_name=self.os_name,
            consecutive_spec_updates=self._consec_spec_updates,
            last_spec_update_reason=self._last_spec_update_reason,
        )
        plan = extract_json(resp) if isinstance(resp, str) else resp
        if not isinstance(plan, dict):
            plan = {}

        # Comment translated to English.
        route = plan.get("route", "coding")
        if route == "meta":
            route = "spec_update"
        plan["route"] = route
        plan.setdefault("packages", [])
        plan.setdefault("bash_cmds", [])
        plan.setdefault("reason", "")
        plan.setdefault("notes", "")
        if "pip_extra" in plan and not isinstance(plan["pip_extra"], str):
            plan["pip_extra"] = str(plan["pip_extra"])

        # Route decision is handled primarily by the ERROR_TRIAGE prompt logic.
        # Below we add a few hard guardrails for critical cases (timeouts/OOM).

        # 3.1) Hard guardrail: dependency-missing errors must route to install first.
        missing_pkg = self._extract_missing_package(stderr or "", stdout or "")
        dependency_missing = bool(missing_pkg)
        if dependency_missing:
            pkg_key = (missing_pkg or "").lower().strip()
            fail_count = self._install_fail_counts.get(pkg_key, 0)
            if fail_count >= self._max_install_fails:
                # Package has failed to install too many times — stop retrying.
                # Route to coding so the agent can use an alternative library.
                plan["route"] = "coding"
                plan["reason"] = (
                    f"Package '{missing_pkg}' failed to install after {fail_count} attempts "
                    f"(likely unavailable in this environment). "
                    f"Rewrite code to avoid this dependency — use a pre-installed alternative."
                )
                plan["notes"] = (
                    (plan.get("notes") or "") +
                    f" | install_hard_blocked: '{missing_pkg}' exceeded max retries ({self._max_install_fails})"
                ).strip()
                dependency_missing = False  # do not force install route below
            else:
                plan["route"] = "install"
                packages = plan.get("packages") or []
                if missing_pkg not in packages:
                    packages = [missing_pkg] + list(packages)
                # keep order, dedupe
                seen = set()
                norm = [self._normalize_package_name(p) for p in packages]
                plan["packages"] = [p for p in norm if p and not (p in seen or seen.add(p))]
                if not plan.get("reason"):
                    plan["reason"] = f"Missing dependency detected: {missing_pkg}"

        # Comment translated to English.
        if route == self._last_route:
            self._route_repeats += 1
        else:
            self._last_route = route
            self._route_repeats = 0

        patch_sig = None
        if route == "spec_update":
            try:
                patch_sig = self._sig(json.dumps(plan.get("spec_patch", {}), sort_keys=True, ensure_ascii=False))
            except Exception:
                patch_sig = self._sig(str(plan.get("spec_patch", "")))
            if patch_sig == self._last_patch_sig:
                self._patch_repeats += 1
            else:
                self._last_patch_sig = patch_sig
                self._patch_repeats = 0
        else:
            self._last_patch_sig = None
            self._patch_repeats = 0

        # Defense-in-depth: break spec_update loops at router level
        if route == "spec_update" and self._route_repeats >= 2:
            plan["route"] = "coding"
            plan["reason"] = (
                (plan.get("reason") or "")
                + f" | router forced coding: spec_update repeated {self._route_repeats + 1} times"
            ).strip()
            route = "coding"
            self._route_repeats = 0

        # Comment translated to English.
        need_hints = False
        # Comment translated to English.
        if route == "install" and not plan.get("packages"):
            need_hints = True
        # Comment translated to English.
        if "ModuleNotFoundError" in (stderr or "") or "No module named" in (stderr or ""):
            need_hints = True
        # Comment translated to English.
        if self._repeats >= 1 and route in ("install", "coding", "bash"):
            need_hints = True

        if need_hints:
            hints = self._agent_hints(stderr=stderr, stdout=stdout, spec=spec, code_text=code_text)
            if isinstance(hints, dict) and hints:
                # Comment translated to English.
                pkgs = (plan.get("packages") or []) + (hints.get("packages") or [])
                # Comment translated to English.
                seen = {}
                norm_pkgs = [self._normalize_package_name(p) for p in pkgs]
                plan["packages"] = [seen.setdefault(p.lower(), p) for p in norm_pkgs if p and p.lower() not in seen][:4]
                if hints.get("pip_extra") and not plan.get("pip_extra"):
                    plan["pip_extra"] = hints["pip_extra"]
                if hints.get("bash_cmds"):
                    # Robust merge: guard against None from LLM
                    existing_cmds = plan.get("bash_cmds") or []
                    hint_cmds = hints.get("bash_cmds") or []
                    if not isinstance(existing_cmds, list):
                        existing_cmds = [existing_cmds]
                    if not isinstance(hint_cmds, list):
                        hint_cmds = [hint_cmds]
                    plan["bash_cmds"] = existing_cmds + hint_cmds
                if route != "spec_update" and hints.get("spec_patch"):
                    # Comment translated to English.
                    plan["notes"] = (plan.get("notes", "") + " | hint: spec_patch suggested").strip()
                if hints.get("reason"):
                    plan["reason"] = plan.get("reason") or hints["reason"]

            # Comment translated to English.
            if not plan.get("packages"):
                q = self._derive_query(stderr or "")
                plan["notes"] = (plan.get("notes", "") + f" | search_hint: {q}").strip()
                
                # Comment translated to English.
                if ("ModuleNotFoundError" in (stderr or "") or "No module named" in (stderr or "")) and "pip install" in q:
                    # Comment translated to English.
                    import re
                    m = re.search(r"pip install ([a-zA-Z0-9_\-\.]+)", q)
                    if m:
                        pkg_name = self._normalize_package_name(m.group(1))
                        plan["route"] = "install"
                        plan["packages"] = [pkg_name]
                        plan["reason"] = f"Auto-detected missing module: {pkg_name}"

        # 3.3) Hard guardrail: resource/timeout errors should prefer spec_update route.
        lo_err = (stderr or "").lower()
        if any(
            key in lo_err
            for key in [
                "timeoutexpired",
                "timed out",
                "cuda out of memory",
                "out of memory",
                "killed",
            ]
        ):
            if route not in ("install", "lead"):
                plan["route"] = "spec_update"
                notes = plan.get("notes", "") or ""
                plan["notes"] = (notes + " | auto spec_update due to timeout/resource error").strip()
                # Ensure spec_patch exists so caller can adjust config (batch_size, epochs, model, etc.).
                plan.setdefault("spec_patch", {})

        # Comment translated to English.
        should_escalate = False
        if self._repeats >= self._repeat_to_lead:
            should_escalate = True
        if self._route_repeats >= self._repeat_to_lead:
            should_escalate = True
        if route == "spec_update" and self._patch_repeats >= self._repeat_to_lead:
            should_escalate = True

        # Never escalate missing-dependency errors to lead before install is attempted.
        if should_escalate and route != "lead" and not dependency_missing:
            plan["route"] = "lead"
            notes = plan.get("notes", "") or ''
            plan["notes"] = (notes + " | escalated after repeated same error/route/patch").strip()

        # Track consecutive spec_updates — reset on any other final route, increment on spec_update.
        final_route = plan.get("route", "coding")
        if final_route == "spec_update":
            # Defense-in-depth: if we already did 2 spec_updates in a row, flip to coding.
            if self._consec_spec_updates >= 2:
                plan["route"] = "coding"
                plan["notes"] = (plan.get("notes", "") + " | forced coding: 2+ consecutive spec_updates").strip()
                self._consec_spec_updates = 0
                self._last_spec_update_reason = ""
            else:
                self._consec_spec_updates += 1
                self._last_spec_update_reason = str(plan.get("reason", ""))[:400]
        else:
            self._consec_spec_updates = 0
            self._last_spec_update_reason = ""

        return plan
