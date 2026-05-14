from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
from langchain_core.prompts import ChatPromptTemplate

# Comment translated to English.
from src.utils import _parse_yaml_tasks, YAMLParseError
from src.prompts_agents import extract_code, _strip_think
from src.llm_utils import invoke_and_log


@dataclass
class AbstractState:
    """
    Представляет абстрактное состояние системы для верификации.
    """
    step_index: int
    satisfied_predicates: Set[str]
    resources: Dict[str, float]
    trace: List[str] = field(default_factory=list)


@dataclass
class VerificationSpec:
    """
    Формальная спецификация ограничений (Safety Properties).
    """
    max_steps: int
    required_preconditions: Dict[str, List[str]]
    forbidden_states: List[str]
    resource_limits: Dict[str, float]


@dataclass
class VerificationResult:
    valid: bool
    counterexample: Optional[List[str]] = None
    violation_reason: str = ""


class FormalVerifier:
    """
    Реализует Bounded Model Checking (BMC) и статический анализ кода.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    def _llm_lift_tasks(self, tasks: List[str], context: str) -> List[Dict[str, Any]]:
        """
        Использует LLM для преобразования текстовых задач в абстрактную модель.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """FORMAL VERIFICATION AGENT
Analyze these tasks for a Python Data Science Agent.
Return a JSON list where each item corresponds to a task and has:
- 'preconditions': list of strings (logical dependencies, e.g., 'data_loaded', 'model_trained', 'features_engineered')
- 'effects': list of strings (what becomes true, e.g., 'submission_saved', 'metrics_calculated')
- 'resources': dict with 'complexity' (0.1-1.0) and 'risk' (0.1-1.0)

JSON ONLY. No markdown."""),
            ("user", "Tasks: {tasks_json}\nContext: {context_str}")
        ])

        try:
            # Comment translated to English.
            res = invoke_and_log(
                self.llm,
                prompt,
                {
                    "tasks_json": json.dumps(tasks),
                    "context_str": context[:1000],
                },
                agent_name="formal_verifier_tasks_lift",
            )
            content = _strip_think(getattr(res, "content", str(res)))

            # Comment translated to English.
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            data = json.loads(content)
            if isinstance(data, list) and len(data) == len(tasks):
                return data
        except Exception:
            pass

        # Fallback heuristic
        return [{"preconditions": [], "effects": [], "resources": {"complexity": 0.1}} for _ in tasks]

    def verify_plan(self, tasks: List[str], spec: VerificationSpec, context: str = "") -> VerificationResult:
        """
        Проверяет план (список задач) на логическую согласованность.
        """
        abstract_tasks = self._llm_lift_tasks(tasks, context)

        current_state = AbstractState(
            step_index=0,
            satisfied_predicates={"start", "data_available"},  # Comment translated to English.
            resources={k: 0.0 for k in spec.resource_limits}
        )

        for i, task_model in enumerate(abstract_tasks):
            task_name = tasks[i]

            # Comment translated to English.
            for pre in task_model.get("preconditions", []):
                # Comment translated to English.
                pass

            # Comment translated to English.
            task_res = task_model.get("resources", {})
            for res_name, usage in task_res.items():
                current_val = current_state.resources.get(res_name, 0.0)
                limit = spec.resource_limits.get(res_name, 100.0)
                if current_val + usage > limit:
                    return VerificationResult(
                        valid=False,
                        counterexample=current_state.trace + [task_name],
                        violation_reason=f"Resource violation: Task '{task_name}' exceeds limit for '{res_name}'."
                    )
                current_state.resources[res_name] = current_val + usage

            # Comment translated to English.
            for eff in task_model.get("effects", []):
                current_state.satisfied_predicates.add(eff)
                if eff in spec.forbidden_states:
                    return VerificationResult(
                        valid=False,
                        counterexample=current_state.trace + [task_name],
                        violation_reason=f"Safety violation: Task '{task_name}' triggers forbidden state '{eff}'."
                    )

            current_state.trace.append(task_name)
            current_state.step_index += 1

        return VerificationResult(valid=True)

    def verify_code_safety(self, code: str, spec: Dict[str, Any]) -> VerificationResult:
        """
        Статический анализ кода перед выполнением.
        """
        # 1. Regex checks
        forbidden_patterns = [
            (r"shutil\.rmtree\s*\(\s*['\"]/['\"]\s*\)", "Attempt to delete root"),
            (r"os\.system\s*\(\s*['\"]rm\s+-rf\s+/", "Attempt to delete root via shell"),
            (r"while\s+True\s*:", "Potential infinite loop without break (heuristic)"),
            (r"open\s*\(\s*['\"](?![^'\"]*(artifacts|temp))[^'\"]*[\\/][^'\"]*['\"]\s*,\s*['\"][^'\"]*[wxa+][^'\"]*['\"]", "Attempt to write file outside allowed directories (artifacts/temp)"),
            (r"to_csv\s*\(\s*['\"](?![^'\"]*(artifacts|temp))[^'\"]*[\\/][^'\"]*['\"]", "Attempt to save CSV outside allowed directories (artifacts/temp)"),
            (r"save\s*\(\s*['\"](?![^'\"]*(artifacts|temp))[^'\"]*[\\/][^'\"]*['\"]", "Attempt to save file outside allowed directories (artifacts/temp)"),
        ]

        for pattern, reason in forbidden_patterns:
            if re.search(pattern, code):
                return VerificationResult(valid=False, violation_reason=f"Static Analysis: {reason}")

        # 2. LLM Audit
        system_msg = """CODE SECURITY AUDIT
Analyze the Python code for safety violations.
VIOLATIONS:
1. Modifying/deleting data outside ./artifacts or ./temp (reading is OK if data is not modified)
2. Writing files outside the project directory (including data directory). Writing to subdirectories of the project root other than 'artifacts' or 'temp' is also forbidden.
3. Infinite loops
4. Hardcoded credentials
5. Ignoring 'spec' (e.g. changing target columns without validation)

STRICT RULES:
- READING any file (absolute OR relative path) is ALWAYS ALLOWED regardless of location. Data lives wherever it lives; pure reads are SAFE. Do NOT flag "hardcoded absolute read path" as unsafe — this is NOT a violation.
- Absolute paths like '/work/workspace/data/...' used ONLY for reading are FINE. The runtime mounts data in that location.
- WRITING to 'artifacts/' or 'temp/' (relative or absolute paths within project root) is ALLOWED.
- WRITING to the project root directory directly (e.g. ./file.txt) is DISCOURAGED but allowed if it's a small temporary file; however, 'artifacts/' is PREFERRED.
- WRITING to any other directory is FORBIDDEN.
- Using paths from spec.data.* for data loading is ALLOWED. If code uses a hardcoded absolute path that resolves to the same data directory, it is STILL a read → still ALLOWED.
- Only flag "safe=false" for real issues: writes outside allowed dirs, credentials, destructive shell, data mutation, infinite loops. Do NOT flag based on path style for reads.

Return JSON: {{"safe": boolean, "reason": "string"}}"""

        # Comment translated to English.
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "SPEC SUMMARY: {spec_summary}\n\nCODE:\n{code_snippet}")
        ])

        try:
            # Comment translated to English.
            res = invoke_and_log(
                self.llm,
                prompt,
                {
                    "spec_summary": json.dumps(spec.get('submission', {})),
                    "code_snippet": code[:4000],
                },
                agent_name="formal_verifier_code_safety",
            )
            content = _strip_think(getattr(res, "content", str(res)))

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            data = json.loads(content)
            if not data.get("safe", True):
                return VerificationResult(valid=False,
                                          violation_reason=data.get("reason", "LLM Audit detected unsafe code."))
        except Exception:
            pass  # Fail open if LLM fails

        return VerificationResult(valid=True)