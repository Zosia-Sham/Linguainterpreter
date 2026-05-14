"""Regression: reviewer prompt JSON example must not add spurious LangChain variables."""
from langchain_core.prompts import ChatPromptTemplate


def test_reviewer_prompt_variables_match_invoke_keys():
    reviewer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are REVIEWER_AGENT.\n"
                "Return ONLY JSON: {{\"pass\": true|false, \"issues\": [\"...\"], \"must_fix\": [\"...\"]}}.",
            ),
            (
                "user",
                "SUBTASK:\n{subtask}\n\nSPEC:\n{spec}\n\nCONTEXT:\n{ctx}\n\nCANDIDATE_CODE:\n```python\n{code}\n```",
            ),
        ]
    )
    assert set(reviewer_prompt.input_variables) == {"subtask", "spec", "ctx", "code"}
    reviewer_prompt.format_messages(
        subtask="t", spec={}, ctx="c", code="print(1)"
    )


def test_verification_code_gen_prompt_only_ctx():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "SYS"),
            (
                "user",
                "CONTEXT:\n{ctx}\n\n"
                "SPEC_SOURCE_JSON: {{\"spec_path\":\"<path>\",\"loaded\":true|false}}\n",
            ),
        ]
    )
    assert set(prompt.input_variables) == {"ctx"}
    prompt.format_messages(ctx="N/A")


def test_verification_code_safety_prompt_no_spurious_vars():
    """verification.py verify_code_safety — JSON example must not declare {safe} as variable."""
    system_msg = """CODE SECURITY AUDIT
Return JSON: {{"safe": boolean, "reason": "string"}}"""
    p = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("user", "SPEC SUMMARY: {spec_summary}\n\nCODE:\n{code_snippet}"),
        ]
    )
    assert set(p.input_variables) == {"spec_summary", "code_snippet"}


def test_helpers_react_final_submission_sys_escaping():
    sys = (
        "You are a ReAct final-submission repair agent.\n"
        "Return strict JSON: {{\"ok\": bool, \"used_code_rel\": str, \"summary\": str, \"last_errors\": [str]}}"
    )
    p = ChatPromptTemplate.from_messages(
        [
            ("system", sys),
            (
                "user",
                "TASK:\n{task}\n\nCANDIDATE_TAG:\n{candidate_tag}\n\n"
                "CURRENT_CODE_REL:\n{code_rel}\n\nMAX_ROUNDS:\n{max_rounds}",
            ),
        ]
    )
    assert set(p.input_variables) == {
        "task",
        "candidate_tag",
        "code_rel",
        "max_rounds",
    }


def test_orchestrator_pip_resolver_sys_escaping():
    sys = (
        "You are a Python packaging expert. "
        'return ONLY JSON: {{"replacements": {{"bad_pkg": "good_pkg"}}, "reason": "..."}}.'
    )
    p = ChatPromptTemplate.from_messages(
        [
            ("system", sys),
            ("user", "Requested packages:\n{requested}\n\nPip error:\n{err}\n\nExtra."),
        ]
    )
    assert set(p.input_variables) == {"requested", "err"}


def test_optimizer_proposal_sys_escaping():
    sys = """You are a Senior ML Tech Lead.
Return ONLY JSON:
{{
  "improvements": [
    {{
      "title": "short",
      "rationale": "why",
      "hint": "hint",
      "risk": "low|medium",
      "allow_stack_switch": false
    }}
  ]
}}"""
    p = ChatPromptTemplate.from_messages(
        [
            ("system", sys),
            ("user", "SPEC:\n{spec}\n\nMETRICS_JSON:\n{metrics}\n\nCODE_HEAD:\n{code_head}"),
        ]
    )
    assert set(p.input_variables) == {"spec", "metrics", "code_head"}
