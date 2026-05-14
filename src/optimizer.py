from __future__ import annotations
import json, math, uuid
from typing import Any, Dict, Tuple, List
from colorama import Fore
from langchain_core.prompts import ChatPromptTemplate

from .validators import parse_metrics_from_stdout, detect_mixed_stacks
from .prompts_agents import finetune_code_v2
from .llm_utils import invoke_and_log

def _proposal_agent(llm_strong, spec: Dict[str,Any], metrics: Dict[str,Any], code_head: str) -> List[Dict[str,Any]]:
    """
    Просим старшего предложить до 3 точечных улучшений. Возвращаем список dict'ов.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior ML Tech Lead focused on QUICK metric improvements.
Given SPEC, current METRICS_JSON and a code summary (head), propose up to 3 SMALL, TARGETED modifications that could improve the primary metric.

Return ONLY JSON:
{{
  "improvements": [
    {{
      "title": "short",
      "rationale": "why this helps",
      "hint": "precise coding hint (hyperparams/augmentations/thresholds/backbone/etc.)",
      "risk": "low|medium",
      "allow_stack_switch": false
    }}
  ]
}}"""),
        ("user", "SPEC:\n{spec}\n\nMETRICS_JSON:\n{metrics}\n\nCODE_HEAD:\n{code_head}")
    ])
    res = invoke_and_log(
        llm_strong,
        prompt,
        {
            "spec": json.dumps(spec, ensure_ascii=False, indent=2),
            "metrics": json.dumps(metrics or {}, ensure_ascii=False, indent=2),
            "code_head": (code_head or "")[:1200],
        },
        agent_name="metrics_opt_proposal",
    )
    try:
        data = json.loads(getattr(res, "content", "") or "{}")
        imps = data.get("improvements", [])
        return imps if isinstance(imps, list) else []
    except Exception:
        return []

def _judge_better_agent(llm_strong, metric_name: str, maximize: bool, old: Dict[str,Any], new: Dict[str,Any]) -> bool:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an impartial metric judge. Return ONLY True if NEW is better than OLD given metric semantics; else False."),
        ("user", "metric_name={m}\nmaximize={mx}\nOLD={old}\nNEW={new}")
    ])
    res = invoke_and_log(
        llm_strong,
        prompt,
        {"m": metric_name, "mx": str(bool(maximize)), "old": json.dumps(old, ensure_ascii=False), "new": json.dumps(new, ensure_ascii=False)},
        agent_name="metrics_opt_judge",
    )
    t = (getattr(res, "content", "") or "").strip().lower()
    return ("true" in t) and ("false" not in t)

def _is_better_value(llm_strong, metric_name: str, maximize: bool,
                     new_val: float, old_val: float,
                     abs_thr: float, rel_thr: float) -> bool:
    # Comment translated to English.
    if rel_thr > 0.0 and math.isfinite(old_val) and abs(old_val) > 1e-12:
        if maximize:
            return new_val > old_val * (1.0 + rel_thr)
        else:
            return new_val < old_val * (1.0 - rel_thr)
    # Comment translated to English.
    if abs_thr > 0.0:
        if maximize:
            return (new_val - old_val) > abs_thr
        else:
            return (old_val - new_val) > abs_thr
    # Comment translated to English.
    return _judge_better_agent(llm_strong, metric_name, maximize,
                               {"primary": old_val}, {"primary": new_val})

def optimize_metrics(
    orch,
    llm_strong,
    llm_fast,       # Comment translated to English.
    code_llm,
    spec: Dict[str,Any],
    base_code: str,
    base_metrics: Dict[str,Any],
    max_iters: int = 2,
    min_improvement_abs: float = 0.0,
    min_improvement_rel: float = 0.0,
) -> Tuple[str, str, Dict[str,Any]]:
    """
    Короткий цикл улучшения метрики.
    Возвращает (best_stdout, best_code, best_metrics). Файлы best_* сохраняются изнутри при улучшении.
    """
    if not base_code or not isinstance(base_code, str):
        return "", "", {}

    maximize = bool(spec.get("primary_metric", {}).get("maximize", True))
    metric_name = (spec.get("primary_metric", {}) or {}).get("name", "primary")

    # Comment translated to English.
    best_code = base_code
    best_metrics = base_metrics or {}
    try:
        best_primary = float(best_metrics.get("primary", 0.0) or 0.0)
    except Exception:
        best_primary = 0.0
    best_stdout = ""

    # Comment translated to English.
    improvements = _proposal_agent(llm_strong, spec, best_metrics, best_code[:1200])
    if not improvements:
        return best_stdout, best_code, best_metrics

    # Comment translated to English.
    for i, imp in enumerate(improvements[:max_iters], start=1):
        title = str(imp.get("title", f"improvement_{i}"))
        hint = str(imp.get("hint", ""))
        allow_switch = bool(imp.get("allow_stack_switch", False))

        ctx = f"OPTIMIZE: {title}\nHINT: {hint}\nCURRENT_PRIMARY={best_primary}"
        new_code = finetune_code_v2(code_llm, f"Improve metric - {title}", best_code, spec, error=ctx)

        # Comment translated to English.
        orc = getattr(orch.cfg, "orchestration", None)
        if isinstance(orc, dict):
            enforce = bool(orc.get("enforce_single_stack", True))
            allow_ens = bool(orc.get("allow_ensembles", True))
        else:
            enforce = bool(getattr(orc, "enforce_single_stack", True))
            allow_ens = bool(getattr(orc, "allow_ensembles", True))
        if enforce and detect_mixed_stacks(new_code) and not (allow_ens or allow_switch):
            print(Fore.RED + f"[OPT] skip '{title}': mixed stack not allowed")
            continue

        rel_script = f"{orch.cfg.paths.scripts_dir}/opt_{i}_{uuid.uuid4().hex[:6]}.py"
        orch.write_file(rel_script, new_code)
        res = orch.run_python_file(rel_script, stream=True)
        stdout = res.get("output","") or ""
        metrics = parse_metrics_from_stdout(stdout)
        if not metrics:
            print(Fore.YELLOW + f"[OPT] '{title}': no METRICS_JSON — skip")
            continue

        try:
            new_primary = float(metrics.get("primary", 0.0) or 0.0)
        except Exception:
            new_primary = best_primary

        better = _is_better_value(llm_strong, metric_name, maximize, new_primary, best_primary,
                                  abs_thr=min_improvement_abs, rel_thr=min_improvement_rel)
        print(Fore.CYAN + f"[OPT] {title}: {new_primary:.6f} vs {best_primary:.6f} -> {'better' if better else 'no better'}")
        if better:
            best_primary = new_primary
            best_code = new_code
            best_metrics = metrics
            best_stdout = stdout
            # Comment translated to English.
            orch.write_file(f"{orch.cfg.paths.artifacts_dir}/best_code.py", best_code)
            orch.write_file(f"{orch.cfg.paths.artifacts_dir}/best_metrics.json", json.dumps(best_metrics, ensure_ascii=False, indent=2))

    return best_stdout, best_code, best_metrics
