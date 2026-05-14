# Prompts (agents)

## Where things live

| Piece | Role |
|--------|------|
| **`templates.py`** | System prompts (strings). Single source of truth for agent *instructions*. |
| **`../prompts_agents.py`** | Python wrappers: builds messages, calls LLMs, parses outputs. |

Orchestration injects **`format_spec_constraints_block(spec)`** (see `../utils.py`) for **meta-planner**, **tasks_generation**, **task_ordering**, and code agents; plus deadline, Improve mode, artifacts snapshot, `project_context` in **`../pipeline.py`** — *runtime guidance*, not a second copy of full prompts.

## Design principle

- **`spec` and templates give guidelines** (metrics, paths, safety). **Concrete choices** (e.g. holdout vs 3-fold, which extras to log) are for the **code-generating agent** unless the task text locks something in.
- **`spec.validation`** describes a *target* protocol; agents **adapt** to time, modality, and data size. Group/time leakage rules from the task still apply.
- **DEBUG lines** (`VALIDATION_DECISION`, `VALIDATION_PROTOCOL_SUMMARY`, `DATA_ROLE_SUMMARY`) are **recommended** when they clarify what was run — not a rigid checklist on every tiny sub-task.

## Main agent roles (high level)

| Area | Typical entry in `prompts_agents.py` | Template constant (in `templates.py`) |
|------|--------------------------------------|--------------------------------------|
| Spec from brief | `problem_spec_from_text` | e.g. problem-spec prompt |
| High-level plan | `meta_planner_agent` | `DS_META_PLANNER_SYS` |
| Decompose to sub-tasks | `tasks_generation`, `task_ordering` | task-generation / ordering templates |
| **Code for a sub-task** | `perform_task_python_v2` | `PERFORM_TASK_PYTHON_SYS` |
| Fix after error | `finetune_code_v2`, `error_triage_agent` | repair / triage templates |
| Paths | `datapath_agent`, `datapath_consistency_check_agent` | `DATAPATH_SYS`, `DATAPATH_CONSISTENCY_CHECK_SYS` |
| Dataset probe (tools) | Implemented in `src/dataset_checker.py` (`dataset_react_probe` logs), not in `prompts_agents.py` | System prompt string inside `_run_dataset_react_agent`; uses `invoke_with_tools` + `shell_exec` |
| Pip / CUDA recovery (tools) | Implemented in `src/router.py` (`ErrorRouter`, `build_pip_react_tools`, `repair_torch_cuda_with_react`); triggered from router and `src/bootstrap.py` | Inline system strings in router; uses `invoke_with_tools` + `StructuredTool` |
| Fast validation (tools) | Implemented dynamically inside `perform_task_python_v2` (`generate_and_execute` tool) | Inline string tool description; accessible by `planner_agent`, `reviewer_agent`, and `coder_agent` |
| Replan / Improve | `replanning_agent`, `improvement_replanning_agent`, `improver_head_agent`, `improvement_tasks_generation` (also gets **`constraints_block`**) | improvement / lead templates |
| Metrics recovery (stdout tail) | `metrics_recover_from_stdout` | `METRICS_RECOVER_FROM_STDOUT_SYS` |
| Quality gates | `review_artifacts_agent`, `runtime_output_ok_agent`, `checker_code_agent`, `verification_code_gen` | review / verification templates |
| Lead escalation | `lead_incident_manager_agent`, `lead_agent_propose_changes` | lead templates |

For the full list of functions, open `prompts_agents.py`.

## Graph / task visibility

Lifecycle transitions from orchestration (`SKIPPED`, `PRUNED`, `ADDED`) are written to `artifacts/task_graph_events.jsonl` and surfaced in `task_plan.md` under **Graph Events (latest)**.

## Agent trace logs (under `{artifacts_dir}/agents/`)

All LLM calls in the codebase are routed through `src/llm_utils.py` helpers:
- `invoke_and_log(...)` (regular calls)
- `invoke_with_tools(...)` (tool-calling / ReAct-style loops)

If `agent_name` is not explicitly provided, it is derived from the **caller function name**, so **every agent wrapper in `src/prompts_agents.py` automatically produces a dedicated JSONL file** at:
`{project_root}/{artifacts_dir}/agents/<agent_name>.jsonl`.

| File | Source |
|------|--------|
| `dataset_react_probe.jsonl` | Shell ReAct in `dataset_checker.py` (`log_agent_trace` / `invoke_with_tools`) |
| `datapath_agent.jsonl` | Path suggestions from `datapath_agent` (`invoke_and_log`) |
| `pip_install_react.jsonl` | `ErrorRouter` install/CUDA tool loop |
| `bootstrap_cuda_react.jsonl` | `bootstrap_gpu_stack` → `repair_torch_cuda_with_react` |
| `checker_code_agent.jsonl` | LLM-generated verifier script (`checker_code_agent`) |
| `verification_code_gen.jsonl` | Metric recomputation verifier script (`verification_code_gen`) |
| `final_metric_selector_agent.jsonl` | Final-by-all-metrics chooser for submission materialization |
| `token_usage.jsonl` | Token usage + estimated cost (written on every logged call) |

## Editing workflow

1. Change the string in `templates.py` (or the injected prefix in `pipeline.py` if it is cross-cutting context only).
2. If you add a **new** system prompt variable, import it where the chat is built (`prompts_agents.py` or orchestrator).
3. Restart the run; no separate “compile” step for prompts.

## Related docs

- `docs/03_AGENTS_AND_LLM.md` — LLM factory, bash execution, agent overview.
- `docs/04_DATA_AND_HARDWARE.md` — `probe_dataset_with_bash`, zip unpack, dataset ReAct probe.
- Root `AGENT_INTERACTIONS.md` — end-to-end data-path and probing sequence.
- Root `PROMPTS.md` — historical / illustrative excerpts; **canonical** wording is in `templates.py`.
- `docs/08_SPECIFICATIONS_AND_INTERFACES.md` — **`METRICS_JSON`** / **`primary`** / ledger contract.
