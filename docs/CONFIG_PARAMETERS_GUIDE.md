# Configuration Parameters Guide

This guide explains which parameters control runtime behavior and where to set the most important knobs.

## Where To Change Tree Width/Depth

Set these in `config.yaml` under `orchestration`:

```yaml
orchestration:
  max_tree_width: 3
  max_tree_depth: 5
```

- `max_tree_width`: max number of subtasks for non-root decomposition.
- `max_tree_depth`: recursion depth ceiling for task splitting.

Root-level plan is intentionally immutable in current pipeline logic (no root truncation/replanning).

## Parameter Groups

### runtime
- `project_name`: base directory for run artifacts.
- `create_env`: create virtual environment on startup.
- `code_timeout_min`, `pip_timeout_min`: base operation timeouts.
- `predictive_buffer_pct`: execution predictor timeout buffer.
- `checker_timeout_cap_sec`, `verifier_timeout_cap_sec`: caps for checker/verifier script runs.
- `default_task_budget_sec`: fallback per-task budget if not provided by planner.
- `prediction_fallback_sec`: default expected runtime when predictor fails.
- `min_exec_timeout_sec`: lower timeout floor for task execution.
- `bash_timeout_sec`: timeout for router-triggered bash actions.
- `metric_validation_retry_limit`: max retries when metrics are missing/invalid.
- `router_retry_limit`: max routing retries inside improvement loops.
- `generation_retry_limit`: hard cap on generation loop retries.
- `execution_output_shorten_threshold`, `execution_output_shorten_target`: stdout truncation controls.
- `replan_context_chars`: max chars passed into replanning context.
- `aggregate_tail_chars`: max chars injected into aggregate context tails.
- `attach_hardware_limit_files`: file-scan limit for hardware attachment step.

Execution timeout behavior (important):

- Effective script timeout is selected from:
  1) explicit `timeout` argument (if passed),
  2) per-task budget (`spec["_current_task_budget_sec"]`, when set by pipeline),
  3) `runtime.code_timeout_sec` fallback,
  then additionally constrained by global remaining deadline.
- Streamed runs also have monitor-side no-output kill behavior (idle-timeout) to avoid silent hangs.

### orchestration
- `enforce_single_stack`, `allow_ensembles`, `metric_source`, `require_metrics_json`
- `check_fail_threshold`: checker failure ratio threshold.
- `min_metric_improvement_rel`, `optimize_iters`: improvement loop controls.
- `meta_planner_time_pct`, `meta_planner_max_attempts`: ReAct meta-planner time slice + retries (used inside the improver/improvement loop for deep task discovery).
- `max_tree_width`, `max_tree_depth`: tree controls.
- `replan_max_calls`: max tail-replanning calls per branch before forced execution-first behavior.
- `replan_cooldown_steps`: minimum executed subtasks between replanning calls (prevents replan storms).
- `main_verifier_max_steps`, `improve_verifier_max_steps`: verifier bounds.
- `total_budget_min`, `improve_budget_min`: global/main and improve budgets.

### data_check
- `enabled`: enable dataset probing.
- `max_samples_per_dir`: sample cap while scanning data.
- `probe_timeout_sec`: timeout for the host probe command used by `probe_dataset_with_bash()`.
- `react_max_rounds`: maximum outer rounds for the **dataset ReAct** tool-calling probe (`dataset_react_probe` in `src/dataset_checker.py`; each round may issue multiple tool calls until the model returns final JSON).

### paths
- `artifacts_dir`, `data_dir`, `logs_dir`, `scripts_dir`, `src_dir`, `tests_dir`, `venv_dir`
- Relative paths resolve under `runtime.project_name`.
- Submission path is controlled by:
  - `submission_dir`
  - `submission_filename`
  Canonical target = `project_root / submission_dir / submission_filename`.
  Default remains root-level `project_root/submission.csv` when `submission_dir: ""`.

Finalization path reconciliation:

- Even with root-level canonical path, finalization may mirror to/from `project_root/submission/submission.csv`
  for external checker compatibility.
- This mirroring does not redefine canonical path in config; it is a runtime reconciliation step.

### llm
- `prefer`: backend selection (`openai`, `anthropic`, `google`, `vertex`, `ollama`).
- `model_pricing_file`: JSON file (default `llm_model_pricing.json`) next to `config.yaml` — USD estimates for `token_usage` logs (see `src/llm_utils.py`). `.yaml` is still accepted if the path ends with `.yaml`.
- Backend-specific model names, temperature, keys, and endpoints.

### hardware
- `require_cuda`, `fail_if_no_cuda`, `cuda_devices`.

### mcp
- `enabled`, `servers[]` definitions for MCP tool backends.

### preinstall
- `enable`, `pkgs[]`, `torch_cuda_index_url`.

### proxy
- `http`, `https`.

## Validation Rules At Startup

`main.py` now validates key constraints before execution:
- `max_tree_depth > 0`
- `max_tree_width > 0`
- `min_exec_timeout_sec >= 1`
- retry limits are non-negative/positive where required
- `data_check.max_samples_per_dir >= 1`
- `data_check.probe_timeout_sec >= 1`

If invalid, startup exits with configuration errors.

## Notes

- Use `config.yaml.example` as the canonical template with inline comments.
- Keep secrets in environment variables (`${...}`) instead of plaintext in repo files.

## Error Recovery and Lead Logic

Recovery now follows layered control:

1) `ErrorRouter` proposes a route (`install` / `bash` / `coding` / `spec_update` / `lead`).
2) If route is `lead` (or the main loop starts thrashing routes), `lead_incident_manager_agent` is invoked.
3) Lead manager receives:
   - triage plan,
   - recent attempt history,
   - stderr/stdout tails,
   - code head,
   - task/spec context,
   and returns a structured next action route.

Per-node attempt history is persisted under `artifacts/state/node_<id>.json` and used to avoid repeated тупняк loops.

Additional hard guardrails:

- Missing-dependency errors force `install` route first.
- Timeout/OOM class errors force `spec_update` (except when install/lead takes precedence), allowing retries with adapted resource-related spec patches.
