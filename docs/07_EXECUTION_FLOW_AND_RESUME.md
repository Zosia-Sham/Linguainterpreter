# 7. Execution Flow and State Management (--resume)

This document provides a detailed sequence and state flow of how `LinguaInterpreter` executes tasks, manages state, handles errors, and optimizes metrics. It specifically details how the `--resume` flag works and where artifacts are read from or written to, ensuring future modifications don't break the state machine.

## High-Level Execution Flow (Mermaid)

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant Pipe as main_pipeline (pipeline.py)
    participant Orch as GlobalOrchestrator
    participant State as state.py / Filesystem
    participant Agent as LLMs (prompts_agents.py)
    participant Bash as BashAgent (bash_agent.py)
    participant Opt as Optimizer (optimizer.py)

    CLI->>Pipe: Start with config & --resume flag
    Pipe->>Orchestrator: Initialize (load tree if resume)
    Orch->>State: load_state() / read tree.json
    
    alt If not resuming or tree empty
        Pipe->>Agent: problem_spec_from_text()
        Agent-->>Pipe: spec.json
        Pipe->>Agent: tasks_generation()
        Agent-->>Pipe: Sub-tasks list
        Pipe->>Orch: add_node() for each sub-task
        Orch->>State: write tasks_tree.json
    end

    loop Over pending nodes in Orchestrator
        Pipe->>Orch: update_node(status="running")
        Pipe->>Agent: perform_task_python_v2(previous_code)
        Agent-->>Pipe: python_script.py
        
        Pipe->>Bash: run(python_script.py)
        Bash-->>Pipe: stdout, stderr, exit_code
        
        Pipe->>Agent: evaluate_run_ok_with_retry()
        alt Execution Failed
            Pipe->>Agent: error_triage_agent()
            Agent-->>Pipe: Fix Plan (e.g., pip install, code fix)
            Pipe->>Bash: Execute fix (or loop back to code generation)
        else Execution Succeeded
            Pipe->>Pipe: parse_metrics_from_stdout()
            alt Metrics Found
                Pipe->>Opt: optimize_metrics(base_code, metrics)
                Opt->>Agent: _proposal_agent() & finetune_code_v2()
                loop For each proposal
                    Agent-->>Opt: optimized_script.py
                    Opt->>Bash: run(optimized_script.py)
                    Bash-->>Opt: new_metrics
                    Opt->>Opt: compare metrics (_is_better_value)
                end
                Opt-->>Pipe: best_code, best_metrics
            end
            Pipe->>State: _update_best_from_candidate() -> artifacts/best/
            Pipe->>State: _record_metrics_version() -> artifacts/versions/
            Pipe->>Orch: update_node(status="done")
        end
        Orch->>State: write tasks_tree.json
    end
    
    Pipe->>State: _finalize_single_submission_by_all_metrics_llm() -> artifacts/final/
    CLI->>Agent: checker_code_agent() (best-effort final verifier)
    Agent-->>CLI: final_verifier.py
    CLI->>Bash: run(final_verifier.py)
    Bash-->>CLI: CHECK PASS/FAIL + CHECK_SUMMARY
```

## Specification, data paths, and probing (before the task loop)

When a full spec is built (not loading a frozen `spec.json` from resume), the pipeline typically runs:

1. **`problem_spec_from_text`** — initial `spec.json` fields from the task brief.
2. **`datapath_agent`** (+ optional **`datapath_consistency_check_agent`**) — align `spec.data` paths with a FILETREE snapshot.
3. **`probe_dataset_with_bash`** (`src/dataset_checker.py`) — host probe, optional **`_auto_unpack_zip_archives`**, optional **`dataset_react_probe`** tool loop (`shell_exec`, etc.).

Details: `AGENT_INTERACTIONS.md` (Data paths section), `docs/04_DATA_AND_HARDWARE.md`.

## How `--resume` Works

The `--resume` flag is critical for long-running ML tasks. It prevents the system from starting from scratch if a script crashes, the machine reboots, or the user manually stops the process via the dashboard.

1. **State Loading**: When `main.py` is invoked with `--resume`, the `GlobalOrchestrator` attempts to load existing state from the `artifacts/` directory instead of purging it.
2. **Artifacts Read**:
    * `artifacts/spec.json`: Loaded to skip the initial specification generation phase.
    * `artifacts/data_meta.json` & `artifacts/hardware.json`: Loaded to skip filesystem and hardware probing.
    * `artifacts/tree.json` & `artifacts/tasks_tree.json`: Loaded into the orchestrator's internal `self.nodes` dictionary.
3. **Execution Resumption**:
    * The `main_pipeline` iterates through `orch.get_ordered_nodes()`.
    * It checks the status of each node. If a node's status is `"done"`, the pipeline **skips it entirely**.
    * If a node is `"running"` or `"failed"`, the pipeline assumes it was interrupted or needs fixing, and begins execution from that exact node.
    * It automatically injects the `previous_code` from the last successful node (read from `artifacts/last/code.py` or the `versions/` index) into the LLM prompt for the resuming node.

**WARNING for LLM Modifications**:
- Do not alter the structure of `tree.json` or `tasks_tree.json` without updating `GlobalOrchestrator.load_tree()` and the frontend `ui.html`.
- If you change how nodes are marked as `"done"`, ensure it only happens *after* all artifacts (`best/`, `versions/`, `last/`) are successfully written to disk. Otherwise, `--resume` will skip the node but subsequent nodes will lack the necessary `previous_code`.

## Artifacts Ecosystem Breakdown

Understanding where files are written and read is vital for maintaining the pipeline.

### 1. `artifacts/last/`
* **Written by**: `helpers._update_best_from_candidate` (Always, on every successful node execution).
* **Contents**: `code.py`, `metrics.json`, `submission.csv`.
* **Used for**: This is the immediate state. When the pipeline moves to the next node, it reads `artifacts/last/code.py` to provide the `previous_code` context to the LLM (`perform_task_python_v2`).

### 2. `artifacts/best/`
* **Written by**: `helpers._update_best_from_candidate` (ONLY if the `primary` metric strictly improves compared to the existing `artifacts/best/metrics.json`).
* **Contents**: `best_code.py`, `metrics.json`, `submission.csv`.
* **Used for**: Finalizing the project. This directory guarantees that regardless of how many optimization steps degraded performance, the absolute best attempt is preserved.

### 3. `artifacts/versions/`
* **Written by**: `helpers._record_metrics_version` (On every successful run, inside an immutable timestamped folder like `20260310_120000_tag/`).
* **Contents**: Snapshots of code, metrics, and submissions. Also updates `index.json` and `ledger.csv`.
* **Used for**: History tracking, rollback (if needed), and debugging LLM degradation over time.

### 4. `artifacts/final/`
* **Written by**: `helpers._finalize_single_submission_by_all_metrics_llm` (At the very end of the pipeline and also from `main.py` `finally`, best-effort).
* **Contents**: Copies the absolute best submission and code from the `versions` index or `best` folder to provide a clean, user-facing output directory.
* **Additional outputs**:
  - `output_gate_report.json` from `run_final_output_gate(...)` with final path checks and validation errors/warnings.
  - `metrics.json` final snapshot materialized by pipeline finalization code (for external checker compatibility).

## Final Output ReAct Recovery (Submission + Metrics)

The final stage no longer assumes a single submission location. Recovery logic actively searches and reconciles:

- canonical path from config: `canonical_submission_path(orch)`
- legacy root path: `project_root/submission.csv`
- bench path: `project_root/submission/submission.csv`
- known artifacts (`artifacts/final`, `artifacts/versions`)

Execution order (simplified):

1. Attempt canonical copy (`ensure_canonical_submission_copy`).
2. If missing, try legacy root submission.
3. If still missing, try bench path.
4. Mirror valid submission across canonical + legacy + bench-friendly path.
5. Run strict final submission validation and write `output_gate_report.json`.

This behavior is designed to prevent false failures when training succeeded but file was saved in an alternate expected location.

## Verifier/Checker Anti-Hardcode Policy

During metrics recovery, generated verifier scripts are audited before execution:

- detect hardcoded spec dictionaries / serialized `spec_json` patterns
- require dynamic loading from `artifacts/spec.json`

If policy fails, the verifier code is auto-repaired via agentic repair loop before executing `metric_check.py`.

Additionally, `main.py` runs a **best-effort final verifier** via `checker_code_agent` and stores its raw output under:
- `artifacts/final/final_verifier_stdout.txt`

## The `Improver` (Optimizer) Flow

The optimization loop (`src/optimizer.py:optimize_metrics`) acts as a micro-pipeline within a single sub-task node.

1. **Trigger**: If a node executes successfully and its stdout contains `METRICS_JSON:{...}`, the pipeline passes the code and metrics to the optimizer.
2. **Proposals**: The LLM (`_proposal_agent`) generates up to 3 JSON proposals for targeted tweaks (e.g., "Increase learning rate from 1e-3 to 5e-4", "Change loss function to Focal Loss").
3. **Finetuning**: For each proposal, the LLM (`finetune_code_v2`) rewrites the full Python script to include the tweak.
4. **Execution**: The new script is saved to `artifacts/scripts/opt_<id>.py` and executed via `BashAgent`.
5. **Evaluation**: If the new script outputs better metrics (evaluated by `_is_better_value`), it immediately overwrites the `best_code` and `best_metrics` variables in memory.
6. **Return**: The optimizer returns the best found code and metrics back to the `main_pipeline`, which then commits them to disk via the `helpers` module.

## Improvement Loop Meta-Planning (ReAct Improver Meta-Planner)

Besides the optimizer micro-loop (`src/optimizer.py:optimize_metrics`), the improver/improvement loop (`src/pipeline.py:improvement_pipeline`) may need to *generate new improvement tasks* for an iteration.

When there are no existing improvement children for the current iteration:
- The pipeline calls `react_improver_meta_planner_agent` (ReAct) to audit the current baseline in `artifacts/best/` and `artifacts/last/`, discover where the metric values were computed (typically via `artifacts/versions/*/metrics.json` + the corresponding producing `code.py`), and then output a hierarchical deep task list.
- The meta-planner is budgeted by `orchestration.meta_planner_time_pct` and retries are capped by `orchestration.meta_planner_max_attempts`.

Resume behavior (`--resume`):
- If the improvement tasks already exist (children in the tree are present), the generation branch is skipped and the meta-planner is not re-run.
- This keeps `--resume` deterministic: previously created nodes continue, and only incomplete/missing parts are regenerated.
