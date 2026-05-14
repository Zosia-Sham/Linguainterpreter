# 6. Artifacts, Dependencies, and Weaknesses

## Artifacts Structure
As the pipeline runs, it generates a strict directory structure within the configured `artifacts/` folder:
```text
artifacts/
├── spec.json                # The formalized JSON definition of the ML task.
├── tree.json                # Orchestrator state history.
├── tasks_tree.json          # Granular sub-task tree used by the UI.
├── agents/                  # Append-only JSONL traces (e.g. datapath_agent, dataset_react_probe, pip_install_react, bootstrap_cuda_react, token_usage.jsonl).
├── data_meta.json           # Discovered metadata about the dataset.
├── hardware.json            # Hardware probing results and resource plans.
├── best/                    # Overwritten with the highest-scoring run (calculated metrics only).
│   ├── code.py
│   ├── metrics.json         # Same canonical shape as METRICS_JSON (type, primary, name, maximize, extras?)
│   └── submission.csv
├── last/                    # Overwritten with the results of the very last run.
├── versions/                # Immutable history of all successful runs.
│   ├── index.json           # JSON list of all runs.
│   ├── ledger.csv           # One row per successful calculated run: ts, tag, primary, maximize, paths.
│   ├── ledger.md            # Human-readable experiments table (refreshed by orchestrator).
│   └── 20260310_120000_tag/ # Snapshot of a specific run.
└── scripts/                 # Temporary generated Python scripts before execution.
```

## Project Dependencies
The project relies on standard, modern Python libraries defined in `requirements.txt`:
- **LangChain Ecosystem**: `langchain`, `langchain-community`, `langchain-openai`, `langchain-ollama`, `langchain-google-genai`, `langchain-google-vertexai`.
- **FastAPI Ecosystem**: `fastapi`, `uvicorn[standard]` for the dashboard and real-time SSE streaming.
- **Observability**: `langfuse` (for tracing LLM latency and costs).
- **Utilities**: `PyYAML` (configuration), `colorama` (terminal formatting), `psutil` (hardware tracking, dynamically imported), `pynvml` (NVIDIA GPU tracking, dynamically imported).

## Weak Points & Vulnerabilities

### 1. Arbitrary Code Execution (Security)
The system is built to generate and execute Python and Shell code locally via the `BashAgent`. **There is no strict sandboxing (like a Docker container or a chroot jail).** Although `FormalVerifier` in `src/verification.py` attempts to catch malicious commands (`rm -rf /`) using Regex and LLM audits, this is easily bypassed by obfuscated code. If the LLM hallucinates or is subjected to prompt injection via the `task.txt`, it can execute destructive commands on the host machine.

### 2. State Synchronization Race Conditions
The web dashboard (`server.py`) uses a while-loop to poll the modification time (`os.path.getmtime`) of `artifacts/tree.json`. If the `GlobalOrchestrator` writes to the file extremely rapidly, the server might read a partially written file or skip an update frame, leading to JSON parsing errors or a desynced UI. There are no file-locks utilized during the writing/reading phase.

### 3. Infinite Loops in Error Recovery
The `ErrorRouter` attempts to fix runtime errors automatically. While it has a counter (`_repeats`) to escalate to a "Lead Agent", obscure errors (e.g., complex CUDA memory leaks or very subtle tensor shape mismatches) can cause the LLM to generate the exact same incorrect fix repeatedly, burning through API tokens and execution time.

### 4. Stdout parsing and metrics recovery
The orchestrator expects `METRICS_JSON: {...}` with **canonical keys** (`primary`, not `primary_score`). **`parse_metrics_from_stdout`** uses brace-balanced extraction and **last-wins** among multiple blocks. If parsing fails but logs look like training, **`metrics_recover_from_stdout`** (LLM on stdout **tail** only) may reconstruct metrics; otherwise **`verification_code_gen`** runs. Wrong keys prevent **`_validate_and_normalize_metrics`** from accepting the payload, so **ledger.csv** / **versions/** are not updated — keep generated scripts aligned with `templates.py` and `docs/08_SPECIFICATIONS_AND_INTERFACES.md`.

### 5. Lack of Asynchronous Task Execution
The `GlobalOrchestrator` processes tasks sequentially in a blocking manner. While the generated Python scripts can utilize multithreading (e.g., PyTorch DataLoaders), the orchestration framework itself cannot run two independent exploratory tasks (like EDA and simple baseline training) concurrently, under-utilizing system resources.

## Reliability Enhancements (Applied)

### A) Submission path reconciliation

To reduce path-mismatch failures with external checkers, finalization now reconciles real files across:

- canonical path from config (`paths.submission_dir` + `paths.submission_filename`)
- `project_root/submission.csv`
- `project_root/submission/submission.csv`
- discovered candidates in `artifacts/final` and `artifacts/versions`

Once a valid submission is found, it is mirrored to expected locations.

### B) Final metrics materialization

Pipeline finalization now always writes `artifacts/final/metrics.json` with the current metrics snapshot.
This avoids checker failures caused by missing metrics files even when metrics were computed/recovered in logs.

### C) Router robustness + resource-aware rerouting

- Fixed `NoneType + list` crash in `ErrorRouter` when merging `bash_cmds`.
- Timeout/OOM signatures (`TimeoutExpired`, `timed out`, `CUDA out of memory`, `out of memory`, `killed`) now force `spec_update` routing (unless install/lead path has priority), enabling adaptive retry instead of immediate dead-end failure.

### D) Execution timeout enforcement improvements

- Per-task `time_budget_sec` is propagated into runtime execution timeout via `spec["_current_task_budget_sec"]`.
- Stream monitoring adds no-output (idle) timeout handling for long silent hangs.
