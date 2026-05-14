# 2. Core Orchestration

This section details the primary files responsible for controlling the execution loop, managing state, and providing the user interface.

## `main.py`
**Purpose**: The CLI entry point for the application.
**How it works**:
- Uses `argparse` to read `--config` (default `config.yaml`), `--task_file`, `--resume`, `--no_log_file` (disable tee to `logs/run_*.log`), etc.
- Parses the YAML configuration into an `AppConfig` dataclass object.
- Applies environment variables (API keys, proxies) using `cfg.apply_env()`.
- Instantiates three LLM roles (Strong, Fast, Code) via `build_llms(cfg)`.
- Calls `main_pipeline` to start the autonomous execution.

## `server.py`
**Purpose**: A FastAPI-based local dashboard for monitoring the orchestrator's progress in real-time.
**How it works**:
- Exposes a static `ui.html` on the root `/` endpoint.
- **`/api/run`**: Starts the execution by launching `main.py` as a non-blocking `subprocess.Popen` background task.
- **`/api/stop`**: Terminates the background process.
- **`_watch_artifacts_json()`**: An asynchronous background loop that continuously checks the modification time (`os.path.getmtime`) of `artifacts/tree.json` and `artifacts/tasks_tree.json`.
- **`/api/stream_artifacts`**: Uses Server-Sent Events (SSE) to push the updated JSON trees to the frontend whenever the files change.

## `src/pipeline.py`
**Purpose**: The central nervous system of the project containing the `main_pipeline` function.
**How it works**:
- Reads the initial task from disk.
- Generates a structured JSON specification (`problem_spec_from_text`).
- Resolves and validates data paths (`datapath_agent`, `datapath_consistency_check_agent`, then `probe_dataset_with_bash` in `src/dataset_checker.py` — see `AGENT_INTERACTIONS.md`).
- Probes hardware and dataset metadata (`attach_hardware_to_spec`).
- Breaks down the task into sub-tasks and inserts them into the `GlobalOrchestrator`.
- **Execution Loop**:
  - Iterates over the `orchestrator.nodes`.
  - For each task, it invokes the code-generation LLM (`perform_task_python_v2`).
  - Writes the generated code to `artifacts/scripts/`.
  - Executes the code via `orch.run_python_file(...)`.
  - Checks if the runtime output is healthy using `evaluate_run_ok_with_retry`.
  - Parses metrics via **`parse_metrics_from_stdout`** (balanced JSON, last valid wins); on failure, optional **`metrics_recover_from_stdout`** (LLM on stdout tail), then **`verification_code_gen`** if still missing.
  - Persists valid **calculated** metrics through **`_update_best_from_candidate`** (`src/helpers.py`) into **`artifacts/last/`**, **`artifacts/versions/`**, and **`artifacts/versions/ledger.csv`** (see `docs/08_SPECIFICATIONS_AND_INTERFACES.md`).
  - Prepends **`format_spec_constraints_block(spec)`** (`src/utils.py`) into **meta-planner**, **task generation**, **ordering**, and **code** prompts; the code path also gets an **artifacts snapshot** (`src/pipeline.py`).
  - If the execution fails, it passes the stderr/stdout to the `ErrorRouter` which can decide to install dependencies via `pip`, modify the code, or ask a "Lead Agent" for advice.
  - If it succeeds and produces metrics, it triggers `optimize_metrics` to see if a better score can be achieved.

## `src/orchestrator.py`
**Purpose**: Manages the state machine, the task tree, and the execution of bash commands.
**How it works**:
- **`GlobalOrchestrator` Class**: Stores the project configuration and maintains the state of the task tree in `self.nodes`.
- **Task Management**: Nodes are added via `add_node(id, parent_id, label, task_text)`. Their status transitions between `pending`, `running`, `done`, and `failed` via `update_node(...)`.
- **Artifact Serialization**: The `_dump_tree()` method recursively serializes the nodes and writes them to `artifacts/tasks_tree.json` so the FastAPI server can pick them up.
- **Execution**: The `run_python_file()` method delegates the actual OS-level execution to the `BashAgent`.
