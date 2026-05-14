# 3. Agents and LLM Management

This section details the interaction with Large Language Models and the execution of generated code.

## `src/llm_factory.py`
**Purpose**: Abstracts the instantiation of LLMs.
**How it works**:
- Uses LangChain bindings (`langchain_openai`, `langchain_google_genai`, `langchain_google_vertexai`, `langchain_ollama`).
- Provides a fallback mechanism. The `build_llms(cfg)` function tries to initialize the preferred provider (e.g., OpenAI). If API keys are missing or it fails, it falls back to the next available provider.
- Returns a tuple of three models: `(llm_strong, llm_fast, code_llm)`. Strong is used for complex reasoning and task generation, Fast is used for quick parsing and error triaging, Code is used with low temperature for writing Python scripts.

## `src/bash_agent.py`
**Purpose**: A secure wrapper for executing shell commands and Python scripts locally.
**How it works**:
- **`BashAgent` Class**: Determines the OS (`Windows` vs `Linux/Darwin`) and sets up the appropriate shell (`powershell` vs `bash`).
- Configures environments to force UTF-8 encoding (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`) to prevent character encoding crashes in the console.
- **`run(...)`**: Executes commands using `subprocess`. It supports two modes:
  1. `stream=False`: Blocks and captures stdout/stderr using `capture_output=True`.
  2. `stream=True`: Uses `subprocess.Popen` with background threads (`threading.Thread`) to stream stdout and stderr line-by-line in real-time while capturing it into memory and a log file simultaneously.
- Handles timeouts (`subprocess.TimeoutExpired`) gracefully.

## `src/prompts_agents.py`
**Purpose**: The core repository of all LLM prompts and specific agent functions. Prompt **text** lives in `src/prompts/templates.py`; this module wires templates to LangChain and parses outputs.

**How it works (Key Agents)**:
- **`problem_spec_from_text`**: Produces JSON for modalities, metrics, constraints (`spec.json`).
- **`meta_planner_agent`**: High-level stage list (YAML); must adapt to modality, not over-specify implementation. Receives the same **`format_spec_constraints_block`** prefix as the code agent (`src/utils.py` + `src/pipeline.py`).
- **`tasks_generation` / `task_ordering`**: Decomposes work into sub-tasks and orders them; same **COMPETITION CONSTRAINTS** block is prepended from the pipeline so plans stay aligned with `internet_allowed` / `pretrained_allowed` / external-data flags.
- **`perform_task_python_v2`**: Main **code** agent — executable Python, dynamic spec load, data safety, **`METRICS_JSON`** contract on training/eval steps. Incorporates a ReAct-style flow with `planner_agent`, `reviewer_agent`, and `coder_agent` that can use read-only tools and the **`generate_and_execute`** tool for fast runtime validation. **Validation strategy** (holdout vs k-fold vs LOO) is chosen by the agent within guidelines (`spec.validation`, time, data size); group/time rules from the task are not optional shortcuts.
- **`replanning_agent`**: Adjusts remaining tasks when stuck or over budget.
- **`improvement_replanning_agent` / `improver_head_agent` / `improvement_tasks_generation`**: Improve-mode planning and task refresh without full replan from scratch.
- **`react_improver_meta_planner_agent`**: ReAct meta-planner inside the improver/improvement loop. It audits `artifacts/best/` + `artifacts/last/`, discovers where metrics were computed (e.g. `artifacts/versions/*/metrics.json` + producing `code.py`), then outputs deep tasks tailored to `spec.modalities`. Designed to be safe under `--resume`.
- **`error_triage_agent`**: Classifies failures (`install`, `coding`, `bash`, …).
- **`finetune_code_v2`**: Fixes Python from error log + previous script.
- **`datapath_agent` / `datapath_consistency_check_agent`**: Path resolution against file tree (templates in `src/prompts/templates.py`).
- **Dataset ReAct probe** (`dataset_react_probe`, implemented in `src/dataset_checker.py`): After `probe_dataset_with_bash()` runs, a **tool-calling** pass may execute. It uses `StructuredTool` + `invoke_with_tools()` (`src/llm_utils.py`) with `shell_ls`, `shell_find_archives`, `shell_head_csv`, and **`shell_exec`** so the model can run real PowerShell/bash commands (e.g. unzip / `Expand-Archive`) when needed. **LangGraph is not required** for this path.
- **`metrics_recover_from_stdout`**: Fast LLM pass to extract or repair **`METRICS_JSON`** from a **truncated stdout tail** when regex parsing fails (only when heuristics suggest training/eval output); validated by **`validate_recovered_metrics`** before acceptance.
- **`review_artifacts_agent`**, **`runtime_output_ok_agent`**, **`checker_code_agent`**, **`verification_code_gen`**: Review and verification steps.
- **`lead_incident_manager_agent`**, **`lead_agent_propose_changes`**: Escalation when errors repeat or pipeline needs a strategic change.

See `src/prompts/README.md` for a compact table and editing notes.

**Runtime context (not full prompts)**: `src/pipeline.py` prepends short notes (deadline, Improve rules, evaluation guidance, **`format_spec_constraints_block(spec)`** for meta-planner / task gen / ordering / code) so agents see budget and competition rules without duplicating `templates.py`.

## `src/optimizer.py`
**Purpose**: Iteratively improves the primary metric of the machine learning model.
**How it works**:
- **`optimize_metrics(...)`**: Calls a `_proposal_agent` to suggest up to 3 targeted modifications to the code (e.g., "Change learning rate", "Add SMOTE").
- For each proposal, it calls `finetune_code_v2` to rewrite the script, executes it via the orchestrator, and parses the new metrics.
- Uses `_is_better_value` (which compares numerical values based on `maximize=True/False` or uses an LLM judge) to determine if the new code should replace the old `best_code.py`.

## `src/router.py`
**Purpose**: Routes execution failures to the appropriate recovery action.
**How it works**:
- **`ErrorRouter` Class**: Keeps track of identical errors to prevent infinite loops (`_last_sig`, `_repeats`). It receives **`orch`** (`GlobalOrchestrator`) so recovery tools run in the real project venv.
- Uses **`error_triage_agent`** for a JSON plan (`route`, `packages`, `reason`).
- **Install / CUDA ReAct** (no LangGraph on this path): when hints are needed (empty `packages` on `install`, `ModuleNotFoundError`, repeated same error), runs **`pip_install_react`** — a loop via **`StructuredTool`** + **`invoke_with_tools`** (`src/llm_utils.py`). Tools from **`build_pip_react_tools`**: **`pip_install`** → `orch.pip_install(...)`, **`cuda_probe`** → embedded **`CUDA_PROBE_PY`** via `orch.run_python_code`, **`google_search`** → optional if `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are set.
- Escalates to a **Lead** route if the same error repeats multiple times despite previous fixes.

**Bootstrap CUDA repair** (`src/bootstrap.py`): **`bootstrap_gpu_stack(orch, cfg, llm_fast=...)`** may call **`repair_torch_cuda_with_react`** when the post-preinstall probe still shows CPU-only PyTorch while CUDA is configured. Wired from **`src/pipeline.py`** (fresh spec runs pass `llm_fast`). Trace logs: `agents/pip_install_react.jsonl`, `agents/bootstrap_cuda_react.jsonl`.
