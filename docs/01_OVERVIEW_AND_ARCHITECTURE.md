# 1. Overview and Architecture

## High-Level Goal
LinguaInterpreter is an advanced, autonomous LLM-driven orchestration framework designed to act as a "Virtual Data Scientist." It interprets natural language tasks (like "train a classification model on this dataset"), formalizes them into a JSON specification, breaks the task down into an ordered series of code-generation steps, executes the generated Python scripts locally, verifies their output via metrics, and automatically applies iterative optimizations and error triage.

## Core Components
The system consists of several independent but tightly coupled components:
1.  **Pipeline (`src/pipeline.py`)**: The main control loop. It handles everything from understanding the initial task to executing sub-tasks, optimizing them, and recovering from runtime errors.
2.  **Orchestrator (`src/orchestrator.py`)**: The state manager. It tracks the progress of tasks and sub-tasks in a hierarchical tree (saved as `tasks_tree.json` and `tree.json`).
3.  **Agents & LLMs (`src/prompts_agents.py`, `src/llm_factory.py`)**: The reasoning engines. They generate code, write tests, diagnose errors, and propose metric optimizations.
4.  **Bash Execution (`src/bash_agent.py`)**: A local shell wrapper that executes the generated Python scripts, capturing their stdout and stderr safely.
5.  **Dashboard (`server.py`, `ui.html`)**: A FastAPI web server providing a real-time, D3.js-based visualization of the task tree and the agent's thought process.

## Architectural Flow & Interfaces
1.  **Initialization**: `main.py` is invoked with a task file (`task.txt`) and a configuration (`config.yaml`). It initializes the LLMs via `src/llm_factory.py` (which supports OpenAI, Google Gemini, Vertex, and Ollama).
2.  **Specification Generation**: The `main_pipeline` calls `problem_spec_from_text` to convert the natural language task into a rigid JSON structure containing the modalities, metrics, validation strategy, and constraints.
3.  **Hardware & Data Probe**: The pipeline probes the system hardware (`src/hardware.py`) and dataset structure (`src/dataset_checker.py`, `src/data_meta.py`) to build a resource plan (e.g., number of threads, batch sizes). Data paths are first proposed by LLM agents (`datapath_agent`, consistency check), then validated and enriched by `probe_dataset_with_bash()` (including optional zip extraction and a tool-calling **dataset ReAct** probe). See `AGENT_INTERACTIONS.md` (Data paths section) and `docs/04_DATA_AND_HARDWARE.md`.
4.  **Task Decomposition**: The main task is recursively broken down using `tasks_generation` into actionable sub-tasks (e.g., "Load Data -> EDA -> Train XGBoost"). These tasks are ordered and added to the `GlobalOrchestrator` tree.
5.  **Execution Loop**: For each sub-task:
    *   The LLM writes a Python script (`perform_task_python_v2`), incorporating previous successful code.
    *   The `BashAgent` executes the script.
    *   The stdout is parsed for a specific signal (`METRICS_JSON:{...}`).
    *   If the code fails or an error is detected, the `ErrorRouter` kicks in to install missing `pip` packages, fix the code, or adjust the path.
6.  **Optimization**: If a sub-task calculates a metric (e.g., accuracy), the `optimize_metrics` module can propose tweaks to improve the score, iteratively generating and testing new code versions.
7.  **Real-Time UI**: While the CLI runs in the background, `server.py` watches the generated `artifacts/tasks_tree.json` and streams updates to the browser via Server-Sent Events (SSE) and WebSockets.

## Call Interface
- **CLI**: `python main.py --config config.yaml --task_file data/task.txt`
- **Dashboard**: `uvicorn server:app --reload` (or running `server.py` directly, which internally launches `main.py` as a subprocess).
