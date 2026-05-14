# 9. Deep Dive: Memory, Context Passing, and `--resume`

This document provides a detailed explanation of how LinguaInterpreter manages memory (LLM context window), passes state between different modules and agents, and ensures stable recovery using the `--resume` flag. Understanding this is crucial for preventing context window explosion and ensuring flawless task continuation.

## 1. The `--resume` Logic: State Recovery

The `--resume` flag allows the orchestrator to continue a previously interrupted session without restarting from scratch. This is vital for long-running ML tasks (e.g., training deep neural networks) that might fail midway due to OOM errors or timeouts.

### How it works step-by-step:

1. **CLI Invocation**: When `main.py` is called with `--resume`, the `AppConfig` and `GlobalOrchestrator` are initialized, but they **do not purge** the `artifacts/` directory.
2. **State Hydration**:
   - `orchestrator.load_tree()` reads `artifacts/tree.json` and `artifacts/tasks_tree.json`.
   - The orchestrator rebuilds its internal `self.nodes` dictionary, restoring the exact state (`pending`, `running`, `done`, `failed`) of every task and sub-task.
3. **Spec and Meta Loading**:
   - The pipeline checks if `artifacts/spec.json` exists. If it does, it loads it directly into memory, skipping the expensive `problem_spec_from_text` LLM call.
   - It also loads `data_meta.json` and `hardware.json` to skip filesystem and hardware probing.
4. **Execution Loop Resumption**:
   - The `main_pipeline` iterates through the ordered nodes.
   - **Crucial Rule**: If a node's status is `"done"`, the pipeline **skips it entirely**.
   - If a node is `"running"` or `"failed"`, it assumes the task was interrupted or needs fixing, and begins execution from that exact node.
5. **Context Re-injection**:
   - To resume a task, the LLM needs to know what code was written before the crash. The pipeline reads `artifacts/last/code.py` (or the last successful version from `artifacts/versions/index.json`).
   - This script is injected into the LLM prompt as `previous_code`.

**Why this is safe:**
Because the state is saved to disk *only after* all artifacts (metrics, code, submissions) are successfully written, a crash during execution will leave the node in a `"running"` state, ensuring it gets re-executed on resume with the last known good context.

## 2. Memory Management: Passing Context Between Modules

A common issue with autonomous agents is "context window explosion," where the agent is fed its entire history, causing it to hallucinate, slow down, or hit token limits. LinguaInterpreter solves this by **aggressively compressing and compartmentalizing state**.

### A. The Specification (`spec.json`) as Ground Truth
Instead of passing the entire conversation history to every agent, the system compresses the initial user request and the data structure into a single, compact JSON object: `spec.json`.
- **How it's passed**: Every major code-generation agent (`perform_task_python_v2`, `finetune_code_v2`, `error_triage_agent`) receives the `spec.json` as a stringified variable in its system prompt.
- **Why it works**: It provides the exact constraints (e.g., `maximize: true`, `internet_allowed: false`) without the noise of how those constraints were derived.

### B. Code Context (`previous_code`)
Instead of keeping a running chat history of every generated Python script, the system only passes the **absolute best, most recent, fully functional script**.
- **How it's passed**: When generating code for Task N, the prompt for `perform_task_python_v2` receives the source code generated in Task N-1 as `previous_code`.
- **The LLM's Job**: The LLM is instructed to act like it's writing the next cell in a Jupyter Notebook. It must take the `previous_code` (which contains imports, data loading, and previous processing) and append the new logic for Task N, returning a single cohesive script.

### C. Error Context (`ErrorRouter`)
When a script fails, the entire codebase and the full stdout/stderr are not dumped blindly into the context.
- **How it's passed**: The `ErrorRouter` passes the `spec.json`, the **head** of the code (to check imports), and the **tail** of stdout/stderr (to catch the actual exception) to the `error_triage_agent`.
- **Actionable Output**: The agent returns a tiny JSON object (e.g., `{"route": "install", "packages": ["pandas"]}`). This JSON is executed by the orchestrator (via Bash), keeping the LLM out of the loop until it needs to write code again.

### D. The Optimizer (`Improver`) Memory
When optimizing metrics, the system avoids generating entirely new scripts from scratch.
- **How it's passed**: The optimizer passes the `best_code.py` (truncated if too long) and the `best_metrics.json` to the `_proposal_agent`.
- **The Output**: The agent returns specific, targeted hints (e.g., "Change learning rate to 0.001"). These hints are then fed into `finetune_code_v2` along with the original script, ensuring the LLM only focuses on surgical changes.

## 3. The Role of `utils.clean_specs`
To further save tokens, the `clean_specs` function (in `src/utils.py`) recursively scrubs the JSON structures before they are passed to the LLM. It removes any keys that map to `null`, empty strings `""`, empty lists `[]`, or empty dictionaries `{}`. This ensures the LLM doesn't waste attention on irrelevant or missing data fields.

## Summary of Context Flow
1. **User Prompt** -> `problem_spec_from_text` -> **`spec.json`** (Stored)
2. **`spec.json`** + Filesystem -> `build_data_meta` -> **`data_meta.json`** (Merged into `spec`)
3. **`spec`** -> `tasks_generation` -> **`tasks_tree.json`** (Stored)
4. **Task N** + **`spec`** + **`artifacts/last/code.py`** -> `perform_task_python_v2` -> **New Python Script**
5. **New Script** -> `BashAgent` -> **`METRICS_JSON`**
6. **`METRICS_JSON`** + **New Script** -> `optimizer.py` -> **`best_code.py`** (Stored in `artifacts/best/`)
