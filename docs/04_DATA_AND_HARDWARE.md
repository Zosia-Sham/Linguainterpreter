# 4. Data and Hardware Modules

These modules are responsible for understanding the environment and the data without executing heavy, memory-intensive operations prematurely.

## `src/data_meta.py`
**Purpose**: Safely explores and summarizes the dataset provided to the agent.
**How it works**:
- **`collect_raw_observations()`**: Performs a lightweight filesystem scan. It locates the train/test directories and CSV files. It uses `glob` to sample a small number of files (e.g., 3 images, 3 audio files) without reading their full contents into memory. It generates a histogram of file extensions.
- **`summarize_data_meta_llm()`**: Passes the raw filesystem observations to an LLM, asking it to construct a clean JSON `meta` object that infers the task modality (`image`, `tabular`), identifies the target columns, and maps absolute path aliases.
- This creates the `data_meta.json` artifact, preventing the code-generation LLM from hallucinating paths or making wrong assumptions about the dataset.

## `src/hardware.py`
**Purpose**: Probes the host machine's hardware capabilities to set safe resource constraints for the generated code.
**How it works**:
- **`probe_hardware()`**: Uses the `psutil` library (if installed) or parses `/proc/meminfo` on Linux to determine available RAM and physical/logical CPU cores.
- Uses `pynvml`, the `torch` module, or parses `nvidia-smi` shell output to detect available GPUs and free VRAM.
- **`estimate_dataset_footprint()`**: Evaluates the size of the dataset on disk by summing up file sizes, up to a limit of 5000 files, to prevent hanging on massive datasets.
- **`recommend_resource_plan()`**: Applies heuristics based on the hardware and dataset footprint to recommend the number of DataLoader workers, thread counts, maximum batch sizes, and whether to use mixed precision (AMP). This plan is attached to the `spec.json`.

## `src/dataset_checker.py`
**Purpose**: Validate and enrich `spec.data` using the real filesystem: static probing, optional **zip extraction**, and an optional **LLM + tools** pass that can run shell commands.

**How it works**:
- **`probe_dataset_with_bash()`** (despite the name, the implementation is OS-aware): Builds and runs a **single host-native command** via `BashAgent` — on Windows this is **PowerShell**; on Linux/macOS it uses a **Python heredoc** driven by `bash -lc` to scan the tree and print one JSON blob to stdout. The result includes existence flags, sample paths, extension histograms, and `archives_found`.
- **`_find_archives()`**: Locates archive files under the resolved root (used to decide whether unpacking is needed).
- **`_auto_unpack_zip_archives()`**: When expected folders (e.g. `train/`, `test/`) are missing but matching `train.zip` / `test.zip` exist, unpacks with **`zipfile`** inside **`GlobalOrchestrator.run_python_code()`** (venv Python). This is **reliable on Windows** (no reliance on bash-style `python <<'PY'` heredocs inside PowerShell). Idempotency uses `.unpacked_markers/<stem>.done`; if a marker exists but the expected directory is still absent, the marker is removed and extraction is retried.
- **`_run_dataset_react_agent()`**: If `orch.monitor_llm` is available, runs up to `data_check.react_max_rounds` rounds of **tool calling** using `StructuredTool` definitions and **`invoke_with_tools`** in `src/llm_utils.py`. Tools execute through **`orch.bash.run`** (`shell_exec`). The model is instructed to call **`shell_exec`** with OS-appropriate unzip/`Expand-Archive` commands when zips are present and directories are missing. **LangGraph / `create_react_agent` are not used**; only `langchain_core` tools + `bind_tools` are required.
- **`_scrub_missing_data_paths()`**: Drops stale path fields that do not exist on disk after probing.

Together, this isolates heavy or interactive filesystem work from the main process and keeps `spec.data` aligned with reality before `data_meta` summarization.
