# src/data_meta.py
from __future__ import annotations

import glob
import os, json, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Counter
from collections import Counter
from langchain_core.prompts import ChatPromptTemplate

from .utils import shorten_string_middle
from .llm_utils import invoke_and_log

_IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def _safe_head_csv(path: Path, nrows: int = 100) -> Dict[str, Any]:
    """Читает маленькую шапку CSV, пытаясь определить delimiter/колонки/dtypes без больших затрат."""
    info = {"path": path.as_posix(), "exists": path.exists(), "delimiter": ",", "columns": [], "dtypes": {}}
    if not path.exists(): return info
    # Comment translated to English.
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        info["delimiter"] = dialect.delimiter
    except Exception:
        pass
    # Comment translated to English.
    try:
        import pandas as pd
        df = pd.read_csv(path, nrows=nrows, sep=info["delimiter"])
        info["columns"] = list(df.columns)
        # Comment translated to English.
        info["dtypes"] = {c: str(dt) for c, dt in df.dtypes.items()}
    except Exception:
        # Comment translated to English.
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip()
            if header:
                info["columns"] = [h.strip() for h in header.split(info["delimiter"])]
        except Exception:
            pass
    return info

def _walk_exts(root: Path, max_files_per_dir: int = 200) -> Dict[str, Any]:
    """Обходит директорию, считая расширения; не смотрит содержимое файлов."""
    exts = Counter()
    examples: Dict[str, List[str]] = {}
    total = 0
    for dp, _, files in os.walk(root):
        cnt = 0
        for fn in files:
            ext = Path(fn).suffix.lower()
            exts[ext] += 1
            total += 1
            if ext not in examples:
                examples[ext] = []
            if len(examples[ext]) < 5:
                examples[ext].append(os.path.join(dp, fn))
            cnt += 1
            if cnt >= max_files_per_dir:
                break
        # Comment translated to English.
        if sum(len(v) for v in examples.values()) > 200:
            break
    return {"total_files_seen": total, "ext_counts": dict(exts), "examples": {k:[str(x) for x in v] for k,v in examples.items()}}

def _guess_id_col(cols: List[str]) -> Optional[str]:
    for cand in ["id","image_id","img_id","filename","file","image","name"]:
        if cand in [c.lower() for c in cols]:
            # Comment translated to English.
            for c in cols:
                if c.lower()==cand: return c
    return None

def _as_path_or_default(candidate, default: Path) -> Path:
    """
    candidate: str | Path | None
    Если candidate пустой/None -> вернуть default.
    Если это строка с пробелами -> trim и проверить.
    """
    if isinstance(candidate, Path):
        return candidate
    if isinstance(candidate, str):
        cand = candidate.strip()
        if cand:
            try:
                return Path(cand)
            except Exception:
                return default
    return default

def _as_str(p: Path | str | None) -> str:
    if isinstance(p, Path):
        return p.as_posix()
    if isinstance(p, str):
        return p
    return ""

def collect_raw_observations(spec: Dict[str, Any], project_root: Path, max_samples_per_dir: int = 3) -> Dict[str, Any]:
    """
    Безопасно собираем первичные наблюдения по структуре данных.
    Никаких Path(None) — везде дефолты.
    """
    data = spec.get("data") or {}
    # ROOT
    root_hint = data.get("resolved_root") or data.get("root_hint") or "data"
    # Comment translated to English.
    root = Path(root_hint) if os.path.isabs(str(root_hint)) else (project_root / root_hint)
    root = root.resolve()

    # Comment translated to English.
    train_dir  = _as_path_or_default(data.get("train_dir"),  root / "train").resolve()
    test_dir   = _as_path_or_default(data.get("test_dir"),   root / "test").resolve()
    train_csv  = _as_path_or_default(data.get("train_csv"),  root / "train.csv").resolve()
    labels_csv = _as_path_or_default(data.get("labels_csv"), root / "labels.csv").resolve()

    # Comment translated to English.
    exists = {
        "root": root.is_dir(),
        "train_dir": train_dir.is_dir(),
        "test_dir": test_dir.is_dir(),
        "train_csv": train_csv.is_file(),
        "labels_csv": labels_csv.is_file(),
    }

    # Comment translated to English.
    def _sample_files(dir_path: Path, patterns: List[str], k: int) -> List[str]:
        if not dir_path.is_dir():
            return []
        out: List[str] = []
        for pat in patterns:
            # Comment translated to English.
            out.extend(glob.glob(str(dir_path / "**" / pat), recursive=True))
            if len(out) >= k:
                break
        return [os.path.abspath(p).replace("\\", "/") for p in out[:k]]

    img_patterns   = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff"]
    audio_patterns = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]
    video_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
    doc_patterns   = ["*.pdf", "*.txt", "*.json", "*.xml"]
    tab_patterns   = ["*.csv", "*.xml", "*.tsv", "*.parquet", "*.feather"]


    train_samples = _sample_files(train_dir, img_patterns + audio_patterns + video_patterns + doc_patterns + tab_patterns,
                                  max_samples_per_dir)
    test_samples  = _sample_files(test_dir,  img_patterns + audio_patterns + video_patterns + doc_patterns + tab_patterns,
                                  max_samples_per_dir)

    # Comment translated to English.
    def _ext_hist(paths: List[str]) -> Dict[str, int]:
        h: Dict[str, int] = {}
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext:
                h[ext] = h.get(ext, 0) + 1
        return h

    observations = {
        "abs_root": _as_str(root),
        "exists": exists,
        "files": {
            "train_dir": _as_str(train_dir),
            "test_dir": _as_str(test_dir),
            "train_csv": _as_str(train_csv),
            "labels_csv": _as_str(labels_csv),
            "train_samples": train_samples,
            "test_samples": test_samples,
        },
        "ext_hist": {
            "train": _ext_hist(train_samples),
            "test": _ext_hist(test_samples),
        },
        # Comment translated to English.
        "tree_sample": {
            "train_dir": _as_str(train_dir),
            "test_dir": _as_str(test_dir),
            "train_csv_exists": exists["train_csv"],
            "labels_csv_exists": exists["labels_csv"],
        }
    }
    return observations


def summarize_data_meta_llm(llm_fast, task_text: str, spec: Dict[str, Any], observations: Dict[str, Any]) -> Dict[
    str, Any]:
    """
    Использует LLM для сжатия наблюдений в компактный и полезный объект "meta".
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert data analyst assistant. Your task is to summarize raw observations about a dataset into a structured JSON object called "meta".

Given the TASK description, the project's `spec.data` configuration, and RAW OBSERVATIONS from the filesystem, output ONLY a single, compact JSON object with the following fields:

- `path_aliases`: A dictionary mapping semantic names to absolute file paths. Include `root`, `train_dir`, `test_dir`, `train_csv`, `test_csv`, `labels_csv`. If a path does not exist or is not applicable, its value must be `null`.
- `has_train_dir`, `has_test_dir`: Simple booleans indicating if these directories exist, based on observations.
- `modalities_detected`: An array of strings, e.g., ["image", "tabular", "text"].
- `csv_summaries`: A dictionary where keys are CSV filenames (e.g., "train.csv"). Each value is an object with `delimiter`, `columns` (an array of strings), and `dtypes` (a dictionary mapping column names to their types as strings).
- `id_column`: The best-guess name of the ID column, if any.
- `target_columns`: An array of column names that appear to be prediction targets.
- `notes`: A brief, one or two-sentence summary of the dataset's structure and the task.
- `notes_tree`: A concise, text-based representation of the key directory structure and files, to help orient a developer.

**Rules:**
1.  **Absolute Paths Only:** All paths in `path_aliases` must be absolute.
2.  **Fact-Based:** Base your output strictly on the provided `OBSERVATIONS` and `SPEC_DATA`. Do not invent files or properties.
3.  **Handle Missing Info:** If a value is unknown or not present (e.g., no CSV files), use `null`, an empty string `""`, or an empty list `[]` as appropriate for the field type.
4.  **JSON Only:** Your entire output must be a single, valid JSON object, and nothing else. Start with `{{` and end with `}}`."""),
        ("user", "TASK:\n{task}\n\nSPEC_DATA:\n{spec_data}\n\nOBSERVATIONS:\n{obs}")
    ])

    res = invoke_and_log(
        llm_fast,
        prompt,
        {
            "task": task_text,
            "spec_data": json.dumps(spec.get("data", {}), ensure_ascii=False, indent=2),
            "obs": json.dumps(observations, ensure_ascii=False, indent=2),
        },
        agent_name="data_meta_summarizer",
    )

    # Robustly parse the JSON from the LLM's response
    try:
        text = getattr(res, "content", "") or ""
        
        # Ensure text is a string if LangChain returns a list of blocks
        if isinstance(text, list):
            text_parts = []
            for part in text:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(str(part['text']))
                elif isinstance(part, str):
                    text_parts.append(part)
            text = "".join(text_parts)
        elif not isinstance(text, str):
            text = str(text)

        # Find the outermost JSON object
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1:
            json_str = text[s:e + 1]
            meta = json.loads(json_str)
            # Ensure it's a dictionary
            return meta if isinstance(meta, dict) else {}
        return {}
    except json.JSONDecodeError:
        # If parsing fails, return an empty dict to avoid crashing
        return {}


def build_data_meta(orch, llm_fast, spec: Dict[str, Any], task_text: str, max_samples_per_dir: int) -> Dict[str, Any]:
    """Собирает наблюдения, вызывает LLM для их сжатия в 'meta' и сохраняет результат."""
    # Comment translated to English.
    obs = collect_raw_observations(spec, orch.project_root, max_samples_per_dir=max_samples_per_dir)

    # Comment translated to English.
    meta = summarize_data_meta_llm(llm_fast, task_text, spec, obs)

    # Comment translated to English.
    if not meta or not isinstance(meta, dict):
        # Comment translated to English.
        meta = {
            "path_aliases": obs.get("files", {}),
            "has_train_dir": obs.get("exists", {}).get("train_dir", False),
            "has_test_dir": obs.get("exists", {}).get("test_dir", False),
            "modalities_detected": [],
            "csv_summaries": {},
            "id_column": "",
            "target_columns": [],
            "notes": "Failed to generate data meta via LLM. This is a fallback structure.",
            "notes_tree": "Could not be generated."
        }

    # Comment translated to English.
    meta_path = Path(orch.cfg.paths.artifacts_dir) / "data_meta.json"
    orch.write_file(meta_path.as_posix(), json.dumps(meta, ensure_ascii=False, indent=2))

    # Comment translated to English.
    if "data" not in spec: spec["data"] = {}
    spec["data"]["meta"] = meta

    return spec
