# src/dataset_checker.py
from __future__ import annotations
import json
import os
import platform
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from langchain_core.tools import StructuredTool
except Exception:
    StructuredTool = None

try:
    from colorama import Fore
except Exception:
    class Fore:
        CYAN = ""
        YELLOW = ""
        GREEN = ""

from src.llm_utils import log_agent_trace, invoke_with_tools


def _parse_json_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    if not stdout:
        return None
    s = stdout.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except Exception:
        return None


def _find_archives(root: str) -> list[str]:
    exts = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar"}
    out: list[str] = []
    try:
        rp = Path(root)
        if not rp.exists():
            return out
        for p in rp.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(str(p.resolve()))
            if len(out) >= 100:
                break
    except Exception:
        return []
    return out


def _auto_unpack_zip_archives(orch, root: str, archives: list[str]) -> list[str]:
    """
    Unpack zip archives into the dataset root (idempotent best-effort).
    Uses project venv Python via run_python_code (works on Windows PowerShell; bash heredocs do not).
    Returns list of extracted zip paths.
    """
    extracted: list[str] = []
    if not archives:
        return extracted
    root_abs = str(Path(root).resolve())
    for ap in archives:
        p = Path(ap)
        if p.suffix.lower() != ".zip":
            continue
        stem = p.stem.lower()
        marker = Path(root_abs) / ".unpacked_markers"
        marker.mkdir(parents=True, exist_ok=True)
        done = marker / f"{p.stem}.done"
        if done.exists():
            if stem == "train" and not (Path(root_abs) / "train").is_dir():
                try:
                    done.unlink()
                except Exception:
                    pass
            elif stem == "test" and not (Path(root_abs) / "test").is_dir():
                try:
                    done.unlink()
                except Exception:
                    pass
            else:
                continue
        safe_zip = repr(str(p.resolve()))
        safe_root = repr(root_abs)
        code = f"""import zipfile
from pathlib import Path
z = Path({safe_zip})
dst = Path({safe_root})
dst.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(z, "r") as f:
    f.extractall(dst)
print("OK")
"""
        res = orch.run_python_code(code, filename="unpack_zip.py", timeout=600)
        if int(res.get("exit_code", 1)) == 0:
            extracted.append(str(p))
            try:
                done.write_text("ok", encoding="utf-8")
            except Exception:
                pass
        else:
            err = (res.get("errors") or res.get("stderr") or "")[:800]
            print(Fore.YELLOW + f"[DATASET] Unpack failed for {p}: {err}")
    return extracted


def _path_exists_for_key(key: str, value: str) -> bool:
    p = Path(value)
    if key.endswith("_dir"):
        return p.is_dir()
    return p.is_file() or p.is_dir()


def _scrub_missing_data_paths(data: Dict[str, Any]) -> None:
    """Drop spec.data path fields that point to non-existent files/dirs (stale LLM / ReAct paths)."""
    if not isinstance(data, dict):
        return
    for k in ("train_csv", "labels_csv", "train_dir", "test_dir"):
        v = data.get(k)
        if not isinstance(v, str) or not v.strip():
            continue
        p = Path(v)
        if not _path_exists_for_key(str(k), str(p.resolve())):
            data[k] = None
    rr = data.get("resolved_root")
    if isinstance(rr, str) and rr.strip() and not Path(rr).exists():
        data["resolved_root"] = ""


def _run_dataset_react_agent(orch, abs_root: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    ReAct-style loop with shell-backed tools (shell_ls, shell_exec, …) via bind_tools + invoke_with_tools.
    Does not require langgraph. The model must call shell_exec to unpack zips when dirs are missing.
    """
    if StructuredTool is None:
        msg = "dataset_react skipped: langchain_core.tools.StructuredTool unavailable"
        print(Fore.YELLOW + f"[DATASET_REACT] {msg}")
        log_agent_trace("dataset_react_probe", "skipped", msg)
        return {}
    llm = getattr(orch, "monitor_llm", None)
    if llm is None:
        msg = "dataset_react skipped: orch.monitor_llm is None (set after build_llms in main)"
        print(Fore.YELLOW + f"[DATASET_REACT] {msg}")
        log_agent_trace("dataset_react_probe", "skipped", msg)
        return {}
    if not abs_root or not os.path.exists(abs_root):
        msg = f"dataset_react skipped: bad resolved_root {abs_root!r}"
        print(Fore.YELLOW + f"[DATASET_REACT] {msg}")
        log_agent_trace("dataset_react_probe", "skipped", msg)
        return {}
    try:
        root = Path(abs_root).resolve()
    except Exception as ex:
        msg = f"dataset_react skipped: resolve root failed: {ex}"
        print(Fore.YELLOW + f"[DATASET_REACT] {msg}")
        log_agent_trace("dataset_react_probe", "skipped", msg)
        return {}

    def shell_ls(rel_path: str = ".") -> str:
        """List files/directories under a path. Input: relative path from dataset root."""
        p = (root / (rel_path or ".")).resolve()
        if root not in p.parents and p != root:
            return "Path is outside dataset root."
        cmd = f'Get-ChildItem -Force "{p.as_posix()}" | Select-Object Mode,Length,Name | Format-Table -AutoSize' if platform.system() == "Windows" else f'ls -lah "{p.as_posix()}"'
        res = orch.bash.run(cmd, timeout=45)
        return (res.get("stdout", "") or "")[:8000]

    def shell_exec(command: str) -> str:
        """Execute a shell command for dataset investigation/fix (unzip/list/move). Keep it scoped to dataset root."""
        cmd = str(command or "").strip()
        if not cmd:
            return "Empty command."
        blocked = (" rm ", " rmdir ", " del ", " format ", " mkfs ", " shutdown ", " reboot ", "git reset", "git clean")
        low = f" {cmd.lower()} "
        if any(b in low for b in blocked):
            return "Blocked unsafe command."
        res = orch.bash.run(cmd, timeout=300)
        out = (res.get("stdout", "") or "")[:6000]
        err = (res.get("stderr", "") or "")[:2000]
        return f"exit={res.get('exit_code', 1)}\nSTDOUT:\n{out}\nSTDERR:\n{err}"

    def shell_find_archives(rel_path: str = ".") -> str:
        """Find archive files (.zip/.tar/.gz/.7z). Input: relative path from dataset root."""
        p = (root / (rel_path or ".")).resolve()
        if root not in p.parents and p != root:
            return "Path is outside dataset root."
        if platform.system() == "Windows":
            cmd = f'Get-ChildItem "{p.as_posix()}" -Recurse -File -Include *.zip,*.tar,*.gz,*.7z,*.rar | Select-Object -First 40 FullName,Length'
        else:
            cmd = f'find "{p.as_posix()}" -type f \\( -name "*.zip" -o -name "*.tar" -o -name "*.gz" -o -name "*.7z" -o -name "*.rar" \\) | head -n 40'
        res = orch.bash.run(cmd, timeout=60)
        return (res.get("stdout", "") or "")[:8000]

    def shell_head_csv(rel_path: str) -> str:
        """Show first rows of a CSV file. Input: relative CSV path from dataset root."""
        p = (root / (rel_path or "")).resolve()
        if not p.exists():
            return f"Missing file: {p.as_posix()}"
        if root not in p.parents:
            return "Path is outside dataset root."
        if platform.system() == "Windows":
            cmd = f'Import-Csv "{p.as_posix()}" | Select-Object -First 5 | ConvertTo-Json -Depth 4'
        else:
            cmd = (
                f"{orch.bash.python_exec} -c \"import pandas as pd; "
                f"print(pd.read_csv(r'{p.as_posix()}').head(5).to_json(orient='records'))\""
            )
        res = orch.bash.run(cmd, timeout=45)
        return (res.get("stdout", "") or "")[:8000]

    tools = [
        StructuredTool.from_function(shell_ls, name="shell_ls", description=shell_ls.__doc__ or ""),
        StructuredTool.from_function(shell_find_archives, name="shell_find_archives", description=shell_find_archives.__doc__ or ""),
        StructuredTool.from_function(shell_head_csv, name="shell_head_csv", description=shell_head_csv.__doc__ or ""),
        StructuredTool.from_function(shell_exec, name="shell_exec", description=shell_exec.__doc__ or ""),
    ]

    try:
        from langchain_core.prompts import ChatPromptTemplate

        max_rounds = int(getattr(getattr(orch.cfg, "data_check", object()), "react_max_rounds", 3) or 3)
        max_rounds = max(1, min(max_rounds, 8))
        latest: Dict[str, Any] = {}
        prompt = ChatPromptTemplate.from_messages([("system", "{sys_prompt}"), ("user", "{user_prompt}")])
        print(
            Fore.CYAN
            + f"[DATASET_REACT] bind_tools ReAct starting ({max_rounds} round(s)), root={root.as_posix()} "
            + "(tools: shell_ls, shell_find_archives, shell_head_csv, shell_exec)"
        )
        log_agent_trace(
            "dataset_react_probe",
            "start",
            {"root": root.as_posix(), "max_rounds": max_rounds, "spec_data_keys": list((spec or {}).get("data", {}).keys())},
        )
        for i in range(max_rounds):
            sys_prompt = (
                "You are a dataset inspector and fixer. You MUST use tools (no guessed paths). "
                "If train/ or test/ are missing but train.zip and/or test.zip exist in the dataset root, "
                "you MUST unpack them by calling shell_exec with OS-appropriate commands: "
                "on Windows use PowerShell, e.g. Expand-Archive -Path '<abs>\\\\train.zip' -DestinationPath '<dataset_root>' -Force; "
                "on Linux/macOS use unzip. Then shell_ls to verify train/ and test/ exist. "
                "List directories with shallow commands only; do not dump huge trees. "
                "In recommendations, state what you unpacked and which folder is train vs test. "
                "Never put labels_csv in resolved_paths unless that file exists on disk. "
                "Return strictly one JSON object with keys: "
                '{"recommendations":[],"resolved_paths":{},"warnings":[],"done":false}. '
                "All resolved_paths entries must be absolute paths that exist. "
                "Call at least 2 tools (including shell_exec when zips need unpacking) before final JSON each round."
            )
            user_prompt = (
                f"Round {i+1}/{max_rounds}\n"
                f"Dataset root: {root.as_posix()}\n"
                f"Current spec.data: {json.dumps((spec or {}).get('data', {}), ensure_ascii=False)}\n"
                f"Previous result: {json.dumps(latest, ensure_ascii=False)}\n"
                "Inspect, unpack if needed via shell_exec, and re-check. Mark done=true only if key data paths are valid."
            )
            res = invoke_with_tools(
                llm,
                prompt,
                {"sys_prompt": sys_prompt, "user_prompt": user_prompt},
                tools=tools,
                agent_name="dataset_react_probe",
            )
            content = getattr(res, "content", None) or ""
            log_agent_trace("dataset_react_probe", f"round_{i + 1}_last_content", str(content)[:12000])
            if not isinstance(content, str):
                continue
            parsed = _parse_json_stdout(content)
            if isinstance(parsed, dict):
                latest = parsed
                log_agent_trace("dataset_react_probe", f"round_{i + 1}_parsed_json", parsed)
                if bool(parsed.get("done", False)):
                    print(Fore.GREEN + "[DATASET_REACT] done=true from model JSON")
                    break
            print(Fore.CYAN + f"[DATASET_REACT] round {i + 1}/{max_rounds} complete")
        if latest:
            return latest
        print(Fore.YELLOW + "[DATASET_REACT] no structured JSON from ReAct; check dataset_react_probe.jsonl")
    except Exception as ex:
        log_agent_trace("dataset_react_probe", "error", str(ex))
        print(Fore.YELLOW + f"[DATASET_REACT] exception: {ex}")
        return {}
    return {}


def _coerce_react_result(raw: Dict[str, Any], default_root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(raw, dict):
        raw = {}
    rp = raw.get("resolved_paths", {}) if isinstance(raw.get("resolved_paths", {}), dict) else {}
    out["resolved_root"] = str(rp.get("root") or default_root or "").strip()
    out["train_dir"] = str(rp.get("train_dir") or "").strip()
    out["test_dir"] = str(rp.get("test_dir") or "").strip()
    out["train_csv"] = str(rp.get("train_csv") or "").strip()
    out["labels_csv"] = str(rp.get("labels_csv") or "").strip()
    recs = raw.get("recommendations", [])
    warns = raw.get("warnings", [])
    out["recommendations"] = [str(x) for x in recs[:30]] if isinstance(recs, list) else []
    out["warnings"] = [str(x) for x in warns[:30]] if isinstance(warns, list) else []
    out["done"] = bool(raw.get("done", False))
    return out


def probe_dataset_with_bash(orch, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Лёгкая проверка структуры данных.
    Расширено: возвращаем samples_by_type, ext_hist, tree_sample, помимо абсолютных путей.
    """
    try:
        max_samples = int(getattr(getattr(orch.cfg, "data_check", object()), "max_samples_per_dir", 3))
    except Exception:
        max_samples = 3

    data = spec.get("data", {}) or {}
    cfg_data_dir = getattr(orch.cfg.paths, 'data_dir', 'data')
    if not data.get("resolved_root"):
        root_hint = cfg_data_dir if os.path.isabs(cfg_data_dir) else f"./{cfg_data_dir}"
    else:
        root_hint = data["resolved_root"]

    os_name = platform.system() or "Windows"

    # Comment translated to English.
    img_pats = "*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff,*.webp"
    aud_pats = "*.wav,*.mp3,*.flac,*.ogg,*.m4a"
    vid_pats = "*.mp4,*.avi,*.mov,*.mkv,*.webm"
    txt_pats = "*.txt,*.md"
    tab_pats = "*.csv,*.tsv,*.parquet,*.feather,*.json"
    doc_pats = "*.pdf,*.docx,*.pptx,*.xlsx"

    if os_name == "Windows":
        cmd = f'''
$ErrorActionPreference = "SilentlyContinue"
$hint = @"
{root_hint}
"@
try {{
  $abs = (Resolve-Path $hint -ErrorAction Stop).Path
}} catch {{
  $abs = [System.IO.Path]::GetFullPath($hint)
}}

function Get-SamplesFrom {{
  param(
    [string]$Base,
    [string[]]$Include,
    [int]$K = {max_samples}
  )
  if (-not (Test-Path $Base)) {{ return @() }}
  # Comment translated to English.
  Get-ChildItem -Path $Base -File -Recurse -Include $Include -ErrorAction SilentlyContinue |
    Select-Object -First $K |
    ForEach-Object {{ $_.FullName }}
}}

$trainCsv = Join-Path $abs "train.csv"
$labelsCsv = Join-Path $abs "labels.csv"
$trainDir = Join-Path $abs "train"
$testDir  = Join-Path $abs "test"

$exists = [PSCustomObject]@{{
  root       = (Test-Path $abs)
  train_csv  = (Test-Path $trainCsv)
  labels_csv = (Test-Path $labelsCsv)
  train_dir  = (Test-Path $trainDir)
  test_dir   = (Test-Path $testDir)
}}

# Comment translated to English.
$baseForSamples = if ($exists.train_dir -or $exists.test_dir) {{ @($trainDir,$testDir) }} else {{ @($abs) }}

# Comment translated to English.
$imgs = @(); $auds=@(); $vids=@(); $txts=@(); $tabs=@(); $docs=@()
foreach($b in $baseForSamples){{
  $imgs += Get-SamplesFrom -Base $b -Include @({img_pats.split(",")}) -K {max_samples}
  $auds += Get-SamplesFrom -Base $b -Include @({aud_pats.split(",")}) -K {max_samples}
  $vids += Get-SamplesFrom -Base $b -Include @({vid_pats.split(",")}) -K {max_samples}
  $txts += Get-SamplesFrom -Base $b -Include @({txt_pats.split(",")}) -K {max_samples}
  $tabs += Get-SamplesFrom -Base $b -Include @({tab_pats.split(",")}) -K {max_samples}
  $docs += Get-SamplesFrom -Base $b -Include @({doc_pats.split(",")}) -K {max_samples}
}}

# Comment translated to English.
$topDirs = Get-ChildItem -Path $abs -Directory -ErrorAction SilentlyContinue | Select-Object -First 10 | ForEach-Object {{ $_.FullName }}
$topFiles = Get-ChildItem -Path $abs -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First {max_samples*10} | ForEach-Object {{ $_.FullName }}

# Comment translated to English.
function Get-Ext([string]$p) {{ [System.IO.Path]::GetExtension($p).ToLowerInvariant() }}
$allSamples = @() + $imgs + $auds + $vids + $txts + $tabs + $docs
$hist = @{{}}
foreach($p in $allSamples){{
  $e = Get-Ext $p
  if (-not $hist.ContainsKey($e)) {{ $hist[$e] = 0 }}
  $hist[$e] += 1
}}

$result = [PSCustomObject]@{{
  abs_root = $abs
  exists   = $exists
  files    = [PSCustomObject]@{{
    train_csv  = $trainCsv
    labels_csv = $labelsCsv
    train_dir  = $trainDir
    test_dir   = $testDir
    train_samples = if ($exists.train_dir) {{ Get-SamplesFrom -Base $trainDir -Include @({img_pats.split(",")}) -K {max_samples} }} else {{ @() }}
    test_samples  = if ($exists.test_dir)  {{ Get-SamplesFrom -Base $testDir  -Include @({img_pats.split(",")}) -K {max_samples} }} else {{ @() }}
  }}
  samples_by_type = [PSCustomObject]@{{
    images = $imgs
    audio  = $auds
    video  = $vids
    text   = $txts
    tabular= $tabs
    docs   = $docs
  }}
  ext_hist = $hist
  tree_sample = [PSCustomObject]@{{ top_dirs = $topDirs; top_files = $topFiles }}
}}
$result | ConvertTo-Json -Depth 8
'''
    else:
        # Comment translated to English.
        cmd = rf'''
python - <<'PY'
import os, json, itertools

hint = r"{root_hint}"
abs_root = os.path.abspath(hint)
train_csv = os.path.join(abs_root, "train.csv")
labels_csv = os.path.join(abs_root, "labels.csv")
train_dir = os.path.join(abs_root, "train")
test_dir  = os.path.join(abs_root, "test")

exists = {{
  "root": os.path.isdir(abs_root),
  "train_csv": os.path.isfile(train_csv),
  "labels_csv": os.path.isfile(labels_csv),
  "train_dir": os.path.isdir(train_dir),
  "test_dir": os.path.isdir(test_dir),
}}

IMG = {{".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}}
AUD = {{".wav",".mp3",".flac",".ogg",".m4a"}}
VID = {{".mp4",".avi",".mov",".mkv",".webm"}}
TXT = {{".txt",".md"}}
TAB = {{".csv",".tsv",".parquet",".feather",".json"}}
DOC = {{".pdf",".docx",".pptx",".xlsx"}}

def sample_under(base, exts, k={max_samples}):
    out = []
    if not os.path.isdir(base): return out
    # Comment translated to English.
    for dp, dn, fn in os.walk(base):
        for f in fn:
            if len(out) >= k: return out
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                out.append(os.path.join(dp, f))
        if len(out) >= k: break
    return out

bases = [p for p in (train_dir, test_dir) if os.path.isdir(p)] or [abs_root]

images, audio, video, text, tabular, docs = [], [], [], [], [], []
for b in bases:
    images  += sample_under(b, IMG)
    audio   += sample_under(b, AUD)
    video   += sample_under(b, VID)
    text    += sample_under(b, TXT)
    tabular += sample_under(b, TAB)
    docs    += sample_under(b, DOC)

def tree_sample(root, max_dirs=10, max_files={max_samples*10}):
    dirs, files = [], []
    try:
        for dp, dn, fn in os.walk(root):
            # Comment translated to English.
            if dp == root:
                dirs = [os.path.join(dp, d) for d in dn[:max_dirs]]
            # Comment translated to English.
            for f in fn:
                if len(files) >= max_files: break
                files.append(os.path.join(dp, f))
            break
    except Exception:
        pass
    return {{"top_dirs": dirs, "top_files": files}}

# Comment translated to English.
def ext_histogram(paths):
    hist = {{}}
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        hist[ext] = hist.get(ext, 0) + 1
    return hist

all_samples = images + audio + video + text + tabular + docs
out = {{
  "abs_root": abs_root,
  "exists": exists,
  "files": {{
    "train_csv": train_csv,
    "labels_csv": labels_csv,
    "train_dir": train_dir,
    "test_dir":  test_dir,
    "train_samples": sample_under(train_dir, IMG) if exists["train_dir"] else [],
    "test_samples":  sample_under(test_dir,  IMG) if exists["test_dir"]  else [],
  }},
  "samples_by_type": {{
    "images": images, "audio": audio, "video": video, "text": text, "tabular": tabular, "docs": docs
  }},
  "ext_hist": ext_histogram(all_samples),
  "tree_sample": tree_sample(abs_root),
}}
print(json.dumps(out, ensure_ascii=False))
PY
'''
    probe_timeout = int(getattr(getattr(orch.cfg, "data_check", object()), "probe_timeout_sec", 180))
    res = orch.bash.run(cmd, timeout=probe_timeout)
    info = _parse_json_stdout(res.get("stdout", "")) or {}
    abs_root = str(info.get("abs_root", data.get("resolved_root", root_hint)) or "")

    # If expected structure is missing but archives exist, unpack zip archives and re-probe.
    archives = _find_archives(abs_root) if abs_root else []
    exists0 = info.get("exists", {}) if isinstance(info.get("exists", {}), dict) else {}
    missing_expected = not bool(exists0.get("train_dir") or exists0.get("test_dir") or exists0.get("train_csv"))
    # train.csv may exist at root while images still live only inside train.zip / test.zip — unpack anyway.
    needs_zip_layout = False
    for ap in archives:
        try:
            if Path(ap).suffix.lower() != ".zip":
                continue
            stem = Path(ap).stem.lower()
            if stem == "train" and not exists0.get("train_dir"):
                needs_zip_layout = True
            if stem == "test" and not exists0.get("test_dir"):
                needs_zip_layout = True
        except Exception:
            continue
    extracted = []
    if archives and (missing_expected or needs_zip_layout):
        extracted = _auto_unpack_zip_archives(orch, abs_root, archives)
        if extracted:
            res = orch.bash.run(cmd, timeout=probe_timeout)
            info = _parse_json_stdout(res.get("stdout", "")) or info

    # Comment translated to English.
    exists_info = info.get("exists", {})
    
    # Only keep paths that actually exist, remove non-existent ones
    train_dir_path = info.get("files", {}).get("train_dir", data.get("train_dir"))
    test_dir_path = info.get("files", {}).get("test_dir", data.get("test_dir"))
    train_csv_path = info.get("files", {}).get("train_csv", data.get("train_csv"))
    labels_csv_path = info.get("files", {}).get("labels_csv", data.get("labels_csv"))
    
    # Remove directory paths if they don't exist
    if exists_info.get("train_dir") is False:
        train_dir_path = None
    if exists_info.get("test_dir") is False:
        test_dir_path = None

    if exists_info.get("train_csv") is False:
        train_csv_path = None
    if exists_info.get("labels_csv") is False:
        labels_csv_path = None
    
    data.update({
        "resolved_root": info.get("abs_root", data.get("resolved_root", root_hint)),
        "train_csv": train_csv_path,
        "labels_csv": labels_csv_path,
        "train_dir": train_dir_path,
        "test_dir": test_dir_path,
        "train_samples": info.get("files", {}).get("train_samples", []),
        "test_samples": info.get("files", {}).get("test_samples", []),
        "samples_by_type": info.get("samples_by_type", {}),
        "ext_hist": info.get("ext_hist", {}),
        "tree_sample": info.get("tree_sample", {}),
        "exists": exists_info,
        "archives_found": archives[:50],
        "archives_unpacked": extracted,
    })
    # Ensure full absolute paths only.
    for key in ("resolved_root", "train_csv", "labels_csv", "train_dir", "test_dir"):
        v = data.get(key)
        if isinstance(v, str) and v:
            data[key] = os.path.abspath(v)

    # After bash probe + optional static zip unpack: LangGraph ReAct (tools) for inspect/unpack narrative.
    # IMPORTANT: Do not rerun the ReAct probe on every spec_update/resume loop if it already succeeded.
    # It is expensive and becomes noisy when the router forces spec_update repeatedly.
    try:
        already_done = bool(data.get("react_done", False))
        exists_now = data.get("exists", {}) if isinstance(data.get("exists", {}), dict) else {}
        has_minimum_paths = bool(exists_now.get("root")) and bool(exists_now.get("train_csv"))
    except Exception:
        already_done, has_minimum_paths = False, False

    if not (already_done and has_minimum_paths):
        react_info = _run_dataset_react_agent(orch, data.get("resolved_root", ""), spec)
        if isinstance(react_info, dict) and react_info:
            structured = _coerce_react_result(react_info, data.get("resolved_root", ""))
            recs = react_info.get("recommendations", [])
            if isinstance(recs, list):
                data["recommendations"] = [str(x) for x in recs[:20]]
            warns = react_info.get("warnings", [])
            if isinstance(warns, list):
                data["warnings"] = [str(x) for x in warns[:20]]
            for k in ("resolved_root", "train_dir", "test_dir", "train_csv", "labels_csv"):
                v = str(structured.get(k, "") or "").strip()
                if not v:
                    continue
                vp = os.path.abspath(v)
                if k == "resolved_root":
                    if Path(vp).exists():
                        data[k] = vp
                    continue
                if _path_exists_for_key(str(k), vp):
                    data[k] = vp
            # Keep final status marker for downstream prompts/debugging.
            data["react_done"] = bool(structured.get("done", False))

    _scrub_missing_data_paths(data)
    spec["data"] = data
    return spec
