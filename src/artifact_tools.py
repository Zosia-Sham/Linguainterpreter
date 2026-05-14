"""
Git-anchored artifact tools.

The artifacts directory (``<project_root>/<cfg.paths.artifacts_dir>``) is the
ONE writable sandbox that agents may touch for persisted outputs. This module
exposes:

- ``ensure_artifacts_repo(orch)``  : idempotent ``git init`` + baseline commit
  for the artifacts dir. Safe to call many times.
- ``snapshot_artifacts(orch, msg)`` : commits current state, returns the commit
  sha. Use before/after a subtask to get a stable reference for diffing.
- ``artifacts_diff_since(orch, sha)`` : list files added/modified/deleted since
  a given commit sha. This is the ground-truth replacement for regex-based
  ``_verify_claimed_artifacts``.
- ``tool_save_artifact(orch, path, content)`` : write a file strictly under
  artifacts_dir (traversal-blocked). Returns sha256+size. The ONLY sanctioned
  write path for agent-produced artifacts.
- ``tool_list_artifacts(orch, subdir)`` : enumerate files under artifacts_dir
  with size and sha256.
- ``tool_read_artifact(orch, path, n_bytes)`` : read a file under artifacts_dir
  (head, capped).
- ``tool_artifacts_diff(orch, since=None)`` : report added/modified/deleted
  since a commit sha (or HEAD^1 if omitted).

All tools follow the read-only/write-sandbox split; all paths are anchored
to artifacts_dir with strict path-traversal protection.

Git usage is intentionally minimal: we shell out to ``git`` via subprocess
with ``-C <artifacts_dir>``; no heavy libraries. If git is unavailable, the
tools degrade gracefully (snapshot returns ``None``; diff returns empty).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_MAX_READ_BYTES = 16000
_MAX_LIST_ENTRIES = 500


class ArtifactToolError(Exception):
    """Raised when a tool call is invalid (bad args, traversal, missing path)."""


# ---------------------------------------------------------------------------
# Path safety + helpers
# ---------------------------------------------------------------------------
def _artifacts_dir(orch: Any) -> Path:
    root = getattr(orch, "project_root", None)
    if root is None:
        raise ArtifactToolError("orch has no project_root")
    cfg = getattr(orch, "cfg", None)
    rel = None
    try:
        rel = cfg.paths.artifacts_dir  # type: ignore[attr-defined]
    except Exception:
        rel = "artifacts"
    return (Path(root) / rel).resolve()


def _safe_resolve(orch: Any, rel_path: str) -> Path:
    if rel_path is None:
        raise ArtifactToolError("path is required")
    p = str(rel_path).strip().strip('"').strip("'")
    if not p:
        raise ArtifactToolError("path is empty")
    normalized = p.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise ArtifactToolError(f"path traversal blocked: {p!r}")
    base = _artifacts_dir(orch)
    candidate = Path(p) if os.path.isabs(p) else (base / p)
    try:
        resolved = candidate.resolve()
    except Exception as e:
        raise ArtifactToolError(f"cannot resolve {p!r}: {e}")
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ArtifactToolError(f"path escapes artifacts_dir: {p!r}")
    return resolved


def _have_git() -> bool:
    return shutil.which("git") is not None


def _git(base: Path, *args: str, check: bool = False, timeout: int = 30) -> Tuple[int, str, str]:
    """Run ``git -C <base> <args...>``. Returns (rc, stdout, stderr)."""
    if not _have_git():
        return 127, "", "git binary not found"
    try:
        cp = subprocess.run(
            ["git", "-C", str(base), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
        return cp.returncode, cp.stdout, cp.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"git timeout: {e}"
    except Exception as e:
        return 1, "", f"git error: {e}"


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except Exception:
        return "?"
    return h.hexdigest()


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n}{unit}"
        n //= 1024
    return f"{n}T"


# ---------------------------------------------------------------------------
# Repo lifecycle
# ---------------------------------------------------------------------------
def ensure_artifacts_repo(orch: Any) -> Optional[Path]:
    """
    Idempotently ensure artifacts_dir exists AND contains a git repo.

    Returns the artifacts_dir path on success, ``None`` if git is unavailable.
    Safe to call repeatedly. Never fails the pipeline — any git error is
    swallowed and logged via stderr print.
    """
    base = _artifacts_dir(orch)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ARTIFACTS-GIT] cannot create {base}: {e}")
        return None
    if not _have_git():
        return None
    git_dir = base / ".git"
    if git_dir.exists():
        return base
    # Initialize. Use --initial-branch=main to avoid git 2.28+ prompts.
    rc, _out, err = _git(base, "init", "--initial-branch=main", timeout=15)
    if rc != 0:
        # Fallback for older git versions that don't support --initial-branch.
        rc2, _o2, err2 = _git(base, "init", timeout=15)
        if rc2 != 0:
            print(f"[ARTIFACTS-GIT] init failed: {err or err2}")
            return None
    # Minimal local identity so commits don't fail on fresh containers.
    _git(base, "config", "user.email", "artifacts@linguainterpreter.local")
    _git(base, "config", "user.name", "linguainterpreter-artifacts")
    # Tell git to track only this dir; no .gitignore by default (we want all
    # artifact files tracked as ground truth).
    # Create a baseline empty commit so HEAD exists for later diffs.
    rc3, _o3, err3 = _git(base, "commit", "--allow-empty", "-m", "artifacts:init", timeout=15)
    if rc3 != 0:
        print(f"[ARTIFACTS-GIT] baseline commit failed: {err3}")
    return base


def snapshot_artifacts(orch: Any, message: str = "snapshot") -> Optional[str]:
    """
    Stage all current artifacts and commit. Returns commit sha, or None on
    error / no-git / empty diff.
    """
    base = ensure_artifacts_repo(orch)
    if base is None:
        return None
    # Stage everything inside artifacts_dir.
    rc, _o, err = _git(base, "add", "-A", timeout=30)
    if rc != 0:
        print(f"[ARTIFACTS-GIT] add failed: {err}")
        return None
    # Use --allow-empty so we always advance HEAD (makes diffs deterministic).
    rc2, _o2, err2 = _git(
        base, "commit", "--allow-empty", "-m", f"artifacts: {message}", timeout=30
    )
    if rc2 != 0:
        # On race conditions where nothing to commit and --allow-empty is ignored,
        # fall back to reading current HEAD.
        pass
    rc3, out3, err3 = _git(base, "rev-parse", "HEAD", timeout=10)
    if rc3 != 0:
        print(f"[ARTIFACTS-GIT] rev-parse HEAD failed: {err3}")
        return None
    return (out3 or "").strip() or None


def artifacts_diff_since(orch: Any, since_sha: Optional[str]) -> Dict[str, List[str]]:
    """
    Return files added/modified/deleted since ``since_sha`` up to HEAD.

    If ``since_sha`` is None, diffs against HEAD^1 (i.e. the last commit).
    All paths are returned as POSIX relative to artifacts_dir.
    """
    out: Dict[str, List[str]] = {"added": [], "modified": [], "deleted": []}
    base = _artifacts_dir(orch)
    if not (base / ".git").exists():
        return out
    ref = since_sha or "HEAD^"
    rc, data, err = _git(base, "diff", "--name-status", f"{ref}..HEAD", timeout=15)
    if rc != 0:
        # Fallback: maybe there's no HEAD^ yet (fresh repo) — use an empty tree.
        rc2, data2, _err2 = _git(
            base,
            "diff",
            "--name-status",
            "4b825dc642cb6eb9a060e54bf8d69288fbee4904..HEAD",  # empty-tree sha
            timeout=15,
        )
        if rc2 != 0:
            print(f"[ARTIFACTS-GIT] diff failed: {err}")
            return out
        data = data2
    for line in (data or "").splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts[0].strip(), parts[1].strip()
        key = {"A": "added", "M": "modified", "D": "deleted"}.get(status[:1])
        if key is None:
            # Renames etc. surface as "R100\told\tnew" — treat as modified.
            rn_parts = line.split("\t")
            if status.startswith(("R", "C")) and len(rn_parts) >= 3:
                out["added"].append(rn_parts[-1].strip())
                out["deleted"].append(rn_parts[-2].strip())
            continue
        out[key].append(path.replace("\\", "/"))
    return out


# ---------------------------------------------------------------------------
# Write-sandbox tool
# ---------------------------------------------------------------------------
def tool_save_artifact(orch: Any, path: str, content: Any) -> str:
    """
    Write ``content`` (str or bytes) to ``path`` under artifacts_dir.

    Creates parent directories. Rejects traversal. Returns a short JSON-ish
    summary with sha256 + size so the agent gets hard proof of the write.
    """
    target = _safe_resolve(orch, path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ArtifactToolError(f"mkdir failed: {e}")
    if isinstance(content, bytes):
        data = content
    elif isinstance(content, str):
        data = content.encode("utf-8")
    else:
        raise ArtifactToolError(
            f"content must be str or bytes, got {type(content).__name__}"
        )
    try:
        with open(target, "wb") as fh:
            fh.write(data)
    except Exception as e:
        raise ArtifactToolError(f"write failed: {e}")
    rel = target.relative_to(_artifacts_dir(orch)).as_posix()
    sha = _sha256_bytes(data)
    return (
        "ARTIFACT_SAVED\n"
        f"path: {rel}\n"
        f"bytes: {len(data)}\n"
        f"sha256: {sha}"
    )


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------
def tool_list_artifacts(orch: Any, subdir: str = ".") -> str:
    base = _artifacts_dir(orch)
    if not base.exists():
        return "(no artifacts dir)"
    target = _safe_resolve(orch, subdir) if subdir and subdir != "." else base
    if not target.exists():
        raise ArtifactToolError(f"not found: {subdir}")
    lines: List[str] = [f"artifacts_dir: {base}"]
    count = 0
    truncated = False
    for p in sorted(target.rglob("*")):
        if p.is_dir():
            continue
        if ".git" in p.relative_to(base).parts:
            continue
        if count >= _MAX_LIST_ENTRIES:
            truncated = True
            break
        try:
            st = p.stat()
            rel = p.relative_to(base).as_posix()
            sha = _sha256_file(p)[:12]
            lines.append(f"{_fmt_bytes(st.st_size):>8}  {sha}  {rel}")
            count += 1
        except OSError as e:
            lines.append(f"    ?   ?  {p}  [stat error: {e}]")
    if count == 0:
        lines.append("(empty)")
    if truncated:
        lines.append(f"... [truncated to {_MAX_LIST_ENTRIES} entries]")
    return "\n".join(lines)


def tool_read_artifact(orch: Any, path: str, n_bytes: int = 4000) -> str:
    target = _safe_resolve(orch, path)
    if not target.exists():
        raise ArtifactToolError(f"not found: {path}")
    if not target.is_file():
        raise ArtifactToolError(f"not a file: {path}")
    try:
        limit = int(n_bytes)
    except (TypeError, ValueError):
        limit = 4000
    limit = max(200, min(limit, _MAX_READ_BYTES))
    try:
        with open(target, "rb") as fh:
            raw = fh.read(limit + 1)
    except Exception as e:
        raise ArtifactToolError(f"read failed: {e}")
    truncated = len(raw) > limit
    raw = raw[:limit]
    try:
        data = raw.decode("utf-8", errors="replace")
    except Exception:
        data = raw.decode("latin-1", errors="replace")
    rel = target.relative_to(_artifacts_dir(orch)).as_posix()
    header = f"--- artifacts/{rel} (first {limit}B)"
    if truncated:
        header += " [truncated]"
    return header + "\n" + data


def tool_artifacts_diff(orch: Any, since: Optional[str] = None) -> str:
    diff = artifacts_diff_since(orch, since)
    parts = []
    for key in ("added", "modified", "deleted"):
        items = diff.get(key) or []
        if items:
            parts.append(f"{key} ({len(items)}):\n  " + "\n  ".join(items))
    if not parts:
        return "(no changes since reference commit)"
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Dispatcher (mirrors lead_tools layout)
# ---------------------------------------------------------------------------
_TOOL_REGISTRY = {
    "save_artifact": tool_save_artifact,
    "list_artifacts": tool_list_artifacts,
    "read_artifact": tool_read_artifact,
    "artifacts_diff": tool_artifacts_diff,
}


ARTIFACT_TOOLS_HELP = """Git-anchored artifact tools (writes restricted to artifacts_dir):

- save_artifact(path, content)     : write a file strictly under artifacts_dir.
                                     Returns ARTIFACT_SAVED with sha256+size.
                                     This is the ONLY sanctioned write path for
                                     agent-produced artifacts. Traversal blocked.
- list_artifacts(subdir=".")       : enumerate files under artifacts_dir with
                                     size + sha256 prefix. Use this instead of
                                     guessing shapes/columns.
- read_artifact(path, n_bytes=4000): read a file under artifacts_dir (head).
                                     Use this to INSPECT a file before writing
                                     a passport; never fabricate shapes.
- artifacts_diff(since=<sha>)      : list files added/modified/deleted since a
                                     reference commit. This is the ground-truth
                                     signal for "did my code actually save X?".

All paths relative to artifacts_dir. `..` blocked. `.git/` hidden from listing.
Writes OUTSIDE artifacts_dir will fail — that is the intended guarantee."""


# ---------------------------------------------------------------------------
# Modality-aware inspection
# ---------------------------------------------------------------------------
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
_TEXT_EXT = {".txt", ".md", ".log", ".yaml", ".yml", ".ini", ".cfg"}
_TABULAR_EXT = {".csv", ".tsv", ".parquet"}


def _probe_tabular(p: Path) -> Dict[str, Any]:
    """Columns/dtypes/row_count via polars (cheap), fallback to pandas."""
    info: Dict[str, Any] = {"kind": "tabular"}
    ext = p.suffix.lower()
    try:
        import polars as pl  # type: ignore
        if ext == ".parquet":
            lf = pl.scan_parquet(str(p))
        elif ext == ".tsv":
            lf = pl.scan_csv(str(p), separator="\t")
        else:
            lf = pl.scan_csv(str(p))
        schema = lf.collect_schema()
        info["columns"] = list(schema.names())
        info["dtypes"] = {c: str(schema[c]) for c in schema.names()}
        try:
            info["n_rows"] = int(lf.select(pl.len()).collect().item())
        except Exception:
            pass
        try:
            head_df = lf.head(3).collect()
            info["head"] = head_df.to_dicts()
        except Exception:
            pass
        return info
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore
        if ext == ".parquet":
            df = pd.read_parquet(p)
        elif ext == ".tsv":
            df = pd.read_csv(p, sep="\t", nrows=3)
        else:
            df = pd.read_csv(p, nrows=3)
        info["columns"] = list(df.columns.astype(str))
        info["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
        info["head"] = df.head(3).to_dict(orient="records")
        info["note"] = "columns/head via pandas nrows=3 (n_rows unknown)"
        return info
    except Exception as e:
        info["error"] = f"tabular probe failed: {e}"
        return info


def _probe_pickle(p: Path) -> Dict[str, Any]:
    """Load pickle/joblib head; describe top-level type without traversing heavy tensors."""
    info: Dict[str, Any] = {"kind": "pickle"}
    try:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as e:
        try:
            import joblib  # type: ignore
            obj = joblib.load(p)
        except Exception as ee:
            info["error"] = f"pickle/joblib load failed: {e} / {ee}"
            return info
    info["type"] = type(obj).__name__
    if isinstance(obj, dict):
        info["keys"] = [str(k) for k in list(obj.keys())[:50]]
        info["n_keys"] = len(obj)
        shapes: Dict[str, Any] = {}
        for k, v in list(obj.items())[:20]:
            if hasattr(v, "shape"):
                shapes[str(k)] = {"shape": list(getattr(v, "shape", ())), "dtype": str(getattr(v, "dtype", ""))}
            elif isinstance(v, (list, tuple)):
                shapes[str(k)] = {"len": len(v), "type": type(v).__name__}
            else:
                shapes[str(k)] = type(v).__name__
        info["value_shapes"] = shapes
    elif hasattr(obj, "shape"):
        info["shape"] = list(getattr(obj, "shape", ()))
        info["dtype"] = str(getattr(obj, "dtype", ""))
    elif isinstance(obj, (list, tuple)):
        info["len"] = len(obj)
        info["item0_type"] = type(obj[0]).__name__ if obj else None
    else:
        cls = obj.__class__
        info["class"] = f"{cls.__module__}.{cls.__name__}"
        for attr in ("estimators_", "classes_", "feature_names_in_", "n_features_in_", "feature_importances_"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if hasattr(val, "shape"):
                    info[attr] = {"shape": list(val.shape)}
                elif isinstance(val, (list, tuple)):
                    info[attr] = {"len": len(val), "sample": [str(x) for x in val[:10]]}
                else:
                    info[attr] = str(val)[:200]
    return info


def _probe_npy(p: Path) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        return {"kind": "npy", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception as e:
        return {"kind": "npy", "error": f"{e}"}


def _probe_npz(p: Path) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        with np.load(p, allow_pickle=False) as z:
            keys = list(z.files)[:30]
            shapes = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in keys}
        return {"kind": "npz", "keys": keys, "shapes": shapes}
    except Exception as e:
        return {"kind": "npz", "error": f"{e}"}


def _probe_torch_state(p: Path) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        return {"kind": "torch", "error": f"{e}"}
    info: Dict[str, Any] = {"kind": "torch", "type": type(obj).__name__}
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if hasattr(sd, "items"):
        keys: List[str] = []
        shapes: Dict[str, List[int]] = {}
        for i, (k, v) in enumerate(sd.items()):
            if i >= 40:
                break
            keys.append(str(k))
            if hasattr(v, "shape"):
                shapes[str(k)] = list(v.shape)
        info["keys_head"] = keys
        info["shapes_head"] = shapes
        try:
            info["n_params"] = sum(1 for _ in sd.items())
        except Exception:
            pass
    if isinstance(obj, dict):
        extra = [k for k in obj.keys() if k != "state_dict"]
        if extra:
            info["dict_keys"] = [str(k) for k in extra]
    return info


def _probe_image(p: Path) -> Dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(p) as im:
            return {"kind": "image", "size_wh": list(im.size), "mode": im.mode, "format": im.format}
    except Exception as e:
        return {"kind": "image", "error": f"PIL probe failed: {e}"}


def _probe_audio(p: Path) -> Dict[str, Any]:
    try:
        import soundfile as sf  # type: ignore
        with sf.SoundFile(str(p)) as f:
            return {"kind": "audio", "samplerate": f.samplerate, "channels": f.channels, "frames": int(f.frames), "seconds": round(f.frames / float(f.samplerate or 1), 3)}
    except Exception:
        pass
    try:
        import wave  # stdlib
        with wave.open(str(p), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            return {"kind": "audio", "samplerate": sr, "channels": w.getnchannels(), "frames": n, "seconds": round(n / float(sr or 1), 3), "note": "stdlib wave (non-wav files not probed)"}
    except Exception as e:
        return {"kind": "audio", "error": f"audio probe failed: {e}"}


def _probe_json(p: Path) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            obj = json.load(fh)
    except Exception as e:
        return {"kind": "json", "error": f"{e}"}
    info: Dict[str, Any] = {"kind": "json", "type": type(obj).__name__}
    if isinstance(obj, dict):
        info["keys"] = list(obj.keys())[:50]
        info["n_keys"] = len(obj)
    elif isinstance(obj, list):
        info["len"] = len(obj)
        if obj and isinstance(obj[0], dict):
            info["item0_keys"] = list(obj[0].keys())[:30]
    return info


def _probe_text(p: Path, n: int = 1500) -> Dict[str, Any]:
    try:
        with open(p, "rb") as fh:
            raw = fh.read(n + 1)
        truncated = len(raw) > n
        txt = raw[:n].decode("utf-8", errors="replace")
        return {"kind": "text", "head": txt, "truncated": truncated, "size_bytes": p.stat().st_size}
    except Exception as e:
        return {"kind": "text", "error": f"{e}"}


def _probe_folder(p: Path, max_samples: int = 3) -> Dict[str, Any]:
    """Summarize a directory: counts by extension, per-subfolder file count, sample probes."""
    info: Dict[str, Any] = {"kind": "folder", "path": str(p)}
    ext_counts: Dict[str, int] = {}
    sub_counts: Dict[str, int] = {}
    first_by_kind: Dict[str, Path] = {}
    n_files = 0
    for child in p.rglob("*"):
        if not child.is_file():
            continue
        if ".git" in child.parts:
            continue
        n_files += 1
        ext = child.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        try:
            rel_sub = child.relative_to(p).parts[0]
            sub_counts[rel_sub] = sub_counts.get(rel_sub, 0) + 1
        except Exception:
            pass
        # pick first sample per kind
        kind = None
        if ext in _IMAGE_EXT:
            kind = "image"
        elif ext in _AUDIO_EXT:
            kind = "audio"
        elif ext in _TABULAR_EXT:
            kind = "tabular"
        if kind and kind not in first_by_kind:
            first_by_kind[kind] = child
        if n_files > 200000:
            info["note"] = "truncated file scan at 200000"
            break
    info["n_files"] = n_files
    info["ext_counts"] = dict(sorted(ext_counts.items(), key=lambda kv: -kv[1])[:15])
    info["subfolders_top"] = dict(sorted(sub_counts.items(), key=lambda kv: -kv[1])[:20])
    # Heuristic: image class folder = many subdirs each with many images
    img_subs = [k for k, v in sub_counts.items() if v >= 5 and (p / k).is_dir()]
    if img_subs and sum(ext_counts.get(e, 0) for e in _IMAGE_EXT) > 10:
        info["looks_like_image_classes"] = True
        info["class_names_head"] = img_subs[:30]
    samples: Dict[str, Any] = {}
    for kind, f in list(first_by_kind.items())[:max_samples]:
        try:
            if kind == "image":
                samples[f.name] = _probe_image(f)
            elif kind == "audio":
                samples[f.name] = _probe_audio(f)
            elif kind == "tabular":
                samples[f.name] = _probe_tabular(f)
        except Exception as e:
            samples[f.name] = {"error": str(e)}
    if samples:
        info["sample_probes"] = samples
    return info


def tool_inspect_artifact(orch: Any, path: str) -> str:
    """
    Modality-aware probe of a file OR directory under artifacts_dir (or any absolute
    path the agent passes). Returns compact JSON string with structural metadata:

    - tabular (.csv/.tsv/.parquet): columns, dtypes, n_rows, head(3)
    - pickle (.pkl/.joblib): top-level type; if dict → keys + value shapes/dtypes;
      if numpy/torch tensor → shape+dtype; if sklearn-like → classes_/feature_names_in_
    - numpy (.npy/.npz): shape + dtype (mmap, no full load)
    - torch (.pt/.pth/.ckpt): state_dict key names + shapes (head)
    - json: top-level keys or list length + item0 keys
    - image: size, mode, format (PIL)
    - audio: samplerate, channels, duration seconds
    - folder: file counts by extension, top subfolders, sample probes
    - text/log/md: head bytes

    Use this BEFORE merge/join/groupby to confirm exact column names, and before
    loading a pkl/ckpt to confirm the structure matches what your code expects.
    Never fabricate shapes — always inspect.
    """
    if not path:
        raise ArtifactToolError("path required")
    raw = str(path).strip().strip('"').strip("'")
    candidate: Path
    if os.path.isabs(raw):
        candidate = Path(raw).resolve()
    else:
        # Try artifacts_dir-relative first (safe), then project_root-relative.
        try:
            candidate = _safe_resolve(orch, raw)
        except ArtifactToolError:
            root = getattr(orch, "project_root", None)
            if not root:
                raise
            candidate = (Path(root) / raw).resolve()
    if not candidate.exists():
        raise ArtifactToolError(f"not found: {raw}")
    if candidate.is_dir():
        info = _probe_folder(candidate)
    else:
        ext = candidate.suffix.lower()
        if ext in _TABULAR_EXT:
            info = _probe_tabular(candidate)
        elif ext in (".pkl", ".joblib"):
            info = _probe_pickle(candidate)
        elif ext == ".npy":
            info = _probe_npy(candidate)
        elif ext == ".npz":
            info = _probe_npz(candidate)
        elif ext in (".pt", ".pth", ".ckpt", ".safetensors"):
            info = _probe_torch_state(candidate)
        elif ext in _IMAGE_EXT:
            info = _probe_image(candidate)
        elif ext in _AUDIO_EXT:
            info = _probe_audio(candidate)
        elif ext == ".json":
            info = _probe_json(candidate)
        elif ext in _TEXT_EXT or ext in (".py", ".ipynb"):
            info = _probe_text(candidate)
        else:
            info = {"kind": "unknown_ext", "ext": ext, "size_bytes": candidate.stat().st_size}
        info.setdefault("size_bytes", candidate.stat().st_size)
    info["resolved_path"] = str(candidate)
    try:
        return json.dumps(info, ensure_ascii=False, default=str)[:6000]
    except Exception:
        return str(info)[:6000]


# ---------------------------------------------------------------------------
# Git history tools (for agents who want to see what changed / roll back)
# ---------------------------------------------------------------------------
def tool_git_log(orch: Any, path: Optional[str] = None, n: int = 10) -> str:
    """Return last N commits touching `path` (or whole artifacts dir if None)."""
    base = _artifacts_dir(orch)
    if not (base / ".git").exists():
        return "(no artifacts git repo)"
    try:
        n_ = max(1, min(int(n or 10), 50))
    except Exception:
        n_ = 10
    args: List[str] = ["log", f"-n{n_}", "--format=%h %ad %s", "--date=iso-strict"]
    if path:
        p = str(path).strip().strip('"').strip("'")
        if p and ".." not in p.replace("\\", "/").split("/"):
            args += ["--", p]
    rc, out, err = _git(base, *args, timeout=15)
    if rc != 0:
        return f"git_log_error: {err or out}"
    return (out or "(empty)")[:6000]


def tool_git_show(orch: Any, commit: str, path: str, n_bytes: int = 4000) -> str:
    """Show `path` as it was at `commit`. Use to recover a lost artifact version."""
    base = _artifacts_dir(orch)
    if not (base / ".git").exists():
        return "(no artifacts git repo)"
    c = str(commit or "").strip().strip('"').strip("'")
    p = str(path or "").strip().strip('"').strip("'")
    if not c or not p or ".." in p.replace("\\", "/").split("/"):
        raise ArtifactToolError("commit + safe path required")
    rc, out, err = _git(base, "show", f"{c}:{p}", timeout=30)
    if rc != 0:
        return f"git_show_error: {err or out[:300]}"
    try:
        limit = max(200, min(int(n_bytes or 4000), _MAX_READ_BYTES))
    except Exception:
        limit = 4000
    truncated = len(out) > limit
    head = f"--- {p}@{c[:8]} ({min(len(out), limit)}B)"
    if truncated:
        head += " [truncated]"
    return head + "\n" + out[:limit]


# ---------------------------------------------------------------------------
# Schema snapshot builder (modality-aware) — for coder/reviewer prompts
# ---------------------------------------------------------------------------
def _parse_data_audit_report(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Best-effort parser for artifacts/data_audit_report.md.
    Expects per-file sections like:
      ### FILENAME.csv
      - **Shape**: 198577 rows × 8 columns
      - **Columns**: Season, DayNum, ...
      #### Data Types
      - `Season`: Int64
    Returns {filename: {shape, columns, dtypes}}.
    """
    result: Dict[str, Dict[str, Any]] = {}
    cur: Optional[str] = None
    buf: Dict[str, Any] = {}
    in_dtypes = False
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if line.startswith("### ") and not line.startswith("#### "):
            if cur and buf:
                result[cur] = buf
            cur = line[4:].strip()
            buf = {}
            in_dtypes = False
            continue
        if line.startswith("#### Data Types"):
            in_dtypes = True
            buf["dtypes"] = {}
            continue
        if line.startswith("####"):
            in_dtypes = False
            continue
        if cur is None:
            continue
        low = line.lstrip("- ").strip()
        if low.startswith("**Shape**:"):
            buf["shape"] = low[len("**Shape**:"):].strip()
        elif low.startswith("**Columns**:"):
            cols_raw = low[len("**Columns**:"):].strip()
            buf["columns"] = [c.strip() for c in cols_raw.split(",") if c.strip()]
        elif in_dtypes and low.startswith("`") and "`" in low[1:]:
            try:
                name_end = low.index("`", 1)
                name = low[1:name_end]
                rest = low[name_end + 1:].lstrip(": ").strip()
                buf.setdefault("dtypes", {})[name] = rest
            except Exception:
                pass
    if cur and buf:
        result[cur] = buf
    return result


def build_schema_snapshot(orch: Any, spec: Dict[str, Any], max_files: int = 40) -> str:
    """
    Build a compact, authoritative DATA SCHEMA block for injection into coder /
    reviewer / triage prompts. Modality-agnostic:

    - Tabular: pulls columns/dtypes from (a) spec.data.meta.csv_summaries,
      (b) artifacts/data_audit_report.md.
    - Images/Audio/Text/Video: falls back to spec.data hints + probes if cheap.
    - Always includes the modality list and a short "verify-before-use" note.

    Returns markdown-style block under 6000 chars. Empty string if nothing known.
    """
    safe = spec or {}
    modalities = safe.get("modalities") or []
    data = safe.get("data") or {}
    meta = (data.get("meta") or {}) if isinstance(data, dict) else {}
    csv_summaries = meta.get("csv_summaries") if isinstance(meta, dict) else {}
    if not isinstance(csv_summaries, dict):
        csv_summaries = {}

    # Pull data_audit_report.md if present
    audit_sections: Dict[str, Dict[str, Any]] = {}
    base = _artifacts_dir(orch)
    audit_path = base / "data_audit_report.md"
    if audit_path.exists():
        try:
            audit_sections = _parse_data_audit_report(
                audit_path.read_text(encoding="utf-8", errors="ignore")
            )
        except Exception:
            audit_sections = {}

    lines: List[str] = []
    lines.append("DATA SCHEMA (authoritative; column names are CASE-SENSITIVE):")
    if modalities:
        lines.append(f"Modalities declared in spec: {modalities}")
    lines.append("")

    # Tabular files — union of csv_summaries and audit_sections
    tabular_names = list(dict.fromkeys(list(csv_summaries.keys()) + list(audit_sections.keys())))
    if tabular_names:
        lines.append("[TABULAR]")
        for name in tabular_names[:max_files]:
            cs = csv_summaries.get(name) or {}
            aud = audit_sections.get(name) or {}
            cols = cs.get("columns") or aud.get("columns") or []
            dtypes = cs.get("dtypes") or aud.get("dtypes") or {}
            shape = cs.get("shape") or aud.get("shape") or ""
            if isinstance(cols, list) and cols and dtypes:
                col_block = ", ".join(f"{c}({dtypes.get(c, '?')})" for c in cols)
            elif isinstance(cols, list) and cols:
                col_block = ", ".join(cols)
            else:
                col_block = "(columns unknown — probe with inspect_artifact)"
            shape_str = f" [{shape}]" if shape else ""
            lines.append(f"- {name}{shape_str}: {col_block}")
        lines.append("")

    # Non-tabular data paths declared in spec.data
    other_hints: List[str] = []
    for key in ("train_dir", "test_dir", "labels_csv", "train_csv"):
        val = data.get(key) if isinstance(data, dict) else None
        if not val:
            continue
        path = Path(str(val))
        if path.exists() and path.is_dir():
            try:
                probe = _probe_folder(path)
                ext = probe.get("ext_counts") or {}
                tags: List[str] = []
                if probe.get("looks_like_image_classes"):
                    tags.append(f"image-classes(n={len(probe.get('class_names_head') or [])})")
                if any(e in _IMAGE_EXT for e in ext):
                    tags.append(f"images({sum(ext.get(e, 0) for e in _IMAGE_EXT)})")
                if any(e in _AUDIO_EXT for e in ext):
                    tags.append(f"audio({sum(ext.get(e, 0) for e in _AUDIO_EXT)})")
                if any(e in _VIDEO_EXT for e in ext):
                    tags.append(f"video({sum(ext.get(e, 0) for e in _VIDEO_EXT)})")
                tag_str = ", ".join(tags) if tags else f"n_files={probe.get('n_files')}"
                other_hints.append(f"- {key}={val} → {tag_str}")
            except Exception as e:
                other_hints.append(f"- {key}={val} (probe failed: {e})")
    if other_hints:
        lines.append("[NON-TABULAR DATA]")
        lines.extend(other_hints)
        lines.append("")

    lines.append("RULES:")
    lines.append("- Column names above are exact. Do NOT guess (e.g. 'TeamID' vs 'WTeamID').")
    lines.append("- If a file you need is not listed, call inspect_artifact(path) BEFORE you")
    lines.append("  write merge/join/groupby/read_csv. Never fabricate columns or keys.")
    lines.append("- For image/audio/text tasks, inspect a sample file (or folder) with")
    lines.append("  inspect_artifact to confirm dims / sample_rate / class layout.")
    block = "\n".join(lines)
    if len(block) > 6000:
        block = block[:5990] + "\n…(truncated)"
    return block


# ---------------------------------------------------------------------------
# Artifacts index — structured catalog for improver
# ---------------------------------------------------------------------------
def write_artifacts_index(orch: Any, max_files: int = 80) -> Optional[Path]:
    """
    Scan artifacts_dir and write artifacts_index.json describing each file with
    modality-aware structural metadata. Best-effort; skips files that fail to
    probe. Returns the path on success, None otherwise.

    Consumed by improver_head_agent and react_improver_meta_planner: machine-
    readable supplement to the human .md reports so the improver can reason
    about SHAPES and KEYS without re-inspecting.
    """
    base = _artifacts_dir(orch)
    if not base.exists():
        return None
    index: Dict[str, Any] = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifacts_dir": str(base),
        "files": {},
    }
    count = 0
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if rel.as_posix() == "artifacts_index.json":
            continue
        if count >= max_files:
            index["truncated"] = True
            break
        ext = p.suffix.lower()
        entry: Dict[str, Any] = {"size_bytes": p.stat().st_size, "ext": ext}
        try:
            if ext in _TABULAR_EXT:
                entry.update(_probe_tabular(p))
            elif ext in (".pkl", ".joblib"):
                entry.update(_probe_pickle(p))
            elif ext == ".npy":
                entry.update(_probe_npy(p))
            elif ext == ".npz":
                entry.update(_probe_npz(p))
            elif ext in (".pt", ".pth", ".ckpt"):
                entry.update(_probe_torch_state(p))
            elif ext == ".json":
                entry.update(_probe_json(p))
            elif ext in _IMAGE_EXT:
                entry.update(_probe_image(p))
            elif ext in _AUDIO_EXT:
                entry.update(_probe_audio(p))
            elif ext in _TEXT_EXT or ext == ".py":
                entry["kind"] = "text"
            else:
                entry["kind"] = "other"
        except Exception as e:
            entry["probe_error"] = str(e)[:200]
        index["files"][rel.as_posix()] = entry
        count += 1
    out_path = base / "artifacts_index.json"
    try:
        out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        print(f"[ARTIFACTS-INDEX] write failed: {e}")
        return None
    return out_path


# ---------------------------------------------------------------------------
# Knowledge Curator: canonical .md file schema.
# These five files are the ONLY markdown files owned by the Curator agent.
# No other agent/coder is allowed to write to these paths (policy enforced).
# ---------------------------------------------------------------------------
CURATOR_MD_SCHEMA: Dict[str, List[str]] = {
    "competition_brief.md": [
        "meta", "modality", "metric", "stages",
        "constraints", "submission", "leakage_rules",
    ],
    "data_schema.md": [
        "modality", "files", "reference_tables", "notes",
    ],
    "experiments_ledger.md": [
        # Section names are created dynamically per iteration: "iter_<N>".
        # An "index" section holds a compact summary table.
        "index",
    ],
    "lessons.md": [
        # Tag-based sections, e.g. "schema", "cv", "model", "feature",
        # "metric", "submission", "infra", "cost", "routing".
        "schema", "cv", "model", "feature",
        "metric", "submission", "infra", "cost", "routing",
    ],
    "pruned_tasks.md": [
        "index",
    ],
}

_CURATOR_DIR_NAME = "curator"  # artifacts/curator/*.md


def _curator_dir(orch: Any) -> Path:
    d = Path(orch.project_root) / orch.cfg.paths.artifacts_dir / _CURATOR_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _curator_md_path(orch: Any, file_name: str) -> Path:
    if file_name not in CURATOR_MD_SCHEMA:
        raise ArtifactToolError(
            f"'{file_name}' is not a canonical curator file. "
            f"Allowed: {sorted(CURATOR_MD_SCHEMA)}"
        )
    return _curator_dir(orch) / file_name


def _iso_now() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_curator_file(path: Path) -> None:
    """Create the file with a frontmatter header if missing."""
    if path.exists():
        return
    fm = (
        "---\n"
        "owner: curator\n"
        "schema: v1\n"
        f"created_at: {_iso_now()}\n"
        f"updated_at: {_iso_now()}\n"
        "---\n\n"
        f"# {path.stem.replace('_', ' ').title()}\n\n"
    )
    path.write_text(fm, encoding="utf-8")


def _split_sections(md_text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split a markdown doc into (header_block, [(section_name, section_body), ...]).
    Section headers are lines matching '## <name>' (exactly H2).
    """
    lines = md_text.splitlines(keepends=True)
    header: List[str] = []
    sections: List[Tuple[str, List[str]]] = []
    current_name: Optional[str] = None
    current_body: List[str] = []
    in_header = True
    for ln in lines:
        stripped = ln.rstrip("\n")
        if stripped.startswith("## ") and not stripped.startswith("### "):
            if in_header:
                in_header = False
            else:
                sections.append((current_name or "", current_body))
            current_name = stripped[3:].strip()
            current_body = []
            continue
        if in_header:
            header.append(ln)
        else:
            current_body.append(ln)
    if current_name is not None:
        sections.append((current_name, current_body))
    return ("".join(header), [(n, "".join(b)) for n, b in sections])


def _rewrite_frontmatter_updated(header: str) -> str:
    """Bump updated_at inside the YAML frontmatter (best-effort)."""
    import re as _re
    if header.startswith("---"):
        return _re.sub(
            r"updated_at:\s*[^\n]+",
            f"updated_at: {_iso_now()}",
            header,
            count=1,
        )
    return header


def curator_write_section(orch: Any, file: str, section: str, content: str) -> str:
    """
    Replace (or create) a single ## section inside a canonical curator .md file.
    Validates both file and section name against CURATOR_MD_SCHEMA (lessons/ledger
    allow dynamic section names as documented).
    Returns a short confirmation string.
    """
    path = _curator_md_path(orch, file)
    _ensure_curator_file(path)

    allowed = CURATOR_MD_SCHEMA[file]
    dynamic_ok = (
        (file == "experiments_ledger.md" and section.startswith("iter_"))
        or (file == "pruned_tasks.md" and section.startswith("iter_"))
        or file == "lessons.md"  # tag-based, free-form
    )
    if section not in allowed and not dynamic_ok:
        raise ArtifactToolError(
            f"section '{section}' not allowed in {file}. "
            f"Allowed: {allowed}"
        )

    text = path.read_text(encoding="utf-8")
    header, sections = _split_sections(text)
    header = _rewrite_frontmatter_updated(header)
    names = [n for n, _ in sections]
    new_body = content if content.endswith("\n") else content + "\n"
    if section in names:
        sections = [(n, new_body if n == section else b) for n, b in sections]
    else:
        sections.append((section, new_body))

    out = header
    for n, b in sections:
        out += f"## {n}\n{b if b else ''}"
        if not out.endswith("\n"):
            out += "\n"
        out += "\n"
    path.write_text(out, encoding="utf-8")
    return f"curator: wrote {file}#{section} ({len(content)} chars)"


def curator_append_line(orch: Any, file: str, section: str, line: str) -> str:
    """Append one markdown line under an existing or new section."""
    path = _curator_md_path(orch, file)
    _ensure_curator_file(path)
    text = path.read_text(encoding="utf-8")
    header, sections = _split_sections(text)
    header = _rewrite_frontmatter_updated(header)
    names = [n for n, _ in sections]
    ln = line if line.startswith("- ") else f"- {line}"
    if not ln.endswith("\n"):
        ln += "\n"
    if section in names:
        sections = [
            (n, (b + ln) if n == section else b) for n, b in sections
        ]
    else:
        sections.append((section, ln))
    out = header
    for n, b in sections:
        out += f"## {n}\n{b}"
        if not out.endswith("\n"):
            out += "\n"
        out += "\n"
    path.write_text(out, encoding="utf-8")
    return f"curator: appended to {file}#{section}"


def curator_read_section(orch: Any, file: str, section: str = "") -> str:
    """Read one section ('' = whole file, capped at 8 KB)."""
    path = _curator_md_path(orch, file)
    if not path.exists():
        return f"curator: {file} does not exist yet"
    text = path.read_text(encoding="utf-8")
    if not section:
        return text[:8000]
    _, sections = _split_sections(text)
    for n, b in sections:
        if n == section:
            return f"## {n}\n{b}"[:8000]
    return f"curator: section '{section}' not present in {file}"


def build_curator_tools(orch: Any) -> List[Any]:
    """
    Tools EXCLUSIVE to the knowledge_curator_agent: includes read helpers
    shared with the coder + write helpers that only the curator may use.
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:
        return []

    base = build_structured_artifact_tools(orch)

    def _write(file: str, section: str, content: str) -> str:
        """Replace a ## section inside a canonical curator .md file."""
        try:
            return curator_write_section(orch, file, section, content)
        except ArtifactToolError as e:
            return f"write_error: {e}"

    def _append(file: str, section: str, line: str) -> str:
        """Append one bullet line to a section in a canonical curator .md file."""
        try:
            return curator_append_line(orch, file, section, line)
        except ArtifactToolError as e:
            return f"append_error: {e}"

    def _read_section(file: str, section: str = "") -> str:
        """Read one section of a canonical curator .md file ('' = whole file)."""
        try:
            return curator_read_section(orch, file, section)
        except ArtifactToolError as e:
            return f"read_error: {e}"

    extras = [
        StructuredTool.from_function(
            _write,
            name="write_md_section",
            description=(
                "CURATOR-ONLY. Overwrite a single '## section' of a canonical "
                "curator .md file under artifacts/curator/. Validated against schema."
            ),
        ),
        StructuredTool.from_function(
            _append,
            name="append_md_line",
            description=(
                "CURATOR-ONLY. Append one bullet line to a section of a canonical "
                "curator .md file (used for lessons / ledger rows)."
            ),
        ),
        StructuredTool.from_function(
            _read_section,
            name="read_md_section",
            description=(
                "Read one '## section' (or the whole file) of a canonical curator .md "
                "file. Use before deciding whether to patch."
            ),
        ),
    ]
    return base + extras


def build_structured_artifact_tools(orch: Any) -> List[Any]:
    """
    Return a list of LangChain StructuredTools wrapping our inspect/list/git
    helpers, for injection into coder/improver tool-calling agents.

    Degrades gracefully if langchain_core is missing (returns []).
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception:
        return []

    def _inspect(path: str) -> str:
        """Probe a file OR folder. Supports csv/parquet/pkl/npy/pt/json/image/audio/folder."""
        try:
            return tool_inspect_artifact(orch, path)
        except ArtifactToolError as e:
            return f"inspect_error: {e}"

    def _list(subdir: str = ".") -> str:
        """List files under artifacts_dir with size+sha prefix."""
        try:
            return tool_list_artifacts(orch, subdir)
        except ArtifactToolError as e:
            return f"list_error: {e}"

    def _read(path: str, n_bytes: int = 4000) -> str:
        """Read head bytes of a file under artifacts_dir."""
        try:
            return tool_read_artifact(orch, path, n_bytes)
        except ArtifactToolError as e:
            return f"read_error: {e}"

    def _log(path: str = "", n: int = 10) -> str:
        """Git history of artifacts — last N commits (optionally for a specific path)."""
        return tool_git_log(orch, path or None, n)

    def _show(commit: str, path: str, n_bytes: int = 4000) -> str:
        """Show `path` as it was at `commit` (use git_log to find commit sha)."""
        try:
            return tool_git_show(orch, commit, path, n_bytes)
        except ArtifactToolError as e:
            return f"show_error: {e}"

    return [
        StructuredTool.from_function(
            _inspect,
            name="inspect_artifact",
            description=(
                "Probe a file or folder and return machine-readable structural metadata. "
                "For TABULAR (.csv/.tsv/.parquet): returns {columns, dtypes, n_rows, head}. "
                "For .pkl/.joblib: type + dict keys + value shapes/dtypes. "
                "For .npy/.npz: shape + dtype. For .pt/.pth/.ckpt: state_dict keys + shapes. "
                "For images: size+mode. For audio: samplerate+duration. "
                "For folders: file counts by extension + top subfolders + sample probes. "
                "USE BEFORE any merge/join/groupby to confirm exact column names."
            ),
        ),
        StructuredTool.from_function(
            _list,
            name="list_artifacts",
            description="List files under artifacts_dir (optional subdir). Returns size+sha+path per line.",
        ),
        StructuredTool.from_function(
            _read,
            name="read_artifact",
            description="Read first N bytes of a file under artifacts_dir (text/log/md inspection).",
        ),
        StructuredTool.from_function(
            _log,
            name="git_log_artifact",
            description="Last N commits touching a path (or all artifacts). Use to find a best past commit sha.",
        ),
        StructuredTool.from_function(
            _show,
            name="git_show_artifact",
            description="Show a file's content at a specific commit (recover a lost/regressed version).",
        ),
    ]


# Extend dispatcher registry
_TOOL_REGISTRY.update({
    "inspect_artifact": tool_inspect_artifact,
    "git_log": tool_git_log,
    "git_show": tool_git_show,
})


def list_tool_names() -> List[str]:
    return list(_TOOL_REGISTRY.keys())


def call_tool(orch: Any, name: str, args: Dict[str, Any]) -> str:
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        raise ArtifactToolError(
            f"unknown artifact tool: {name!r}. Available: {', '.join(_TOOL_REGISTRY)}"
        )
    kwargs = dict(args or {})
    try:
        return fn(orch, **kwargs)
    except ArtifactToolError:
        raise
    except TypeError as e:
        raise ArtifactToolError(f"bad arguments to {name}: {e}")
    except Exception as e:
        raise ArtifactToolError(f"{name} failed: {e}")
