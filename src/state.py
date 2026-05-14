# src/state.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, Optional

STATE_FILENAME = "state.json"

def _state_path(artifacts_dir: str | Path) -> Path:
    p = Path(artifacts_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / STATE_FILENAME

def save_state(artifacts_dir: str | Path, **kwargs) -> None:
    path = _state_path(artifacts_dir)
    state: Dict[str, Any] = {}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state.update(kwargs)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def load_state(artifacts_dir: str | Path) -> Dict[str, Any]:
    path = _state_path(artifacts_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def clear_state(artifacts_dir: str | Path) -> None:
    path = _state_path(artifacts_dir)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
