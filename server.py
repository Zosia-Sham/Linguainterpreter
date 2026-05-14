# server.py
# Comment translated to English.
# Comment translated to English.
# Comment translated to English.
# Comment translated to English.
# - WebSocket + SSE fallback
#
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocketState

# -------------------- LOGGING --------------------
logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(_handler)

# -------------------- PATHS / CONFIG --------------------
PROJECT_ROOT = Path(__file__).parent.resolve()


def _config_yaml_path() -> Path:
    """
    Active config for the dashboard and path resolution.
    Override with env LINGUA_CONFIG (absolute or relative to repo root) so server matches `main.py --config`.
    """
    raw = os.environ.get("LINGUA_CONFIG", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return (PROJECT_ROOT / "config.yaml").resolve()


def _load_cfg_obj() -> Dict[str, Any]:
    """Ленивая загрузка config.yaml (кэш по mtime + путь)."""
    cfg_path = _config_yaml_path()
    ts = getattr(_load_cfg_obj, "_mtime", 0.0)
    obj = getattr(_load_cfg_obj, "_obj", None)
    cached_p = getattr(_load_cfg_obj, "_path", None)
    try:
        cur = cfg_path.stat().st_mtime if cfg_path.exists() else 0.0
        path_s = str(cfg_path)
        if obj is None or cur != ts or cached_p != path_s:
            text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
            data = yaml.safe_load(text) if text.strip() else {}
            if not isinstance(data, dict):
                data = {}
            _load_cfg_obj._obj = data
            _load_cfg_obj._mtime = cur
            _load_cfg_obj._path = path_s
            logger.info("config reloaded from %s", path_s)
        return getattr(_load_cfg_obj, "_obj", {}) or {}
    except Exception as e:
        logger.error(f"config load error: {e}")
        return {}

def _resolve_path(v: str) -> Path:
    """Абсолютные пути — как есть; относительные — относительно PROJECT_ROOT."""
    p = Path(v)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _runtime_project_root() -> Path:
    cfg = _load_cfg_obj()
    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    configured = str((runtime or {}).get("project_root", "") or "").strip()
    if not configured:
        project_name = str((runtime or {}).get("project_name", "ml_project") or "ml_project").strip() or "ml_project"
        return (PROJECT_ROOT / project_name).resolve()
    p = Path(configured).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

def _cfg_paths() -> Dict[str, Path]:
    """ВАЖНО: пути только из config.yaml. Никакого поиска в корне."""
    cfg = _load_cfg_obj()
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    run_root = _runtime_project_root()
    def _under_root(v: str) -> Path:
        p = Path(v)
        return p if p.is_absolute() else (run_root / p).resolve()

    artifacts_dir = _under_root(paths.get("artifacts_dir", "artifacts"))
    logs_dir      = _under_root(paths.get("logs_dir", "logs"))
    scripts_dir   = _under_root(paths.get("scripts_dir", "scripts"))
    src_dir       = _under_root(paths.get("src_dir", "src"))
    data_dir      = _under_root(paths.get("data_dir", "data"))
    task_txt      = run_root
    return {
        "artifacts": artifacts_dir,
        "logs": logs_dir,
        "scripts": scripts_dir,
        "src": src_dir,
        "data": data_dir,
        "task_txt": task_txt
    }

def _tree_json_path() -> Path:
    return _cfg_paths()["artifacts"] / "tree.json"


def _tasks_tree_json_path() -> Path:
    return _cfg_paths()["artifacts"] / "tasks_tree.json"


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen: Set[str] = set()
    out: List[Path] = []
    for p in paths:
        try:
            k = str(p.resolve())
        except Exception:
            k = str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _tree_json_search_paths() -> List[Path]:
    """
    Prefer config-derived path, then common locations (main often writes under ml_project/artifacts
    while config paths or cwd differ).
    """
    return _dedupe_paths(
        [
            _tree_json_path(),
            PROJECT_ROOT / "ml_project" / "artifacts" / "tree.json",
            PROJECT_ROOT / "artifacts" / "tree.json",
        ]
    )


def _tasks_tree_search_paths() -> List[Path]:
    return _dedupe_paths(
        [
            _tasks_tree_json_path(),
            PROJECT_ROOT / "ml_project" / "artifacts" / "tasks_tree.json",
            PROJECT_ROOT / "artifacts" / "tasks_tree.json",
        ]
    )


# Last successfully loaded sources (for API debug).
_last_tree_json_source: str = ""
_last_tasks_tree_source: str = ""

def _task_txt_path() -> Path:
    return _cfg_paths()['task_txt'] / 'task.txt'


def _status_color(status: str) -> str:
    s = str(status or "").lower()
    return {
        "done": "#16a34a",      # green
        "failed": "#dc2626",    # red
        "running": "#2563eb",   # blue
        "skipped": "#d97706",   # amber
        "pending": "#6b7280",   # gray
    }.get(s, "#6b7280")


def _event_color(event: str) -> str:
    e = str(event or "").upper()
    return {
        "ADDED": "#16a34a",
        "PRUNED": "#dc2626",
        "SKIPPED": "#d97706",
    }.get(e, "#6b7280")


def _task_graph_events_path() -> Path:
    return _cfg_paths()["artifacts"] / "task_graph_events.jsonl"


def _load_task_graph_events(limit: int = 100) -> List[Dict[str, Any]]:
    p = _task_graph_events_path()
    if not p.exists():
        return []
    try:
        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for ln in rows[-max(1, int(limit)):]:
        try:
            ev = json.loads(ln)
            out.append({
                "ts": ev.get("ts"),
                "event": str(ev.get("event", "")).upper(),
                "task": ev.get("task", ""),
                "node_id": ev.get("node_id", ""),
                "parent_node_id": ev.get("parent_node_id", ""),
                "reason": ev.get("reason", ""),
                "color": _event_color(ev.get("event", "")),
            })
        except Exception:
            continue
    return out

# -------------------- PROCESS CONTROL --------------------
PYTHON_EXE = sys.executable
CLI_ENTRY = PROJECT_ROOT / "main.py"  # Comment translated to English.

@dataclass
class ProcState:
    process: Optional[subprocess.Popen] = None
    args: List[str] = field(default_factory=list)
    start_ts: float = 0.0
    resume: bool = False
    capturing_thread: Optional[threading.Thread] = None
    killed: bool = False

proc_state = ProcState()


def _spawn_process(resume: bool) -> subprocess.Popen:
    cfg_path = str(_config_yaml_path())
    task_path = str(_task_txt_path())
    args = [PYTHON_EXE, str(CLI_ENTRY), "--config", cfg_path, "--task_file", task_path]
    if resume:
        args.append("--resume")

    env = os.environ.copy()
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    p = subprocess.Popen(
        args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        creationflags=creation,
    )
    proc_state.args = args
    proc_state.start_ts = time.time()
    proc_state.resume = resume
    proc_state.killed = False
    return p

def _capture_io():
    p = proc_state.process
    if not p:
        return
    try:
        for line in p.stdout:
            asyncio.run(broker.broadcast({"type": "log", "stream": "stdout", "line": line}))
        for line in p.stderr:
            asyncio.run(broker.broadcast({"type": "log", "stream": "stderr", "line": line}))
    except Exception as e:
        logger.exception(f"capture_io error: {e}")

# -------------------- WS / SSE BROKER --------------------
class Broker:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def attach(self, ws: WebSocket):
        await ws.accept()
        async with self.lock:
            self.clients.add(ws)

    async def detach(self, ws: WebSocket):
        async with self.lock:
            self.clients.discard(ws)

    async def broadcast(self, payload: Dict[str, Any]):
        dead: List[WebSocket] = []
        msg = json.dumps(payload, ensure_ascii=False)
        async with self.lock:
            for ws in list(self.clients):
                try:
                    if ws.application_state == WebSocketState.CONNECTED:
                        await ws.send_text(msg)
                    else:
                        dead.append(ws)
                except Exception:
                    dead.append(ws)
            for d in dead:
                self.clients.discard(d)

broker = Broker()

# -------------------- TREE LOAD (CONFIG PATH + FALLBACK DISCOVERY) --------------------
def _load_tree_json() -> Dict[str, Any]:
    global _last_tree_json_source
    _last_tree_json_source = ""
    empty: Dict[str, Any] = {"nodes": {}, "roots": [], "completed": False}
    best_doc: Optional[Dict[str, Any]] = None
    best_rank: tuple = (-1, -1, 0.0)  # (n_nodes, n_roots, mtime)

    for p in _tree_json_search_paths():
        if not p.exists():
            continue
        try:
            t = json.loads(p.read_text(encoding="utf-8"))
            mtime = p.stat().st_mtime
        except Exception as e:
            logger.warning("tree.json unreadable %s: %s", p, e)
            continue
        if not isinstance(t, dict):
            continue
        nodes = t.get("nodes") or {}
        roots = list(t.get("roots") or [])
        if not roots and nodes:
            roots = [
                nid
                for nid, n in nodes.items()
                if not n.get("parent_node_id") or n.get("parent_node_id") not in nodes
            ]
            t["roots"] = roots
        if not nodes and not roots:
            continue
        rank = (len(nodes), len(roots), mtime)
        if rank > best_rank:
            best_rank = rank
            best_doc = t
            try:
                _last_tree_json_source = str(p.resolve())
            except Exception:
                _last_tree_json_source = str(p)

    if best_doc is not None:
        nn = len(best_doc.get("nodes") or {})
        nr = len(best_doc.get("roots") or [])
        logger.info("tree.json using %s (%d nodes, %d roots)", _last_tree_json_source, nn, nr)
        return best_doc
    return empty


def _parse_tasks_tree_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(doc, dict) or "trees" not in doc:
        return {"nodes": {}, "roots": [], "completed": False}

    out_nodes: Dict[str, Dict[str, Any]] = {}
    out_roots: List[str] = []

    trees: Dict[str, Any] = doc.get("trees") or {}
    for _key, t in trees.items():
        nodes = t.get("nodes", {}) or {}
        root_id = t.get("root_id")
        for nid, n in nodes.items():
            node_id = n.get("id", nid)
            parent_id = n.get("parent_id")
            children = list(n.get("children", []))
            status = n.get("status", "pending")
            task = n.get("task", "")
            level = n.get("level", 0)
            kind = n.get("kind", "subtask" if level > 0 else "root")

            out_nodes[node_id] = {
                "node_id": node_id,
                "parent_node_id": parent_id,
                "children": children,
                "kind": f"tasks_tree:{kind}",
                "task": task,
                "status": status,
                "created_at": n.get("created_at"),
                "started_at": n.get("updated_at"),
                "finished_at": None,
                "meta": {"source": "tasks_tree"},
            }
        if root_id and root_id not in out_roots:
            out_roots.append(root_id)

    return {"nodes": out_nodes, "roots": out_roots, "completed": False}


def _load_tasks_tree_json() -> Dict[str, Any]:
    global _last_tasks_tree_source
    _last_tasks_tree_source = ""
    empty: Dict[str, Any] = {"nodes": {}, "roots": [], "completed": False}
    best_parsed: Optional[Dict[str, Any]] = None
    best_rank: tuple = (-1, -1, 0.0)

    for p in _tasks_tree_search_paths():
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            mtime = p.stat().st_mtime
        except Exception as e:
            logger.warning("tasks_tree.json unreadable %s: %s", p, e)
            continue
        parsed = _parse_tasks_tree_document(doc if isinstance(doc, dict) else {})
        nodes = parsed.get("nodes") or {}
        roots = parsed.get("roots") or []
        if not nodes and not roots:
            continue
        rank = (len(nodes), len(roots), mtime)
        if rank > best_rank:
            best_rank = rank
            best_parsed = parsed
            try:
                _last_tasks_tree_source = str(p.resolve())
            except Exception:
                _last_tasks_tree_source = str(p)

    if best_parsed is not None:
        nn = len(best_parsed.get("nodes") or {})
        nr = len(best_parsed.get("roots") or [])
        logger.info("tasks_tree.json using %s (%d nodes, %d roots)", _last_tasks_tree_source, nn, nr)
        return best_parsed
    return empty

def _combine_orchestrator_and_tasks() -> Dict[str, Any]:
    a = _load_tree_json()
    b = _load_tasks_tree_json()
    nodes: Dict[str, Any] = {}
    nodes.update(a.get("nodes") or {})
    for nid, node in (b.get("nodes") or {}).items():
        if nid not in nodes:
            nodes[nid] = node
    roots = list(a.get("roots") or [])
    for r in b.get("roots") or []:
        if r not in roots:
            roots.append(r)

    # Safety: include orphaned top-level nodes even if roots list is stale/incomplete.
    for nid, node in nodes.items():
        pid = (node or {}).get("parent_node_id")
        if (not pid or pid not in nodes) and nid not in roots:
            roots.append(nid)

    # Hide synthetic/empty unknown roots when normal roots exist.
    cleaned_roots: List[str] = []
    for r in roots:
        rn = nodes.get(r) or {}
        kind = str(rn.get("kind", "")).lower()
        task = str(rn.get("task", "")).strip()
        if kind == "unknown" and not task:
            continue
        cleaned_roots.append(r)
    if cleaned_roots:
        roots = cleaned_roots

    # Root label fallback: avoid empty/unknown root cards in UI.
    task_hint = ""
    try:
        tp = _task_txt_path()
        if tp.exists():
            for ln in tp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    task_hint = ln.strip()
                    break
    except Exception:
        task_hint = ""

    # Enrich nodes with UI-friendly colors.
    enriched_nodes: Dict[str, Any] = {}
    for nid, node in nodes.items():
        n = dict(node or {})
        status = str(n.get("status", "pending")).lower()
        n["status"] = status
        # If the root node has no task text, provide a stable display label.
        if nid in roots and not str(n.get("task") or "").strip():
            n["task"] = task_hint or "Main Task"
        if nid in roots and str(n.get("kind") or "").lower() in {"", "unknown"}:
            n["kind"] = "root"
        n["status_color"] = _status_color(status)
        n["ui"] = {
            "status_color": n["status_color"],
            "status_badge_bg": n["status_color"] + "22",
            "status_badge_border": n["status_color"] + "66",
        }
        enriched_nodes[nid] = n

    # Normalize children ordering for deterministic UI/API behavior.
    # Priority: explicit `order` (planned index) -> finished/started time -> created time.
    # This helps when children nodes are created before they actually execute (created_at skew).
    def _safe_order(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 10 ** 9

    _max_ts = "9999-12-31T23:59:59.999999Z"

    def _child_sort_key(child_id: str) -> tuple:
        cn = enriched_nodes.get(child_id) or {}
        # Planned order: from top-level `order` or from meta.order if present.
        order_val = cn.get("order")
        if order_val is None and isinstance(cn.get("meta"), dict):
            order_val = cn.get("meta", {}).get("order")
        order_key = _safe_order(order_val)
        # Execution time first: finished_at, else started_at, else created_at.
        ts = cn.get("finished_at") or cn.get("started_at") or cn.get("created_at") or _max_ts
        ts = str(ts) if ts else _max_ts
        return (order_key, ts)

    for nid, node in enriched_nodes.items():
        ch = list(node.get("children") or [])
        if len(ch) <= 1:
            continue
        try:
            ch.sort(key=_child_sort_key)
            node["children"] = ch
        except Exception:
            # Best-effort only; leave original order on any parsing issues.
            pass

    graph_events = _load_task_graph_events(limit=100)
    payload = {
        "nodes": enriched_nodes,
        "roots": roots,
        "graph_events": graph_events,
        "palette": {
            "status": {
                "done": _status_color("done"),
                "failed": _status_color("failed"),
                "running": _status_color("running"),
                "skipped": _status_color("skipped"),
                "pending": _status_color("pending"),
            },
            "events": {
                "ADDED": _event_color("ADDED"),
                "PRUNED": _event_color("PRUNED"),
                "SKIPPED": _event_color("SKIPPED"),
            },
        },
        "completed": bool(a.get("completed") and b.get("completed")),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Helps debug “empty tree” when config path ≠ run artifacts path.
        "tree_sources": {
            "config_yaml": str(_config_yaml_path()),
            "tree_json": _last_tree_json_source or None,
            "tasks_tree_json": _last_tasks_tree_source or None,
        },
    }

    # Starlette/FastAPI will serialize with strict JSON (no NaN/Inf).
    # Some artifacts (metrics/meta) can contain NaN, which would crash the UI API.
    def _sanitize_json(x: Any) -> Any:
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        if isinstance(x, dict):
            return {k: _sanitize_json(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize_json(v) for v in x]
        if isinstance(x, tuple):
            return [_sanitize_json(v) for v in x]
        return x

    return _sanitize_json(payload)

# -------------------- BACKGROUND WATCHERS --------------------
# Snapshot of mtimes for every candidate tree/tasks_tree path (config + repo fallbacks).
ARTIFACTS_TREE_WATCH_SIG: tuple = tuple()
TASK_MTIME = 0.0


def _artifacts_tree_watch_signature() -> tuple:
    sig: List[tuple] = []
    for p in _dedupe_paths(_tree_json_search_paths() + _tasks_tree_search_paths()):
        try:
            if p.exists():
                sig.append((str(p.resolve()), p.stat().st_mtime))
            else:
                sig.append((str(p.resolve()), 0.0))
        except OSError:
            sig.append((str(p), 0.0))
    return tuple(sorted(sig, key=lambda x: x[0]))


async def _watch_artifacts_json():
    """Следит за tree.json / tasks_tree.json по всем кандидатным путям."""
    global ARTIFACTS_TREE_WATCH_SIG
    while True:
        try:
            sig = _artifacts_tree_watch_signature()
            if sig != ARTIFACTS_TREE_WATCH_SIG:
                ARTIFACTS_TREE_WATCH_SIG = sig
                await broker.broadcast({"type": "tree", "data": _combine_orchestrator_and_tasks()})
        except Exception as e:
            logger.exception(f"watch artifacts error: {e}")
        await asyncio.sleep(1.0)

async def _watch_task_file():
    """Следит за artifacts/task.txt и пушит текст при изменениях."""
    global TASK_MTIME
    while True:
        try:
            p = _task_txt_path()
            m = p.stat().st_mtime if p.exists() else 0.0
            if m != TASK_MTIME:
                TASK_MTIME = m
                txt = p.read_text(encoding="utf-8") if p.exists() else ""
                await broker.broadcast({"type": "task", "text": txt})
        except Exception:
            pass
        await asyncio.sleep(1.0)

# Comment translated to English.
def _ensure_project_dirs():
    paths = _cfg_paths()
    # Comment translated to English.
    for key in ("artifacts", "logs", "scripts", "src", "data"):
        try:
            paths[key].mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    # Comment translated to English.
    try:
        task_path = _task_txt_path()
        task_path.parent.mkdir(parents=True, exist_ok=True)
        # If runtime task.txt is missing or empty, seed it from repo-root task.txt (if present).
        seed_from = PROJECT_ROOT / "task.txt"
        if (not task_path.exists()) or (task_path.exists() and task_path.stat().st_size == 0):
            if seed_from.exists():
                try:
                    seed_text = seed_from.read_text(encoding="utf-8")
                except Exception:
                    seed_text = ""
                if seed_text.strip():
                    task_path.write_text(seed_text, encoding="utf-8")
                else:
                    if not task_path.exists():
                        task_path.write_text("", encoding="utf-8")
            else:
                if not task_path.exists():
                    task_path.write_text("", encoding="utf-8")
    except Exception:
        pass

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="ML Orchestrator Dashboard", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

INDEX_HTML = PROJECT_ROOT / "ui.html"

@app.on_event("startup")
async def _startup():
    _ensure_project_dirs()  # Comment translated to English.
    asyncio.create_task(_watch_artifacts_json())
    asyncio.create_task(_watch_task_file())


# -------------------- ROUTES: UI --------------------
@app.get("/", include_in_schema=False)
async def root_redirect():
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/ui">')

@app.get("/ui", include_in_schema=False)
async def ui_page():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

# -------------------- ROUTES: WS + SSE --------------------
def _current_status() -> Dict[str, Any]:
    p = proc_state.process
    running = p is not None and p.poll() is None
    return {
        "running": running,
        "pid": p.pid if p else None,
        "args": proc_state.args,
        "start_ts": proc_state.start_ts,
        "resume": proc_state.resume,
        "killed": proc_state.killed,
    }

@app.websocket("/ws")
async def ws_main(ws: WebSocket):
    logger.info("WS: connection incoming")
    await broker.attach(ws)
    logger.info("WS: connected")
    try:
        await ws.send_text(json.dumps({"type": "status", "data": _current_status()}, ensure_ascii=False))
        await ws.send_text(json.dumps({"type": "tree", "data": _combine_orchestrator_and_tasks()}, ensure_ascii=False))
        tp = _task_txt_path()
        if tp.exists():
            await ws.send_text(json.dumps({"type": "task", "text": tp.read_text(encoding="utf-8")}, ensure_ascii=False))
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info("WS: disconnected")
    except Exception as e:
        logger.exception(f"WS error: {e}")
    finally:
        await broker.detach(ws)

@app.get("/events")
async def sse_events(request: Request):
    async def event_stream():
        last_tree_sent = 0.0
        while True:
            if await request.is_disconnected():
                break
            try:
                yield f"event: status\ndata: {json.dumps(_current_status(), ensure_ascii=False)}\n\n"
                # Comment translated to English.
                t = _combine_orchestrator_and_tasks()
                yield f"event: tree\ndata: {json.dumps(t, ensure_ascii=False)}\n\n"
                # task
                p = _task_txt_path()
                if p.exists():
                    yield f"event: task\ndata: {json.dumps({'text': p.read_text(encoding='utf-8')}, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.exception(f"SSE error: {e}")
            await asyncio.sleep(1.0)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# -------------------- ROUTES: API --------------------
@app.get("/api/status")
def api_status():
    return _current_status()

@app.get("/api/tree")
def api_tree():
    return _combine_orchestrator_and_tasks()

@app.get("/api/node/{node_id}")
def api_node(node_id: str):
    t = _combine_orchestrator_and_tasks()
    n = (t.get("nodes") or {}).get(node_id)
    if not n:
        raise HTTPException(404, "node not found")
    nodes = t.get("nodes") or {}
    ancestors: List[str] = []
    cur = n
    seen = set()
    while cur and cur.get("parent_node_id") and cur["parent_node_id"] not in seen:
        pid = cur["parent_node_id"]; seen.add(pid)
        p = nodes.get(pid)
        if not p: break
        ancestors.append(p["node_id"])
        cur = p
    children = list(n.get("children") or [])
    return {"node": n, "ancestors": ancestors, "children": children}

# Comment translated to English.
@app.put("/api/task")
def api_put_task(text: str = Body(..., media_type="text/plain")):
    _ensure_project_dirs()  # Comment translated to English.
    p = _task_txt_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return {"ok": True}


@app.get("/api/task")
def api_get_task() -> PlainTextResponse:
    """Return current task.txt content (runtime project root)."""
    p = _task_txt_path()
    if not p.exists():
        return PlainTextResponse("")
    try:
        return PlainTextResponse(p.read_text(encoding="utf-8"))
    except Exception:
        return PlainTextResponse("")

@app.get("/api/config")
def api_get_config():
    p = _config_yaml_path()
    return PlainTextResponse(p.read_text(encoding="utf-8") if p.exists() else "")

@app.put("/api/config")
def api_put_config(text: str = Body(..., media_type="text/plain"), validate: bool = Query(True)):
    if validate:
        try:
            obj = yaml.safe_load(text) if text.strip() else {}
            if not isinstance(obj, dict):
                raise ValueError("YAML root must be a mapping")
        except Exception as e:
            raise HTTPException(400, f"YAML error: {e}")
    p = _config_yaml_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return {"ok": True}


@app.get("/api/task_plan")
def api_get_task_plan():
    """Return the current task_plan.md if it exists (for UI/agents)."""
    # task_plan.md is produced under the runtime project root (same root as task.txt).
    path = _runtime_project_root() / "task_plan.md"
    if not path.exists():
        return PlainTextResponse("")
    try:
        return PlainTextResponse(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"failed to read task_plan.md: {e}")

# Comment translated to English.
@app.get("/api/file")
def api_file(path: str):
    _ensure_project_dirs()  # Comment translated to English.
    allowed = _cfg_paths()
    target = Path(path)

    def _under(target: Path, base: Path) -> bool:
        try:
            return target.resolve().is_file() and target.resolve().is_relative_to(base.resolve())
        except Exception:
            # Comment translated to English.
            try:
                target.resolve().relative_to(base.resolve())
                return True
            except Exception:
                return False

    if not any(_under(target, base) for base in allowed.values()):
        raise HTTPException(400, "path out of allowed roots (not in config paths)")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(target))


@app.post("/api/run")
def api_run(resume: bool = Query(False)):
    if proc_state.process and proc_state.process.poll() is None:
        raise HTTPException(409, "process already running")

    # Comment translated to English.
    _ensure_project_dirs()

    # Comment translated to English.
    tp = _task_txt_path()
    tp.parent.mkdir(parents=True, exist_ok=True)
    if not tp.exists():
        tp.write_text("Demo: print('Hello from generated script')", encoding="utf-8")

    # Comment translated to English.
    lp = _cfg_paths()["artifacts"] / "last_run.json"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({"resume": resume}, ensure_ascii=False, indent=2), encoding="utf-8")

    p = _spawn_process(resume=resume)
    proc_state.process = p
    th = threading.Thread(target=_capture_io, daemon=True)
    proc_state.capturing_thread = th
    th.start()
    return {"ok": True, "pid": p.pid, "args": proc_state.args}


@app.post("/api/stop")
def api_stop(force: bool = Query(False)):
    p = proc_state.process
    if not p or p.poll() is not None:
        return {"ok": True, "message": "not running"}
    try:
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if force:
                    p.kill()
        else:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if force:
                    p.kill()
        proc_state.killed = True
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"stop failed: {e}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

