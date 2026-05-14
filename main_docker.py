from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

from src.config import AppConfig, load_dotenv_from_cwd


PROJECT_ROOT = Path(__file__).parent.resolve()


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _default_task_from_cfg(cfg: AppConfig) -> Path:
    # Keep compatibility with existing workflow where task.txt is in repo root.
    return (PROJECT_ROOT / "task.txt").resolve()


def _prepare_container_config(cfg_path: Path, cfg: AppConfig, container_data_dir: str, container_task_file: str) -> Path:
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}

    paths = raw.get("paths")
    if not isinstance(paths, dict):
        paths = {}
        raw["paths"] = paths

    # Inside container we use mounted input directory and task file.
    paths["data_dir"] = container_data_dir

    # Keep runtime/preinstall behavior from host config.yaml.
    # Docker-specific adjustments should only touch mounted paths/task file.
    # Keep task file path in config for tools that may inspect it.
    runtime = raw.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        raw["runtime"] = runtime
    # In container all run artifacts should stay in mounted workspace root.
    runtime["project_root"] = "/workspace"
    runtime["project_name"] = ""

    raw.setdefault("docker", {})
    if isinstance(raw["docker"], dict):
        raw["docker"]["task_file"] = container_task_file

    out_dir = PROJECT_ROOT / cfg.paths.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = out_dir / "config.docker.yaml"
    out_cfg.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_cfg


def _build_image(image: str, dockerfile: str, rebuild: bool) -> None:
    args = ["docker", "build", "-f", dockerfile, "-t", image, "."]
    if rebuild:
        args.insert(2, "--no-cache")
    print(f"[DOCKER] Building image: {image}")
    subprocess.run(args, cwd=str(PROJECT_ROOT), check=True)


def _collect_env(cfg: AppConfig) -> Dict[str, str]:
    env: Dict[str, str] = {}
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ]
    # Export from config first.
    cfg.apply_env()
    for k in keys:
        v = os.getenv(k, "")
        if v:
            env[k] = v
    return env


def _run_container(
    image: str,
    cfg: AppConfig,
    host_cfg: Path,
    host_data_dir: Path,
    host_task_file: Path,
    resume: bool,
    gpus: str,
    log_file: Path,
) -> int:
    container_cfg = "/workspace/config.docker.yaml"
    container_data = "/workspace/data_input"
    container_task = f"/workspace/task_input/{host_task_file.name}"

    run_args: List[str] = [
        "docker", "run", "--rm",
        "--gpus", gpus,
        "-v", f"{PROJECT_ROOT}:/workspace",
        "-v", f"{host_data_dir}:/workspace/data_input:ro",
        "-v", f"{host_task_file.parent}:/workspace/task_input:ro",
        "-v", f"{host_cfg}:/workspace/config.docker.yaml:ro",
        "-w", "/workspace",
    ]

    for k, v in _collect_env(cfg).items():
        run_args.extend(["-e", f"{k}={v}"])

    cmd = ["python", "main.py", "--config", container_cfg, "--task_file", container_task]
    if resume:
        cmd.append("--resume")
    run_args.extend([image, *cmd])

    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOCKER] Running container with task: {host_task_file}")
    print(f"[DOCKER] Logs: {log_file}")

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write("\n=== docker run start ===\n")
        lf.write(
            "# Host-side capture of container stdout/stderr. "
            "Inside the container, main.py also writes project_root/logs/run_*.log and last_run.log by default.\n"
        )
        lf.write(" ".join(run_args) + "\n")
        proc = subprocess.Popen(
            run_args,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()
        lf.write(f"=== docker run end (exit={proc.returncode}) ===\n")
        return int(proc.returncode or 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml", help="Path to host config.yaml")
    p.add_argument("--task_file", default="", help="Path to host task file (defaults to ./task.txt)")
    p.add_argument("--resume", action="store_true", help="Resume mode for pipeline")
    p.add_argument("--image", default="linguainterpreter:cuda", help="Docker image tag")
    p.add_argument("--dockerfile", default="Dockerfile.cuda", help="Dockerfile path")
    p.add_argument("--rebuild", action="store_true", help="Force no-cache image rebuild")
    p.add_argument("--gpus", default="all", help="Value for docker --gpus")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_from_cwd()
    cfg_path = _resolve(args.config)
    cfg = AppConfig.from_yaml(cfg_path)
    cfg.apply_env()

    host_data_dir = _resolve(cfg.paths.data_dir)
    if not host_data_dir.exists():
        print(f"[DOCKER][ERROR] data_dir does not exist: {host_data_dir}")
        return 2

    host_task_file = _resolve(args.task_file) if args.task_file else _default_task_from_cfg(cfg)
    if not host_task_file.exists():
        print(f"[DOCKER][ERROR] task file does not exist: {host_task_file}")
        return 2

    container_task_file = f"/workspace/task_input/{host_task_file.name}"
    docker_cfg = _prepare_container_config(
        cfg_path=cfg_path,
        cfg=cfg,
        container_data_dir="/workspace/data_input",
        container_task_file=container_task_file,
    )

    _build_image(args.image, args.dockerfile, args.rebuild)

    log_file = PROJECT_ROOT / cfg.paths.artifacts_dir / "docker_run.log"
    return _run_container(
        image=args.image,
        cfg=cfg,
        host_cfg=docker_cfg,
        host_data_dir=host_data_dir,
        host_task_file=host_task_file,
        resume=args.resume,
        gpus=args.gpus,
        log_file=log_file,
    )


if __name__ == "__main__":
    sys.exit(main())
