from __future__ import annotations
import os, sys, json, platform, shutil, subprocess, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

def _gb(x: float) -> float:
    return round(float(x) / (1024**3), 3)

def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

def probe_hardware(respect_constraints: bool = True) -> Dict[str, Any]:
    """
    Универсальный сбор окружения: CPU/ОЗУ/Диск/GPU.
    Ничего не ставит в сеть. Пытается torch/psutil/pynvml — если есть.
    """
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": sys.version.split()[0],
        "cuda_available": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cpu": {"logical_cores": os.cpu_count() or 1, "physical_cores": None, "max_freq_mhz": None},
        "ram": {"total_gb": None, "available_gb": None},
        "disk": {},
        "gpus": []
    }

    # Comment translated to English.
    psutil = _safe_import("psutil")
    if psutil:
        try:
            vm = psutil.virtual_memory()
            info["ram"]["total_gb"] = _gb(vm.total)
            info["ram"]["available_gb"] = _gb(vm.available)
        except Exception:
            pass
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["cpu"]["max_freq_mhz"] = int(cpu_freq.max or cpu_freq.current or 0)
        except Exception:
            pass
        try:
            # Comment translated to English.
            info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False) or None
        except Exception:
            pass
    else:
        # Comment translated to English.
        try:
            if info["os"] == "Linux":
                with open("/proc/meminfo", "r") as f:
                    mem = f.read()
                for line in mem.splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram"]["total_gb"] = round(kb/1024/1024, 3)
                        break
            # Comment translated to English.
        except Exception:
            pass

    # Comment translated to English.
    try:
        root = Path(".").resolve()
        usage = shutil.disk_usage(str(root))
        info["disk"] = {
            "mount": str(root.drive if info["os"] == "Windows" else root.anchor),
            "total_gb": _gb(usage.total),
            "free_gb": _gb(usage.free)
        }
    except Exception:
        info["disk"] = {}

    # Comment translated to English.
    torch = _safe_import("torch")
    if torch and getattr(torch, "cuda", None) and torch.cuda.is_available():
        info["cuda_available"] = True
        try:
            cnt = torch.cuda.device_count()
            for i in range(cnt):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_gb = round(props.total_memory/1024/1024/1024, 3)
                free_gb = None
                try:
                    # Comment translated to English.
                    pynvml = _safe_import("pynvml")
                    if pynvml:
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        free_gb = round(mem.free/1024/1024/1024, 3)
                        pynvml.nvmlShutdown()
                except Exception:
                    pass
                info["gpus"].append({
                    "id": i, "name": name, "vram_total_gb": total_gb, "vram_free_gb": free_gb
                })
        except Exception:
            pass
    else:
        # Comment translated to English.
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if out.returncode == 0:
                for idx, line in enumerate(out.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        name, total, free = parts[0], parts[1], parts[2]
                        info["gpus"].append({
                            "id": idx, "name": name,
                            "vram_total_gb": round(float(total)/1024, 3),
                            "vram_free_gb": round(float(free)/1024, 3)
                        })
                info["cuda_available"] = len(info["gpus"]) > 0
        except Exception:
            pass

    return info

def _sum_dir(path: Path, limit_files: int = 5000) -> Dict[str, Any]:
    """
    Быстрая оценка размера директории. Скан ограничиваем limit_files (для больших датасетов),
    и помечаем approximate=True.
    """
    total_bytes = 0
    files = 0
    approx = False
    for root, _, fnames in os.walk(path):
        for fn in fnames:
            files += 1
            try:
                total_bytes += (path / Path(root).relative_to(path) / fn).stat().st_size
            except Exception:
                pass
            if files >= limit_files:
                approx = True
                break
        if approx:
            break
    return {"bytes": int(total_bytes), "files_scanned": files, "approximate": approx}

def estimate_dataset_footprint(spec: Dict[str, Any], limit_files: int = 5000) -> Dict[str, Any]:
    """
    Грубо оценивает объём датасета на диске и даёт простые предположения об ОЗУ.
    """
    data = spec.get("data", {}) or {}
    meta = (data.get("meta") or {})
    root = Path(data.get("resolved_root") or ".").resolve()

    out: Dict[str, Any] = {"root": str(root), "sizes": {}, "totals": {}, "notes": ""}

    # CSV/TSV
    for key in ["train_csv", "test_csv", "labels_csv", "sample_submission_csv"]:
        p = data.get(key)
        if p:
            fp = Path(p)
            try:
                sz = fp.stat().st_size
                out["sizes"][key] = {"bytes": int(sz), "approximate": False}
            except Exception:
                out["sizes"][key] = {"bytes": 0, "approximate": True}

    # Comment translated to English.
    for key in ["train_dir", "test_dir"]:
        p = data.get(key)
        if p and isinstance(p, str):
            d = Path(p)
            if d.is_dir():
                s = _sum_dir(d, limit_files=limit_files)
                out["sizes"][key] = s

    # Comment translated to English.
    total_bytes = sum(v.get("bytes", 0) for v in out["sizes"].values())
    out["totals"]["bytes_on_disk"] = int(total_bytes)
    out["totals"]["gb_on_disk"] = _gb(total_bytes)

    # Comment translated to English.
    csv_bytes = sum(v.get("bytes", 0) for k, v in out["sizes"].items() if k.endswith("_csv"))
    out["totals"]["est_ram_needed_gb_for_tabular"] = round((csv_bytes * 2.0) / (1024**3), 2)

    # Comment translated to English.
    out["notes"] = "Tabular memory ~2x csv size; images/audio/videos typically streamed in batches."

    return out

def recommend_resource_plan(spec: Dict[str, Any], hw: Dict[str, Any], ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Простые эвристики: воркеры, потоки, batch-size hint, AMP.
    Без знания модели — только безопасные базовые правила.
    """
    cpu_cores = int(hw.get("cpu", {}).get("logical_cores") or 1)
    avail_ram = float(hw.get("ram", {}).get("available_gb") or 0.0)
    gpus = hw.get("gpus", []) or []
    cuda = bool(hw.get("cuda_available", False))

    # Comment translated to English.
    # Comment translated to English.
    if avail_ram and avail_ram < 4:
        workers = min(2, max(1, cpu_cores//4))
    else:
        workers = max(1, min(8, cpu_cores - 1))

    threads = max(1, min(16, cpu_cores))
    use_amp = bool(cuda)  # Comment translated to English.
    prefetch = 2
    persistent = workers > 0

    # Comment translated to English.
    bs_hints: Dict[str, int] = {"tabular_rows_per_chunk": 8192}
    if cuda and gpus:
        # Comment translated to English.
        free = gpus[0].get("vram_free_gb") or gpus[0].get("vram_total_gb") or 4.0
        # Comment translated to English.
        bs_hints.update({
            "image_bs": int(max(8, min(64, math.floor(free * 0.8)))),
            "text_bs": int(max(8, min(64, math.floor(free * 1.5)))),
        })
    else:
        bs_hints.update({"image_bs": 8, "text_bs": 16})

    # Comment translated to English.
    free_disk = float(hw.get("disk", {}).get("free_gb") or 0.0)
    ds_gb = float(ds.get("totals", {}).get("gb_on_disk") or 0.0)
    disk_ok = free_disk > max(5.0, ds_gb * 0.5)  # Comment translated to English.

    plan = {
        "cpu": {
            "logical_cores": cpu_cores,
            "workers": workers,
            "threads": threads,
        },
        "ram": {
            "available_gb": avail_ram,
            "low_memory_mode": avail_ram and avail_ram < 8,
        },
        "gpu": {
            "use_cuda": cuda,
            "count": len(gpus),
            "amp": use_amp,
            "gpus": gpus,
        },
        "io": {
            "prefetch_factor": prefetch,
            "persistent_workers": persistent,
        },
        "batch_size_hint": bs_hints,
        "disk_ok": disk_ok,
        "notes": "Heuristics only; model-specific tuning may override."
    }
    return plan

def attach_hardware_to_spec(orch, spec: Dict[str, Any], limit_files: int = 5000) -> Dict[str, Any]:
    """
    Высокоуровневая обвязка для пайплайна:
    - снимаем hardware env
    - оцениваем footprint датасета
    - генерим resource plan
    - сохраняем на диск и возвращаем обновлённый spec
    """
    # Comment translated to English.
    constraints = (spec.get("constraints") or {})
    env = probe_hardware(respect_constraints=True)
    ds = estimate_dataset_footprint(spec, limit_files=limit_files)
    plan = recommend_resource_plan(spec, env, ds)

    spec.setdefault("hardware", {})
    spec["hardware"]["env"] = env
    spec["hardware"]["dataset"] = ds
    spec["hardware"]["plan"] = plan

    try:
        payload = {"env": env, "dataset": ds, "plan": plan}
        orch.write_file(f"{orch.cfg.paths.artifacts_dir}/hardware.json", json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        pass

    return spec
