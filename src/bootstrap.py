from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

def bootstrap_gpu_stack(orch, cfg, llm_fast: Optional[Any] = None) -> None:
    """
    1) Ставит только пакеты из config.preinstall.pkgs
    2) Для torch-пакетов использует CUDA index-url из config.preinstall.torch_cuda_index_url
    3) Пробует краткий CUDA-пробник и печатает статус
    """
    pre = getattr(cfg, "preinstall", None) or {}
    if not pre or not getattr(pre, 'enable', True):
        print("[bootstrap] preinstall disabled")
        return

    pkgs: List[str] = list(getattr(pre, 'pkgs', []))
    # Idempotency guard to avoid re-installing the same deps multiple times.
    marker_path = orch.dir_paths["logs"] / "preinstall_done.marker"
    should_skip = marker_path.exists()
    if should_skip:
        # venv is recreated per run; verify required modules are importable in current venv.
        # If imports fail, force reinstall (do NOT trust marker blindly).
        required_modules: List[str] = []
        for p in pkgs:
            pl = str(p).strip().lower()
            if pl.startswith("scikit-learn"):
                required_modules.append("sklearn")
            elif pl.startswith("pandas"):
                required_modules.append("pandas")
            elif pl.startswith("numpy"):
                required_modules.append("numpy")
            elif pl.startswith("torchvision"):
                required_modules.append("torchvision")
            elif pl.startswith("torchaudio"):
                required_modules.append("torchaudio")
            elif pl.startswith("torch"):
                required_modules.append("torch")
            elif pl.startswith("timm"):
                required_modules.append("timm")
            else:
                # Fallback: assume importable module has same name as pip package.
                required_modules.append(pl.replace("-", "_"))

        try:
            check_script = []
            for m in sorted(set(required_modules)):
                check_script.append(f"import {m} as _m_{m}")
            if any(str(p).strip().lower().startswith("langchain-anthropic") for p in pkgs):
                check_script.append("from langchain_anthropic import ChatAnthropic")
            code = "\n".join(check_script)
            res = orch.run_python_code(
                code,
                filename="preinstall_import_check.py",
                timeout=30,
                stream=False,
            )
            should_skip = int(res.get("exit_code", 1)) == 0
        except Exception:
            should_skip = False

    if should_skip:
        print("[bootstrap] preinstall marker exists and venv looks ready, skipping package install")
    else:
        if pkgs:
            index_url = getattr(pre, 'torch_cuda_index_url', "")
            torch_roots = ("torch", "torchvision", "torchaudio")

            other_pkgs: List[str] = []
            torch_pkgs: List[str] = []
            for p in pkgs:
                pl = str(p).strip().lower()
                if any(pl.startswith(root) for root in torch_roots):
                    torch_pkgs.append(p)
                else:
                    other_pkgs.append(p)

            # Install non-torch packages from default index (important: CUDA index hosts only torch wheels).
            if other_pkgs:
                print(f"[bootstrap] Installing preinstall non-torch pkgs: {other_pkgs}")
                orch.pip_install(other_pkgs, extra="")

            # Install torch packages from CUDA index (if configured).
            if torch_pkgs:
                extra = ""
                if index_url:
                    extra = f"--index-url {index_url}"
                print(f"[bootstrap] Installing preinstall torch pkgs: {torch_pkgs} {('with ' + extra) if extra else ''}")
                orch.pip_install(torch_pkgs, extra=extra)
        else:
            print("[bootstrap] No preinstall.pkgs configured, skipping package install")

    # Only write marker after we successfully proved imports (or marker existed and imports passed above).
    if not should_skip:
        try:
            test_modules: List[str] = []
            for p in pkgs:
                pl = str(p).strip().lower()
                if pl.startswith("scikit-learn"):
                    test_modules.append("sklearn")
                elif pl.startswith("pandas"):
                    test_modules.append("pandas")
                elif pl.startswith("numpy"):
                    test_modules.append("numpy")
                elif pl.startswith("torchvision"):
                    test_modules.append("torchvision")
                elif pl.startswith("torchaudio"):
                    test_modules.append("torchaudio")
                elif pl.startswith("torch"):
                    test_modules.append("torch")
                elif pl.startswith("timm"):
                    test_modules.append("timm")
                else:
                    test_modules.append(pl.replace("-", "_"))
            check_script = "\n".join([f"import {m}" for m in sorted(set(test_modules))])
            res = orch.run_python_code(check_script, filename="preinstall_import_check_2.py", timeout=30, stream=False)
            if int(res.get("exit_code", 1)) == 0:
                marker_path.write_text(str(time.time()), encoding="utf-8")
        except Exception:
            # If verification fails, do not write marker; the next run should retry.
            pass

    from src.router import CUDA_PROBE_PY

    res = orch.run_python_code(CUDA_PROBE_PY, filename="probe_cuda.py")
    out = (res.get("output") or "") + (res.get("errors") or "")
    print(out)
    cuda_bad = "cuda_available\": false" in out.lower() or '"cuda_available": false' in out.lower()
    if cuda_bad:
        print("[bootstrap] WARNING: PyTorch does not see CUDA in this venv (torch.cuda.is_available() is False).")
        if llm_fast is not None:
            try:
                from src.router import repair_torch_cuda_with_react

                ok = repair_torch_cuda_with_react(llm_fast, orch, cfg)
                if ok:
                    print("[bootstrap] CUDA repair ReAct: torch now sees CUDA — re-run probe.")
                    res = orch.run_python_code(CUDA_PROBE_PY, filename="probe_cuda.py")
                    print((res.get("output") or "") + (res.get("errors") or ""))
                else:
                    print("[bootstrap] CUDA repair ReAct did not enable CUDA; check preinstall.torch_cuda_index_url and drivers.")
            except Exception as e:
                print(f"[bootstrap] CUDA repair ReAct failed: {e}")
        else:
            print("[bootstrap] Hint: set preinstall.torch_cuda_index_url and torch* in preinstall.pkgs, or run pipeline after LLMs load.")
