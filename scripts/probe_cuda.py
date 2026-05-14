import json

try:
    import torch

    info = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_name_0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
except Exception as e:
    info = {"torch_version": None, "cuda_available": False, "error": str(e)}
print("BOOTSTRAP_CUDA_PROBE_JSON:", json.dumps(info))
