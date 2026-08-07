"""GPU, disk and runtime preflight for formal-like generator runs."""

from __future__ import annotations

import inspect
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        first = output.stdout.strip().splitlines()[0]
        name, total, free, driver = [part.strip() for part in first.split(",")]
        return {
            "visible": True,
            "name": name,
            "memory_total_mib": int(total),
            "memory_free_mib": int(free),
            "driver_version": driver,
        }
    except Exception as exc:
        return {"visible": False, "error": f"{type(exc).__name__}: {exc}"}


def generator_preflight(project_root: str | Path) -> dict[str, Any]:
    import sdmetrics
    import sdv
    import sklearn
    import torch
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

    root = Path(project_root).resolve()
    disk = shutil.disk_usage(root)
    cuda_available = bool(torch.cuda.is_available())
    tensor_test: dict[str, Any]
    if cuda_available:
        try:
            tensor = torch.tensor([1.0], device="cuda") * 2
            tensor_test = {"passed": float(tensor.cpu().item()) == 2.0}
        except Exception as exc:
            tensor_test = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        tensor_test = {"passed": False, "error": "torch.cuda.is_available() is False"}
    signatures = {
        "CTGAN": str(inspect.signature(CTGANSynthesizer.__init__)),
        "TVAE": str(inspect.signature(TVAESynthesizer.__init__)),
    }
    enable_gpu_supported = all("enable_gpu" in value for value in signatures.values())
    hard_gpu_passed = cuda_available and tensor_test["passed"] and enable_gpu_supported
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "sdv": sdv.__version__,
        "sdmetrics": sdmetrics.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "torch_gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "tensor_gpu_test": tensor_test,
        "nvidia_smi": _nvidia_smi(),
        "sdv_constructor_signatures": signatures,
        "enable_gpu_supported": enable_gpu_supported,
        "hard_gpu_passed": hard_gpu_passed,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "estimate": {
            "method": "linear CPU extrapolation from Adult 3000-row/20-epoch smoke; GPU runtime may differ",
            "ctgan_full_300_seconds": 79.1147939 * (29273 / 3000) * (300 / 20),
            "tvae_full_300_seconds": 45.4438992 * (29273 / 3000) * (300 / 20),
            "pilot_artifact_bytes_each": 220000000,
        },
        "gpu_environment_repair": {
            "executed": False,
            "recommended_environment": "tabpollution-gpu",
            "note": "Create an isolated conda environment, install a PyTorch CUDA build compatible with driver, then reinstall pinned SDV/SDMetrics; verify with the commands below before use.",
            "commands": [
                "conda create -n tabpollution-gpu python=3.11 -y",
                "conda activate tabpollution-gpu",
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
                "pip install sdv==1.37.3 sdmetrics==0.28.0 pandas==2.3.3 scikit-learn==1.9.0",
                "python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\"",
            ],
            "estimated_download_and_disk": "Several GB; verify official PyTorch selector against the installed driver before executing.",
            "rollback": "Remove only the isolated tabpollution-gpu environment; do not modify tabpollution.",
        },
    }


def ensure_disk_space(project_root: str | Path, minimum_free_bytes: int) -> None:
    free = shutil.disk_usage(Path(project_root)).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"Insufficient disk before training: free={free}, required={minimum_free_bytes}")
