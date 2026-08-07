"""Non-invasive environment capture for reproducibility reports."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ["pandas", "numpy", "scikit-learn", "PyYAML", "pytest", "sdv", "sdmetrics", "torch"]


def capture_environment(path: str | Path) -> None:
    lines = [
        f"captured_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
        f"machine={platform.machine()}",
        f"logical_cpu_count={os.cpu_count()}",
    ]
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        lines.append(f"package.{package}={version}")
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        lines.append(f"nvidia_smi={gpu.stdout.strip() or gpu.stderr.strip() or 'UNAVAILABLE'}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        lines.append("nvidia_smi=UNAVAILABLE")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

