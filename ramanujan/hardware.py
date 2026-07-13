"""Local hardware detection.

For local execution, the agents should know what the machine actually offers:
an NVIDIA GPU with a working PyTorch/CUDA stack means the engineer may train
on `device='cuda'`; otherwise it must stick to CPU libraries. The note this
module produces is appended to the task's environment_notes, so the planner,
EDA agent and engineer all see the same truth.

Set RAMANUJAN_FORCE_CPU=1 to keep runs on CPU even when a GPU is present.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache


def _detect_gpu_name() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return ""


def _torch_cuda_available() -> bool:
    try:
        import torch  # noqa: F401  (optional dependency)

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def local_hardware_note() -> str:
    """One sentence describing the local training hardware, for prompt injection."""
    if os.environ.get("RAMANUJAN_FORCE_CPU"):
        return "LOCAL HARDWARE: treat this machine as CPU-only (RAMANUJAN_FORCE_CPU is set)."
    gpu = _detect_gpu_name()
    if not gpu:
        return "LOCAL HARDWARE: no NVIDIA GPU detected; CPU-only."
    if _torch_cuda_available():
        return (
            f"LOCAL HARDWARE: NVIDIA GPU available ({gpu}) with working PyTorch CUDA - "
            "you MAY train on the GPU (device='cuda') when the approach benefits from it."
        )
    return (
        f"LOCAL HARDWARE: an NVIDIA GPU ({gpu}) is present but PyTorch/CUDA is not usable "
        "from this Python environment - use CPU libraries."
    )
