"""Seed utilities for CPU checks and future explicitly selected devices."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, device: str = "cpu", deterministic: bool = True) -> None:
    """Seed RNGs; CPU mode never requests CUDA execution.

    Reproducibility is scoped to the same software and hardware. Set
    PYTHONHASHSEED before launching Python if hash ordering matters.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a boolean")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda or mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; no CPU fallback")
    random.seed(seed)
    np.random.seed(seed)
    # Direct CPU generator seeding avoids torch.manual_seed's accelerator seeding.
    torch.random.default_generator.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    elif device == "mps":
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
