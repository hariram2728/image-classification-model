"""
Reproducibility utilities for consistent results.
"""

import random
import os
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Use deterministic algorithms (may impact performance)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_rng_state() -> dict:
    """
    Get current random number generator states.

    Returns:
        Dictionary containing RNG states
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(state: dict) -> None:
    """
    Load random number generator states.

    Args:
        state: Dictionary containing RNG states from get_rng_state()
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class SeedContext:
    """Context manager for temporary seed setting."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.saved_state = None

    def __enter__(self):
        self.saved_state = get_rng_state()
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.saved_state:
            load_rng_state(self.saved_state)
        return False


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader.

    Ensures each worker has a different seed.

    Args:
        worker_id: ID of the DataLoader worker
    """
    # Get base seed from main process
    base_seed = torch.initial_seed() % 2**32

    # Set worker-specific seed
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
