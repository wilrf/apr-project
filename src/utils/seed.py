"""Global seeding for reproducible runs.

Seeds Python's `random`, NumPy, and (when installed) PyTorch from a single
entry point so experiment scripts can pin determinism in one call.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> int:
    """Seed Python, NumPy, and Torch (if available). Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
