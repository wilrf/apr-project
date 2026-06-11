from __future__ import annotations

import numpy as np

from src.utils.seed import set_global_seed


def test_set_global_seed_makes_numpy_reproducible():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_global_seed_returns_the_seed():
    assert set_global_seed(7) == 7
