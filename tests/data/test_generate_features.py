"""Tests for feature-generation orchestration helpers."""

from __future__ import annotations

import pandas as pd

from src.data.generate_features import _atomic_write_csv


def test_atomic_write_csv_replaces_target_file(tmp_path):
    output = tmp_path / "train.csv"
    output.write_text("stale")

    _atomic_write_csv(pd.DataFrame({"a": [1, 2]}), output)

    written = pd.read_csv(output)
    assert list(written["a"]) == [1, 2]
    assert list(tmp_path.glob("*.tmp")) == []
