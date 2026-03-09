"""Canonical target helpers for upset prediction."""

from __future__ import annotations

import pandas as pd

from src.features.pipeline import (
    _identify_underdog as _pipeline_identify_underdog,
)
from src.features.pipeline import (
    calculate_upset_target as _pipeline_calculate_upset_target,
)


def identify_underdog(row: pd.Series) -> str | None:
    """Identify the underdog team when favorite metadata is available."""
    return _pipeline_identify_underdog(row)


def calculate_upset_target(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the canonical upset target used by the feature pipeline."""
    return _pipeline_calculate_upset_target(df)
