"""Shared helpers for prediction formatting and reporting."""

from __future__ import annotations

import pandas as pd


def safe_team_str(value: object) -> str:
    """Convert a team name value to string, handling NaN/None."""
    return "" if pd.isna(value) else str(value)
