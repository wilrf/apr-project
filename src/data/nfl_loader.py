"""NFL data loading utilities using nfl_data_py."""

from __future__ import annotations

import pandas as pd
import nfl_data_py as nfl
from typing import List


def load_schedules(
    seasons: List[int],
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """
    Load NFL schedule data for specified seasons.

    Args:
        seasons: List of seasons to load (e.g., [2022, 2023])
        regular_season_only: If True, filter to regular season games only

    Returns:
        DataFrame with game schedule information
    """
    df = nfl.import_schedules(seasons)

    if regular_season_only:
        df = df[df["game_type"] == "REG"].copy()

    return df


def load_pbp_data(seasons: List[int]) -> pd.DataFrame:
    """
    Load play-by-play data for specified seasons.

    Args:
        seasons: List of seasons to load

    Returns:
        DataFrame with play-by-play data including EPA
    """
    df = nfl.import_pbp_data(seasons)
    return df
