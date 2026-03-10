"""NFL data loading utilities."""

from __future__ import annotations

from typing import List

import nfl_data_py as nfl
import pandas as pd

NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)
REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


def _load_schedule_source() -> pd.DataFrame:
    """Load the canonical nflverse schedule dataset over HTTPS."""
    try:
        return pd.read_csv(NFLVERSE_GAMES_URL, low_memory=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download NFL schedules from nflverse: {e}"
        ) from e


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
    df = _load_schedule_source()

    missing_columns = sorted(REQUIRED_SCHEDULE_COLUMNS - set(df.columns))
    if missing_columns:
        raise RuntimeError(
            "Failed to load NFL schedules from nflverse. "
            f"Missing required columns: {missing_columns}"
        )

    df = df[df["season"].isin(seasons)].copy()

    if regular_season_only:
        df = df[df["game_type"] == "REG"].copy()

    return df.reset_index(drop=True)


def load_pbp_data(seasons: List[int]) -> pd.DataFrame:
    """
    Load play-by-play data for specified seasons.

    Args:
        seasons: List of seasons to load

    Returns:
        DataFrame with play-by-play data including EPA
    """
    try:
        df = nfl.import_pbp_data(seasons)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load play-by-play data for {seasons}. "
            f"Check your network connection and nfl_data_py installation: {e}"
        ) from e
    return df
