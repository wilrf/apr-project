"""Betting data loading utilities for Kaggle spreadspoke dataset."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

# Team abbreviation mapping for relocations and inconsistencies
TEAM_ABBR_MAP = {
    # Relocations
    "STL": "LA",   # Rams to LA (2016)
    "SD": "LAC",   # Chargers to LA (2017)
    "OAK": "LV",   # Raiders to Vegas (2020)
    # Naming inconsistencies
    "JAC": "JAX",
    "WSH": "WAS",
}


def normalize_team_abbr(abbr: str) -> str:
    """
    Normalize team abbreviation to nflverse standard.

    Args:
        abbr: Team abbreviation to normalize

    Returns:
        Normalized team abbreviation
    """
    return TEAM_ABBR_MAP.get(abbr, abbr)


def load_betting_data(
    filepath: Optional[Path] = None,
    min_season: Optional[int] = None,
    max_season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load betting data from Kaggle spreadspoke dataset.

    Args:
        filepath: Path to CSV file. Defaults to data/raw/spreadspoke_scores.csv
        min_season: Minimum season to include
        max_season: Maximum season to include

    Returns:
        DataFrame with betting data
    """
    if filepath is None:
        filepath = Path("data/raw/spreadspoke_scores.csv")

    df = pd.read_csv(filepath)

    # Filter by season if specified
    if min_season is not None:
        df = df[df["schedule_season"] >= min_season]
    if max_season is not None:
        df = df[df["schedule_season"] <= max_season]

    # Normalize team abbreviations
    df["team_home"] = df["team_home"].apply(normalize_team_abbr)
    df["team_away"] = df["team_away"].apply(normalize_team_abbr)

    return df
