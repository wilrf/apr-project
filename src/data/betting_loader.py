"""Betting data loading utilities for Kaggle spreadspoke dataset."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

# Full team name to nflverse abbreviation mapping
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Colts": "IND",
    "Baltimore Ravens": "BAL",
    "Boston Patriots": "NE",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Oilers": "TEN",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Raiders": "LV",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Oakland Raiders": "LV",
    "Philadelphia Eagles": "PHI",
    "Phoenix Cardinals": "ARI",
    "Pittsburgh Steelers": "PIT",
    "San Diego Chargers": "LAC",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "St. Louis Cardinals": "ARI",
    "St. Louis Rams": "LA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Oilers": "TEN",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}

# Team abbreviation mapping for relocations and inconsistencies
TEAM_ABBR_MAP = {
    # Relocations
    "STL": "LA",   # Rams to LA (2016)
    "SD": "LAC",   # Chargers to LA (2017)
    "OAK": "LV",   # Raiders to Vegas (2020)
    # Naming inconsistencies
    "JAC": "JAX",
    "WSH": "WAS",
    "LVR": "LV",
    "LAR": "LA",
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

    # Convert week to numeric, filtering out playoff games (named rounds)
    df["schedule_week"] = pd.to_numeric(df["schedule_week"], errors="coerce")
    df = df[df["schedule_week"].notna()].copy()
    df["schedule_week"] = df["schedule_week"].astype(int)

    # Convert team names to abbreviations (handles full names like "Kansas City Chiefs")
    # First try full name mapping, then apply abbreviation normalization
    df["team_home"] = df["team_home"].apply(
        lambda x: normalize_team_abbr(TEAM_NAME_TO_ABBR.get(x, x))
    )
    df["team_away"] = df["team_away"].apply(
        lambda x: normalize_team_abbr(TEAM_NAME_TO_ABBR.get(x, x))
    )

    return df
