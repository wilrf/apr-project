"""Betting data loading utilities for Kaggle spreadspoke dataset."""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional

# Full team name to nflverse abbreviation (includes historical names)
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Colts": "IND", "Baltimore Ravens": "BAL",
    "Boston Patriots": "NE", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Oilers": "TEN", "Houston Texans": "HOU",
    "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Raiders": "LV", "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ", "Oakland Raiders": "LV", "Philadelphia Eagles": "PHI",
    "Phoenix Cardinals": "ARI", "Pittsburgh Steelers": "PIT", "San Diego Chargers": "LAC", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "St. Louis Cardinals": "ARI", "St. Louis Rams": "LA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Oilers": "TEN", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
}

# Abbreviation normalization (relocations + inconsistencies)
TEAM_ABBR_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "LAR": "LA"}


def normalize_team_abbr(abbr: str) -> str:
    """Normalize team abbreviation to nflverse standard."""
    return TEAM_ABBR_MAP.get(abbr, abbr)


def load_betting_data(filepath: Optional[Path] = None, min_season: Optional[int] = None,
                      max_season: Optional[int] = None) -> pd.DataFrame:
    """Load betting data from Kaggle spreadspoke dataset."""
    df = pd.read_csv(filepath or Path("data/raw/spreadspoke_scores.csv"))

    if min_season:
        df = df[df["schedule_season"] >= min_season]
    if max_season:
        df = df[df["schedule_season"] <= max_season]

    # Filter to numeric weeks (exclude playoff rounds)
    df["schedule_week"] = pd.to_numeric(df["schedule_week"], errors="coerce")
    df = df[df["schedule_week"].notna()].copy()
    df["schedule_week"] = df["schedule_week"].astype(int)

    # Normalize team names/abbreviations
    convert = lambda n: normalize_team_abbr(TEAM_NAME_TO_ABBR.get(n, n)) if pd.notna(n) else n
    for col in ["team_home", "team_away"] + (["team_favorite_id"] if "team_favorite_id" in df.columns else []):
        df[col] = df[col].apply(convert)

    return df
