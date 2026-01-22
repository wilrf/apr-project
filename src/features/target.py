"""Target variable calculation for upset prediction."""

from __future__ import annotations

import pandas as pd
from typing import Optional

MINIMUM_SPREAD = 3.0  # Minimum spread to qualify as underdog


def identify_underdog(row: pd.Series) -> Optional[str]:
    """
    Identify the underdog team for a game.

    Args:
        row: Game row with spread and team information

    Returns:
        Underdog team abbreviation, or None if spread < 3 or data is missing
    """
    # Check for missing spread or favorite data first
    if pd.isna(row["spread_favorite"]) or pd.isna(row["team_favorite_id"]):
        return None

    spread = abs(row["spread_favorite"])

    if spread < MINIMUM_SPREAD:
        return None

    favorite = row["team_favorite_id"]

    if favorite == row["home_team"]:
        return row["away_team"]
    else:
        return row["home_team"]


def calculate_upset_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate binary upset target variable.

    Upset = 1 if underdog wins outright, 0 otherwise.
    Games with spread < 3 are excluded (NaN target).

    Args:
        df: DataFrame with game results and spread information

    Returns:
        DataFrame with added 'upset' and 'underdog' columns
    """
    df = df.copy()

    # Identify underdog for each game
    df["underdog"] = df.apply(identify_underdog, axis=1)

    # Calculate winner using vectorized operations
    df["winner"] = None
    home_wins = df["home_score"] > df["away_score"]
    away_wins = df["away_score"] > df["home_score"]
    df.loc[home_wins, "winner"] = df.loc[home_wins, "home_team"]
    df.loc[away_wins, "winner"] = df.loc[away_wins, "away_team"]

    # Identify favorite for each game (for matchup differentials)
    df["favorite"] = df["team_favorite_id"].where(df["underdog"].notna())

    # Upset = 1 if underdog won
    df["upset"] = (df["underdog"] == df["winner"]).astype(int)

    # Set NaN for excluded games (small spreads)
    df.loc[df["underdog"].isna(), "upset"] = None

    # Set NaN for tie games (winner is None but underdog exists)
    df.loc[df["winner"].isna() & df["underdog"].notna(), "upset"] = None

    return df
