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
        Underdog team abbreviation, or None if spread < 3
    """
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

    # Calculate winner
    df["winner"] = df.apply(
        lambda r: r["home_team"] if r["home_score"] > r["away_score"]
        else (r["away_team"] if r["away_score"] > r["home_score"] else None),
        axis=1
    )

    # Upset = 1 if underdog won
    df["upset"] = (df["underdog"] == df["winner"]).astype(int)

    # Set NaN for excluded games (small spreads or ties)
    df.loc[df["underdog"].isna(), "upset"] = None

    return df
