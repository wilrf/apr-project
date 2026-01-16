"""Matchup differential feature calculations."""

from __future__ import annotations

import pandas as pd
from typing import Optional, Dict, Tuple


def calculate_matchup_differentials(
    df: pd.DataFrame,
    underdog_prefix: Optional[str] = None,
    favorite_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate matchup differential features (underdog - favorite).

    These features capture the relative strengths between the underdog's
    offense and the favorite's defense, helping identify potential upset
    opportunities where the underdog has a favorable matchup.

    Args:
        df: DataFrame with team rolling stats for both teams
        underdog_prefix: Column prefix for underdog stats (e.g., "det_")
        favorite_prefix: Column prefix for favorite stats (e.g., "kc_")

    Returns:
        DataFrame with added matchup differential columns
    """
    result = df.copy()

    # If prefixes not provided, try to infer from underdog/favorite columns
    if underdog_prefix is None:
        underdog_prefix = result["underdog"].iloc[0].lower() + "_"
    if favorite_prefix is None:
        favorite_prefix = result["favorite"].iloc[0].lower() + "_"

    # Define differential calculations: (underdog_col, favorite_col)
    differentials: Dict[str, Tuple[str, str]] = {
        "offense_defense_mismatch": (
            f"{underdog_prefix}pass_epa_roll5",
            f"{favorite_prefix}pass_def_epa_roll5",
        ),
        "rush_attack_advantage": (
            f"{underdog_prefix}rush_yards_roll5",
            f"{favorite_prefix}rush_yards_allowed_roll5",
        ),
        "turnover_edge": (
            f"{underdog_prefix}turnover_margin_roll5",
            f"{favorite_prefix}turnover_margin_roll5",
        ),
    }

    for feature_name, (underdog_col, favorite_col) in differentials.items():
        if underdog_col in result.columns and favorite_col in result.columns:
            result[feature_name] = result[underdog_col] - result[favorite_col]

    return result
