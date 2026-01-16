"""Rolling average calculations for team performance features."""

from __future__ import annotations

import pandas as pd
from typing import List

ROLLING_WINDOW = 5  # Use last 5 games


def get_team_game_sequence(
    games_df: pd.DataFrame,
    team: str,
    season: int,
) -> pd.DataFrame:
    """
    Get sequential games for a team in a season.

    Args:
        games_df: DataFrame with all games
        team: Team abbreviation
        season: Season year

    Returns:
        DataFrame of team's games ordered by week
    """
    season_games = games_df[games_df["season"] == season].copy()

    # Find games where team played (home or away)
    team_games = season_games[
        (season_games["home_team"] == team) |
        (season_games["away_team"] == team)
    ].copy()

    return team_games.sort_values("week").reset_index(drop=True)


def calculate_rolling_stats(
    team_games: pd.DataFrame,
    columns: List[str],
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Calculate rolling statistics for specified columns.

    Uses shift(1) to ensure we only use PRIOR games (no leakage).
    Early season games use all available prior games.

    Args:
        team_games: Team's games in order
        columns: Columns to calculate rolling stats for
        window: Rolling window size

    Returns:
        DataFrame with added rolling stat columns
    """
    result = team_games.copy()

    for col in columns:
        # Shift by 1 to only use prior games
        shifted = result[col].shift(1)

        # Calculate rolling mean with min_periods=1 for early season
        rolling = shifted.rolling(window=window, min_periods=1).mean()

        # Week 1 has no prior games, set to NaN
        rolling.iloc[0] = pd.NA

        result[f"{col}_roll{window}"] = rolling

    return result
