"""Simple pre-game Elo computation for NFL schedule data."""

from __future__ import annotations

from typing import Dict

import pandas as pd


DEFAULT_ELO = 1500.0
DEFAULT_K_FACTOR = 20.0
DEFAULT_HOME_ADVANTAGE = 50.0


def _expected_home_win_probability(
    home_elo: float,
    away_elo: float,
    home_advantage: float,
) -> float:
    """Return the expected home win probability under Elo."""
    adjusted_home = home_elo + home_advantage
    return 1.0 / (1.0 + 10 ** ((away_elo - adjusted_home) / 400.0))


def compute_pre_game_elo(
    games: pd.DataFrame,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    base_elo: float = DEFAULT_ELO,
) -> pd.DataFrame:
    """
    Compute pre-game home/away Elo ratings for each game row.

    Ratings carry across seasons. Teams not seen before start at `base_elo`.
    """
    sort_cols = [
        col for col in ["season", "week", "gameday", "game_id"] if col in games
    ]
    ordered = games.sort_values(sort_cols).copy()
    ratings: Dict[str, float] = {}
    records = []

    for _, row in ordered.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_elo = ratings.get(home_team, base_elo)
        away_elo = ratings.get(away_team, base_elo)

        records.append(
            {
                "game_id": row["game_id"],
                "home_elo_pre": home_elo,
                "away_elo_pre": away_elo,
            }
        )

        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue

        if row["home_score"] > row["away_score"]:
            actual_home = 1.0
        elif row["home_score"] < row["away_score"]:
            actual_home = 0.0
        else:
            actual_home = 0.5

        expected_home = _expected_home_win_probability(
            home_elo,
            away_elo,
            home_advantage,
        )
        delta = k_factor * (actual_home - expected_home)
        ratings[home_team] = home_elo + delta
        ratings[away_team] = away_elo - delta

    return pd.DataFrame(records)
