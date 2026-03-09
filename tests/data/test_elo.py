"""Tests for Elo feature generation."""

import pandas as pd

from src.data.elo import compute_pre_game_elo


def test_compute_pre_game_elo_initializes_teams_at_base_rating():
    games = pd.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2023],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["DET"],
            "home_score": [20],
            "away_score": [21],
        }
    )

    result = compute_pre_game_elo(games)

    assert result.iloc[0]["home_elo_pre"] == 1500.0
    assert result.iloc[0]["away_elo_pre"] == 1500.0


def test_compute_pre_game_elo_carries_updates_forward():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2023, 2023],
            "week": [1, 2],
            "home_team": ["KC", "KC"],
            "away_team": ["DET", "BUF"],
            "home_score": [20, 24],
            "away_score": [21, 17],
        }
    )

    result = compute_pre_game_elo(games)

    assert result.iloc[1]["home_elo_pre"] != 1500.0
    assert result.iloc[1]["away_elo_pre"] == 1500.0
