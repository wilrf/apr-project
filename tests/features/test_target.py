"""Tests for the canonical upset target helpers."""

import numpy as np
import pandas as pd
import pytest

from src.features.target import calculate_upset_target, identify_underdog


class TestIdentifyUnderdog:
    def test_home_team_underdog_when_away_is_favorite(self):
        row = pd.Series(
            {
                "home_team": "DET",
                "away_team": "KC",
                "spread_favorite": -6.0,
                "team_favorite_id": "KC",
            }
        )
        assert identify_underdog(row) == "DET"

    def test_away_team_underdog_when_home_is_favorite(self):
        row = pd.Series(
            {
                "home_team": "KC",
                "away_team": "DET",
                "spread_favorite": -6.0,
                "team_favorite_id": "KC",
            }
        )
        assert identify_underdog(row) == "DET"

    def test_small_spreads_still_identify_underdog(self):
        row = pd.Series(
            {
                "home_team": "DET",
                "away_team": "KC",
                "spread_favorite": -2.5,
                "team_favorite_id": "KC",
            }
        )
        assert identify_underdog(row) == "DET"

    def test_missing_spread_returns_none(self):
        row = pd.Series(
            {
                "home_team": "DET",
                "away_team": "KC",
                "spread_favorite": np.nan,
                "team_favorite_id": "KC",
            }
        )
        assert identify_underdog(row) is None

    def test_missing_favorite_returns_none(self):
        row = pd.Series(
            {
                "home_team": "DET",
                "away_team": "KC",
                "spread_favorite": -6.0,
                "team_favorite_id": np.nan,
            }
        )
        assert identify_underdog(row) is None


class TestCalculateUpsetTarget:
    @pytest.fixture
    def sample_games(self):
        return pd.DataFrame(
            {
                "game_id": ["g1", "g2", "g3", "g4", "g5"],
                "home_team": ["DET", "KC", "BUF", "PHI", "DAL"],
                "away_team": ["KC", "DET", "MIA", "SF", "NYG"],
                "home_score": [24, 31, 21, 20, 17],
                "away_score": [21, 17, 28, 20, 14],
                "spread_favorite": [-6.0, -2.5, -7.0, -3.0, np.nan],
                "team_favorite_id": ["KC", "KC", "BUF", "PHI", "DAL"],
            }
        )

    def test_upset_when_underdog_wins(self, sample_games):
        result = calculate_upset_target(sample_games)
        assert result.loc[result["game_id"] == "g1", "upset"].item() == 1.0

    def test_small_spreads_excluded_from_target(self, sample_games):
        result = calculate_upset_target(sample_games)
        row = result.loc[result["game_id"] == "g2"].iloc[0]
        assert row["underdog"] == "DET"
        assert pd.isna(row["upset"])

    def test_away_underdog_upset(self, sample_games):
        result = calculate_upset_target(sample_games)
        assert result.loc[result["game_id"] == "g3", "upset"].item() == 1.0

    def test_favorite_win_has_zero_upset_target(self):
        games = pd.DataFrame(
            {
                "game_id": ["g6"],
                "home_team": ["KC"],
                "away_team": ["DET"],
                "home_score": [31],
                "away_score": [17],
                "spread_favorite": [-6.0],
                "team_favorite_id": ["KC"],
            }
        )

        result = calculate_upset_target(games)
        row = result.iloc[0]

        assert row["favorite"] == "KC"
        assert row["underdog"] == "DET"
        assert row["winner"] == "KC"
        assert row["upset"] == 0.0

    def test_ties_have_missing_target(self, sample_games):
        result = calculate_upset_target(sample_games)
        assert pd.isna(result.loc[result["game_id"] == "g4", "upset"].item())

    def test_missing_spread_keeps_target_missing(self, sample_games):
        result = calculate_upset_target(sample_games)
        row = result.loc[result["game_id"] == "g5"].iloc[0]
        assert pd.isna(row["underdog"])
        assert pd.isna(row["upset"])
