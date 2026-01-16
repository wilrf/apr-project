# tests/features/test_rolling.py
import pytest
import pandas as pd
import numpy as np
from src.features.rolling import (
    calculate_rolling_stats,
    get_team_game_sequence,
    ROLLING_WINDOW,
)


class TestGetTeamGameSequence:
    @pytest.fixture
    def sample_games(self):
        """Sample games for a team across multiple weeks."""
        return pd.DataFrame({
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            "season": [2023, 2023, 2023, 2023, 2023],
            "week": [1, 2, 3, 5, 6],  # Note: week 4 is bye
            "home_team": ["KC", "DET", "KC", "KC", "BUF"],
            "away_team": ["DET", "KC", "BAL", "NE", "KC"],
            "home_score": [21, 17, 28, 31, 24],
            "away_score": [20, 24, 14, 10, 17],
        })

    def test_sequence_ordered_by_week(self, sample_games):
        """Team game sequence should be ordered by week."""
        seq = get_team_game_sequence(sample_games, "KC", 2023)
        assert list(seq["week"]) == [1, 2, 3, 5, 6]

    def test_sequence_contains_all_team_games(self, sample_games):
        """Sequence should contain all games where team played."""
        seq = get_team_game_sequence(sample_games, "KC", 2023)
        assert len(seq) == 5


class TestCalculateRollingStats:
    @pytest.fixture
    def sample_team_games(self):
        """Sample sequential games for rolling calculation."""
        return pd.DataFrame({
            "week": [1, 2, 3, 4, 5, 6, 7],
            "points_scored": [21, 28, 14, 35, 17, 24, 31],
            "points_allowed": [17, 24, 21, 14, 28, 10, 14],
        })

    def test_rolling_mean_calculation(self, sample_team_games):
        """Test rolling mean is calculated correctly."""
        result = calculate_rolling_stats(
            sample_team_games,
            columns=["points_scored"],
            window=ROLLING_WINDOW,
        )

        # Week 6 rolling mean (weeks 2-6 shifted, so prior 5 games: weeks 1-5): (21+28+14+35+17)/5 = 23.0
        assert abs(result.loc[5, "points_scored_roll5"] - 23.0) < 0.1

    def test_early_season_uses_available_games(self, sample_team_games):
        """Early season should use all available prior games (< window)."""
        result = calculate_rolling_stats(
            sample_team_games,
            columns=["points_scored"],
            window=ROLLING_WINDOW,
        )

        # Week 3 should use weeks 1-2 only: (21+28)/2 = 24.5
        assert abs(result.loc[2, "points_scored_roll5"] - 24.5) < 0.1

    def test_week_1_is_nan(self, sample_team_games):
        """Week 1 has no prior games, should be NaN."""
        result = calculate_rolling_stats(
            sample_team_games,
            columns=["points_scored"],
            window=ROLLING_WINDOW,
        )
        assert pd.isna(result.loc[0, "points_scored_roll5"])
