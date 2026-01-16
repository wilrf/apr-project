# tests/features/test_target.py
import pytest
import pandas as pd
from src.features.target import calculate_upset_target, identify_underdog


class TestIdentifyUnderdog:
    def test_home_team_underdog_positive_spread(self):
        """Home team is underdog when spread is positive (favorite is away)."""
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": -6.0,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) == "DET"

    def test_away_team_underdog_negative_spread(self):
        """Away team is underdog when they're not the favorite."""
        row = pd.Series({
            "home_team": "KC",
            "away_team": "DET",
            "spread_favorite": -6.0,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) == "DET"

    def test_excludes_small_spreads(self):
        """Returns None for spreads < 3 (pick'em or close games)."""
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": -2.5,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) is None

    def test_handles_nan_spread(self):
        """Returns None when spread_favorite is NaN."""
        import numpy as np
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": np.nan,
            "team_favorite_id": "KC",
        })
        assert identify_underdog(row) is None

    def test_handles_nan_favorite(self):
        """Returns None when team_favorite_id is NaN."""
        import numpy as np
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": -6.0,
            "team_favorite_id": np.nan,
        })
        assert identify_underdog(row) is None

    def test_handles_both_nan(self):
        """Returns None when both spread and favorite are NaN."""
        import numpy as np
        row = pd.Series({
            "home_team": "DET",
            "away_team": "KC",
            "spread_favorite": np.nan,
            "team_favorite_id": np.nan,
        })
        assert identify_underdog(row) is None


class TestCalculateUpsetTarget:
    @pytest.fixture
    def sample_games(self):
        return pd.DataFrame({
            "game_id": ["g1", "g2", "g3"],
            "home_team": ["DET", "KC", "BUF"],
            "away_team": ["KC", "DET", "MIA"],
            "home_score": [24, 31, 21],
            "away_score": [21, 17, 28],
            "spread_favorite": [-6.0, -3.0, -7.0],
            "team_favorite_id": ["KC", "KC", "BUF"],
        })

    def test_upset_when_underdog_wins(self, sample_games):
        """Target is 1 when underdog wins outright."""
        result = calculate_upset_target(sample_games)
        # g1: DET (underdog) beat KC -> upset = 1
        assert result.loc[result["game_id"] == "g1", "upset"].values[0] == 1

    def test_no_upset_when_favorite_wins(self, sample_games):
        """Target is 0 when favorite wins."""
        result = calculate_upset_target(sample_games)
        # g2: KC (favorite) beat DET -> upset = 0
        assert result.loc[result["game_id"] == "g2", "upset"].values[0] == 0

    def test_away_underdog_upset(self, sample_games):
        """Away underdog winning is still an upset."""
        result = calculate_upset_target(sample_games)
        # g3: MIA (away underdog) beat BUF -> upset = 1
        assert result.loc[result["game_id"] == "g3", "upset"].values[0] == 1
