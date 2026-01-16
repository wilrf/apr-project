# tests/features/test_matchup.py
import pytest
import pandas as pd
from src.features.matchup import calculate_matchup_differentials


class TestCalculateMatchupDifferentials:
    @pytest.fixture
    def sample_game_features(self):
        """Sample game with team rolling stats."""
        return pd.DataFrame({
            "game_id": ["g1"],
            "underdog": ["DET"],
            "favorite": ["KC"],
            # Underdog (DET) stats
            "det_pass_epa_roll5": [0.15],
            "det_rush_yards_roll5": [120.0],
            "det_turnover_margin_roll5": [0.5],
            # Favorite (KC) stats
            "kc_pass_def_epa_roll5": [0.05],
            "kc_rush_yards_allowed_roll5": [100.0],
            "kc_turnover_margin_roll5": [1.0],
        })

    def test_offense_defense_mismatch(self, sample_game_features):
        """Test offense/defense mismatch calculation."""
        result = calculate_matchup_differentials(sample_game_features)
        # DET pass EPA (0.15) - KC pass def EPA (0.05) = 0.10
        assert abs(result["offense_defense_mismatch"].values[0] - 0.10) < 0.01

    def test_rush_attack_advantage(self, sample_game_features):
        """Test rush attack advantage calculation."""
        result = calculate_matchup_differentials(sample_game_features)
        # DET rush (120) - KC rush allowed (100) = 20
        assert result["rush_attack_advantage"].values[0] == 20.0

    def test_turnover_edge(self, sample_game_features):
        """Test turnover edge calculation."""
        result = calculate_matchup_differentials(sample_game_features)
        # DET margin (0.5) - KC margin (1.0) = -0.5
        assert result["turnover_edge"].values[0] == -0.5

    def test_handles_different_teams(self):
        """Test that differential works with different team prefixes."""
        df = pd.DataFrame({
            "game_id": ["g2"],
            "underdog": ["MIA"],
            "favorite": ["BUF"],
            "mia_pass_epa_roll5": [0.20],
            "mia_rush_yards_roll5": [100.0],
            "mia_turnover_margin_roll5": [0.0],
            "buf_pass_def_epa_roll5": [-0.10],
            "buf_rush_yards_allowed_roll5": [90.0],
            "buf_turnover_margin_roll5": [0.5],
        })
        result = calculate_matchup_differentials(df)
        # MIA pass EPA (0.20) - BUF pass def EPA (-0.10) = 0.30
        assert abs(result["offense_defense_mismatch"].values[0] - 0.30) < 0.01
