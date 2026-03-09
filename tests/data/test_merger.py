# tests/data/test_merger.py
import pytest
import pandas as pd
from src.data.merger import merge_nfl_betting_data


class TestMergeNflBettingData:
    @pytest.fixture
    def sample_nfl_data(self):
        """Create sample NFL schedule data."""
        return pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BAL_HOU"],
                "season": [2023, 2023],
                "week": [1, 1],
                "home_team": ["DET", "HOU"],
                "away_team": ["KC", "BAL"],
                "home_score": [20, 25],
                "away_score": [21, 9],
            }
        )

    @pytest.fixture
    def sample_betting_data(self):
        """Create sample betting data."""
        return pd.DataFrame(
            {
                "schedule_season": [2023, 2023],
                "schedule_week": [1, 1],
                "team_home": ["DET", "HOU"],
                "team_away": ["KC", "BAL"],
                "spread_favorite": [-6.0, -9.5],
                "team_favorite_id": ["KC", "BAL"],
            }
        )

    def test_merge_returns_dataframe(self, sample_nfl_data, sample_betting_data):
        """Test that merge returns a DataFrame."""
        merged, _ = merge_nfl_betting_data(sample_nfl_data, sample_betting_data)
        assert isinstance(merged, pd.DataFrame)

    def test_merge_preserves_game_ids(self, sample_nfl_data, sample_betting_data):
        """Test that all game_ids are preserved after merge."""
        merged, _ = merge_nfl_betting_data(sample_nfl_data, sample_betting_data)
        assert set(merged["game_id"]) == set(sample_nfl_data["game_id"])

    def test_merge_adds_spread_column(self, sample_nfl_data, sample_betting_data):
        """Test that spread column is added after merge."""
        merged, _ = merge_nfl_betting_data(sample_nfl_data, sample_betting_data)
        assert "spread_favorite" in merged.columns


class TestMergeAudit:
    def test_audit_tracks_unmatched_rows(self):
        """Test that audit captures unmatched rows."""
        nfl_df = pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET"],
                "season": [2023],
                "week": [1],
                "home_team": ["DET"],
                "away_team": ["KC"],
            }
        )
        betting_df = pd.DataFrame(
            {
                "schedule_season": [2023],
                "schedule_week": [1],
                "team_home": ["XXX"],  # Won't match
                "team_away": ["YYY"],
            }
        )
        _, audit = merge_nfl_betting_data(nfl_df, betting_df)
        assert len(audit["unmatched_nfl"]) > 0

    def test_audit_reports_merge_rate(self):
        """Test that audit includes merge rate."""
        nfl_df = pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BAL_HOU"],
                "season": [2023, 2023],
                "week": [1, 1],
                "home_team": ["DET", "HOU"],
                "away_team": ["KC", "BAL"],
            }
        )
        betting_df = pd.DataFrame(
            {
                "schedule_season": [2023],
                "schedule_week": [1],
                "team_home": ["DET"],  # Only matches one game
                "team_away": ["KC"],
                "spread_favorite": [-6.0],
            }
        )
        _, audit = merge_nfl_betting_data(nfl_df, betting_df)
        assert audit["merge_rate"] == 0.5  # 1 of 2 matched
