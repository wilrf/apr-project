# tests/data/test_nfl_loader.py
import pytest
import pandas as pd
from src.data.nfl_loader import load_schedules, load_pbp_data


class TestLoadSchedules:
    def test_load_schedules_returns_dataframe(self):
        """Test that load_schedules returns a pandas DataFrame."""
        df = load_schedules(seasons=[2023])
        assert isinstance(df, pd.DataFrame)

    def test_load_schedules_has_required_columns(self):
        """Test that schedules have required columns for merging."""
        df = load_schedules(seasons=[2023])
        required_cols = [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_schedules_filters_regular_season(self):
        """Test that only regular season games are returned."""
        df = load_schedules(seasons=[2023], regular_season_only=True)
        assert all(df["game_type"] == "REG")

    def test_load_schedules_multiple_seasons(self):
        """Test loading multiple seasons."""
        df = load_schedules(seasons=[2022, 2023])
        assert set(df["season"].unique()) == {2022, 2023}


class TestLoadPbpData:
    def test_load_pbp_returns_dataframe(self):
        """Test that load_pbp_data returns a pandas DataFrame."""
        df = load_pbp_data(seasons=[2023])
        assert isinstance(df, pd.DataFrame)

    def test_load_pbp_has_epa_columns(self):
        """Test that pbp data has EPA columns."""
        df = load_pbp_data(seasons=[2023])
        assert "epa" in df.columns
