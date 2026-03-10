# tests/data/test_nfl_loader.py
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.nfl_loader import NFLVERSE_GAMES_URL, load_pbp_data, load_schedules


class TestLoadSchedules:
    def test_load_schedules_returns_dataframe(self):
        """Test that load_schedules returns a pandas DataFrame."""
        with patch(
            "src.data.nfl_loader.pd.read_csv",
            return_value=pd.DataFrame(
                {
                    "game_id": ["g1"],
                    "season": [2023],
                    "week": [1],
                    "game_type": ["REG"],
                    "home_team": ["KC"],
                    "away_team": ["BUF"],
                    "home_score": [24],
                    "away_score": [21],
                }
            ),
        ) as mock_read_csv:
            df = load_schedules(seasons=[2023])
        assert isinstance(df, pd.DataFrame)
        mock_read_csv.assert_called_once_with(NFLVERSE_GAMES_URL, low_memory=False)

    def test_load_schedules_has_required_columns(self):
        """Test that schedules have required columns for merging."""
        with patch(
            "src.data.nfl_loader.pd.read_csv",
            return_value=pd.DataFrame(
                {
                    "game_id": ["g1"],
                    "season": [2023],
                    "week": [1],
                    "game_type": ["REG"],
                    "home_team": ["KC"],
                    "away_team": ["BUF"],
                    "home_score": [24],
                    "away_score": [21],
                }
            ),
        ):
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
        with patch(
            "src.data.nfl_loader.pd.read_csv",
            return_value=pd.DataFrame(
                {
                    "game_id": ["g1", "g2"],
                    "season": [2023, 2023],
                    "week": [1, 19],
                    "game_type": ["REG", "POST"],
                    "home_team": ["KC", "KC"],
                    "away_team": ["BUF", "BUF"],
                    "home_score": [24, 24],
                    "away_score": [21, 21],
                }
            ),
        ):
            df = load_schedules(seasons=[2023], regular_season_only=True)
        assert all(df["game_type"] == "REG")

    def test_load_schedules_multiple_seasons(self):
        """Test loading multiple seasons."""
        with patch(
            "src.data.nfl_loader.pd.read_csv",
            return_value=pd.DataFrame(
                {
                    "game_id": ["g1", "g2", "g3"],
                    "season": [2022, 2023, 2024],
                    "week": [1, 1, 1],
                    "game_type": ["REG", "REG", "REG"],
                    "home_team": ["KC", "BUF", "DET"],
                    "away_team": ["BUF", "KC", "GB"],
                    "home_score": [24, 21, 17],
                    "away_score": [21, 24, 20],
                }
            ),
        ):
            df = load_schedules(seasons=[2022, 2023])
        assert set(df["season"].unique()) == {2022, 2023}
        assert 2024 not in set(df["season"])


class TestLoadPbpData:
    def test_load_pbp_returns_dataframe(self):
        """Test that load_pbp_data returns a pandas DataFrame."""
        with patch(
            "src.data.nfl_loader.nfl.import_pbp_data",
            return_value=pd.DataFrame({"epa": [0.1], "season": [2023]}),
        ):
            df = load_pbp_data(seasons=[2023])
        assert isinstance(df, pd.DataFrame)

    def test_load_pbp_has_epa_columns(self):
        """Test that pbp data has EPA columns."""
        with patch(
            "src.data.nfl_loader.nfl.import_pbp_data",
            return_value=pd.DataFrame({"epa": [0.1], "season": [2023]}),
        ):
            df = load_pbp_data(seasons=[2023])
        assert "epa" in df.columns


class TestErrorHandling:
    """H13: API calls should produce clear errors on failure."""

    def test_load_schedules_wraps_network_error(self):
        """Network failures should raise RuntimeError with guidance."""
        with patch(
            "src.data.nfl_loader.pd.read_csv",
            side_effect=ConnectionError("no network"),
        ):
            with pytest.raises(
                RuntimeError,
                match="Failed to download NFL schedules from nflverse",
            ):
                load_schedules(seasons=[2023])

    def test_load_schedules_rejects_schema_drift(self):
        """Missing required schedule columns should fail clearly."""
        with patch(
            "src.data.nfl_loader.pd.read_csv",
            return_value=pd.DataFrame(
                {
                    "game_id": ["g1"],
                    "season": [2023],
                    "week": [1],
                    "home_team": ["KC"],
                    "away_team": ["BUF"],
                    "home_score": [24],
                    "away_score": [21],
                }
            ),
        ):
            with pytest.raises(RuntimeError, match="Missing required columns"):
                load_schedules(seasons=[2023])

    def test_load_pbp_wraps_network_error(self):
        """Network failures should raise RuntimeError with guidance."""
        with patch(
            "src.data.nfl_loader.nfl.import_pbp_data",
            side_effect=ConnectionError("no network"),
        ):
            with pytest.raises(RuntimeError, match="Failed to load play-by-play"):
                load_pbp_data(seasons=[2023])
