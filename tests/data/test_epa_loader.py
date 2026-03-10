"""Tests for EPA data loader."""

from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.epa_loader import (
    _PBP_COLUMNS,
    PBP_URL_TEMPLATE,
    load_game_advanced_stats,
    load_game_epa,
)


def _mock_pbp_data():
    """Create mock PBP data with multiple plays per game."""
    return pd.DataFrame(
        {
            "game_id": ["2023_01_KC_DET"] * 3 + ["2023_01_BAL_CLE"] * 2,
            "play_id": [1, 2, 3, 1, 2],
            "home_team": ["DET"] * 3 + ["CLE"] * 2,
            "away_team": ["KC"] * 3 + ["BAL"] * 2,
            "posteam": ["DET", "DET", "KC", "CLE", "BAL"],
            "play_type": ["pass", "run", "pass", "run", "pass"],
            "epa": [0.1, 0.2, -0.3, 0.4, 0.5],
            "success": [1.0, 0.0, 1.0, 1.0, 0.0],
            "cpoe": [3.0, np.nan, -2.0, np.nan, 1.5],
            "interception": [0.0, 0.0, 1.0, 0.0, 0.0],
            "fumble_lost": [0.0, 0.0, 0.0, 0.0, 1.0],
            # Cumulative totals — last row has the final values
            "total_home_pass_epa": [5.0, 10.0, 15.2, 3.0, 8.5],
            "total_away_pass_epa": [4.0, 8.0, 12.1, 2.0, 7.3],
            "total_home_rush_epa": [1.0, 2.0, 3.4, 0.5, 2.1],
            "total_away_rush_epa": [0.5, 1.5, 2.8, 1.0, 3.0],
        }
    )


class TestLoadGameEpa:
    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_returns_one_row_per_game(self, mock_read_parquet, _mock_dns):
        mock_read_parquet.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        assert len(result) == 2
        assert set(result["game_id"]) == {"2023_01_KC_DET", "2023_01_BAL_CLE"}

    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_correct_column_names(self, mock_read_parquet, _mock_dns):
        mock_read_parquet.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        expected_cols = [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
        assert list(result.columns) == expected_cols

    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_takes_last_row_cumulative_values(self, mock_read_parquet, _mock_dns):
        mock_read_parquet.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        kc_det = result[result["game_id"] == "2023_01_KC_DET"].iloc[0]
        assert kc_det["home_off_pass_epa"] == 15.2
        assert kc_det["home_off_rush_epa"] == 3.4
        assert kc_det["away_off_pass_epa"] == 12.1
        assert kc_det["away_off_rush_epa"] == 2.8

    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_handles_nan_values(self, mock_read_parquet, _mock_dns):
        pbp = _mock_pbp_data()
        pbp.loc[4, "total_home_pass_epa"] = np.nan
        mock_read_parquet.return_value = pbp

        result = load_game_epa([2023])

        bal_cle = result[result["game_id"] == "2023_01_BAL_CLE"].iloc[0]
        assert pd.isna(bal_cle["home_off_pass_epa"])

    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_reads_expected_nflverse_parquet(self, mock_read_parquet, _mock_dns):
        mock_read_parquet.return_value = _mock_pbp_data()

        load_game_epa([2023])

        mock_read_parquet.assert_called_once_with(
            PBP_URL_TEMPLATE.format(season=2023),
            columns=_PBP_COLUMNS,
        )


class TestLoadGameAdvancedStats:
    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_returns_expected_advanced_columns(self, mock_read_parquet, _mock_dns):
        mock_read_parquet.return_value = _mock_pbp_data()

        result = load_game_advanced_stats([2023])

        expected = {
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
            "home_success_rate",
            "away_success_rate",
            "home_cpoe",
            "away_cpoe",
            "home_turnover_margin",
            "away_turnover_margin",
        }
        assert expected.issubset(result.columns)

    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch("src.data.epa_loader.pd.read_parquet")
    def test_computes_turnover_margin_from_offensive_turnovers(
        self,
        mock_read_parquet,
        _mock_dns,
    ):
        mock_read_parquet.return_value = _mock_pbp_data()

        result = load_game_advanced_stats([2023])

        kc_det = result[result["game_id"] == "2023_01_KC_DET"].iloc[0]
        bal_cle = result[result["game_id"] == "2023_01_BAL_CLE"].iloc[0]

        assert kc_det["home_turnover_margin"] == 1.0
        assert kc_det["away_turnover_margin"] == -1.0
        assert bal_cle["home_turnover_margin"] == 1.0
        assert bal_cle["away_turnover_margin"] == -1.0


class TestEpaErrorHandling:
    """H13: EPA loader should produce clear errors on network failure."""

    @patch("src.data.epa_loader._github_dns_override", return_value=nullcontext())
    @patch(
        "src.data.epa_loader.pd.read_parquet",
        side_effect=ConnectionError("no network"),
    )
    def test_load_game_advanced_stats_wraps_network_error(
        self,
        _mock_read_parquet,
        _mock_dns,
    ):
        with pytest.raises(
            RuntimeError,
            match="Failed to load play-by-play data for season 2023",
        ):
            load_game_advanced_stats([2023])
