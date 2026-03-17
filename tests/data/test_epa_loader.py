"""Tests for EPA data loader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.epa_loader import (
    _PBP_COLUMNS,
    _download_pbp_season,
    _load_pbp_season,
    _pbp_cache_path,
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
    @patch("src.data.epa_loader._load_pbp_data")
    def test_returns_one_row_per_game(self, mock_load_pbp):
        mock_load_pbp.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        assert len(result) == 2
        assert set(result["game_id"]) == {"2023_01_KC_DET", "2023_01_BAL_CLE"}

    @patch("src.data.epa_loader._load_pbp_data")
    def test_correct_column_names(self, mock_load_pbp):
        mock_load_pbp.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        expected_cols = [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
        assert list(result.columns) == expected_cols

    @patch("src.data.epa_loader._load_pbp_data")
    def test_takes_last_row_cumulative_values(self, mock_load_pbp):
        mock_load_pbp.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        kc_det = result[result["game_id"] == "2023_01_KC_DET"].iloc[0]
        assert kc_det["home_off_pass_epa"] == 15.2
        assert kc_det["home_off_rush_epa"] == 3.4
        assert kc_det["away_off_pass_epa"] == 12.1
        assert kc_det["away_off_rush_epa"] == 2.8

    @patch("src.data.epa_loader._load_pbp_data")
    def test_handles_nan_values(self, mock_load_pbp):
        pbp = _mock_pbp_data()
        pbp.loc[4, "total_home_pass_epa"] = np.nan
        mock_load_pbp.return_value = pbp

        result = load_game_epa([2023])

        bal_cle = result[result["game_id"] == "2023_01_BAL_CLE"].iloc[0]
        assert pd.isna(bal_cle["home_off_pass_epa"])

    @patch("src.data.epa_loader._load_pbp_data")
    def test_requests_expected_seasons(self, mock_load_pbp):
        mock_load_pbp.return_value = _mock_pbp_data()

        load_game_epa([2023])

        mock_load_pbp.assert_called_once_with([2023])


class TestLoadGameAdvancedStats:
    @patch("src.data.epa_loader._load_pbp_data")
    def test_returns_expected_advanced_columns(self, mock_load_pbp):
        mock_load_pbp.return_value = _mock_pbp_data()

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

    @patch("src.data.epa_loader._load_pbp_data")
    def test_computes_turnover_margin_from_offensive_turnovers(
        self,
        mock_load_pbp,
    ):
        mock_load_pbp.return_value = _mock_pbp_data()

        result = load_game_advanced_stats([2023])

        kc_det = result[result["game_id"] == "2023_01_KC_DET"].iloc[0]
        bal_cle = result[result["game_id"] == "2023_01_BAL_CLE"].iloc[0]

        assert kc_det["home_turnover_margin"] == 1.0
        assert kc_det["away_turnover_margin"] == -1.0
        assert bal_cle["home_turnover_margin"] == 1.0
        assert bal_cle["away_turnover_margin"] == -1.0


class TestLoadPbpSeason:
    @patch("src.data.epa_loader._download_pbp_season")
    @patch("src.data.epa_loader.pd.read_parquet")
    @patch("src.data.epa_loader.PBP_CACHE_DIR")
    def test_uses_existing_cached_parquet(
        self,
        mock_cache_dir,
        mock_read_parquet,
        mock_download,
        tmp_path,
    ):
        mock_cache_dir.__truediv__.side_effect = (
            lambda season_file: tmp_path / season_file
        )
        cached = _pbp_cache_path(2023)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"cached")
        mock_read_parquet.return_value = _mock_pbp_data()

        result = _load_pbp_season(2023)

        assert len(result) == 5
        mock_download.assert_not_called()
        mock_read_parquet.assert_called_once_with(cached, columns=_PBP_COLUMNS)

    @patch("src.data.epa_loader._download_pbp_season")
    @patch("src.data.epa_loader.pd.read_parquet")
    @patch("src.data.epa_loader.PBP_CACHE_DIR")
    def test_downloads_missing_season_before_reading(
        self,
        mock_cache_dir,
        mock_read_parquet,
        mock_download,
        tmp_path,
    ):
        mock_cache_dir.__truediv__.side_effect = (
            lambda season_file: tmp_path / season_file
        )
        cached = _pbp_cache_path(2024)
        mock_download.side_effect = lambda season, destination: destination.write_bytes(
            b"downloaded"
        )
        mock_read_parquet.return_value = _mock_pbp_data()

        _load_pbp_season(2024)

        mock_download.assert_called_once_with(2024, cached)
        mock_read_parquet.assert_called_once_with(cached, columns=_PBP_COLUMNS)

    @patch("src.data.epa_loader._download_pbp_season")
    @patch("src.data.epa_loader.pd.read_parquet")
    @patch("src.data.epa_loader.PBP_CACHE_DIR")
    def test_bad_cached_file_is_deleted_and_re_downloaded(
        self,
        mock_cache_dir,
        mock_read_parquet,
        mock_download,
        tmp_path,
    ):
        mock_cache_dir.__truediv__.side_effect = (
            lambda season_file: tmp_path / season_file
        )
        cached = _pbp_cache_path(2025)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"bad-cache")

        def _download(season: int, destination: Path) -> None:
            destination.write_bytes(b"fresh-cache")

        mock_download.side_effect = _download
        mock_read_parquet.side_effect = [
            ValueError("corrupt parquet"),
            _mock_pbp_data(),
        ]

        result = _load_pbp_season(2025)

        assert len(result) == 5
        assert cached.exists()
        mock_download.assert_called_once_with(2025, cached)
        assert mock_read_parquet.call_count == 2


class TestDownloadPbpSeason:
    @patch("src.data.epa_loader._github_dns_override")
    @patch("src.data.epa_loader.urlopen")
    def test_download_writes_parquet_to_destination(
        self,
        mock_urlopen,
        mock_dns,
        tmp_path,
    ):
        mock_dns.return_value.__enter__.return_value = None
        mock_dns.return_value.__exit__.return_value = False

        response = MagicMock()
        response.read.side_effect = [b"abc", b"123", b""]
        mock_urlopen.return_value.__enter__.return_value = response

        destination = tmp_path / "play_by_play_2023.parquet"
        result = _download_pbp_season(2023, destination)

        assert result == destination
        assert destination.read_bytes() == b"abc123"

    @patch("src.data.epa_loader.time.sleep")
    @patch("src.data.epa_loader._github_dns_override")
    @patch("src.data.epa_loader.urlopen")
    def test_download_retries_transient_timeouts(
        self,
        mock_urlopen,
        mock_dns,
        mock_sleep,
        tmp_path,
    ):
        mock_dns.return_value.__enter__.return_value = None
        mock_dns.return_value.__exit__.return_value = False

        response = MagicMock()
        response.read.side_effect = [b"ok", b""]
        mock_urlopen.side_effect = [
            TimeoutError("timed out"),
            MagicMock(
                __enter__=MagicMock(return_value=response),
                __exit__=MagicMock(return_value=False),
            ),
        ]

        destination = tmp_path / "play_by_play_2011.parquet"
        _download_pbp_season(2011, destination)

        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once()


class TestEpaErrorHandling:
    """H13: EPA loader should produce clear errors on network failure."""

    @patch("src.data.epa_loader._load_pbp_data", side_effect=RuntimeError("no network"))
    def test_load_game_advanced_stats_wraps_network_error(
        self,
        _mock_load_pbp,
    ):
        with pytest.raises(RuntimeError, match="no network"):
            load_game_advanced_stats([2023])
