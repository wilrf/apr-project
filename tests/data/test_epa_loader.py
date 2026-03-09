"""Tests for EPA data loader."""

import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data.epa_loader import load_game_advanced_stats, load_game_epa


def _mock_pbp_data():
    """Create mock PBP data with multiple plays per game."""
    return pd.DataFrame(
        {
            "game_id": ["2023_01_KC_DET"] * 3 + ["2023_01_BAL_CLE"] * 2,
            "home_team": ["DET"] * 3 + ["CLE"] * 2,
            "away_team": ["KC"] * 3 + ["BAL"] * 2,
            # Cumulative totals — last row has the final values
            "total_home_pass_epa": [5.0, 10.0, 15.2, 3.0, 8.5],
            "total_away_pass_epa": [4.0, 8.0, 12.1, 2.0, 7.3],
            "total_home_rush_epa": [1.0, 2.0, 3.4, 0.5, 2.1],
            "total_away_rush_epa": [0.5, 1.5, 2.8, 1.0, 3.0],
        }
    )


class TestLoadGameEpa:
    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_returns_one_row_per_game(self, mock_import):
        mock_import.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        assert len(result) == 2
        assert set(result["game_id"]) == {"2023_01_KC_DET", "2023_01_BAL_CLE"}

    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_correct_column_names(self, mock_import):
        mock_import.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        expected_cols = [
            "game_id",
            "home_off_pass_epa",
            "home_off_rush_epa",
            "away_off_pass_epa",
            "away_off_rush_epa",
        ]
        assert list(result.columns) == expected_cols

    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_takes_last_row_cumulative_values(self, mock_import):
        mock_import.return_value = _mock_pbp_data()

        result = load_game_epa([2023])

        kc_det = result[result["game_id"] == "2023_01_KC_DET"].iloc[0]
        assert kc_det["home_off_pass_epa"] == 15.2
        assert kc_det["home_off_rush_epa"] == 3.4
        assert kc_det["away_off_pass_epa"] == 12.1
        assert kc_det["away_off_rush_epa"] == 2.8

    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_handles_nan_values(self, mock_import):
        pbp = _mock_pbp_data()
        pbp.loc[4, "total_home_pass_epa"] = np.nan
        mock_import.return_value = pbp

        result = load_game_epa([2023])

        bal_cle = result[result["game_id"] == "2023_01_BAL_CLE"].iloc[0]
        assert pd.isna(bal_cle["home_off_pass_epa"])

    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_passes_columns_to_import(self, mock_import):
        mock_import.return_value = _mock_pbp_data()

        load_game_epa([2023])

        call_kwargs = mock_import.call_args
        assert call_kwargs[0][0] == [2023]
        assert "columns" in call_kwargs[1]


class TestLoadGameAdvancedStats:
    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_returns_expected_advanced_columns(self, mock_import):
        mock_import.return_value = _mock_pbp_data().assign(
            posteam=["DET", "DET", "KC", "CLE", "BAL"],
            play_type=["pass", "run", "pass", "run", "pass"],
            epa=[0.1, 0.2, -0.3, 0.4, 0.5],
            success=[1.0, 0.0, 1.0, 1.0, 0.0],
            cpoe=[3.0, np.nan, -2.0, np.nan, 1.5],
            interception=[0.0, 0.0, 1.0, 0.0, 0.0],
            fumble_lost=[0.0, 0.0, 0.0, 0.0, 1.0],
            play_id=[1, 2, 3, 1, 2],
        )

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

    @patch("src.data.epa_loader.nfl.import_pbp_data")
    def test_computes_turnover_margin_from_offensive_turnovers(self, mock_import):
        mock_import.return_value = _mock_pbp_data().assign(
            posteam=["DET", "DET", "KC", "CLE", "BAL"],
            play_type=["pass", "run", "pass", "run", "pass"],
            epa=[0.1, 0.2, -0.3, 0.4, 0.5],
            success=[1.0, 0.0, 1.0, 1.0, 0.0],
            cpoe=[3.0, np.nan, -2.0, np.nan, 1.5],
            interception=[0.0, 0.0, 1.0, 0.0, 0.0],
            fumble_lost=[0.0, 0.0, 0.0, 0.0, 1.0],
            play_id=[1, 2, 3, 1, 2],
        )

        result = load_game_advanced_stats([2023])

        kc_det = result[result["game_id"] == "2023_01_KC_DET"].iloc[0]
        bal_cle = result[result["game_id"] == "2023_01_BAL_CLE"].iloc[0]

        assert kc_det["home_turnover_margin"] == 1.0
        assert kc_det["away_turnover_margin"] == -1.0
        assert bal_cle["home_turnover_margin"] == 1.0
        assert bal_cle["away_turnover_margin"] == -1.0
