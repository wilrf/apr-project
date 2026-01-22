# tests/data/test_verify_data.py
"""Tests for data verification module."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from tempfile import TemporaryDirectory

from src.data.verify_data import verify_data_coverage, _write_data_readme


class TestVerifyDataCoverage:
    """Tests for verify_data_coverage function."""

    @patch("src.data.verify_data.load_schedules")
    @patch("src.data.verify_data.load_betting_data")
    @patch("src.data.verify_data.merge_nfl_betting_data")
    def test_verify_coverage_returns_dict(
        self,
        mock_merge,
        mock_betting,
        mock_schedules,
    ):
        """Test that verify_data_coverage returns a dictionary."""
        # Setup mocks
        mock_schedules.return_value = pd.DataFrame({
            "game_id": ["2023_01_KC_DET", "2023_01_BAL_CLE"],
            "season": [2023, 2023],
            "week": [1, 1],
            "home_team": ["DET", "CLE"],
            "away_team": ["KC", "BAL"],
        })
        mock_betting.return_value = pd.DataFrame()
        mock_merge.return_value = (
            pd.DataFrame({
                "season": [2023, 2023],
                "spread_favorite": [-3.0, None],
            }),
            {"merge_rate": 1.0, "unmatched_nfl": pd.DataFrame()},
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "README.md"
            result = verify_data_coverage(
                seasons=[2023],
                output_path=output_path,
            )

        assert isinstance(result, dict)
        assert "coverage_by_season" in result
        assert "overall_merge_rate" in result
        assert "unmatched_count" in result

    @patch("src.data.verify_data.load_schedules")
    @patch("src.data.verify_data.load_betting_data")
    @patch("src.data.verify_data.merge_nfl_betting_data")
    def test_verify_coverage_calculates_percentages(
        self,
        mock_merge,
        mock_betting,
        mock_schedules,
    ):
        """Test that coverage percentages are calculated correctly."""
        # 4 games total, 3 with spread data
        mock_schedules.return_value = pd.DataFrame({
            "game_id": [f"2023_01_{i}" for i in range(4)],
            "season": [2023] * 4,
            "week": [1] * 4,
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H"],
        })
        mock_betting.return_value = pd.DataFrame()
        mock_merge.return_value = (
            pd.DataFrame({
                "season": [2023] * 4,
                "spread_favorite": [-3.0, -4.0, -5.0, None],
            }),
            {"merge_rate": 1.0, "unmatched_nfl": pd.DataFrame()},
        )

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "README.md"
            result = verify_data_coverage(
                seasons=[2023],
                output_path=output_path,
            )

        # 3 out of 4 games have spread = 75%
        assert result["coverage_by_season"][2023]["coverage_pct"] == 75.0


class TestWriteDataReadme:
    """Tests for _write_data_readme function."""

    def test_write_readme_creates_file(self):
        """Test that README file is created."""
        coverage = {
            2023: {
                "total_games": 272,
                "games_with_spread": 270,
                "coverage_pct": 99.3,
            }
        }
        audit = {
            "merge_rate": 0.993,
            "unmatched_nfl": pd.DataFrame(),
        }

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data" / "README.md"
            _write_data_readme(coverage, audit, output_path)

            assert output_path.exists()

    def test_write_readme_contains_coverage_table(self):
        """Test that README contains coverage information."""
        coverage = {
            2022: {
                "total_games": 272,
                "games_with_spread": 268,
                "coverage_pct": 98.5,
            },
            2023: {
                "total_games": 272,
                "games_with_spread": 270,
                "coverage_pct": 99.3,
            },
        }
        audit = {
            "merge_rate": 0.99,
            "unmatched_nfl": pd.DataFrame({"game_id": ["a", "b"]}),
        }

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "README.md"
            _write_data_readme(coverage, audit, output_path)

            content = output_path.read_text()

            # Check for table headers
            assert "Season" in content
            assert "Total Games" in content
            assert "Coverage %" in content

            # Check for data
            assert "2022" in content
            assert "2023" in content
            assert "98.5%" in content
            assert "99.3%" in content

    def test_write_readme_contains_merge_stats(self):
        """Test that README contains merge statistics."""
        coverage = {2023: {"total_games": 100, "games_with_spread": 95, "coverage_pct": 95.0}}
        audit = {
            "merge_rate": 0.95,
            "unmatched_nfl": pd.DataFrame({"game_id": ["a", "b", "c", "d", "e"]}),
        }

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "README.md"
            _write_data_readme(coverage, audit, output_path)

            content = output_path.read_text()

            assert "95.0%" in content
            assert "Unmatched NFL games: 5" in content
