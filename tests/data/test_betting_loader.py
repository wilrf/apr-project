# tests/data/test_betting_loader.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.betting_loader import load_betting_data, normalize_team_abbr


class TestNormalizeTeamAbbr:
    """Tests for team abbreviation normalization - no external data needed."""

    def test_normalize_known_relocations(self):
        """Test that relocated teams are normalized."""
        assert normalize_team_abbr("STL") == "LA"  # Rams
        assert normalize_team_abbr("SD") == "LAC"  # Chargers
        assert normalize_team_abbr("OAK") == "LV"  # Raiders

    def test_normalize_unchanged_teams(self):
        """Test that non-relocated teams are unchanged."""
        assert normalize_team_abbr("KC") == "KC"
        assert normalize_team_abbr("NE") == "NE"

    def test_normalize_naming_inconsistencies(self):
        """Test that naming inconsistencies are normalized."""
        assert normalize_team_abbr("JAC") == "JAX"


class TestLoadBettingData:
    """Tests for betting data loading."""

    @pytest.fixture
    def sample_betting_csv(self, tmp_path):
        """Create a sample betting CSV for testing."""
        csv_content = """schedule_date,schedule_season,schedule_week,team_home,team_away,team_favorite_id,spread_favorite,score_home,score_away
2023-09-07,2023,1,DET,KC,KC,-6.0,20,21
2023-09-10,2023,1,HOU,BAL,BAL,-9.5,9,25
2022-09-08,2022,1,LA,BUF,BUF,-2.5,10,31
"""
        csv_file = tmp_path / "test_betting.csv"
        csv_file.write_text(csv_content)
        return csv_file

    def test_load_betting_data_returns_dataframe(self, sample_betting_csv):
        """Test that load_betting_data returns a DataFrame."""
        df = load_betting_data(filepath=sample_betting_csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_betting_data_has_spread_columns(self, sample_betting_csv):
        """Test that betting data has spread columns."""
        df = load_betting_data(filepath=sample_betting_csv)
        assert "spread_favorite" in df.columns

    def test_load_betting_data_filters_by_season(self, sample_betting_csv):
        """Test filtering betting data by season range."""
        df = load_betting_data(filepath=sample_betting_csv, min_season=2023, max_season=2023)
        assert df["schedule_season"].min() >= 2023
        assert df["schedule_season"].max() <= 2023
        assert len(df) == 2  # Only 2023 games

    def test_load_betting_data_normalizes_teams(self, sample_betting_csv):
        """Test that team abbreviations are normalized."""
        # Create CSV with old team abbreviations
        csv_content = """schedule_date,schedule_season,schedule_week,team_home,team_away,team_favorite_id,spread_favorite,score_home,score_away
2023-09-07,2023,1,OAK,SD,-6.0,SD,20,21
"""
        csv_file = sample_betting_csv.parent / "test_betting2.csv"
        csv_file.write_text(csv_content)

        df = load_betting_data(filepath=csv_file)
        assert "LV" in df["team_home"].values  # OAK -> LV
        assert "LAC" in df["team_away"].values  # SD -> LAC
