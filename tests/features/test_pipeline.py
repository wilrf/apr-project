# tests/features/test_pipeline.py
import pytest
import pandas as pd
from src.features.pipeline import FeatureEngineeringPipeline


class TestFeatureEngineeringPipeline:
    def test_pipeline_produces_expected_columns(self):
        """Test that pipeline produces all expected feature columns."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data())

        expected_cols = [
            "game_id",
            "upset",  # Target
            "spread_magnitude",  # Spread feature
            "offense_defense_mismatch",  # Matchup differential
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_pipeline_excludes_week_1(self):
        """Test that Week 1 games are excluded (no prior data)."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_week_1())

        # Week 1 games should be filtered out
        assert not any(result["week"] == 1)

    def test_pipeline_preserves_week_1_when_disabled(self):
        """Test that Week 1 games kept when exclude_week_1=False."""
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(mock_data_with_week_1())

        # Week 1 games should be kept
        assert any(result["week"] == 1)

    def test_pipeline_calculates_upset_target(self):
        """Test that upset target is calculated correctly."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data())

        # DET (underdog at +6) beat KC -> upset = 1
        assert result.loc[0, "upset"] == 1

    def test_pipeline_calculates_spread_magnitude(self):
        """Test that spread magnitude is absolute value."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data())

        # spread_favorite is -6.0, magnitude should be 6.0
        assert result.loc[0, "spread_magnitude"] == 6.0


def mock_data():
    """Create minimal mock data for testing."""
    return pd.DataFrame({
        "game_id": ["2023_02_KC_DET"],
        "season": [2023],
        "week": [2],
        "home_team": ["DET"],
        "away_team": ["KC"],
        "home_score": [24],
        "away_score": [21],
        "spread_favorite": [-6.0],
        "team_favorite_id": ["KC"],
    })


def mock_data_with_week_1():
    """Mock data including Week 1."""
    return pd.DataFrame({
        "game_id": ["2023_01_KC_DET", "2023_02_KC_BAL"],
        "season": [2023, 2023],
        "week": [1, 2],
        "home_team": ["DET", "BAL"],
        "away_team": ["KC", "KC"],
        "home_score": [20, 21],
        "away_score": [21, 17],
        "spread_favorite": [-6.0, -3.0],
        "team_favorite_id": ["KC", "KC"],
    })
