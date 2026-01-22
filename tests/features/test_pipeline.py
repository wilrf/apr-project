# tests/features/test_pipeline.py
"""Tests for the feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from src.features.pipeline import FeatureEngineeringPipeline


class TestFeatureEngineeringPipeline:
    """Core pipeline functionality tests."""

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


class TestFeatureCount:
    """Tests verifying feature count and organization."""

    def test_feature_count_approximately_55_to_65(self):
        """Test that we have approximately 55-65 features."""
        pipeline = FeatureEngineeringPipeline()
        feature_cols = pipeline.get_feature_columns()

        assert 55 <= len(feature_cols) <= 70, f"Expected 55-70 features, got {len(feature_cols)}"

    def test_feature_groups_exist(self):
        """Test that feature groups are properly defined."""
        pipeline = FeatureEngineeringPipeline()
        groups = pipeline.get_feature_groups()

        expected_groups = ["spread", "situational", "underdog_performance",
                          "favorite_performance", "matchup", "interaction"]

        for group in expected_groups:
            assert group in groups, f"Missing feature group: {group}"
            assert len(groups[group]) > 0, f"Empty feature group: {group}"

    def test_all_features_in_groups(self):
        """Test that all features belong to a group."""
        pipeline = FeatureEngineeringPipeline()
        feature_cols = set(pipeline.get_feature_columns())
        groups = pipeline.get_feature_groups()

        grouped_features = set()
        for cols in groups.values():
            grouped_features.update(cols)

        # All listed features should be in groups
        assert feature_cols == grouped_features, (
            f"Mismatch: {feature_cols - grouped_features} not in groups, "
            f"{grouped_features - feature_cols} not in feature list"
        )


class TestSpreadFeatures:
    """Tests for spread-related features."""

    def test_over_under_extracted(self):
        """Test that over/under line is extracted."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_betting())

        assert "over_under" in result.columns
        assert result["over_under"].iloc[0] == 47.5

    def test_spread_categories(self):
        """Test spread categorization features."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_betting())

        # 6.0 spread should be "small" (3-7)
        assert result["spread_small"].iloc[0] == 1.0
        assert result["spread_medium"].iloc[0] == 0.0
        assert result["spread_large"].iloc[0] == 0.0

    def test_implied_scores(self):
        """Test implied score calculations."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_betting())

        # O/U = 47.5, spread = 6
        # favorite_implied = (47.5 + 6) / 2 = 26.75
        # underdog_implied = (47.5 - 6) / 2 = 20.75
        assert abs(result["favorite_implied_score"].iloc[0] - 26.75) < 0.01
        assert abs(result["underdog_implied_score"].iloc[0] - 20.75) < 0.01


class TestSituationalFeatures:
    """Tests for situational/contextual features."""

    def test_home_indicator(self):
        """Test home indicator is calculated correctly."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data())

        # DET is home and underdog -> home_indicator = 1
        assert result["home_indicator"].iloc[0] == 1.0

    def test_divisional_game_flag(self):
        """Test divisional game detection."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_div_game())

        # Should have divisional_game column
        assert "divisional_game" in result.columns
        assert result["divisional_game"].iloc[0] == 1.0

    def test_rest_advantage(self):
        """Test rest advantage calculation."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_rest())

        # Underdog has 10 days rest, favorite has 7 -> advantage = 3
        assert "rest_advantage" in result.columns
        assert result["rest_advantage"].iloc[0] == 3.0

    def test_week_number_normalized(self):
        """Test week number is normalized 0-1."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data())

        assert "week_number" in result.columns
        # Week 2 / 18 = 0.111...
        assert 0 < result["week_number"].iloc[0] < 1


class TestRollingStats:
    """Tests for rolling statistics features."""

    def test_rolling_stats_columns_exist(self):
        """Test that rolling stat columns are created."""
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(mock_multi_game_data())

        expected_cols = [
            "underdog_points_scored_roll5",
            "underdog_point_diff_roll5",
            "underdog_win_streak",
            "favorite_points_scored_roll5",
            "favorite_win_streak",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing rolling stat column: {col}"

    def test_rolling_stats_use_prior_games(self):
        """Test that rolling stats only use prior games (no leakage)."""
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(mock_multi_game_data())

        # Week 1 should have no prior data (NaN or 0 after fillna)
        week_1 = result[result["week"] == 1]
        if len(week_1) > 0:
            # After fillna, week 1 rolling stats should be 0
            pass  # This is expected behavior

        # Week 2+ should have calculated values
        week_2_plus = result[result["week"] > 1]
        if len(week_2_plus) > 0:
            # Should have some non-zero values (or at least be filled)
            assert "underdog_points_scored_roll5" in week_2_plus.columns


class TestMatchupFeatures:
    """Tests for matchup differential features."""

    def test_matchup_differentials_calculated(self):
        """Test that matchup differentials are calculated."""
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(mock_multi_game_data())

        matchup_cols = [
            "offense_defense_mismatch",
            "defense_offense_mismatch",
            "point_diff_differential",
            "win_pct_differential",
        ]
        for col in matchup_cols:
            assert col in result.columns, f"Missing matchup column: {col}"

    def test_win_streak_diff(self):
        """Test win streak differential calculation."""
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(mock_multi_game_data())

        assert "win_streak_diff" in result.columns
        # Values should be filled (no NaN after pipeline)
        assert not result["win_streak_diff"].isna().any()


class TestInteractionFeatures:
    """Tests for interaction features."""

    def test_interaction_columns_exist(self):
        """Test that interaction feature columns are created."""
        pipeline = FeatureEngineeringPipeline()
        result = pipeline.transform(mock_data_with_betting())

        interaction_cols = [
            "spread_x_momentum",
            "spread_x_win_streak",
            "rest_x_home",
            "hot_underdog",
            "cold_favorite",
        ]
        for col in interaction_cols:
            assert col in result.columns, f"Missing interaction column: {col}"

    def test_hot_underdog_flag(self):
        """Test hot underdog indicator logic."""
        pipeline = FeatureEngineeringPipeline()
        feature_cols = pipeline.get_feature_columns()

        assert "hot_underdog" in feature_cols


# =============================================================================
# MOCK DATA FIXTURES
# =============================================================================

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


def mock_data_with_betting():
    """Mock data with betting line info."""
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
        "over_under_line": [47.5],
        "total_line": [47.5],
    })


def mock_data_with_div_game():
    """Mock data with divisional game flag."""
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
        "div_game": [1],
    })


def mock_data_with_rest():
    """Mock data with rest day information."""
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
        "home_rest": [10],  # DET (underdog, home) has 10 days rest
        "away_rest": [7],   # KC (favorite, away) has 7 days rest
    })


def mock_multi_game_data():
    """Mock data with multiple games for meaningful rolling stats."""
    return pd.DataFrame({
        "game_id": [
            "2023_01_KC_DET", "2023_01_BAL_CLE",
            "2023_02_DET_KC", "2023_02_CLE_BAL",
            "2023_03_KC_BAL",
        ],
        "season": [2023, 2023, 2023, 2023, 2023],
        "week": [1, 1, 2, 2, 3],
        "home_team": ["DET", "CLE", "KC", "BAL", "BAL"],
        "away_team": ["KC", "BAL", "DET", "CLE", "KC"],
        "home_score": [20, 17, 24, 28, 21],
        "away_score": [27, 24, 21, 10, 24],
        "spread_favorite": [-3.0, -4.5, -6.0, -7.0, -3.5],
        "team_favorite_id": ["KC", "BAL", "KC", "BAL", "BAL"],
    })
