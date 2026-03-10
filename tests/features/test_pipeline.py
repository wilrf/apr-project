"""Tests for the canonical feature engineering pipeline."""

import pandas as pd

from src.features.pipeline import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_SPREAD,
    XGB_FEATURE_COLUMNS,
    XGB_FEATURE_COLUMNS_NO_SPREAD,
    FeatureEngineeringPipeline,
    _identify_underdog,
)


def _mock_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2023_01_KC_DET",
                "2023_02_DET_BUF",
                "2023_02_KC_BUF",
                "2023_03_DET_KC",
            ],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 2, 3],
            "gameday": ["2023-09-07", "2023-09-14", "2023-09-14", "2023-09-21"],
            "home_team": ["KC", "DET", "KC", "DET"],
            "away_team": ["DET", "BUF", "BUF", "KC"],
            "home_score": [20, 24, 28, 17],
            "away_score": [21, 17, 24, 27],
            "team_favorite_id": ["KC", "DET", "KC", "KC"],
            "spread_favorite": [-4.5, -2.5, -7.0, -3.5],
            "over_under_line": [47.5, 44.0, 48.0, 45.0],
            "home_rest": [7, 7, 7, 7],
            "away_rest": [7, 7, 7, 7],
            "div_game": [0, 0, 0, 0],
            "temp": [72.0, 65.0, 58.0, 49.0],
            "wind": [6.0, 10.0, 12.0, 9.0],
            "roof": ["outdoors", "outdoors", "outdoors", "outdoors"],
            "home_off_pass_epa": [10.0, 8.0, 12.0, 7.0],
            "home_off_rush_epa": [2.0, 3.0, 4.0, 2.0],
            "away_off_pass_epa": [11.0, 6.0, 9.0, 13.0],
            "away_off_rush_epa": [3.0, 1.0, 2.0, 4.0],
            "home_success_rate": [0.48, 0.44, 0.50, 0.42],
            "away_success_rate": [0.51, 0.39, 0.46, 0.55],
            "home_cpoe": [1.2, -0.8, 0.7, -1.0],
            "away_cpoe": [2.1, -2.0, 0.3, 1.5],
            "home_turnover_margin": [-1.0, 1.0, 0.0, -1.0],
            "away_turnover_margin": [1.0, -1.0, 0.0, 1.0],
        }
    )


class TestFeatureEngineeringPipeline:
    def test_pipeline_produces_expected_columns(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(_mock_games())

        expected_cols = [
            "game_id",
            "upset",
            "spread_magnitude",
            "pass_epa_diff",
            "elo_diff",
            "temperature_missing",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_pipeline_excludes_week_1_when_configured(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(_mock_games())
        assert not any(result["week"] == 1)

    def test_pipeline_keeps_week_1_when_disabled(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(_mock_games())
        assert any(result["week"] == 1)

    def test_small_spreads_excluded_from_target(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(_mock_games())

        game = result[result["game_id"] == "2023_02_DET_BUF"].iloc[0]
        assert game["spread_magnitude"] == 2.5
        assert pd.isna(game["upset"])


class TestFeatureCount:
    def test_feature_count_is_46(self):
        pipeline = FeatureEngineeringPipeline()
        assert len(pipeline.get_feature_columns()) == 46
        assert len(FEATURE_COLUMNS) == 46

    def test_no_spread_count_is_42(self):
        pipeline = FeatureEngineeringPipeline()
        assert len(pipeline.get_no_spread_feature_columns()) == 42
        assert len(FEATURE_COLUMNS_NO_SPREAD) == 42

    def test_xgb_feature_count_is_70(self):
        assert len(XGB_FEATURE_COLUMNS) == 70

    def test_xgb_no_spread_count_is_66(self):
        assert len(XGB_FEATURE_COLUMNS_NO_SPREAD) == 66

    def test_xgb_features_are_superset_of_base(self):
        assert set(FEATURE_COLUMNS).issubset(set(XGB_FEATURE_COLUMNS))

    def test_all_features_belong_to_groups(self):
        pipeline = FeatureEngineeringPipeline()
        grouped = set()
        for cols in pipeline.get_feature_groups().values():
            grouped.update(cols)
        assert grouped == set(FEATURE_COLUMNS)


class TestFeatureSemantics:
    def test_market_features_exist(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(_mock_games())

        assert "home_implied_points" in result.columns
        assert "away_implied_points" in result.columns
        assert "total_line" in result.columns

    def test_elo_features_exist(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(_mock_games())

        assert "underdog_elo" in result.columns
        assert "favorite_elo" in result.columns
        assert "elo_diff" in result.columns

    def test_rolling_features_fill_output_schema(self):
        pipeline = FeatureEngineeringPipeline(exclude_week_1=False)
        result = pipeline.transform(_mock_games())

        assert set(FEATURE_COLUMNS).issubset(result.columns)

    def test_identify_underdog_returns_none_for_unknown_favorite(self):
        row = pd.Series(
            {
                "home_team": "KC",
                "away_team": "BUF",
                "team_favorite_id": "PHI",
                "spread_favorite": -3.0,
            }
        )

        assert _identify_underdog(row) is None
