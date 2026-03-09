"""Tests for the canonical no-spread feature configuration."""

from src.features.pipeline import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_NO_SPREAD,
    XGB_FEATURE_COLUMNS,
    XGB_FEATURE_COLUMNS_NO_SPREAD,
    FeatureEngineeringPipeline,
    get_feature_columns,
    get_no_spread_feature_columns,
    get_xgb_feature_columns,
    get_xgb_no_spread_feature_columns,
)


class TestNoSpreadFeatures:
    """Verify the canonical with-spread and no-spread selectors."""

    def setup_method(self):
        self.pipeline = FeatureEngineeringPipeline()

    def test_no_spread_feature_count(self):
        """No-spread set should remove only the direct market features."""
        assert len(self.pipeline.get_feature_columns()) == 46
        assert len(self.pipeline.get_no_spread_feature_columns()) == 42

    def test_module_level_selectors_match_pipeline(self):
        """Module-level selectors should mirror the pipeline methods."""
        assert get_feature_columns() == FEATURE_COLUMNS
        assert get_no_spread_feature_columns() == FEATURE_COLUMNS_NO_SPREAD
        assert self.pipeline.get_feature_columns() == FEATURE_COLUMNS
        assert (
            self.pipeline.get_no_spread_feature_columns() == FEATURE_COLUMNS_NO_SPREAD
        )

    def test_no_spread_is_strict_subset(self):
        """No-spread features must be a strict subset of the full set."""
        full = set(self.pipeline.get_feature_columns())
        no_spread = set(self.pipeline.get_no_spread_feature_columns())
        assert no_spread < full

    def test_market_features_excluded(self):
        """Direct market features must not appear in the no-spread selector."""
        no_spread = set(self.pipeline.get_no_spread_feature_columns())
        for feature in [
            "home_implied_points",
            "away_implied_points",
            "spread_magnitude",
            "total_line",
        ]:
            assert feature not in no_spread

    def test_non_market_features_retained(self):
        """No-spread selector should retain performance and context inputs."""
        no_spread = set(self.pipeline.get_no_spread_feature_columns())
        for feature in [
            "pass_epa_diff",
            "success_rate_diff",
            "underdog_total_epa_std_roll3",
            "elo_diff",
            "temperature",
            "underdog_is_home",
        ]:
            assert feature in no_spread

    def test_feature_groups_cover_schema(self):
        """Feature groups should cover the full canonical schema exactly."""
        grouped = set()
        for cols in self.pipeline.get_feature_groups().values():
            grouped.update(cols)
        assert grouped == set(FEATURE_COLUMNS)

    def test_xgb_no_spread_feature_count(self):
        """XGB no-spread set should remove market features from expanded set."""
        assert len(XGB_FEATURE_COLUMNS) == 70
        assert len(XGB_FEATURE_COLUMNS_NO_SPREAD) == 66

    def test_xgb_module_level_selectors(self):
        """Module-level XGB selectors should match constants."""
        assert get_xgb_feature_columns() == XGB_FEATURE_COLUMNS
        assert get_xgb_no_spread_feature_columns() == XGB_FEATURE_COLUMNS_NO_SPREAD

    def test_xgb_no_spread_is_strict_subset(self):
        """XGB no-spread features must be a strict subset of full XGB set."""
        assert set(XGB_FEATURE_COLUMNS_NO_SPREAD) < set(XGB_FEATURE_COLUMNS)
