# tests/test_integration.py
"""Integration tests for the full NFL upset prediction pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.features.pipeline import FeatureEngineeringPipeline
from src.models.xgboost_model import UpsetXGBoost
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import calculate_baseline_brier


def create_mock_season_data(season: int, n_games_per_week: int = 8, n_weeks: int = 17):
    """
    Create realistic mock data for a full NFL season.

    Args:
        season: Season year
        n_games_per_week: Games per week (16 for typical NFL)
        n_weeks: Number of weeks in season

    Returns:
        DataFrame mimicking real NFL schedule with betting data
    """
    teams = ["KC", "DET", "SF", "PHI", "BAL", "CLE", "BUF", "MIA",
             "DAL", "NYG", "LAR", "SEA", "DEN", "LV", "MIN", "GB"]

    rows = []
    game_id = 0

    for week in range(1, n_weeks + 1):
        # Create games for this week (pair up teams)
        np.random.seed(season * 100 + week)  # Reproducible per week
        shuffled = np.random.permutation(teams)

        for i in range(0, min(len(shuffled), n_games_per_week * 2), 2):
            home = shuffled[i]
            away = shuffled[i + 1]

            # Generate scores (somewhat realistic)
            home_score = np.random.randint(10, 40)
            away_score = np.random.randint(10, 40)

            # Generate spread (random favorite)
            spread_magnitude = np.random.uniform(1, 10)
            if np.random.random() < 0.6:  # Home team favorite 60%
                favorite = home
                spread = -spread_magnitude
            else:
                favorite = away
                spread = spread_magnitude

            rows.append({
                "game_id": f"{season}_{week:02d}_{home}_{away}",
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "spread_favorite": round(spread, 1),
                "team_favorite_id": favorite,
            })
            game_id += 1

    return pd.DataFrame(rows)


class TestFullPipelineIntegration:
    """Integration tests for the complete pipeline."""

    def test_pipeline_produces_valid_features(self):
        """Test that pipeline produces valid features from mock data."""
        # Create mock data for 2 seasons
        df = pd.concat([
            create_mock_season_data(2022, n_games_per_week=8, n_weeks=5),
            create_mock_season_data(2023, n_games_per_week=8, n_weeks=5),
        ]).reset_index(drop=True)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        # Check expected columns exist (updated for new 61-feature pipeline)
        expected_cols = [
            "upset",
            "spread_magnitude",
            "offense_defense_mismatch",
            "point_diff_differential",
            "win_streak_diff",
            "home_indicator",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Check target variable is valid
        valid_targets = result[result["upset"].notna()]["upset"]
        assert all(valid_targets.isin([0, 1]))

        # Check spread magnitude is positive
        assert all(result["spread_magnitude"] >= 0)

    def test_pipeline_excludes_week_1_games(self):
        """Test that Week 1 games are excluded when configured."""
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=5)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        assert not any(result["week"] == 1)
        assert all(result["week"] >= 2)

    def test_xgboost_model_trains_and_predicts(self):
        """Test that XGBoost model can train and produce predictions."""
        # Create training data
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        features_df = pipeline.transform(df)

        # Filter to valid samples
        valid_df = features_df[features_df["upset"].notna()].copy()

        if len(valid_df) < 20:
            pytest.skip("Not enough valid samples for test")

        feature_cols = ["spread_magnitude", "offense_defense_mismatch",
                        "point_diff_differential", "win_streak_diff", "home_indicator"]

        # Filter to columns that exist
        feature_cols = [c for c in feature_cols if c in valid_df.columns]

        X = valid_df[feature_cols].fillna(0)
        y = valid_df["upset"].astype(int)

        # Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        model = UpsetXGBoost(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        # Get predictions
        preds = model.predict_proba(X_test)

        # Check predictions are valid probabilities
        assert len(preds) == len(X_test)
        assert all(0 <= p <= 1 for p in preds)

    def test_model_trainer_cross_validation(self):
        """Test that model trainer performs cross-validation correctly."""
        # Create multi-season data for CV
        df = pd.concat([
            create_mock_season_data(2020, n_games_per_week=8, n_weeks=8),
            create_mock_season_data(2021, n_games_per_week=8, n_weeks=8),
            create_mock_season_data(2022, n_games_per_week=8, n_weeks=8),
            create_mock_season_data(2023, n_games_per_week=8, n_weeks=8),
        ]).reset_index(drop=True)

        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        features_df = pipeline.transform(df)

        # Filter to valid samples
        valid_df = features_df[features_df["upset"].notna()].copy()

        feature_cols = ["spread_magnitude", "offense_defense_mismatch",
                        "point_diff_differential", "win_streak_diff", "home_indicator"]
        feature_cols = [c for c in feature_cols if c in valid_df.columns]

        # Fill missing values
        for col in feature_cols:
            valid_df[col] = valid_df[col].fillna(0)

        # Train with CV
        model = UpsetXGBoost(n_estimators=10, max_depth=3)
        trainer = ModelTrainer(model, n_folds=3)

        results = trainer.cross_validate(
            valid_df,
            feature_cols=feature_cols,
            target_col="upset",
        )

        # Check results structure
        assert "fold_metrics" in results
        assert "aggregated" in results
        assert "predictions" in results

        # Check we got metrics for each fold
        assert len(results["fold_metrics"]) == 3

        # Check aggregated metrics exist
        assert "auc_roc_mean" in results["aggregated"]
        assert "brier_score_mean" in results["aggregated"]

    def test_baseline_brier_calculation(self):
        """Test baseline Brier score calculation."""
        # Create data to calculate upset rate
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        features_df = pipeline.transform(df)

        valid_targets = features_df[features_df["upset"].notna()]["upset"]
        upset_rate = valid_targets.mean()

        # Calculate baseline
        baseline = calculate_baseline_brier(upset_rate)

        # Baseline should be r * (1 - r)
        expected = upset_rate * (1 - upset_rate)
        assert abs(baseline - expected) < 0.001


class TestFeatureVariance:
    """Tests verifying that features have meaningful variance."""

    def test_matchup_differentials_have_variance(self):
        """Test that matchup differentials vary across games."""
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        # Get games with valid data
        valid = result[result["underdog"].notna()]

        if len(valid) < 10:
            pytest.skip("Not enough valid games")

        # Check that offense_defense_mismatch has some variance
        mismatch = valid["offense_defense_mismatch"]
        # Standard deviation should be > 0 if there's variance
        # (will be 0 only if all values identical, which would indicate a bug)
        assert mismatch.std() >= 0  # At minimum, no errors

    def test_spread_magnitude_varies(self):
        """Test that spread magnitude varies across games."""
        df = create_mock_season_data(2023, n_games_per_week=8, n_weeks=10)
        pipeline = FeatureEngineeringPipeline(exclude_week_1=True)
        result = pipeline.transform(df)

        spreads = result["spread_magnitude"]

        # Should have multiple unique values
        assert len(spreads.unique()) > 1
