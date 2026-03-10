# tests/models/test_trainer.py
import numpy as np
import pandas as pd
import pytest

from src.models.trainer import ModelTrainer
from src.models.xgboost_model import UpsetXGBoost


class TestModelTrainer:
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame(
            {
                "season": [2020] * 50 + [2021] * 50 + [2022] * 50 + [2023] * 50,
                "spread_magnitude": np.random.uniform(3, 14, n),
                "offense_diff": np.random.normal(0, 1, n),
                "defense_diff": np.random.normal(0, 1, n),
                "upset": np.random.binomial(1, 0.35, n),
            }
        )

    def test_trainer_runs_cross_validation(self, sample_data):
        """Test that trainer runs all CV folds."""
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=3)

        features = ["spread_magnitude", "offense_diff", "defense_diff"]
        results = trainer.cross_validate(sample_data, features, "upset")

        assert len(results["fold_metrics"]) == 3

    def test_trainer_returns_metrics(self, sample_data):
        """Test that trainer returns expected metrics."""
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=2)

        features = ["spread_magnitude", "offense_diff", "defense_diff"]
        results = trainer.cross_validate(sample_data, features, "upset")

        assert "auc_roc" in results["fold_metrics"][0]
        assert "log_loss" in results["fold_metrics"][0]

    def test_trainer_returns_aggregated_metrics(self, sample_data):
        """Test that trainer returns aggregated metrics."""
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=2)

        features = ["spread_magnitude", "offense_diff", "defense_diff"]
        results = trainer.cross_validate(sample_data, features, "upset")

        assert "auc_roc_mean" in results["aggregated"]
        assert "auc_roc_std" in results["aggregated"]

    def test_trainer_returns_predictions(self, sample_data):
        """Test that trainer returns fold predictions."""
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=2)

        features = ["spread_magnitude", "offense_diff", "defense_diff"]
        results = trainer.cross_validate(sample_data, features, "upset")

        assert len(results["predictions"]) == 2
        assert "y_true" in results["predictions"][0]
        assert "y_pred" in results["predictions"][0]

    def test_calculate_metrics_handles_single_class(self):
        model = UpsetXGBoost()
        trainer = ModelTrainer(model, n_folds=2)

        metrics = trainer._calculate_metrics(
            pd.Series([1, 1, 1]),
            np.array([0.9, 1.0, 0.9999999]),
        )

        assert np.isnan(metrics["auc_roc"])
        assert np.isfinite(metrics["log_loss"])
