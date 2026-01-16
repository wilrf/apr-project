# tests/evaluation/test_comparison.py
import pytest
import pandas as pd
import numpy as np
from src.evaluation.comparison import ModelComparison


class TestModelComparison:
    @pytest.fixture
    def sample_results(self):
        """Create sample model results."""
        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.35, n)

        return {
            "xgboost": {
                "y_true": y_true,
                "y_pred": np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1),
                "metrics": {"auc_roc": 0.72, "brier_score": 0.21},
            },
            "lstm": {
                "y_true": y_true,
                "y_pred": np.clip(y_true + np.random.normal(0, 0.25, n), 0, 1),
                "metrics": {"auc_roc": 0.68, "brier_score": 0.23},
            },
        }

    def test_comparison_calculates_metrics_difference(self, sample_results):
        """Test that comparison calculates metric differences."""
        comparison = ModelComparison(sample_results)
        result = comparison.compare()

        assert "xgboost_vs_lstm" in result
        assert "auc_roc_diff" in result["xgboost_vs_lstm"]

    def test_comparison_identifies_better_model(self, sample_results):
        """Test that comparison identifies the better model."""
        comparison = ModelComparison(sample_results)
        result = comparison.compare()

        # XGBoost has better AUC in sample data
        assert result["xgboost_vs_lstm"]["auc_roc_diff"] > 0

    def test_comparison_handles_single_model(self):
        """Test that single model doesn't crash."""
        results = {
            "xgboost": {
                "y_true": np.array([1, 0, 1]),
                "y_pred": np.array([0.7, 0.3, 0.8]),
                "metrics": {"auc_roc": 0.72},
            }
        }
        comparison = ModelComparison(results)
        result = comparison.compare()

        assert result == {}  # No comparison possible
