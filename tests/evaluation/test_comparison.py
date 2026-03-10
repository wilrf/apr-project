# tests/evaluation/test_comparison.py
import numpy as np
import pytest

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

    def test_agreement_matrix_uses_provided_threshold_not_half(self):
        """H7 regression: agreement matrix must use the caller's threshold.

        With base rate ~0.35, predictions between 0.35 and 0.5 should be
        classified as "upset" at threshold=0.35 but "no upset" at 0.5.
        """
        y_true = np.array([1, 0, 0])
        results = {
            "model_a": {
                "y_true": y_true,
                "y_pred": np.array([0.40, 0.40, 0.10]),
                "metrics": {"auc_roc": 0.6},
            },
            "model_b": {
                "y_true": y_true,
                "y_pred": np.array([0.20, 0.40, 0.10]),
                "metrics": {"auc_roc": 0.6},
            },
        }
        comp = ModelComparison(results)

        # At threshold 0.35: a=[1,1,0], b=[0,1,0] → agree on 2/3
        matrix_035 = comp.get_agreement_matrix(threshold=0.35)
        rate_035 = matrix_035["agreement_rate"].iloc[0]

        # At threshold 0.5 (the buggy default): a=[0,0,0], b=[0,0,0] → agree 3/3
        matrix_050 = comp.get_agreement_matrix(threshold=0.5)
        rate_050 = matrix_050["agreement_rate"].iloc[0]

        # These must differ — proves the threshold is actually used
        assert rate_035 != rate_050
        assert rate_035 == pytest.approx(2 / 3, abs=0.01)
        assert rate_050 == pytest.approx(1.0)

    def test_agreement_matrix_default_threshold_is_base_rate(self):
        """H7: default threshold should be computed from data, not hardcoded 0.5."""
        y_true = np.array([1, 0, 0])  # base rate = 1/3
        results = {
            "model_a": {
                "y_true": y_true,
                "y_pred": np.array([0.40, 0.40, 0.10]),
                "metrics": {},
            },
            "model_b": {
                "y_true": y_true,
                "y_pred": np.array([0.20, 0.40, 0.10]),
                "metrics": {},
            },
        }
        comp = ModelComparison(results)

        # Default should use base rate (~0.333), not 0.5
        # At 0.333: a=[1,1,0], b=[0,1,0] → 2/3 agreement
        # At 0.5:   a=[0,0,0], b=[0,0,0] → 3/3 agreement
        matrix = comp.get_agreement_matrix()
        rate = matrix["agreement_rate"].iloc[0]
        assert rate == pytest.approx(2 / 3, abs=0.01)

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

    def test_constant_prediction_correlation_stays_finite(self):
        results = {
            "xgboost": {
                "y_true": np.array([1, 0, 1]),
                "y_pred": np.array([0.2, 0.2, 0.2]),
                "metrics": {"auc_roc": 0.72},
            },
            "lstm": {
                "y_true": np.array([1, 0, 1]),
                "y_pred": np.array([0.4, 0.4, 0.4]),
                "metrics": {"auc_roc": 0.68},
            },
        }

        comparison = ModelComparison(results)
        result = comparison.compare()
        assert result["xgboost_vs_lstm"]["prediction_correlation"] == 0.0
