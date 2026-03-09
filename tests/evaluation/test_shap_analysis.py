# tests/evaluation/test_shap_analysis.py
import pytest
import pandas as pd
import numpy as np
from src.evaluation.shap_analysis import (
    compute_shap_values,
    get_shap_feature_importance,
)
from src.models.xgboost_model import UpsetXGBoost


class TestSHAPAnalysis:
    @pytest.fixture
    def trained_model(self):
        """Create and train a model."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "spread_magnitude": np.random.uniform(3, 14, n),
                "offense_diff": np.random.normal(0, 1, n),
                "defense_diff": np.random.normal(0, 1, n),
            }
        )
        y = np.random.binomial(1, 0.35, n)

        model = UpsetXGBoost()
        model.fit(X, pd.Series(y))
        return model, X

    def test_shap_values_computed(self, trained_model):
        """Test that SHAP values are computed for all samples."""
        model, X = trained_model
        shap_values = compute_shap_values(model, X)

        assert shap_values.shape == X.shape

    def test_shap_importance_returns_dict(self, trained_model):
        """Test that feature importance returns dictionary."""
        model, X = trained_model
        importance = get_shap_feature_importance(model, X)

        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)

    def test_shap_values_reasonable_magnitude(self, trained_model):
        """Test SHAP values are within reasonable range."""
        model, X = trained_model
        shap_values = compute_shap_values(model, X)

        # SHAP values shouldn't be excessively large
        assert np.abs(shap_values).max() < 10
