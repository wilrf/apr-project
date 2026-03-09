# tests/models/test_xgboost_model.py
import pytest
import pandas as pd
import numpy as np
from src.models.xgboost_model import UpsetXGBoost


class TestUpsetXGBoost:
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame(
            {
                "spread_magnitude": np.random.uniform(3, 14, n),
                "offense_diff": np.random.normal(0, 1, n),
                "defense_diff": np.random.normal(0, 1, n),
                "upset": np.random.binomial(1, 0.35, n),
            }
        )

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns the model instance."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_returns_probabilities(self, sample_data):
        """Test that predict_proba returns values between 0 and 1."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_predict_returns_binary(self, sample_data):
        """Test that predict returns binary predictions."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_feature_importance_available(self, sample_data):
        """Test that feature importance is available after fit."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) == 3

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predicting before fit raises an error."""
        model = UpsetXGBoost()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]

        with pytest.raises(ValueError):
            model.predict_proba(X)
