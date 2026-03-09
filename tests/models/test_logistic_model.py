# tests/models/test_logistic_model.py
import pytest
import pandas as pd
import numpy as np
from src.models.logistic_model import UpsetLogisticRegression


class TestUpsetLogisticRegression:
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
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_returns_probabilities(self, sample_data):
        """Test that predict_proba returns values between 0 and 1."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert all(0 <= p <= 1 for p in probs)
        assert len(probs) == len(X)

    def test_predict_returns_binary(self, sample_data):
        """Test that predict returns binary predictions."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_with_custom_threshold(self, sample_data):
        """Test predict with custom threshold."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        # Higher threshold should predict fewer positives
        preds_low = model.predict(X, threshold=0.3)
        preds_high = model.predict(X, threshold=0.7)
        assert preds_low.sum() >= preds_high.sum()

    def test_get_coefficients_available(self, sample_data):
        """Test that coefficients are available after fit."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        coefficients = model.get_coefficients()
        assert len(coefficients) == 3
        assert all(name in coefficients for name in X.columns)
        assert all(isinstance(v, float) for v in coefficients.values())

    def test_get_feature_importance_available(self, sample_data):
        """Test that feature importance (abs coefficients) is available."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) == 3
        assert all(v >= 0 for v in importance.values())

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predicting before fit raises an error."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]

        with pytest.raises(ValueError):
            model.predict_proba(X)

    def test_get_coefficients_before_fit_raises_error(self):
        """Test that getting coefficients before fit raises an error."""
        model = UpsetLogisticRegression()

        with pytest.raises(ValueError):
            model.get_coefficients()

    def test_default_params(self, sample_data):
        """Test default parameters are applied."""
        model = UpsetLogisticRegression()
        assert model.C == 0.1
        assert model.penalty == "l1"
        assert model.solver == "saga"

    def test_custom_params(self, sample_data):
        """Test custom parameters are applied."""
        model = UpsetLogisticRegression(C=1.0, penalty="l2", solver="lbfgs")
        assert model.C == 1.0
        assert model.penalty == "l2"
        assert model.solver == "lbfgs"

    def test_l1_regularization_produces_sparse_coefficients(self):
        """Test that L1 regularization can produce sparse coefficients."""
        # Create data where signal feature is clearly predictive
        np.random.seed(42)
        n = 500
        signal = np.random.normal(0, 1, n)
        noise1 = np.random.normal(0, 0.1, n)
        noise2 = np.random.normal(0, 0.1, n)

        X = pd.DataFrame(
            {
                "signal": signal,
                "noise1": noise1,
                "noise2": noise2,
            }
        )
        # Target strongly correlated with signal
        y = pd.Series((signal + np.random.normal(0, 0.3, n) > 0).astype(int))

        model = UpsetLogisticRegression(C=0.5)  # Moderate regularization
        model.fit(X, y)

        coefficients = model.get_coefficients()
        # Signal feature should have larger coefficient than noise
        assert abs(coefficients["signal"]) > abs(coefficients["noise1"])
        assert abs(coefficients["signal"]) > abs(coefficients["noise2"])

    def test_scaling_is_applied(self, sample_data):
        """Test that features are scaled internally."""
        model = UpsetLogisticRegression()
        X = sample_data[["spread_magnitude", "offense_diff", "defense_diff"]]
        y = sample_data["upset"]
        model.fit(X, y)

        # Scaler should exist after fitting
        assert model.scaler is not None
        assert model.scaler.mean_ is not None
        assert model.scaler.scale_ is not None
