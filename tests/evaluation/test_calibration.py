"""Tests for post-hoc calibration module."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    CalibrationResult,
    IsotonicCalibrator,
    PlattScaler,
    calibrate_models,
)


class TestPlattScaler:
    def test_fit_transform_returns_probabilities(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.1, 0.9, size=200)
        y = (probs > 0.5).astype(int)

        scaler = PlattScaler().fit(probs, y)
        calibrated = scaler.transform(probs)

        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_transform_before_fit_raises(self):
        scaler = PlattScaler()
        with pytest.raises(ValueError, match="fit"):
            scaler.transform(np.array([0.5]))

    def test_monotonic_output(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.05, 0.95, size=500)
        y = (rng.uniform(size=500) < probs).astype(int)

        scaler = PlattScaler().fit(probs, y)
        test_probs = np.linspace(0.01, 0.99, 50)
        calibrated = scaler.transform(test_probs)

        # Platt scaling is monotonic (logistic sigmoid)
        assert np.all(np.diff(calibrated) >= -1e-10)


class TestIsotonicCalibrator:
    def test_fit_transform_returns_probabilities(self):
        rng = np.random.RandomState(42)
        probs = rng.uniform(0.1, 0.9, size=200)
        y = (probs > 0.5).astype(int)

        scaler = IsotonicCalibrator().fit(probs, y)
        calibrated = scaler.transform(probs)

        assert calibrated.shape == probs.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_transform_before_fit_raises(self):
        scaler = IsotonicCalibrator()
        with pytest.raises(ValueError, match="fit"):
            scaler.transform(np.array([0.5]))


class TestCalibrateModels:
    def test_calibrates_multiple_models(self):
        rng = np.random.RandomState(42)
        y_cal = rng.binomial(1, 0.3, size=200)

        cal_probs = {
            "model_a": rng.uniform(0.1, 0.8, size=200),
            "model_b": rng.uniform(0.05, 0.9, size=200),
        }
        test_probs = {
            "model_a": rng.uniform(0.1, 0.8, size=100),
            "model_b": rng.uniform(0.05, 0.9, size=100),
        }

        results = calibrate_models(cal_probs, y_cal, test_probs, method="platt")

        assert set(results.keys()) == {"model_a", "model_b"}
        for name, res in results.items():
            assert isinstance(res, CalibrationResult)
            assert res.calibrated.shape == test_probs[name].shape
            assert res.method == "platt"
            assert np.all(res.calibrated >= 0)
            assert np.all(res.calibrated <= 1)

    def test_isotonic_method(self):
        rng = np.random.RandomState(42)
        y_cal = rng.binomial(1, 0.3, size=200)
        cal_probs = {"m": rng.uniform(0.1, 0.8, size=200)}
        test_probs = {"m": rng.uniform(0.1, 0.8, size=50)}

        results = calibrate_models(cal_probs, y_cal, test_probs, method="isotonic")
        assert results["m"].method == "isotonic"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            calibrate_models(
                {"m": np.array([0.5])},
                np.array([1]),
                {"m": np.array([0.5])},
                method="unknown",
            )
