# tests/evaluation/test_metrics.py
import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_calibration_metrics,
    calculate_betting_metrics,
    calculate_baseline_brier,
)


class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated predictions."""
        # Predictions at 70% should be correct 70% of the time
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # 7/10 = 70%
        y_pred = np.array([0.7] * 10)

        result = calculate_calibration_metrics(y_true, y_pred, n_bins=1)
        assert abs(result["calibration_error"]) < 0.1


class TestBettingMetrics:
    def test_roi_calculation(self):
        """Test ROI is calculated correctly."""
        y_true = np.array([1, 0, 1, 0, 1])  # 3 wins, 2 losses
        y_pred = np.array([0.8, 0.6, 0.7, 0.9, 0.75])  # All predict upset
        odds = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # Even money

        result = calculate_betting_metrics(y_true, y_pred, odds, threshold=0.5)
        # 3 wins at +100 = +3 units, 2 losses = -2 units, net +1 unit
        # ROI = 1/5 = 20%
        assert abs(result["roi"] - 0.20) < 0.01

    def test_no_bets_placed(self):
        """Test handling when threshold is too high."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.3, 0.2, 0.4])  # All below threshold
        odds = np.array([2.0, 2.0, 2.0])

        result = calculate_betting_metrics(y_true, y_pred, odds, threshold=0.5)
        assert result["n_bets"] == 0


class TestBaselineBrier:
    def test_baseline_brier_formula(self):
        """Test baseline Brier score calculation."""
        # With upset rate r, baseline Brier = r * (1 - r)
        upset_rate = 0.35
        expected = 0.35 * 0.65  # = 0.2275

        result = calculate_baseline_brier(upset_rate)
        assert abs(result - expected) < 0.001

    def test_baseline_at_50_percent(self):
        """Test baseline Brier at 50% upset rate is 0.25."""
        result = calculate_baseline_brier(0.5)
        assert abs(result - 0.25) < 0.001
