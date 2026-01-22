"""Evaluation metrics for upset prediction models."""

from __future__ import annotations
import numpy as np
from sklearn.calibration import calibration_curve
from typing import Dict, Any


def calculate_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Calculate calibration metrics including Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    # ECE: weight by bin occupancy
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])

    bin_weights = np.array([np.sum(bin_indices == np.argmin(np.abs(bin_edges[:-1] + 0.5 / n_bins - mp))) / len(y_pred)
                           for mp in prob_pred])
    if bin_weights.sum() > 0:
        bin_weights /= bin_weights.sum()

    return {"calibration_error": float(np.sum(np.abs(prob_true - prob_pred) * bin_weights)),
            "prob_true": prob_true, "prob_pred": prob_pred}


def calculate_betting_metrics(y_true: np.ndarray, y_pred: np.ndarray, odds: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate betting profitability metrics (ROI, win rate, profit)."""
    bets = y_pred >= threshold
    n_bets = bets.sum()
    if n_bets == 0:
        return {"roi": 0.0, "n_bets": 0, "win_rate": 0.0, "total_profit": 0.0}

    wins = (y_true == 1) & bets
    profit = (wins.sum() * (odds[wins].mean() - 1) if wins.sum() > 0 else 0) - ((y_true == 0) & bets).sum()

    return {"roi": float(profit / n_bets), "n_bets": int(n_bets),
            "win_rate": float(wins.sum() / n_bets), "total_profit": float(profit)}


def calculate_baseline_brier(upset_rate: float) -> float:
    """Baseline Brier score for constant prediction at upset rate."""
    return upset_rate * (1 - upset_rate)
