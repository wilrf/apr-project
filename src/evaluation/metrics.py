"""Evaluation metrics for upset prediction models."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from typing import Dict, Any


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate calibration metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    # Expected Calibration Error (ECE)
    # calibration_curve drops empty bins, so we must compute weights
    # based on which predictions fall into the bins it actually returns
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])  # Assign predictions to bins

    # Count samples in bins that calibration_curve kept (non-empty bins)
    # prob_pred contains the mean predicted probability for each non-empty bin
    bin_weights = np.zeros(len(prob_pred))
    for i, mean_pred in enumerate(prob_pred):
        # Find predictions closest to this bin's mean
        bin_idx = np.argmin(np.abs(bin_edges[:-1] + 0.5 / n_bins - mean_pred))
        bin_weights[i] = np.sum(bin_indices == bin_idx) / len(y_pred)

    # Normalize weights to sum to 1 (only counting non-empty bins)
    if bin_weights.sum() > 0:
        bin_weights = bin_weights / bin_weights.sum()

    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)

    return {
        "calibration_error": float(ece),
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def calculate_betting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    odds: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculate betting profitability metrics.

    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        odds: Decimal odds for each bet
        threshold: Probability threshold for placing bet

    Returns:
        Dictionary with betting metrics
    """
    # Identify bets placed
    bets_placed = y_pred >= threshold
    n_bets = bets_placed.sum()

    if n_bets == 0:
        return {"roi": 0.0, "n_bets": 0, "win_rate": 0.0, "total_profit": 0.0}

    # Calculate returns
    wins = (y_true == 1) & bets_placed
    losses = (y_true == 0) & bets_placed

    # Assuming unit bets
    # For decimal odds: profit = (odds - 1) when winning, -1 when losing
    profit = wins.sum() * (odds[wins].mean() - 1) if wins.sum() > 0 else 0
    loss = losses.sum()  # Lost unit bets

    total_wagered = n_bets
    net_profit = profit - loss

    return {
        "roi": float(net_profit / total_wagered) if total_wagered > 0 else 0.0,
        "n_bets": int(n_bets),
        "win_rate": float(wins.sum() / n_bets) if n_bets > 0 else 0.0,
        "total_profit": float(net_profit),
    }


def calculate_baseline_brier(upset_rate: float) -> float:
    """
    Calculate baseline Brier score for constant prediction.

    The baseline is what you'd get by always predicting the upset rate.
    This is useful for evaluating if a model beats naive predictions.

    Args:
        upset_rate: Historical upset rate (proportion)

    Returns:
        Baseline Brier score
    """
    return upset_rate * (1 - upset_rate)
