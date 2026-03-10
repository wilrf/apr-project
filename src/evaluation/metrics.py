"""Evaluation metrics for upset prediction models."""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import log_loss, roc_auc_score


def clip_probabilities(y_pred: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clip predicted probabilities away from 0 and 1."""
    return np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)


def safe_roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return ROC AUC when defined, otherwise NaN."""
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr)) < 2:
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(roc_auc_score(y_arr, y_pred))


def safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return log loss with clipped probabilities and explicit binary labels."""
    return float(log_loss(y_true, clip_probabilities(y_pred), labels=[0, 1]))


def safe_probability_correlation(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> float:
    """Return a finite Pearson correlation, falling back to 0 for degenerate cases."""
    a = np.asarray(preds_a, dtype=float)
    b = np.asarray(preds_b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return 0.0

    a = a[:n]
    b = b[:n]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0

    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def safe_quantile_buckets(values: np.ndarray, q: int = 5) -> pd.Series:
    """Bucket values into quantiles without failing on constant or empty inputs."""
    series = pd.Series(values, dtype=float)
    if series.empty:
        return pd.Series(dtype="Int64")

    finite = series.dropna()
    if finite.empty or finite.nunique() <= 1:
        buckets = pd.Series(0, index=series.index, dtype="Int64")
        buckets[series.isna()] = pd.NA
        return buckets

    bucketed = pd.qcut(series, q=q, labels=False, duplicates="drop")
    return pd.Series(bucketed, index=series.index, dtype="Int64")


def calculate_calibration_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Calculate calibration metrics including Expected Calibration Error (ECE)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = clip_probabilities(y_pred)
    if len(y_pred_arr) == 0:
        return {
            "calibration_error": 0.0,
            "prob_true": np.array([]),
            "prob_pred": np.array([]),
        }

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_arr, bin_edges[1:-1], right=False)

    prob_true = []
    prob_pred = []
    bin_weights = []
    for bucket in range(n_bins):
        in_bucket = bin_indices == bucket
        if not np.any(in_bucket):
            continue
        prob_true.append(float(np.mean(y_true_arr[in_bucket])))
        prob_pred.append(float(np.mean(y_pred_arr[in_bucket])))
        bin_weights.append(float(np.mean(in_bucket)))

    prob_true_arr = np.asarray(prob_true)
    prob_pred_arr = np.asarray(prob_pred)
    bin_weights_arr = np.asarray(bin_weights)

    return {
        "calibration_error": float(
            np.sum(np.abs(prob_true_arr - prob_pred_arr) * bin_weights_arr)
        ),
        "prob_true": prob_true_arr,
        "prob_pred": prob_pred_arr,
    }


def calculate_betting_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, odds: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculate betting profitability metrics (ROI, win rate, profit)."""
    bets = y_pred >= threshold
    n_bets = bets.sum()
    if n_bets == 0:
        return {"roi": 0.0, "n_bets": 0, "win_rate": 0.0, "total_profit": 0.0}

    wins = (y_true == 1) & bets
    profit = (wins.sum() * (odds[wins].mean() - 1) if wins.sum() > 0 else 0) - (
        (y_true == 0) & bets
    ).sum()

    return {
        "roi": float(profit / n_bets),
        "n_bets": int(n_bets),
        "win_rate": float(wins.sum() / n_bets),
        "total_profit": float(profit),
    }


def calculate_baseline_brier(upset_rate: float) -> float:
    """Baseline Brier score for constant prediction at upset rate."""
    return upset_rate * (1 - upset_rate)
