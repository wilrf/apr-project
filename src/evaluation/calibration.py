"""Post-hoc probability calibration for model outputs.

Fits calibrators on held-out (out-of-fold) predictions so that calibrated
probabilities from different models are comparable for disagreement analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationResult:
    """Holds raw and calibrated probabilities with diagnostics."""

    raw: np.ndarray
    calibrated: np.ndarray
    method: str


class PlattScaler:
    """Platt scaling: fits a logistic sigmoid to map raw probs → calibrated probs."""

    def __init__(self) -> None:
        self._lr: Optional[LogisticRegression] = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        self._lr = LogisticRegression(solver="lbfgs", max_iter=5000)
        self._lr.fit(probs.reshape(-1, 1), y_true)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise ValueError("Call fit() before transform().")
        return self._lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibrator (non-parametric, monotonic)."""

    def __init__(self) -> None:
        self._ir: Optional[IsotonicRegression] = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._ir = IsotonicRegression(out_of_bounds="clip")
        self._ir.fit(probs, y_true)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._ir is None:
            raise ValueError("Call fit() before transform().")
        return self._ir.predict(probs)


def calibrate_models(
    cal_probs: Dict[str, np.ndarray],
    cal_y_true: np.ndarray,
    test_probs: Dict[str, np.ndarray],
    method: str = "platt",
) -> Dict[str, CalibrationResult]:
    """
    Fit calibrators on held-out predictions and apply to test predictions.

    Args:
        cal_probs: {model_name: raw_probs} on calibration set
        cal_y_true: true labels for calibration set
        test_probs: {model_name: raw_probs} on test set
        method: 'platt' or 'isotonic'

    Returns:
        {model_name: CalibrationResult} for each model
    """
    results: Dict[str, CalibrationResult] = {}

    for name in cal_probs:
        if method == "platt":
            scaler = PlattScaler().fit(cal_probs[name], cal_y_true)
        elif method == "isotonic":
            scaler = IsotonicCalibrator().fit(cal_probs[name], cal_y_true)
        else:
            raise ValueError(f"Unknown method: {method}")

        calibrated = scaler.transform(test_probs[name])
        results[name] = CalibrationResult(
            raw=test_probs[name],
            calibrated=calibrated,
            method=method,
        )

    return results


def generate_calibration_predictions(
    train_df: "pd.DataFrame",  # noqa: F821
    feature_cols: list,
    cal_seasons: Tuple[int, ...] = (2021, 2022),
    target_col: str = "upset",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Train models on pre-calibration data, predict on calibration seasons.

    Splits training data into:
      - fit set: seasons before min(cal_seasons)
      - calibration set: cal_seasons

    Returns:
        (model_probs, y_true) on the calibration set
    """
    import torch

    from src.models.lstm_config import TUNED_LSTM_TRAINING_PARAMS
    from src.models.sequence_builder import build_siamese_sequences
    from src.models.unified_trainer import UnifiedTrainer

    cal_set = set(cal_seasons)
    cutoff = min(cal_seasons)

    fit_df = train_df[train_df["season"] < cutoff].copy()
    cal_df = train_df[train_df["season"].isin(cal_set)].copy()

    if len(cal_df) == 0:
        raise ValueError(f"No calibration data for seasons {cal_seasons}")

    y_cal = cal_df[target_col].values

    trainer = UnifiedTrainer()
    models = trainer.train_final(
        fit_df,
        feature_cols,
        target_col=target_col,
        lstm_epochs=int(TUNED_LSTM_TRAINING_PARAMS["epochs"]),
        lstm_batch_size=int(TUNED_LSTM_TRAINING_PARAMS["batch_size"]),
        verbose=False,
    )

    X_cal = cal_df[feature_cols]
    lr_probs = models["lr_model"].predict_proba(X_cal)
    xgb_probs = models["xgb_model"].predict_proba(X_cal)

    cal_seq, _ = build_siamese_sequences(
        cal_df, normalize=True, stats=models["lstm_stats"]
    )
    lstm_model = models["lstm_model"]
    lstm_model.eval()
    with torch.no_grad():
        lstm_probs = (
            lstm_model(
                torch.FloatTensor(cal_seq.underdog_sequences),
                torch.FloatTensor(cal_seq.favorite_sequences),
                torch.FloatTensor(cal_seq.matchup_features),
                torch.FloatTensor(cal_seq.underdog_masks),
                torch.FloatTensor(cal_seq.favorite_masks),
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return (
        {"lr": lr_probs, "xgb": xgb_probs, "lstm": lstm_probs},
        y_cal,
    )
