"""Model training pipeline with cross-validation."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Protocol, Any, Dict, List
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from src.models.cv_splitter import TimeSeriesCVSplitter


class Model(Protocol):
    """Protocol for model interface."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class ModelTrainer:
    """
    Training pipeline with time-series cross-validation.

    Handles:
    - Cross-validation splits
    - Model training per fold
    - Metric calculation
    - MLflow logging (optional)
    """

    def __init__(
        self,
        model: Model,
        n_folds: int = 6,
    ):
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            n_folds: Number of CV folds
        """
        self.model = model
        self.cv_splitter = TimeSeriesCVSplitter(n_folds=n_folds)

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> Dict[str, Any]:
        """
        Run time-series cross-validation.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column

        Returns:
            Dictionary with fold metrics and aggregated results
        """
        X = df[feature_cols]
        y = df[target_col]

        fold_metrics: List[Dict[str, Any]] = []
        fold_predictions: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(df)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            self.model.fit(X_train, y_train)

            # Get predictions
            y_pred_proba = self.model.predict_proba(X_val)

            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred_proba)
            metrics["fold"] = fold_idx
            metrics["train_size"] = len(train_idx)
            metrics["val_size"] = len(val_idx)

            fold_metrics.append(metrics)
            fold_predictions.append(
                {
                    "val_idx": val_idx,
                    "y_true": y_val.values,
                    "y_pred": y_pred_proba,
                }
            )

        # Aggregate metrics
        aggregated = self._aggregate_metrics(fold_metrics)

        return {
            "fold_metrics": fold_metrics,
            "aggregated": aggregated,
            "predictions": fold_predictions,
        }

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
        }

    def _aggregate_metrics(
        self, fold_metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        metric_names = ["auc_roc", "log_loss", "brier_score"]
        aggregated: Dict[str, float] = {}

        for metric in metric_names:
            values = [f[metric] for f in fold_metrics]
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))

        return aggregated
