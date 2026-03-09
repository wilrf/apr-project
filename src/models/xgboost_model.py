"""XGBoost model wrapper for upset prediction."""

from __future__ import annotations

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Any, Dict, List, Optional


class UpsetXGBoost:
    """
    XGBoost classifier wrapper for NFL upset prediction.

    Provides consistent interface with probability outputs
    and feature importance extraction.
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        min_child_weight: int = 1,
        random_state: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize XGBoost model.

        Args:
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum sum of instance weight in child
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            **kwargs,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List] = None,
        verbose: bool = False,
    ) -> "UpsetXGBoost":
        """
        Fit the XGBoost model.

        Args:
            X: Feature DataFrame
            y: Target Series
            eval_set: Optional validation set for early stopping
            verbose: Whether to print training progress

        Returns:
            Self
        """
        self.feature_names = list(X.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=verbose,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict upset probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of upset probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary upset outcomes.

        Args:
            X: Feature DataFrame
            threshold: Probability threshold for positive class

        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Map back to feature names
        # When fitted with DataFrame, get_score() uses actual feature names as keys
        return {name: importance.get(name, 0.0) for name in self.feature_names}
