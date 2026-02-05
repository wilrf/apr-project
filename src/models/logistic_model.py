"""Logistic Regression model wrapper for upset prediction."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional


class UpsetLogisticRegression:
    """
    Logistic Regression classifier wrapper for NFL upset prediction.

    Uses L1 regularization to identify key linear predictors of upsets.
    Captures spread mispricing signal - when the market has systematically
    mispriced certain matchup characteristics.

    Provides consistent interface with XGBoost model:
    - fit(X, y) -> self
    - predict_proba(X) -> np.ndarray
    - predict(X, threshold) -> np.ndarray
    - get_coefficients() -> Dict[str, float]
    """

    def __init__(
        self,
        C: float = 0.1,
        penalty: str = "l1",
        solver: str = "saga",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            C: Inverse of regularization strength (smaller = stronger regularization)
            penalty: Regularization type ('l1', 'l2', 'elasticnet')
            solver: Optimization algorithm ('saga' supports L1)
            max_iter: Maximum iterations for solver
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "UpsetLogisticRegression":
        """
        Fit the Logistic Regression model.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Self
        """
        self.feature_names = list(X.columns)

        # Scale features for logistic regression
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit logistic regression
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict upset probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of upset probabilities
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

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

    def get_coefficients(self) -> Dict[str, float]:
        """
        Get feature coefficients for interpretability.

        Returns:
            Dictionary mapping feature names to coefficient values.
            Positive coefficients increase upset probability.
            Coefficients are on standardized scale.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            name: float(coef)
            for name, coef in zip(self.feature_names, self.model.coef_[0])
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance as absolute coefficient values.

        Provides consistent interface with XGBoost model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        coefficients = self.get_coefficients()
        return {name: abs(coef) for name, coef in coefficients.items()}
