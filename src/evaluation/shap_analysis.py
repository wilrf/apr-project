"""SHAP analysis for XGBoost model interpretability."""

from __future__ import annotations

import shap
import pandas as pd
import numpy as np
from typing import Dict
from src.models.xgboost_model import UpsetXGBoost


def compute_shap_values(
    model: UpsetXGBoost,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for predictions.

    Args:
        model: Trained UpsetXGBoost model
        X: Feature DataFrame

    Returns:
        Array of SHAP values with shape (n_samples, n_features)
    """
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)

    return shap_values


def get_shap_feature_importance(
    model: UpsetXGBoost,
    X: pd.DataFrame,
) -> Dict[str, float]:
    """
    Get feature importance based on mean absolute SHAP values.

    Args:
        model: Trained UpsetXGBoost model
        X: Feature DataFrame

    Returns:
        Dictionary mapping feature names to importance scores
    """
    shap_values = compute_shap_values(model, X)

    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    return {name: float(imp) for name, imp in zip(X.columns, importance)}


def get_shap_summary(
    model: UpsetXGBoost,
    X: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Get SHAP summary statistics for each feature.

    Args:
        model: Trained UpsetXGBoost model
        X: Feature DataFrame

    Returns:
        Dictionary with SHAP summary statistics
    """
    shap_values = compute_shap_values(model, X)

    return {
        "shap_values": shap_values,
        "feature_names": list(X.columns),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "std_shap": np.std(shap_values, axis=0),
    }
