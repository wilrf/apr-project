"""Model comparison utilities for LR, XGBoost, and LSTM."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from itertools import combinations

HIGHER_IS_BETTER = {"auc_roc", "accuracy", "roi"}


class ModelComparison:
    """Compare multiple models on the same data with pairwise metrics and prediction correlation."""

    def __init__(self, model_results: Dict[str, Dict[str, Any]]):
        self.model_results = model_results
        self.model_names = list(model_results.keys())

    def compare(self) -> Dict[str, Dict[str, float]]:
        """Run pairwise model comparisons."""
        if len(self.model_names) < 2:
            return {}
        return {f"{a}_vs_{b}": self._compare_pair(a, b) for a, b in combinations(self.model_names, 2)}

    def _compare_pair(self, model_a: str, model_b: str) -> Dict[str, float]:
        """Compare two models on shared metrics and prediction correlation."""
        ra, rb = self.model_results[model_a], self.model_results[model_b]
        ma, mb = ra.get("metrics", {}), rb.get("metrics", {})

        comparison = {f"{m}_diff": ma[m] - mb[m] for m in set(ma) & set(mb)}

        if "y_pred" in ra and "y_pred" in rb:
            pa, pb = np.array(ra["y_pred"]), np.array(rb["y_pred"])
            if len(pa) == len(pb):
                comparison["prediction_correlation"] = float(np.corrcoef(pa, pb)[0, 1])

        return comparison

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all models' performance with rankings."""
        summary = {"models": {}, "rankings": {}}
        all_metrics = set()

        for name, result in self.model_results.items():
            summary["models"][name] = result.get("metrics", {})
            all_metrics.update(summary["models"][name].keys())

        for metric in all_metrics:
            scores = [(n, r["metrics"][metric]) for n, r in self.model_results.items() if metric in r.get("metrics", {})]
            scores.sort(key=lambda x: x[1], reverse=(metric in HIGHER_IS_BETTER))
            summary["rankings"][metric] = [n for n, _ in scores]

        return summary

    def get_agreement_matrix(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Get pairwise binary prediction agreement rates between all models.

        Args:
            threshold: Probability threshold for binary predictions

        Returns:
            DataFrame with agreement rates for all model pairs
        """
        if len(self.model_names) < 2:
            return pd.DataFrame()

        # Extract predictions for each model
        predictions: Dict[str, np.ndarray] = {}
        for name, result in self.model_results.items():
            if "y_pred" in result:
                probs = np.array(result["y_pred"])
                predictions[name] = (probs >= threshold).astype(int)

        if len(predictions) < 2:
            return pd.DataFrame()

        # Find common length
        min_len = min(len(p) for p in predictions.values())
        predictions = {k: v[:min_len] for k, v in predictions.items()}

        # Calculate pairwise agreement
        rows = []
        model_list = list(predictions.keys())

        for i, model_a in enumerate(model_list):
            for j, model_b in enumerate(model_list):
                if i < j:
                    pa, pb = predictions[model_a], predictions[model_b]
                    agreement = (pa == pb).mean()
                    rows.append({
                        "model_a": model_a,
                        "model_b": model_b,
                        "agreement_rate": agreement,
                    })

        # Add all-agree rate if 3+ models
        if len(model_list) >= 3:
            preds_matrix = np.column_stack([predictions[m] for m in model_list])
            all_agree = (preds_matrix.std(axis=1) == 0).mean()
            rows.append({
                "model_a": "ALL",
                "model_b": "ALL",
                "agreement_rate": all_agree,
            })

        return pd.DataFrame(rows)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of model probability predictions.

        Returns:
            DataFrame with correlation coefficients
        """
        predictions: Dict[str, np.ndarray] = {}
        for name, result in self.model_results.items():
            if "y_pred" in result:
                predictions[name] = np.array(result["y_pred"])

        if len(predictions) < 2:
            return pd.DataFrame()

        # Find common length
        min_len = min(len(p) for p in predictions.values())
        model_names = list(predictions.keys())
        data = np.column_stack([predictions[m][:min_len] for m in model_names])

        corr = np.corrcoef(data.T)
        return pd.DataFrame(corr, index=model_names, columns=model_names)

    def get_ensemble_predictions(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Get weighted ensemble predictions from all models.

        Args:
            weights: Optional model weights (defaults to equal weights)

        Returns:
            Array of ensemble probability predictions
        """
        predictions: Dict[str, np.ndarray] = {}
        for name, result in self.model_results.items():
            if "y_pred" in result:
                predictions[name] = np.array(result["y_pred"])

        if not predictions:
            return np.array([])

        # Default to equal weights
        if weights is None:
            weights = {name: 1.0 / len(predictions) for name in predictions}

        # Normalize weights
        total = sum(weights.get(name, 0) for name in predictions)
        if total == 0:
            return np.array([])

        # Find common length
        min_len = min(len(p) for p in predictions.values())

        # Calculate weighted average
        ensemble = np.zeros(min_len)
        for name, preds in predictions.items():
            w = weights.get(name, 0) / total
            ensemble += w * preds[:min_len]

        return ensemble
