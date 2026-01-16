"""Model comparison utilities for XGBoost vs LSTM."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from itertools import combinations


class ModelComparison:
    """
    Compare multiple models on the same data.

    Supports pairwise comparisons of model performance metrics
    and prediction correlation analysis.
    """

    def __init__(self, model_results: Dict[str, Dict[str, Any]]):
        """
        Initialize comparison.

        Args:
            model_results: Dictionary mapping model names to their results.
                          Each result should have 'y_true', 'y_pred', and 'metrics'.
        """
        self.model_results = model_results
        self.model_names = list(model_results.keys())

    def compare(self) -> Dict[str, Dict[str, float]]:
        """
        Run pairwise model comparisons.

        Returns:
            Dictionary of comparison results for each pair
        """
        if len(self.model_names) < 2:
            return {}

        comparisons = {}

        for model_a, model_b in combinations(self.model_names, 2):
            key = f"{model_a}_vs_{model_b}"
            comparisons[key] = self._compare_pair(model_a, model_b)

        return comparisons

    def _compare_pair(
        self,
        model_a: str,
        model_b: str,
    ) -> Dict[str, float]:
        """
        Compare two models.

        Args:
            model_a: First model name
            model_b: Second model name

        Returns:
            Dictionary with comparison metrics
        """
        result_a = self.model_results[model_a]
        result_b = self.model_results[model_b]

        comparison = {}

        # Compare all shared metrics
        metrics_a = result_a.get("metrics", {})
        metrics_b = result_b.get("metrics", {})

        for metric in set(metrics_a.keys()) & set(metrics_b.keys()):
            comparison[f"{metric}_diff"] = metrics_a[metric] - metrics_b[metric]

        # Prediction correlation
        if "y_pred" in result_a and "y_pred" in result_b:
            pred_a = np.array(result_a["y_pred"])
            pred_b = np.array(result_b["y_pred"])

            if len(pred_a) == len(pred_b):
                comparison["prediction_correlation"] = float(
                    np.corrcoef(pred_a, pred_b)[0, 1]
                )

        return comparison

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all models' performance.

        Returns:
            Summary dictionary with rankings
        """
        summary = {
            "models": {},
            "rankings": {},
        }

        # Collect all metrics
        all_metrics = set()
        for name, result in self.model_results.items():
            metrics = result.get("metrics", {})
            summary["models"][name] = metrics
            all_metrics.update(metrics.keys())

        # Rank models by each metric
        for metric in all_metrics:
            scores = []
            for name, result in self.model_results.items():
                if metric in result.get("metrics", {}):
                    scores.append((name, result["metrics"][metric]))

            # Higher is better for AUC, lower is better for loss/brier
            reverse = metric in ["auc_roc", "accuracy", "roi"]
            scores.sort(key=lambda x: x[1], reverse=reverse)

            summary["rankings"][metric] = [name for name, _ in scores]

        return summary
