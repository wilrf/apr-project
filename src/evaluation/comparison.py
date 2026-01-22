"""Model comparison utilities for XGBoost vs LSTM."""

from __future__ import annotations
import numpy as np
from typing import Dict, Any
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
