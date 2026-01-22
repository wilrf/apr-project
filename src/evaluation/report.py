"""Report generation utilities for model comparison."""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


def _fmt(v, precision=4):
    """Format metric value."""
    return f"{v:.{precision}f}" if isinstance(v, (int, float)) else "N/A"


class ReportGenerator:
    """Generate comparison reports for model evaluation."""

    def __init__(self, model_results: Dict[str, Dict[str, Any]]):
        self.model_results = model_results
        self.model_names = list(model_results.keys())
        self._summary = None

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all model results."""
        if self._summary:
            return self._summary

        summary = {"generated_at": datetime.now().isoformat(), "models": {}, "comparison": {}}

        for name, result in self.model_results.items():
            model_summary = {"metrics": result.get("metrics", {}), "n_samples": len(result.get("y_true", []))}
            if "y_pred" in result:
                y = np.array(result["y_pred"])
                model_summary["prediction_stats"] = {"mean": float(y.mean()), "std": float(y.std()),
                                                     "min": float(y.min()), "max": float(y.max())}
            summary["models"][name] = model_summary

        if len(self.model_names) >= 2:
            summary["comparison"] = self._generate_comparison()

        self._summary = summary
        return summary

    def _generate_comparison(self) -> Dict[str, Any]:
        """Generate pairwise model comparisons."""
        comparison = {"metric_rankings": {}, "pairwise": {}}
        all_metrics = set(m for r in self.model_results.values() for m in r.get("metrics", {}))
        higher_is_better = {"auc_roc", "accuracy", "roi"}

        for metric in all_metrics:
            scores = [(n, r["metrics"][metric]) for n, r in self.model_results.items() if metric in r.get("metrics", {})]
            scores.sort(key=lambda x: x[1], reverse=(metric in higher_is_better))
            comparison["metric_rankings"][metric] = [{"model": n, "value": v} for n, v in scores]

        if len(self.model_names) == 2:
            r1, r2 = [self.model_results[m] for m in self.model_names]
            if "y_pred" in r1 and "y_pred" in r2:
                p1, p2 = np.array(r1["y_pred"]), np.array(r2["y_pred"])
                if len(p1) == len(p2):
                    comparison["pairwise"]["prediction_correlation"] = float(np.corrcoef(p1, p2)[0, 1])
                    comparison["pairwise"]["agreement_rate"] = float(((p1 >= 0.5) == (p2 >= 0.5)).mean())

        return comparison

    def export_markdown(self, output_path: Path, title: str = "NFL Upset Prediction: Model Comparison Report") -> None:
        """Export report as markdown file."""
        s = self.generate_summary()
        lines = [f"# {title}", "", f"*Generated: {s['generated_at']}*", "", "## Model Performance Summary", "",
                 "| Model | AUC-ROC | Brier Score | Accuracy | Samples |", "|-------|---------|-------------|----------|---------|"]

        for name, data in s["models"].items():
            m = data["metrics"]
            lines.append(f"| {name} | {_fmt(m.get('auc_roc'))} | {_fmt(m.get('brier_score'))} | "
                        f"{_fmt(m.get('accuracy'))} | {data['n_samples']:,} |")
        lines.append("")

        if s["comparison"]:
            lines.extend(["## Model Comparison", ""])
            if s["comparison"].get("metric_rankings"):
                lines.extend(["### Metric Rankings", ""])
                for metric, rankings in s["comparison"]["metric_rankings"].items():
                    lines.append(f"**{metric}:**")
                    lines.extend(f"  {i}. {r['model']}: {r['value']:.4f}" for i, r in enumerate(rankings, 1))
                    lines.append("")

            if s["comparison"].get("pairwise"):
                lines.extend(["### Pairwise Analysis", ""])
                pw = s["comparison"]["pairwise"]
                if "prediction_correlation" in pw:
                    lines.append(f"- Prediction Correlation: {pw['prediction_correlation']:.3f}")
                if "agreement_rate" in pw:
                    lines.append(f"- Classification Agreement: {pw['agreement_rate']:.1%}")
                lines.append("")

        lines.extend(["## Model Details", ""])
        for name, data in s["models"].items():
            lines.extend([f"### {name}", ""])
            if "prediction_stats" in data:
                st = data["prediction_stats"]
                lines.extend(["**Prediction Distribution:**", f"- Mean: {st['mean']:.3f}",
                             f"- Std: {st['std']:.3f}", f"- Range: [{st['min']:.3f}, {st['max']:.3f}]", ""])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines))

    def export_dict(self) -> Dict[str, Any]:
        return self.generate_summary()


def generate_report(model_results: Dict[str, Dict[str, Any]], output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to generate a report."""
    gen = ReportGenerator(model_results)
    if output_path:
        gen.export_markdown(output_path)
    return gen.generate_summary()
