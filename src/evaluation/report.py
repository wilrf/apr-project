"""Report generation utilities for model comparison."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """
    Generate comparison reports for model evaluation.

    Creates structured summaries and exports to various formats.
    """

    def __init__(self, model_results: Dict[str, Dict[str, Any]]):
        """
        Initialize report generator.

        Args:
            model_results: Dictionary mapping model names to their results.
                          Each result should have 'y_true', 'y_pred', and 'metrics'.
        """
        self.model_results = model_results
        self.model_names = list(model_results.keys())
        self._summary = None

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all model results.

        Returns:
            Dictionary containing model summaries and comparisons
        """
        if self._summary is not None:
            return self._summary

        summary = {
            "generated_at": datetime.now().isoformat(),
            "models": {},
            "comparison": {},
        }

        # Individual model summaries
        for name, result in self.model_results.items():
            summary["models"][name] = {
                "metrics": result.get("metrics", {}),
                "n_samples": len(result.get("y_true", [])),
            }

            # Add prediction statistics if available
            if "y_pred" in result:
                y_pred = np.array(result["y_pred"])
                summary["models"][name]["prediction_stats"] = {
                    "mean": float(y_pred.mean()),
                    "std": float(y_pred.std()),
                    "min": float(y_pred.min()),
                    "max": float(y_pred.max()),
                }

        # Model comparisons (if multiple models)
        if len(self.model_names) >= 2:
            summary["comparison"] = self._generate_comparison()

        self._summary = summary
        return summary

    def _generate_comparison(self) -> Dict[str, Any]:
        """Generate pairwise model comparisons."""
        comparison = {
            "metric_rankings": {},
            "pairwise": {},
        }

        # Collect all metrics
        all_metrics = set()
        for result in self.model_results.values():
            all_metrics.update(result.get("metrics", {}).keys())

        # Rank models by each metric
        for metric in all_metrics:
            scores = []
            for name, result in self.model_results.items():
                if metric in result.get("metrics", {}):
                    scores.append((name, result["metrics"][metric]))

            # Sort: higher is better for AUC, lower for Brier/loss
            reverse = metric in ["auc_roc", "accuracy", "roi"]
            scores.sort(key=lambda x: x[1], reverse=reverse)

            comparison["metric_rankings"][metric] = [
                {"model": name, "value": value} for name, value in scores
            ]

        # Pairwise prediction correlations
        if len(self.model_names) == 2:
            m1, m2 = self.model_names
            if "y_pred" in self.model_results[m1] and "y_pred" in self.model_results[m2]:
                pred1 = np.array(self.model_results[m1]["y_pred"])
                pred2 = np.array(self.model_results[m2]["y_pred"])

                if len(pred1) == len(pred2):
                    comparison["pairwise"]["prediction_correlation"] = float(
                        np.corrcoef(pred1, pred2)[0, 1]
                    )

                    # Agreement rate at threshold 0.5
                    agree = (pred1 >= 0.5) == (pred2 >= 0.5)
                    comparison["pairwise"]["agreement_rate"] = float(agree.mean())

        return comparison

    def export_markdown(
        self,
        output_path: Path,
        title: str = "NFL Upset Prediction: Model Comparison Report",
    ) -> None:
        """
        Export report as a markdown file.

        Args:
            output_path: Path to write the markdown file
            title: Title for the report
        """
        summary = self.generate_summary()
        output_path = Path(output_path)

        lines = [
            f"# {title}",
            "",
            f"*Generated: {summary['generated_at']}*",
            "",
            "## Model Performance Summary",
            "",
        ]

        # Model metrics table
        lines.append("| Model | AUC-ROC | Brier Score | Accuracy | Samples |")
        lines.append("|-------|---------|-------------|----------|---------|")

        for name, data in summary["models"].items():
            metrics = data["metrics"]
            auc = metrics.get("auc_roc", "N/A")
            brier = metrics.get("brier_score", "N/A")
            acc = metrics.get("accuracy", "N/A")
            n = data["n_samples"]

            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else auc
            brier_str = f"{brier:.4f}" if isinstance(brier, (int, float)) else brier
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else acc

            lines.append(f"| {name} | {auc_str} | {brier_str} | {acc_str} | {n:,} |")

        lines.append("")

        # Comparison section
        if summary["comparison"]:
            lines.append("## Model Comparison")
            lines.append("")

            # Rankings
            if summary["comparison"].get("metric_rankings"):
                lines.append("### Metric Rankings")
                lines.append("")

                for metric, rankings in summary["comparison"]["metric_rankings"].items():
                    lines.append(f"**{metric}:**")
                    for i, rank in enumerate(rankings, 1):
                        lines.append(f"  {i}. {rank['model']}: {rank['value']:.4f}")
                    lines.append("")

            # Pairwise
            if summary["comparison"].get("pairwise"):
                lines.append("### Pairwise Analysis")
                lines.append("")
                pairwise = summary["comparison"]["pairwise"]

                if "prediction_correlation" in pairwise:
                    lines.append(
                        f"- Prediction Correlation: {pairwise['prediction_correlation']:.3f}"
                    )
                if "agreement_rate" in pairwise:
                    lines.append(
                        f"- Classification Agreement: {pairwise['agreement_rate']:.1%}"
                    )
                lines.append("")

        # Individual model details
        lines.append("## Model Details")
        lines.append("")

        for name, data in summary["models"].items():
            lines.append(f"### {name}")
            lines.append("")

            if "prediction_stats" in data:
                stats = data["prediction_stats"]
                lines.append("**Prediction Distribution:**")
                lines.append(f"- Mean: {stats['mean']:.3f}")
                lines.append(f"- Std: {stats['std']:.3f}")
                lines.append(f"- Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                lines.append("")

        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))

    def export_dict(self) -> Dict[str, Any]:
        """
        Export report as a dictionary.

        Returns:
            Complete report as a dictionary
        """
        return self.generate_summary()


def generate_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to generate a report.

    Args:
        model_results: Model results dictionary
        output_path: Optional path to export markdown report

    Returns:
        Report summary dictionary
    """
    generator = ReportGenerator(model_results)

    if output_path:
        generator.export_markdown(output_path)

    return generator.generate_summary()
