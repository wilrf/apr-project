"""Report generation utilities for model comparison."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.evaluation.metrics import safe_probability_correlation

if TYPE_CHECKING:
    from src.evaluation.disagreement import DisagreementAnalyzer


def _fmt(v, precision=4):
    """Format metric value."""
    return f"{v:.{precision}f}" if isinstance(v, (int, float)) else "N/A"


class ReportGenerator:
    """Generate comparison reports for model evaluation."""

    def __init__(
        self,
        model_results: Dict[str, Dict[str, Any]],
        disagreement_analyzer: Optional["DisagreementAnalyzer"] = None,
        threshold: Optional[float] = None,
    ):
        self.model_results = model_results
        self.model_names = list(model_results.keys())
        self.disagreement_analyzer = disagreement_analyzer

        # Threshold: explicit > analyzer > compute from data.
        # If both are provided they must agree, otherwise the report would
        # use one threshold for its own comparisons and a different one for
        # the disagreement section (which delegates to the analyzer).
        if threshold is not None and disagreement_analyzer is not None:
            if not np.isclose(threshold, disagreement_analyzer.threshold):
                raise ValueError(
                    f"Explicit threshold ({threshold}) conflicts with "
                    "disagreement_analyzer.threshold "
                    f"({disagreement_analyzer.threshold}). "
                    f"Pass only one, or ensure they match."
                )
            self.threshold = threshold
        elif threshold is not None:
            self.threshold = threshold
        elif disagreement_analyzer is not None:
            self.threshold = disagreement_analyzer.threshold
        else:
            # Compute base rate from y_true across models
            all_y = [
                y for r in model_results.values() if "y_true" in r for y in r["y_true"]
            ]
            self.threshold = float(np.mean(all_y)) if all_y else 0.5

        self._summary = None

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all model results."""
        if self._summary:
            return self._summary

        summary = {
            "generated_at": datetime.now().isoformat(),
            "models": {},
            "comparison": {},
        }

        for name, result in self.model_results.items():
            model_summary = {
                "metrics": result.get("metrics", {}),
                "n_samples": len(result.get("y_true", [])),
            }
            if "y_pred" in result:
                y = np.array(result["y_pred"])
                if len(y) > 0:
                    model_summary["prediction_stats"] = {
                        "mean": float(y.mean()),
                        "std": float(y.std()),
                        "min": float(y.min()),
                        "max": float(y.max()),
                    }
            summary["models"][name] = model_summary

        if len(self.model_names) >= 2:
            summary["comparison"] = self._generate_comparison()

        if self.disagreement_analyzer:
            summary["disagreement"] = self._generate_disagreement_summary()

        self._summary = summary
        return summary

    def _generate_comparison(self) -> Dict[str, Any]:
        """Generate pairwise model comparisons."""
        comparison = {
            "metric_rankings": {},
            "pairwise": {},
            "agreement": {},
            "correlation": {},
        }
        all_metrics = set(
            m for r in self.model_results.values() for m in r.get("metrics", {})
        )
        higher_is_better = {"auc_roc", "accuracy", "roi"}

        for metric in all_metrics:
            scores = [
                (n, r["metrics"][metric])
                for n, r in self.model_results.items()
                if metric in r.get("metrics", {})
            ]
            scores.sort(key=lambda x: x[1], reverse=(metric in higher_is_better))
            comparison["metric_rankings"][metric] = [
                {"model": n, "value": v} for n, v in scores
            ]

        # Calculate pairwise agreement and correlation for all model pairs
        predictions: Dict[str, np.ndarray] = {}
        for name, result in self.model_results.items():
            if "y_pred" in result:
                predictions[name] = np.array(result["y_pred"])

        if len(predictions) >= 2:
            min_len = min(len(p) for p in predictions.values())
            predictions = {k: v[:min_len] for k, v in predictions.items()}

            for model_a, model_b in combinations(predictions.keys(), 2):
                pa, pb = predictions[model_a], predictions[model_b]
                pair_key = f"{model_a}_vs_{model_b}"
                comparison["pairwise"][pair_key] = {
                    "prediction_correlation": safe_probability_correlation(pa, pb),
                    "agreement_rate": float(
                        ((pa >= self.threshold) == (pb >= self.threshold)).mean()
                    ),
                }

            # All-model agreement if 3+ models
            if len(predictions) >= 3:
                preds_matrix = np.column_stack(
                    [
                        (predictions[m] >= self.threshold).astype(int)
                        for m in predictions
                    ]
                )
                all_agree = (preds_matrix.std(axis=1) == 0).mean()
                comparison["agreement"]["all_models"] = float(all_agree)

        return comparison

    def _generate_disagreement_summary(self) -> Dict[str, Any]:
        """Generate disagreement analysis summary."""
        if not self.disagreement_analyzer:
            return {}

        analyzer = self.disagreement_analyzer
        categories = analyzer.categorize_all()

        summary = {
            "category_counts": {
                cat.value: len(preds) for cat, preds in categories.items()
            },
            "category_stats": analyzer.get_category_stats().to_dict(orient="records"),
            "agreement_matrix": analyzer.get_agreement_matrix().to_dict(
                orient="records"
            ),
            "correlation_matrix": analyzer.get_correlation_matrix().to_dict(),
        }

        # Add exclusive insights
        insights = analyzer.get_exclusive_insights()
        summary["exclusive_insights"] = {}
        for cat, insight in insights.items():
            summary["exclusive_insights"][cat.value] = {
                "description": insight.description,
                "key_features": insight.key_features,
                "example_count": len(insight.example_games),
            }

        return summary

    def export_markdown(
        self,
        output_path: Path,
        title: str = "NFL Upset Prediction: Model Comparison Report",
    ) -> None:
        """Export report as markdown file."""
        s = self.generate_summary()
        lines = [
            f"# {title}",
            "",
            f"*Generated: {s['generated_at']}*",
            "",
            "## Model Performance Summary",
            "",
            "| Model | AUC-ROC | Brier Score | Log Loss | Samples |",
            "|-------|---------|-------------|----------|---------|",
        ]

        for name, data in s["models"].items():
            m = data["metrics"]
            lines.append(
                f"| {name} | {_fmt(m.get('auc_roc'))} | {_fmt(m.get('brier_score'))} | "
                f"{_fmt(m.get('log_loss'))} | {data['n_samples']:,} |"
            )
        lines.append("")

        if s["comparison"]:
            lines.extend(["## Model Comparison", ""])
            if s["comparison"].get("metric_rankings"):
                lines.extend(["### Metric Rankings", ""])
                for metric, rankings in s["comparison"]["metric_rankings"].items():
                    lines.append(f"**{metric}:**")
                    lines.extend(
                        f"  {i}. {r['model']}: {r['value']:.4f}"
                        for i, r in enumerate(rankings, 1)
                    )
                    lines.append("")

            if s["comparison"].get("pairwise"):
                lines.extend(["### Pairwise Analysis", ""])
                for pair, stats in s["comparison"]["pairwise"].items():
                    lines.append(f"**{pair}:**")
                    if "prediction_correlation" in stats:
                        lines.append(
                            "  - Prediction Correlation: "
                            f"{stats['prediction_correlation']:.3f}"
                        )
                    if "agreement_rate" in stats:
                        lines.append(
                            "  - Classification Agreement: "
                            f"{stats['agreement_rate']:.1%}"
                        )
                    lines.append("")

            if s["comparison"].get("agreement", {}).get("all_models"):
                lines.append(
                    "**All Models Agree:** "
                    f"{s['comparison']['agreement']['all_models']:.1%} "
                    "of predictions"
                )
                lines.append("")

        # Disagreement analysis section
        if s.get("disagreement"):
            lines.extend(self._format_disagreement_section(s["disagreement"]))

        lines.extend(["## Model Details", ""])
        for name, data in s["models"].items():
            lines.extend([f"### {name}", ""])
            if "prediction_stats" in data:
                st = data["prediction_stats"]
                lines.extend(
                    [
                        "**Prediction Distribution:**",
                        f"- Mean: {st['mean']:.3f}",
                        f"- Std: {st['std']:.3f}",
                        f"- Range: [{st['min']:.3f}, {st['max']:.3f}]",
                        "",
                    ]
                )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(lines))

    def _format_disagreement_section(self, disagreement: Dict[str, Any]) -> List[str]:
        """Format disagreement analysis for markdown."""
        lines = [
            "## Disagreement Analysis",
            "",
            "This section analyzes where models agree and disagree, "
            "revealing each model's structural biases.",
            "",
            "### Category Breakdown",
            "",
            "| Category | Count | % of Total | Upset Rate |",
            "|----------|-------|------------|------------|",
        ]

        for stat in disagreement.get("category_stats", []):
            lines.append(
                f"| {stat['category']} | {stat['count']} | "
                f"{stat['pct_of_total']:.1f}% | "
                f"{stat['upset_rate']:.1%} |"
            )
        lines.append("")

        # Agreement matrix
        if disagreement.get("agreement_matrix"):
            lines.extend(
                [
                    "### Model Agreement Rates",
                    "",
                    "| Model Pair | Agreement Rate |",
                    "|------------|---------------|",
                ]
            )
            for row in disagreement["agreement_matrix"]:
                lines.append(f"| {row['model_pair']} | {row['agreement_rate']:.1%} |")
            lines.append("")

        # Exclusive insights
        if disagreement.get("exclusive_insights"):
            lines.extend(
                [
                    "### Key Findings: Exclusive Correct Predictions",
                    "",
                    "These categories reveal where only one model captures the signal:",
                    "",
                ]
            )
            for cat, insight in disagreement["exclusive_insights"].items():
                lines.extend(
                    [
                        f"**{cat.upper()}** ({insight['example_count']} games)",
                        "",
                        f"> {insight['description']}",
                        "",
                    ]
                )

        return lines

    def export_dict(self) -> Dict[str, Any]:
        return self.generate_summary()


def generate_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
    disagreement_analyzer: Optional["DisagreementAnalyzer"] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience function to generate a report."""
    gen = ReportGenerator(model_results, disagreement_analyzer, threshold=threshold)
    if output_path:
        gen.export_markdown(output_path)
    return gen.generate_summary()
