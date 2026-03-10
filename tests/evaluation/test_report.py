# tests/evaluation/test_report.py
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestReportGenerator:
    @pytest.fixture
    def sample_results(self):
        """Create sample model results for report."""
        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.35, n)

        return {
            "xgboost": {
                "y_true": y_true,
                "y_pred": np.clip(y_true + np.random.normal(0, 0.2, n), 0, 1),
                "metrics": {
                    "auc_roc": 0.72,
                    "brier_score": 0.21,
                    "accuracy": 0.68,
                },
            },
            "lstm": {
                "y_true": y_true,
                "y_pred": np.clip(y_true + np.random.normal(0, 0.25, n), 0, 1),
                "metrics": {
                    "auc_roc": 0.68,
                    "brier_score": 0.23,
                    "accuracy": 0.65,
                },
            },
        }

    def test_report_generator_creates_summary(self, sample_results):
        """Test that report generator creates a summary."""
        from src.evaluation.report import ReportGenerator

        generator = ReportGenerator(sample_results)
        summary = generator.generate_summary()

        assert "models" in summary
        assert "comparison" in summary

    def test_report_generator_exports_markdown(self, sample_results):
        """Test that report can be exported as markdown."""
        from src.evaluation.report import ReportGenerator

        generator = ReportGenerator(sample_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "xgboost" in content.lower() or "XGBoost" in content

    def test_report_includes_all_metrics(self, sample_results):
        """Test that report includes all metrics."""
        from src.evaluation.report import ReportGenerator

        generator = ReportGenerator(sample_results)
        summary = generator.generate_summary()

        for model_name, model_data in summary["models"].items():
            assert "auc_roc" in model_data["metrics"]
            assert "brier_score" in model_data["metrics"]

    def test_report_handles_single_model(self):
        """Test report works with single model."""
        from src.evaluation.report import ReportGenerator

        results = {
            "xgboost": {
                "y_true": np.array([1, 0, 1, 0]),
                "y_pred": np.array([0.7, 0.3, 0.8, 0.2]),
                "metrics": {"auc_roc": 0.75},
            }
        }

        generator = ReportGenerator(results)
        summary = generator.generate_summary()

        assert "xgboost" in summary["models"]

    def test_threshold_conflict_raises_valueerror(self):
        """Regression: if explicit threshold contradicts analyzer threshold, raise.

        ReportGenerator delegates to disagreement_analyzer for the disagreement
        section, which uses its own threshold internally. If the caller also
        passes an explicit threshold that differs, one report would contain two
        different binary-classification regimes — silently inconsistent.
        """
        from unittest.mock import MagicMock

        from src.evaluation.report import ReportGenerator

        analyzer = MagicMock()
        analyzer.threshold = 0.30

        results = {
            "model_a": {
                "y_true": np.array([1, 0]),
                "y_pred": np.array([0.6, 0.4]),
                "metrics": {"auc_roc": 0.5},
            }
        }

        # Same threshold — should be fine
        gen = ReportGenerator(results, disagreement_analyzer=analyzer, threshold=0.30)
        assert gen.threshold == 0.30

        # Conflicting threshold — must raise
        with pytest.raises(ValueError, match="threshold"):
            ReportGenerator(results, disagreement_analyzer=analyzer, threshold=0.45)

    def test_agreement_uses_base_rate_threshold_not_half(self):
        """H6 regression: agreement must use base-rate threshold, not 0.5.

        With y_true=[1,0,0], base rate = 1/3 ≈ 0.333.
        At threshold 0.333:
          model_a: [0.35→1, 0.35→1, 0.10→0]
          model_b: [0.25→0, 0.40→1, 0.10→0]
          agree on games 2 and 3 → 2/3 ≈ 0.667

        At threshold 0.5 (old bug):
          both: [0,0,0] → agree on all 3 → 1.0
        """
        from src.evaluation.report import ReportGenerator

        y_true = np.array([1, 0, 0])
        results = {
            "model_a": {
                "y_true": y_true,
                "y_pred": np.array([0.35, 0.35, 0.10]),
                "metrics": {"auc_roc": 0.6},
            },
            "model_b": {
                "y_true": y_true,
                "y_pred": np.array([0.25, 0.40, 0.10]),
                "metrics": {"auc_roc": 0.6},
            },
        }
        generator = ReportGenerator(results)
        summary = generator.generate_summary()
        pair = summary["comparison"]["pairwise"]["model_a_vs_model_b"]
        # Must be ~0.667, not 1.0
        assert pair["agreement_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_generate_summary_handles_empty_prediction_arrays(self):
        from src.evaluation.report import ReportGenerator

        generator = ReportGenerator(
            {
                "xgboost": {
                    "y_true": np.array([], dtype=int),
                    "y_pred": np.array([], dtype=float),
                    "metrics": {"auc_roc": 0.5},
                }
            }
        )

        summary = generator.generate_summary()

        assert "prediction_stats" not in summary["models"]["xgboost"]

    def test_pairwise_correlation_stays_finite_for_constant_predictions(self):
        from src.evaluation.report import ReportGenerator

        y_true = np.array([1, 0, 1])
        generator = ReportGenerator(
            {
                "a": {"y_true": y_true, "y_pred": np.array([0.2, 0.2, 0.2])},
                "b": {"y_true": y_true, "y_pred": np.array([0.4, 0.4, 0.4])},
            }
        )

        summary = generator.generate_summary()
        pair = summary["comparison"]["pairwise"]["a_vs_b"]
        assert np.isfinite(pair["prediction_correlation"])
        assert pair["prediction_correlation"] == 0.0
