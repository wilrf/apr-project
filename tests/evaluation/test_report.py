# tests/evaluation/test_report.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


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
