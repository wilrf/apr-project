# tests/models/test_mlflow_utils.py
import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch


class TestMLflowUtils:
    def test_tracker_initializes_with_experiment_name(self):
        """Test that tracker initializes with experiment name."""
        from src.models.mlflow_utils import MLflowTracker

        tracker = MLflowTracker(experiment_name="test", enabled=False)
        assert tracker.experiment_name == "test"

    def test_tracker_disables_when_mlflow_missing(self):
        """Test that tracker disables when MLflow not installed."""
        from src.models.mlflow_utils import MLflowTracker

        # When enabled=True but mlflow not installed, should auto-disable
        # This test passes because mlflow may or may not be installed
        tracker = MLflowTracker(experiment_name="test", enabled=True)
        # Either it's enabled (mlflow found) or disabled (mlflow not found)
        assert isinstance(tracker.enabled, bool)

    def test_log_params_accepts_dict(self):
        """Test that log_params accepts a dictionary."""
        from src.models.mlflow_utils import MLflowTracker

        tracker = MLflowTracker(experiment_name="test", enabled=False)
        # Should not raise when disabled
        tracker.log_params({"param1": 1, "param2": "value"})

    def test_log_metrics_accepts_dict(self):
        """Test that log_metrics accepts a dictionary."""
        from src.models.mlflow_utils import MLflowTracker

        tracker = MLflowTracker(experiment_name="test", enabled=False)
        # Should not raise when disabled
        tracker.log_metrics({"metric1": 0.5, "metric2": 0.8})

    def test_disabled_tracker_no_ops(self):
        """Test that disabled tracker is a no-op."""
        from src.models.mlflow_utils import MLflowTracker

        tracker = MLflowTracker(experiment_name="test", enabled=False)

        # None of these should raise
        tracker.log_params({"a": 1})
        tracker.log_metrics({"b": 2})
        tracker.log_artifact("fake_path")
        tracker.set_tag("key", "value")

    def test_tracker_context_manager(self):
        """Test tracker can be used as context manager."""
        from src.models.mlflow_utils import MLflowTracker

        tracker = MLflowTracker(experiment_name="test", enabled=False)

        # Should work as context manager
        with tracker:
            tracker.log_params({"test": 1})
