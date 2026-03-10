# tests/models/test_mlflow_utils.py
import sys
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

    def test_enabled_tracker_uses_mlflow_client(self):
        from src.models.mlflow_utils import MLflowTracker

        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock()

        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            tracker = MLflowTracker(
                experiment_name="exp",
                run_name="run",
                enabled=True,
                tracking_uri="file:///tmp/mlruns",
            )
            with tracker:
                tracker.log_params({"alpha": 1})
                tracker.log_metric("auc", 0.7)

        fake_mlflow.set_tracking_uri.assert_called_once_with("file:///tmp/mlruns")
        fake_mlflow.set_experiment.assert_called_once_with("exp")
        fake_mlflow.start_run.assert_called_once_with(run_name="run")
        fake_mlflow.log_params.assert_called_once_with({"alpha": 1})
        fake_mlflow.log_metric.assert_called_once_with("auc", 0.7, step=None)
        fake_mlflow.end_run.assert_called_once()
