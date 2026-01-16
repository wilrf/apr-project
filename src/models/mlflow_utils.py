"""MLflow integration utilities for experiment tracking."""

from __future__ import annotations

from typing import Dict, Any, Optional
from pathlib import Path


class MLflowTracker:
    """
    MLflow experiment tracker with optional enable/disable.

    Provides a unified interface for logging experiments,
    with graceful degradation when MLflow is not available
    or tracking is disabled.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        enabled: bool = True,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this specific run
            enabled: Whether to actually log to MLflow
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self._run = None
        self._mlflow = None

        if self.enabled:
            try:
                import mlflow
                self._mlflow = mlflow

                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)

                mlflow.set_experiment(experiment_name)
            except ImportError:
                self.enabled = False

    def __enter__(self):
        """Start MLflow run."""
        if self.enabled and self._mlflow:
            self._run = self._mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        if self.enabled and self._mlflow and self._run:
            self._mlflow.end_run()
        return False

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameter names to values
        """
        if self.enabled and self._mlflow:
            self._mlflow.log_params(params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number for the metrics
        """
        if self.enabled and self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a single metric to MLflow.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.enabled and self._mlflow:
            self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str) -> None:
        """
        Log an artifact file to MLflow.

        Args:
            local_path: Path to the artifact file
        """
        if self.enabled and self._mlflow:
            path = Path(local_path)
            if path.exists():
                self._mlflow.log_artifact(str(path))

    def log_figure(self, figure, artifact_file: str) -> None:
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure object
            artifact_file: Filename for the artifact
        """
        if self.enabled and self._mlflow:
            self._mlflow.log_figure(figure, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag on the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        if self.enabled and self._mlflow:
            self._mlflow.set_tag(key, value)

    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log a model to MLflow.

        Args:
            model: Model object to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: Optional name to register the model
        """
        if self.enabled and self._mlflow:
            # Detect model type and use appropriate flavor
            model_type = type(model).__name__

            if "XGBoost" in model_type or hasattr(model, "model"):
                # XGBoost model
                try:
                    import mlflow.xgboost
                    if hasattr(model, "model"):
                        mlflow.xgboost.log_model(
                            model.model,
                            artifact_path,
                            registered_model_name=registered_model_name,
                        )
                    else:
                        mlflow.xgboost.log_model(
                            model,
                            artifact_path,
                            registered_model_name=registered_model_name,
                        )
                except Exception:
                    # Fallback to generic sklearn
                    import mlflow.sklearn
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path,
                        registered_model_name=registered_model_name,
                    )
            elif "LSTM" in model_type or "Module" in model_type:
                # PyTorch model
                try:
                    import mlflow.pytorch
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path,
                        registered_model_name=registered_model_name,
                    )
                except Exception:
                    pass
            else:
                # Generic sklearn
                try:
                    import mlflow.sklearn
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path,
                        registered_model_name=registered_model_name,
                    )
                except Exception:
                    pass


def create_tracker(
    experiment_name: str = "nfl-upset-prediction",
    run_name: Optional[str] = None,
    enabled: bool = True,
) -> MLflowTracker:
    """
    Factory function to create an MLflow tracker.

    Args:
        experiment_name: Name of the experiment
        run_name: Optional run name
        enabled: Whether tracking is enabled

    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(
        experiment_name=experiment_name,
        run_name=run_name,
        enabled=enabled,
    )
