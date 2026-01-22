"""MLflow integration utilities for experiment tracking."""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracker with graceful degradation when unavailable."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None,
                 enabled: bool = True, tracking_uri: Optional[str] = None):
        self.experiment_name, self.run_name = experiment_name, run_name
        self._run, self._mlflow = None, None
        self.enabled = enabled
        if enabled:
            try:
                import mlflow
                self._mlflow = mlflow
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
            except ImportError:
                self.enabled = False

    def _active(self) -> bool:
        return self.enabled and self._mlflow is not None

    def __enter__(self):
        if self._active():
            self._run = self._mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, *args):
        if self._active() and self._run:
            self._mlflow.end_run()
        return False

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._active():
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._active():
            self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._active():
            self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str) -> None:
        if self._active() and Path(local_path).exists():
            self._mlflow.log_artifact(local_path)

    def log_figure(self, figure, artifact_file: str) -> None:
        if self._active():
            self._mlflow.log_figure(figure, artifact_file)

    def set_tag(self, key: str, value: str) -> None:
        if self._active():
            self._mlflow.set_tag(key, value)

    def log_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None) -> None:
        if not self._active():
            return
        model_type = type(model).__name__
        model_to_log = model.model if hasattr(model, "model") else model

        flavors = []
        if "XGBoost" in model_type or hasattr(model, "model"):
            flavors.append(("mlflow.xgboost", model_to_log))
        if "LSTM" in model_type or "Module" in model_type:
            flavors.append(("mlflow.pytorch", model))
        flavors.append(("mlflow.sklearn", model))

        for flavor_name, model_obj in flavors:
            try:
                flavor = __import__(flavor_name, fromlist=["log_model"])
                flavor.log_model(model_obj, artifact_path, registered_model_name=registered_model_name)
                return
            except Exception as e:
                logger.debug(f"Could not log model with {flavor_name}: {e}")


def create_tracker(experiment_name: str = "nfl-upset-prediction", run_name: Optional[str] = None,
                   enabled: bool = True) -> MLflowTracker:
    """Factory function to create an MLflow tracker."""
    return MLflowTracker(experiment_name=experiment_name, run_name=run_name, enabled=enabled)
