# src/pearlmind/experiment/tracker.py
"""MLflow experiment tracking with fairness metrics."""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, experiment_name: str = "pearlmind"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
    
    def log_model_run(
        self,
        model,
        metrics: Dict[str, float],
        fairness_metrics: Dict[str, float],
        params: Dict[str, Any]
    ):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log standard metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log fairness metrics with prefix
            for key, value in fairness_metrics.items():
                mlflow.log_metric(f"fairness_{key}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
