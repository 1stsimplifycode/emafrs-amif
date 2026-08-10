"""MLOps: training pipeline, model versioning, drift monitoring -- build brief Section 35.

Pipeline: data ingestion (emafrs.data) -> feature generation (emafrs.features) ->
training (emafrs.models.train) -> evaluation -> model versioning (model_registry.py) ->
inference (training_pipeline.py, called by the API) -> monitoring (drift.py).
"""
from emafrs.mlops.drift import DriftReport, detect_feature_drift
from emafrs.mlops.model_registry import load_latest_model, save_model
from emafrs.mlops.training_pipeline import run_training_and_prediction_pipeline

__all__ = [
    "DriftReport", "detect_feature_drift",
    "load_latest_model", "save_model", "run_training_and_prediction_pipeline",
]
