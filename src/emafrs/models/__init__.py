"""Baseline and fusion models -- build brief Sections 4, 22.

Model A (ModelAOpaqueXGBoost) / Model B (ModelBExplainableLogReg): baselines.py
Model C (StaticFusionModel) / Model D (RAMAFusionModel): fusion.py, trained via train.py
Model E (RAMA + XAI): Model D plus the explainability layer added in emafrs.xai (task 9) --
architecturally identical to Model D until that layer exists; not yet a distinct model.
"""
from emafrs.models.baselines import ModelAOpaqueXGBoost, ModelBExplainableLogReg
from emafrs.models.dataset import DatasetBundle, build_daily_dataset
from emafrs.models.fusion import RAMAFusionModel, StaticFusionModel
from emafrs.models.train import train_fusion_model

__all__ = [
    "ModelAOpaqueXGBoost",
    "ModelBExplainableLogReg",
    "DatasetBundle",
    "build_daily_dataset",
    "RAMAFusionModel",
    "StaticFusionModel",
    "train_fusion_model",
]
