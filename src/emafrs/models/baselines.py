"""Baseline models A and B (build brief Sections 4, 22).

Model A -- high-performing opaque ML: gradient-boosted trees (XGBoost) over the full
financial feature set, no interpretability layer.
Model B -- explainable unimodal ML: multinomial logistic regression over the same
feature set -- interpretable by construction (coefficients ARE the explanation), the
"explainable but unimodal" anchor point in the accuracy/explainability comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

_LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
_IDX_TO_LABEL = {v: k for k, v in _LABEL_TO_IDX.items()}
CLASS_NAMES = ["bearish", "neutral", "bullish"]


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map(_LABEL_TO_IDX).to_numpy().astype(int)


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    macro_f1: float
    auc_ovr: float | None
    n_test: int
    class_distribution: dict


def evaluate_predictions(model_name: str, y_true_idx: np.ndarray, y_pred_idx: np.ndarray, y_proba: np.ndarray | None = None) -> EvalResult:
    acc = accuracy_score(y_true_idx, y_pred_idx)
    f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    auc = None
    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true_idx, classes=[0, 1, 2])
            auc = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))
        except ValueError:
            auc = None  # e.g. a class entirely absent from this test split
    dist = pd.Series(y_true_idx).value_counts().sort_index().to_dict()
    return EvalResult(model_name, float(acc), float(f1), auc, len(y_true_idx), dist)


class ModelAOpaqueXGBoost:
    """Model A: strong predictive accuracy, minimal interpretability."""

    def __init__(self, **xgb_params):
        import xgboost as xgb

        params = dict(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42,
        )
        params.update(xgb_params)
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelAOpaqueXGBoost":
        self.model.fit(X.to_numpy(), encode_labels(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.to_numpy())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy())


class ModelBExplainableLogReg:
    """Model B: strong financial model with built-in interpretability (linear
    coefficients), unimodal (financial features only)."""

    def __init__(self, **lr_params):
        params = dict(max_iter=3000, C=1.0, random_state=42)
        params.update(lr_params)
        self.model = LogisticRegression(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelBExplainableLogReg":
        self.model.fit(X.to_numpy(), encode_labels(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X.to_numpy())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy())

    def coefficients(self, feature_names: list[str]) -> pd.DataFrame:
        """Per-class linear coefficients -- the model's own built-in explanation.
        `self.model.classes_` are already in encode_labels' 0/1/2 index space, which by
        construction (see _LABEL_TO_IDX) is exactly CLASS_NAMES' order."""
        coefs = self.model.coef_  # (n_classes, n_features)
        classes = [CLASS_NAMES[c] for c in self.model.classes_]
        return pd.DataFrame(coefs, index=classes, columns=feature_names)
