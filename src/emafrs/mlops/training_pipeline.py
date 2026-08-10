"""End-to-end training + live prediction pipeline: build dataset -> train RAMA -> persist
checkpoint -> compute XAI for the most recent available row -> persist the prediction.

This is the function the API's predict endpoint calls (synchronously, for a demo-scale
deployment) and that a scheduled background job (emafrs.mlops.scheduler) calls
periodically for a production-like refresh pattern. Both paths go through this one
function so there is a single source of truth for "how a prediction gets made."
"""
from __future__ import annotations

import logging

import torch

from emafrs.db.models import PredictionRecord
from emafrs.db.session import get_session
from emafrs.mlops.model_registry import save_model
from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model
from emafrs.regime.hmm_regime import REGIME_NAMES
from emafrs.xai.counterfactual import search_all_feature_counterfactuals
from emafrs.xai.integrated_gradients import compute_integrated_gradients
from emafrs.xai.nlg import generate_explanation

logger = logging.getLogger("emafrs.mlops")

DIRECTION_FROM_CLASS_IDX = {0: -1, 1: 0, 2: 1}


def run_training_and_prediction_pipeline(
    ticker: str, epochs: int = 100, patience: int = 12, seed: int = 42
) -> PredictionRecord:
    """Trains a fresh RAMA model on `ticker` and produces a persisted prediction (with
    explanation) for the most recent row in the dataset. Real computation, real data --
    nothing here is templated or precomputed."""
    logger.info("Building dataset for %s", ticker)
    bundle = build_daily_dataset(ticker, period="5y", horizon=1, deadzone=0.0025)

    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns

    logger.info("Training RAMA model for %s", ticker)
    model, result, _weights = train_fusion_model(
        bundle, "rama", modality_cols, epochs=epochs, patience=patience, seed=seed
    )

    modality_dims = {name: len(cols) for name, cols in modality_cols.items()}
    version = save_model(
        model, "rama", ticker, modality_dims, n_regimes=len(bundle.regime_columns),
        hidden_dim=32, n_classes=3,
        metrics={"accuracy": result.accuracy, "macro_f1": result.macro_f1, "auc_ovr": result.auc_ovr, "n_test": result.n_test},
    )
    logger.info("Saved model version %s (accuracy=%.3f)", version.id, result.accuracy)

    # Predict on the MOST RECENT available row (the closest thing to "today" this dataset supports)
    latest_date = bundle.features.index.max()
    modality_inputs = {
        name: torch.tensor(bundle.features.loc[[latest_date], cols].to_numpy(), dtype=torch.float32)
        for name, cols in modality_cols.items()
    }
    regime_probs = torch.tensor(
        bundle.features.loc[[latest_date], bundle.regime_columns].to_numpy(), dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        logits, weights = model(modality_inputs, regime_probs)
        proba = torch.softmax(logits, dim=1)[0]
        pred_class = int(proba.argmax())

    attrs = compute_integrated_gradients(
        model, modality_inputs, list(modality_cols), target_class=pred_class,
        regime_probs=regime_probs, n_steps=30,
    )
    financial_attrs = list(zip(bundle.financial_columns, attrs["financial"][0]))
    top_financial = sorted(financial_attrs, key=lambda kv: -abs(kv[1]))[:5]

    fixed_extra = {k: v for k, v in modality_inputs.items() if k != "financial"}

    def predict_fn(financial_vec):
        with torch.no_grad():
            inp = dict(fixed_extra)
            inp["financial"] = torch.tensor(financial_vec, dtype=torch.float32).unsqueeze(0)
            logit, _ = model(inp, regime_probs)
            return int(torch.softmax(logit, dim=1).argmax(dim=1).item())

    x0 = modality_inputs["financial"][0].numpy()
    cf_results = search_all_feature_counterfactuals(
        predict_fn, x0, bundle.financial_columns, search_range=(-6.0, 6.0), n_steps=25
    )
    top_cf = cf_results[0] if cf_results else None

    regime_idx = int(regime_probs.argmax(dim=1).item()) + 1
    regime_name = REGIME_NAMES.get(regime_idx, str(regime_idx))
    modality_weight_dict = dict(zip(modality_cols.keys(), weights[0].tolist()))

    explanation = generate_explanation(
        direction=DIRECTION_FROM_CLASS_IDX[pred_class],
        confidence=float(proba[pred_class]),
        regime_name=regime_name,
        modality_weights=modality_weight_dict,
        top_financial_attributions=top_financial,
        counterfactual=top_cf,
    )

    session = get_session()
    try:
        record = PredictionRecord(
            ticker=ticker,
            as_of_date=str(latest_date.date()),
            direction=DIRECTION_FROM_CLASS_IDX[pred_class],
            confidence=float(proba[pred_class]),
            regime=regime_idx,
            regime_name=regime_name,
            modality_weights_json=modality_weight_dict,
            top_attributions_json={name: float(val) for name, val in top_financial},
            counterfactual_json=(
                {
                    "feature_name": top_cf.feature_name, "delta": top_cf.delta,
                    "original_value": top_cf.original_value, "counterfactual_value": top_cf.counterfactual_value,
                }
                if top_cf is not None else None
            ),
            manipulation_risk=None,  # computed separately once real hype/microstructure signals are wired end-to-end live
            explanation_text=explanation,
            model_version_id=version.id,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()
