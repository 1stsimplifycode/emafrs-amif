"""End-to-end proof that the XAI layer (IG, counterfactual, NLG) works against a REAL
trained RAMA model from the modeling pipeline, not just synthetic torch tensors."""
import warnings

import numpy as np
import pytest
import torch

from emafrs.models.dataset import build_daily_dataset
from emafrs.models.train import train_fusion_model
from emafrs.regime.hmm_regime import REGIME_NAMES
from emafrs.xai.counterfactual import search_all_feature_counterfactuals
from emafrs.xai.integrated_gradients import compute_integrated_gradients
from emafrs.xai.nlg import generate_explanation

pytestmark = pytest.mark.integration
warnings.filterwarnings("ignore")


def test_full_xai_chain_on_real_trained_rama_model():
    bundle = build_daily_dataset("SPY", period="3y", horizon=1, deadzone=0.0025)
    modality_cols = {"financial": bundle.financial_columns}
    if len(bundle.microstructure_columns) > 1:
        modality_cols["microstructure"] = bundle.microstructure_columns

    model, result, _weights = train_fusion_model(bundle, "rama", modality_cols, epochs=20, patience=5)

    # take ONE real test-set row for the XAI walkthrough
    test_date = bundle.test_idx[0]
    modality_inputs = {
        name: torch.tensor(bundle.features.loc[[test_date], cols].to_numpy(), dtype=torch.float32)
        for name, cols in modality_cols.items()
    }
    regime_probs = torch.tensor(
        bundle.features.loc[[test_date], bundle.regime_columns].to_numpy(), dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        logits, weights = model(modality_inputs, regime_probs)
        proba = torch.softmax(logits, dim=1)[0]
        pred_class = int(proba.argmax())

    # 1. Integrated Gradients
    attrs = compute_integrated_gradients(
        model, modality_inputs, list(modality_cols), target_class=pred_class,
        regime_probs=regime_probs, n_steps=30,
    )
    financial_attrs = list(zip(bundle.financial_columns, attrs["financial"][0]))
    top_financial = sorted(financial_attrs, key=lambda kv: -abs(kv[1]))[:5]
    assert len(top_financial) == 5

    # 2. Counterfactual search over the financial modality (holding regime/microstructure fixed)
    fixed_regime = regime_probs
    fixed_extra = {k: v for k, v in modality_inputs.items() if k != "financial"}

    def predict_fn(financial_vec: np.ndarray) -> int:
        with torch.no_grad():
            inp = dict(fixed_extra)
            inp["financial"] = torch.tensor(financial_vec, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(inp, fixed_regime)
            return int(torch.softmax(logits, dim=1).argmax(dim=1).item())

    x0 = modality_inputs["financial"][0].numpy()
    cf_results = search_all_feature_counterfactuals(
        predict_fn, x0, bundle.financial_columns, search_range=(-6.0, 6.0), n_steps=25
    )
    top_cf = cf_results[0] if cf_results else None

    # 3. NLG
    regime_idx = int(regime_probs.argmax(dim=1).item()) + 1  # regime columns are 1-indexed
    regime_name = REGIME_NAMES.get(regime_idx, str(regime_idx))
    modality_weight_dict = dict(zip(modality_cols.keys(), weights[0].tolist()))

    text = generate_explanation(
        direction=[-1, 0, 1][pred_class],
        confidence=float(proba[pred_class]),
        regime_name=regime_name,
        modality_weights=modality_weight_dict,
        top_financial_attributions=top_financial,
        counterfactual=top_cf,
    )

    assert isinstance(text, str) and len(text) > 20
    assert regime_name in text
    assert "confidence" in text
    print(f"\nReal generated explanation:\n{text}")
