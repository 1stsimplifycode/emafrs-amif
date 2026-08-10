"""Template-driven natural language explanation generation -- PDF Sec 6.8.3.

Composes a structured explanation from: a market direction summary, top contributing
financial indicators (with attribution magnitudes), modality contribution weights, current
regime, a confidence assessment, and (if available) a counterfactual statement. Every
number in the generated text is passed in from an actual computed value (IG attribution,
RAMA fusion weight, model confidence) -- this module only formats, it never invents.

WORDING RULE (project-wide, see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md's anti-
overclaiming note): attributions are described as "model-attributed evidence," never as
"the reason" or "the cause" -- SHAP/IG/attention are not causal explanations.
"""
from __future__ import annotations

from emafrs.xai.counterfactual import CounterfactualResult

DIRECTION_LABELS = {-1: "bearish", 0: "neutral", 1: "bullish"}


def generate_explanation(
    direction: int,
    confidence: float,
    regime_name: str,
    modality_weights: dict[str, float],
    top_financial_attributions: list[tuple[str, float]],
    counterfactual: CounterfactualResult | None = None,
    manipulation_risk: float | None = None,
) -> str:
    """All inputs must be real, already-computed values from the model/XAI pipeline --
    see tests/unit/test_nlg.py for the contract this function is held to (it must not
    silently fabricate a value for a missing input; callers omit what isn't available)."""
    parts = []

    direction_label = DIRECTION_LABELS.get(direction, "unknown")
    parts.append(f"Predicted market direction: {direction_label} (model confidence {confidence:.0%}).")
    parts.append(f"Current market regime: {regime_name}.")

    if modality_weights:
        ranked = sorted(modality_weights.items(), key=lambda kv: -kv[1])
        weight_str = ", ".join(f"{name} {weight:.0%}" for name, weight in ranked)
        parts.append(f"Modality contribution to this prediction: {weight_str}.")

    if top_financial_attributions:
        feat_str = ", ".join(
            f"{name} ({'+' if val >= 0 else ''}{val:.3f})" for name, val in top_financial_attributions[:5]
        )
        parts.append(f"Model-attributed evidence (top financial indicators, Integrated Gradients): {feat_str}.")

    if manipulation_risk is not None:
        parts.append(f"Manipulation risk score: {manipulation_risk:.2f}.")

    if counterfactual is not None:
        if counterfactual.flipped:
            parts.append(
                f"Counterfactual: had {counterfactual.feature_name} differed by "
                f"{counterfactual.delta:+.3f} (from {counterfactual.original_value:.3f} to "
                f"{counterfactual.counterfactual_value:.3f}), the model's prediction would "
                f"have changed. This describes model sensitivity, not a causal claim about "
                f"the underlying market."
            )
        else:
            parts.append(
                f"No perturbation of {counterfactual.feature_name} within the searched range "
                f"changed the model's prediction."
            )

    return " ".join(parts)
