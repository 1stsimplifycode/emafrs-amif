from emafrs.xai.counterfactual import CounterfactualResult
from emafrs.xai.nlg import generate_explanation


def test_generate_explanation_includes_all_provided_sections():
    cf = CounterfactualResult(
        feature_name="rsi_14", original_value=0.5, counterfactual_value=1.2, delta=0.7,
        original_prediction=2, counterfactual_prediction=1, flipped=True,
    )
    text = generate_explanation(
        direction=1, confidence=0.72, regime_name="High",
        modality_weights={"financial": 0.6, "microstructure": 0.4},
        top_financial_attributions=[("rsi_14", 0.12), ("macd_hist", -0.05)],
        counterfactual=cf, manipulation_risk=0.31,
    )
    assert "bullish" in text
    assert "72%" in text
    assert "High" in text
    assert "financial 60%" in text
    assert "rsi_14" in text
    assert "manipulation" in text.lower()
    assert "0.31" in text
    assert "causal claim" in text  # anti-overclaiming wording must be present


def test_generate_explanation_omits_missing_sections_gracefully():
    text = generate_explanation(
        direction=-1, confidence=0.55, regime_name="Low",
        modality_weights={}, top_financial_attributions=[],
    )
    assert "bearish" in text
    assert "Modality contribution" not in text
    assert "Model-attributed evidence" not in text


def test_generate_explanation_reports_no_flip_found_honestly():
    cf = CounterfactualResult(
        feature_name="volume_zscore", original_value=0.1, counterfactual_value=None,
        delta=None, original_prediction=1, counterfactual_prediction=None, flipped=False,
    )
    text = generate_explanation(
        direction=0, confidence=0.4, regime_name="Medium",
        modality_weights={"financial": 1.0}, top_financial_attributions=[], counterfactual=cf,
    )
    assert "No perturbation" in text
