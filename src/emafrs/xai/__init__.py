"""Explainability layer -- PDF Section 6.8: Integrated Gradients feature attribution,
counterfactual explanation, and template-driven natural language generation."""
from emafrs.xai.counterfactual import CounterfactualResult, search_all_feature_counterfactuals
from emafrs.xai.integrated_gradients import check_completeness, compute_integrated_gradients
from emafrs.xai.nlg import generate_explanation

__all__ = [
    "CounterfactualResult", "search_all_feature_counterfactuals",
    "check_completeness", "compute_integrated_gradients", "generate_explanation",
]
