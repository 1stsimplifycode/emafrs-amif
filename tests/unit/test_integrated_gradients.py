"""Tests for the Captum-based Integrated Gradients wrapper against real (untrained-but-
differentiable) StaticFusionModel/RAMAFusionModel instances -- checking shapes and, most
importantly, the completeness axiom that IS the whole point of using IG over an ad-hoc
attribution method."""
import numpy as np
import torch

from emafrs.models.fusion import RAMAFusionModel, StaticFusionModel
from emafrs.xai.integrated_gradients import check_completeness, compute_integrated_gradients


def _inputs(batch=4, dims={"financial": 6, "microstructure": 3}):
    torch.manual_seed(1)
    return {name: torch.randn(batch, dim) for name, dim in dims.items()}


def test_ig_shapes_match_input_for_static_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = StaticFusionModel(modality_dims, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    attrs = compute_integrated_gradients(model, inputs, list(modality_dims), target_class=1, n_steps=20)
    assert attrs["financial"].shape == (4, 6)
    assert attrs["microstructure"].shape == (4, 3)


def test_ig_shapes_match_input_for_rama_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = RAMAFusionModel(modality_dims, n_regimes=4, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    regime_probs = torch.softmax(torch.randn(4, 4), dim=1)
    attrs = compute_integrated_gradients(
        model, inputs, list(modality_dims), target_class=1, regime_probs=regime_probs, n_steps=20
    )
    assert attrs["financial"].shape == (4, 6)
    assert attrs["microstructure"].shape == (4, 3)
    assert attrs["regime"].shape == (4, 4)


def test_ig_satisfies_completeness_axiom_static_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = StaticFusionModel(modality_dims, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    residual = check_completeness(model, inputs, list(modality_dims), target_class=1, n_steps=100)
    assert np.all(residual < 1e-3)


def test_ig_satisfies_completeness_axiom_rama_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = RAMAFusionModel(modality_dims, n_regimes=4, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    regime_probs = torch.softmax(torch.randn(4, 4), dim=1)
    residual = check_completeness(
        model, inputs, list(modality_dims), target_class=1, regime_probs=regime_probs, n_steps=100
    )
    assert np.all(residual < 1e-3)
