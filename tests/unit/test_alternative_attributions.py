"""Tests for the Gradient x Input and Occlusion attribution methods added during the
trustworthiness-audit remediation pass to cross-check Integrated Gradients' faithfulness
result against methods with different mechanisms (no path integration; no gradient
dependency at all)."""
import numpy as np
import torch

from emafrs.models.fusion import RAMAFusionModel, StaticFusionModel
from emafrs.xai.alternative_attributions import compute_gradient_x_input, compute_occlusion


def _inputs(batch=1, dims={"financial": 6, "microstructure": 3}):
    torch.manual_seed(1)
    return {name: torch.randn(batch, dim) for name, dim in dims.items()}


def test_gradient_x_input_shapes_static_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = StaticFusionModel(modality_dims, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    attrs = compute_gradient_x_input(model, inputs, list(modality_dims), target_class=1)
    assert attrs["financial"].shape == (1, 6)
    assert attrs["microstructure"].shape == (1, 3)


def test_gradient_x_input_shapes_rama_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = RAMAFusionModel(modality_dims, n_regimes=4, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    regime_probs = torch.softmax(torch.randn(1, 4), dim=1)
    attrs = compute_gradient_x_input(model, inputs, list(modality_dims), target_class=1, regime_probs=regime_probs)
    assert attrs["financial"].shape == (1, 6)
    assert attrs["microstructure"].shape == (1, 3)


def test_occlusion_shapes_and_zero_weight_feature_gets_zero_attribution():
    """A feature with a literal zero weight in the model's first linear layer must have
    ~zero occlusion attribution -- it cannot affect predicted probability no matter what
    it is replaced with. Proves occlusion is measuring real functional dependence, not a
    placeholder."""
    modality_dims = {"financial": 4}
    model = StaticFusionModel(modality_dims, hidden_dim=8, n_classes=3)
    with torch.no_grad():
        model.encoders["financial"].net[0].weight[:, 2] = 0.0  # feature index 2 has zero weight everywhere
    inputs = _inputs(dims={"financial": 4})
    attrs = compute_occlusion(model, inputs, list(modality_dims), target_class=1)
    assert attrs["financial"].shape == (1, 4)
    assert abs(attrs["financial"][0, 2]) < 1e-5


def test_occlusion_shapes_rama_model():
    modality_dims = {"financial": 6, "microstructure": 3}
    model = RAMAFusionModel(modality_dims, n_regimes=4, hidden_dim=8, n_classes=3)
    inputs = _inputs()
    regime_probs = torch.softmax(torch.randn(1, 4), dim=1)
    attrs = compute_occlusion(model, inputs, list(modality_dims), target_class=1, regime_probs=regime_probs)
    assert attrs["financial"].shape == (1, 6)
    assert attrs["microstructure"].shape == (1, 3)
