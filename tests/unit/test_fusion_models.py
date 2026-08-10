"""Tests for Model C (static fusion) and Model D (RAMA) on synthetic data -- no network,
no real dataset needed. Proves the disclosed formula correction actually makes RAMA's
fusion weights regime-dependent while StaticFusionModel's are not."""
import torch

from emafrs.models.fusion import RAMAFusionModel, StaticFusionModel


def _dummy_inputs(batch=16, dims={"financial": 5, "microstructure": 3}):
    torch.manual_seed(0)
    return {name: torch.randn(batch, dim) for name, dim in dims.items()}


def test_static_fusion_weights_identical_across_all_rows():
    inputs = _dummy_inputs()
    model = StaticFusionModel({"financial": 5, "microstructure": 3}, hidden_dim=8, n_classes=3)
    logits, weights = model(inputs)
    assert logits.shape == (16, 3)
    assert weights.shape == (16, 2)
    # every row must have the EXACT same weight vector -- that's the definition of "static"
    assert torch.allclose(weights, weights[0].unsqueeze(0).expand_as(weights))


def test_static_fusion_weights_sum_to_one():
    inputs = _dummy_inputs()
    model = StaticFusionModel({"financial": 5, "microstructure": 3}, hidden_dim=8)
    _, weights = model(inputs)
    assert torch.allclose(weights.sum(dim=1), torch.ones(16), atol=1e-5)


def test_rama_weights_differ_across_regimes():
    """The core claim under test: RAMA's fusion weight for a modality actually changes
    with the regime posterior, unlike the PDF's literal (regime-invariant) softmax
    formula. Two batches identical in modality inputs but different in regime one-hot
    must, after training pushes the gate away from its zero-init, produce DIFFERENT
    fusion weights -- verified here by manually setting a non-trivial gate."""
    torch.manual_seed(0)
    model = RAMAFusionModel({"financial": 5, "microstructure": 3}, n_regimes=4, hidden_dim=8)
    # Force a non-trivial, clearly regime-dependent gate (bypassing training, to isolate
    # the mechanism itself from optimization behavior).
    with torch.no_grad():
        model.gate.copy_(torch.tensor([
            [5.0, -5.0],   # regime 0: strongly prefer financial
            [5.0, -5.0],
            [-5.0, 5.0],   # regime 2: strongly prefer microstructure
            [-5.0, 5.0],
        ]))

    inputs = _dummy_inputs(batch=1)
    regime_low_vol = torch.tensor([[1.0, 0.0, 0.0, 0.0]])   # regime 0 -> favors financial
    regime_high_vol = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # regime 2 -> favors microstructure

    _, w_low = model(inputs, regime_low_vol)
    _, w_high = model(inputs, regime_high_vol)

    assert not torch.allclose(w_low, w_high)
    assert w_low[0, 0] > w_low[0, 1]    # regime 0: financial weight > microstructure weight
    assert w_high[0, 1] > w_high[0, 0]  # regime 2: microstructure weight > financial weight


def test_rama_weights_sum_to_one_per_row():
    inputs = _dummy_inputs(batch=10)
    model = RAMAFusionModel({"financial": 5, "microstructure": 3}, n_regimes=4, hidden_dim=8)
    regime_probs = torch.softmax(torch.randn(10, 4), dim=1)
    _, weights = model(inputs, regime_probs)
    assert torch.allclose(weights.sum(dim=1), torch.ones(10), atol=1e-5)


def test_rama_forward_pass_is_differentiable():
    inputs = _dummy_inputs(batch=4)
    model = RAMAFusionModel({"financial": 5, "microstructure": 3}, n_regimes=4, hidden_dim=8)
    regime_probs = torch.softmax(torch.randn(4, 4), dim=1)
    logits, _ = model(inputs, regime_probs)
    loss = logits.sum()
    loss.backward()
    assert model.gate.grad is not None
    assert torch.isfinite(model.gate.grad).all()
