"""Alternative feature-attribution methods, added during the trustworthiness-audit
remediation pass specifically to test whether Integrated Gradients' near-zero faithfulness
result (docs/XAI_EVALUATION.md) is a property of IG specifically or of this model's
decision surface in general. If a removal-based method (Occlusion) and a different
gradient-based method (Gradient x Input) show the SAME near-zero faithfulness, that is
evidence the problem is the model/decision-surface, not the attribution method. If they
diverge, that is evidence IG specifically is unsuited to this architecture.

Both methods reuse the same `_PositionalWrapper` used by
`emafrs.xai.integrated_gradients` so results are directly comparable (same model call
convention, same target-class softmax output).
"""
from __future__ import annotations

import numpy as np
import torch

from emafrs.xai.integrated_gradients import _PositionalWrapper


def compute_gradient_x_input(
    model: torch.nn.Module,
    modality_inputs: dict[str, torch.Tensor],
    modality_names: list[str],
    target_class: int,
    regime_probs: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Gradient x Input: the single backward pass at the actual input, scaled by the
    input itself (Shrikumar et al. 2017). Unlike IG, this does NOT integrate over a path
    from a baseline -- it only reflects the LOCAL gradient at x. If IG's path-integration
    is what causes its faithfulness problem (e.g. because of saturation somewhere along
    the path), Gradient x Input should behave differently; if the local gradient itself is
    uninformative, both will fail similarly."""
    is_rama = regime_probs is not None
    wrapper = _PositionalWrapper(model, modality_names, target_class, is_rama)
    wrapper.eval()

    tensors = tuple(modality_inputs[name].clone().detach().requires_grad_(True) for name in modality_names)
    call_tensors = tensors + ((regime_probs.clone().detach(),) if is_rama else ())

    out = wrapper(*call_tensors)
    out.sum().backward()

    result = {}
    for name, t in zip(modality_names, tensors):
        grad = t.grad.detach().numpy()
        result[name] = grad * t.detach().numpy()
    return result


def compute_occlusion(
    model: torch.nn.Module,
    modality_inputs: dict[str, torch.Tensor],
    modality_names: list[str],
    target_class: int,
    regime_probs: torch.Tensor | None = None,
    baseline_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """Occlusion (Zeiler & Fergus 2014, feature-wise): a pure REMOVAL-based attribution --
    for each feature independently, replace it with `baseline_value` and measure the drop
    in target-class probability. Unlike IG or Gradient x Input, this has no gradient
    dependency at all, so it cannot be affected by gradient saturation -- a useful
    cross-check for whether IG's faithfulness problem is a saturation artifact.

    Batch size must be 1 (one row at a time) -- each feature is occluded independently per
    row, and this implementation is not vectorized across a batch."""
    is_rama = regime_probs is not None
    wrapper = _PositionalWrapper(model, modality_names, target_class, is_rama)
    wrapper.eval()

    with torch.no_grad():
        tensors = tuple(modality_inputs[name] for name in modality_names)
        call_tensors = tensors + ((regime_probs,) if is_rama else ())
        base_out = wrapper(*call_tensors).item()

        result = {}
        for name, t in zip(modality_names, tensors):
            n_features = t.shape[1]
            attrs = np.zeros(n_features, dtype=np.float32)
            for i in range(n_features):
                perturbed = t.clone()
                perturbed[0, i] = baseline_value
                perturbed_tensors = tuple(
                    perturbed if n2 == name else modality_inputs[n2] for n2 in modality_names
                )
                perturbed_call = perturbed_tensors + ((regime_probs,) if is_rama else ())
                perturbed_out = wrapper(*perturbed_call).item()
                attrs[i] = base_out - perturbed_out  # positive = feature supports target_class
            result[name] = attrs.reshape(1, -1)
    return result
