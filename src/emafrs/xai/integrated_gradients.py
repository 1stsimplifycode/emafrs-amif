"""Integrated Gradients feature attribution -- PDF Sec 6.8.1, Eq. 19, via Captum, applied
to the trained RAMA/static fusion models from emafrs.models.fusion.

IG satisfies the completeness axiom: attributions sum to (output at input) - (output at
baseline). This module's tests verify that property directly on real trained-model output
rather than merely asserting it, since a wrapper bug could silently break it.
"""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients


class _PositionalWrapper(torch.nn.Module):
    """Captum's IntegratedGradients needs a model callable with one tensor per input, not
    a dict. Wraps a fusion model's dict-based forward into positional args over a fixed
    modality order (+ a trailing regime-probs tensor for RAMA models)."""

    def __init__(self, model, modality_names: list[str], target_class: int, is_rama: bool):
        super().__init__()
        self.model = model
        self.modality_names = modality_names
        self.target_class = target_class
        self.is_rama = is_rama

    def forward(self, *tensors):
        if self.is_rama:
            *modality_tensors, regime_tensor = tensors
            inputs = dict(zip(self.modality_names, modality_tensors))
            logits, _ = self.model(inputs, regime_tensor)
        else:
            inputs = dict(zip(self.modality_names, tensors))
            logits, _ = self.model(inputs)
        return torch.softmax(logits, dim=1)[:, self.target_class]


def compute_integrated_gradients(
    model: torch.nn.Module,
    modality_inputs: dict[str, torch.Tensor],
    modality_names: list[str],
    target_class: int,
    regime_probs: torch.Tensor | None = None,
    n_steps: int = 50,
) -> dict[str, np.ndarray]:
    """Returns per-modality attribution arrays (batch, n_features_in_modality). Pass
    `regime_probs` only for a RAMAFusionModel; leave None for StaticFusionModel."""
    is_rama = regime_probs is not None
    wrapper = _PositionalWrapper(model, modality_names, target_class, is_rama)
    wrapper.eval()
    ig = IntegratedGradients(wrapper)

    tensors = tuple(modality_inputs[name].clone().detach().requires_grad_(True) for name in modality_names)
    baselines = tuple(torch.zeros_like(t) for t in tensors)
    if is_rama:
        tensors = tensors + (regime_probs.clone().detach(),)
        baselines = baselines + (torch.full_like(regime_probs, 1.0 / regime_probs.shape[1]),)

    attributions = ig.attribute(tensors, baselines=baselines, n_steps=n_steps)

    result = {name: attributions[i].detach().numpy() for i, name in enumerate(modality_names)}
    if is_rama:
        result["regime"] = attributions[-1].detach().numpy()
    return result


def check_completeness(
    model: torch.nn.Module,
    modality_inputs: dict[str, torch.Tensor],
    modality_names: list[str],
    target_class: int,
    regime_probs: torch.Tensor | None = None,
    n_steps: int = 50,
) -> np.ndarray:
    """Per-row |sum(attributions) - (f(x) - f(baseline))|, the completeness-axiom residual.
    Should be small (Captum's Riemann-sum IG approximation is not exact, so a small
    nonzero residual is expected, not a bug -- see the test's tolerance)."""
    is_rama = regime_probs is not None
    wrapper = _PositionalWrapper(model, modality_names, target_class, is_rama)
    wrapper.eval()

    attrs = compute_integrated_gradients(model, modality_inputs, modality_names, target_class, regime_probs, n_steps)
    total_attr = sum(attrs[name].sum(axis=1) for name in modality_names)
    if is_rama:
        total_attr = total_attr + attrs["regime"].sum(axis=1)

    with torch.no_grad():
        tensors = tuple(modality_inputs[name] for name in modality_names)
        if is_rama:
            f_x = wrapper(*tensors, regime_probs).numpy()
            baselines = tuple(torch.zeros_like(t) for t in tensors) + (
                torch.full_like(regime_probs, 1.0 / regime_probs.shape[1]),
            )
            f_baseline = wrapper(*baselines).numpy()
        else:
            f_x = wrapper(*tensors).numpy()
            baselines = tuple(torch.zeros_like(t) for t in tensors)
            f_baseline = wrapper(*baselines).numpy()

    return np.abs(total_attr - (f_x - f_baseline))
