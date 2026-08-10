"""Static multimodal fusion (Model C) and Regime-Adaptive Multimodal Attention (Model D
/ RAMA) -- PDF Section 6.7 / Summary "First Inventive Mechanism".

DISCLOSED FORMULA CORRECTION (docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md): the PDF's RAMA
softmax adds a single scalar regime index R_t identically inside every modality's exponent
(alpha_i(R_t) = softmax_i(I_i - lambda*L_i + gamma*R_t)). Since gamma*R_t does not depend on
i, it is a constant added to every logit before the softmax and cancels out algebraically --
the formula as literally printed cannot make attention weights regime-dependent. This
implementation instead learns a (n_regimes x n_modalities) gate matrix so the regime
conditioning is modality-specific: alpha_i depends on which modality i is AND which regime
is active. This is the smallest change that makes the named mechanism (regime-conditioned
attention) actually do what it is described as doing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticFusionModel(nn.Module):
    """Model C: modality embeddings combined with learned but NOT regime-conditioned
    weights -- a single softmax-over-modalities vector shared by every input row,
    regardless of market regime. This is the "static multimodal fusion" point the RAMA
    ablation (Model D) is contrasted against."""

    def __init__(self, modality_dims: dict[str, int], hidden_dim: int = 32, n_classes: int = 3):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.encoders = nn.ModuleDict(
            {name: ModalityEncoder(dim, hidden_dim) for name, dim in modality_dims.items()}
        )
        self.static_logits = nn.Parameter(torch.zeros(len(self.modality_names)))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, modality_inputs: dict[str, torch.Tensor]):
        embeddings = torch.stack(
            [self.encoders[name](modality_inputs[name]) for name in self.modality_names], dim=1
        )  # (batch, n_mod, hidden)
        weights = torch.softmax(self.static_logits, dim=0)  # (n_mod,) -- identical for every row
        fused = (embeddings * weights.view(1, -1, 1)).sum(dim=1)
        logits = self.head(fused)
        return logits, weights.unsqueeze(0).expand(embeddings.size(0), -1)


class RAMAFusionModel(nn.Module):
    """Model D: Regime-Adaptive Multimodal Attention. Fusion weight for modality i is a
    function of the current regime POSTERIOR (soft probabilities, not just the argmax
    label, so gradients flow through regime uncertainty as well) via a learned
    (n_regimes x n_modalities) gate matrix -- see module docstring for the disclosed
    correction to the PDF's literal (inert) formula."""

    def __init__(self, modality_dims: dict[str, int], n_regimes: int, hidden_dim: int = 32, n_classes: int = 3):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.encoders = nn.ModuleDict(
            {name: ModalityEncoder(dim, hidden_dim) for name, dim in modality_dims.items()}
        )
        self.gate = nn.Parameter(torch.zeros(n_regimes, len(self.modality_names)))
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, modality_inputs: dict[str, torch.Tensor], regime_probs: torch.Tensor):
        embeddings = torch.stack(
            [self.encoders[name](modality_inputs[name]) for name in self.modality_names], dim=1
        )  # (batch, n_mod, hidden)
        regime_gate_logits = regime_probs @ self.gate  # (batch, n_mod)
        weights = torch.softmax(regime_gate_logits, dim=1)  # (batch, n_mod) -- varies per row/regime
        fused = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.head(fused)
        return logits, weights
