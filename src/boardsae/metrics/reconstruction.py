"""Reconstruction-fidelity and utilization proxy metrics (subjects under audit)."""

from __future__ import annotations

import torch


def l1(feature_acts: torch.Tensor) -> float:
    """Mean L1 norm of the feature activations per token."""
    return feature_acts.abs().sum(dim=-1).mean().item()


def fraction_variance_unexplained(
    acts: torch.Tensor, reconstruction: torch.Tensor
) -> float:
    """FVU: residual variance over total variance, in activation space (lower is better).

    Computed directly on cached activations and the SAE's decode(encode(x)), so unlike
    next-move fidelity it needs no model forward pass. Loss recovered (the same splice
    used for the ground-truth fidelity quantity) is deliberately excluded from the proxy
    side: it is the same intervention that defines next-move fidelity, and auditing one
    against the other would validate a metric against itself.
    """
    residual_var = (acts - reconstruction).pow(2).sum(dim=-1)
    total_var = (acts - acts.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)
    return (residual_var.mean() / total_var.mean()).item()


def dead_feature_rate(feature_acts: torch.Tensor) -> float:
    """Fraction of dictionary features that never activate over this evaluation batch.

    A feature is "dead" if it is exactly zero for every token in the batch, so this is a
    property of the evaluation set size: too few tokens will overcount dead features
    (a documented threat to validity, not a bug).
    """
    ever_active = (feature_acts != 0).any(dim=0)
    return 1.0 - ever_active.float().mean().item()
