"""Sparsity proxy metrics (subjects under audit)."""

from __future__ import annotations

import torch


def l0(feature_acts: torch.Tensor) -> float:
    """Mean number of active (nonzero) features per token.

    ``feature_acts`` has shape (..., n_features); the mean is over all leading dims.
    """
    active = (feature_acts != 0).float().sum(dim=-1)
    return active.mean().item()
