"""Shared activation + label collection for the read-out pipelines and sanity tests."""

from __future__ import annotations

import numpy as np
import torch
from transformer_lens import HookedTransformer

from boardsae.models.othello import cache_resid_activations


def collect_acts_and_labels(
    model: HookedTransformer,
    games: list[list[int]],
    boards_BLRRC: torch.Tensor,
    hook_name: str,
    device: str,
    batch_size: int,
    square: tuple[int, int],
) -> tuple[torch.Tensor, np.ndarray]:
    """Cache residual activations and the occupancy label for one square.

    Returns activations flattened to (N, d_model) as CPU float (ready to feed an SAE), and
    the integer occupancy class per token (N,), aligned to the cached sequence length.
    """
    tokens = torch.tensor(games)
    acts_BLD = cache_resid_activations(model, tokens, hook_name, device, batch_size)
    n, length = acts_BLD.shape[0], acts_BLD.shape[1]
    acts = acts_BLD.reshape(n * length, -1).float()
    r, c = square
    labels = (
        boards_BLRRC[:, :length, r, c, :].argmax(dim=-1).reshape(n * length).numpy()
    )
    return acts, labels
