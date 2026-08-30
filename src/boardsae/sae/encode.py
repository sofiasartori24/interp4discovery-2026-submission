"""Encode cached residual activations into SAE feature activations."""

from __future__ import annotations

import torch


@torch.no_grad()
def encode_features(
    sae: torch.nn.Module,
    acts: torch.Tensor,
    device: str,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Encode ``acts`` of shape (N, d_model) into features of shape (N, dict_size).

    Batched with CPU offload so a full game set's activations fit in memory, mirroring the
    activation-cache path.
    """
    features = []
    for start in range(0, acts.shape[0], batch_size):
        batch = acts[start : start + batch_size].to(device)
        features.append(sae.encode(batch).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def decode_features(
    sae: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Decode SAE features of shape (N, dict_size) back into activation space (N, d_model).

    Paired with ``encode_features`` to get the reconstruction ``decode(encode(x))`` that
    the FVU proxy metric and the next-move-fidelity splice both need.
    """
    reconstruction = []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(device)
        reconstruction.append(sae.decode(batch).cpu())
    return torch.cat(reconstruction, dim=0)
