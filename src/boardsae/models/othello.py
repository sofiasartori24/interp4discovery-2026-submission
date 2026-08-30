"""OthelloGPT loading via transformer_lens."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

OTHELLO_GPT_ID = "Baidicoot/Othello-GPT-Transformer-Lens"


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to cuda, then mps, then cpu; pass any explicit value through."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_othello_gpt(
    device: str = "auto", model_id: str = OTHELLO_GPT_ID
) -> HookedTransformer:
    """Load the OthelloGPT world model (the checkpoint SAE_BoardGameEval uses)."""
    model = HookedTransformer.from_pretrained(model_id)
    # nn.Module.to() mutates in place and returns self, but its stub widens the return
    # type to the base Module, not the subtype; use the mutation, not the return value,
    # so the declared HookedTransformer type is preserved for pyright.
    model.to(resolve_device(device))
    return model


@torch.no_grad()
def cache_resid_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Cache activations at one hook, in batches, returning (N, L, D) on CPU.

    Batched and under no_grad so memory stays bounded: without no_grad, run_with_cache
    retains the autograd graph for the whole batch and blows up RAM.
    """
    chunks: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name)
        chunks.append(cache[hook_name].detach().to("cpu"))
    return torch.cat(chunks, dim=0)
