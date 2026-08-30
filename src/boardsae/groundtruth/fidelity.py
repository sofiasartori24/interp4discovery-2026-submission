"""Next-move fidelity: does splicing the SAE reconstruction change the model's own
next-token behaviour?

Treated as ground truth, not a proxy metric, even though it is computed from a
reconstruction: it measures the model's own behaviour (the thing an SAE is supposed to
preserve), not a property of the SAE's activation-space geometry the way FVU is. Loss
recovered / this same splice is deliberately kept off the proxy-metric side for that
reason (see metrics/reconstruction.py), so no metric is validated against itself.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


@torch.no_grad()
def next_move_kl(
    model: HookedTransformer,
    reconstruct_fn: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    hook_name: str,
    device: str,
    batch_size: int = 32,
) -> float:
    """Mean KL(clean || spliced) between the model's next-token distributions with and
    without the SAE reconstruction spliced into the residual stream at ``hook_name``.

    ``reconstruct_fn`` maps a (batch*seq, d_model) activation slice, already on
    ``device``, to its SAE reconstruction (``decode(encode(x))``); kept as a callable so
    this function carries no SAE-specific imports. KL is taken as KL(clean || spliced),
    i.e. how much the reconstructed model's behaviour diverges from its own clean
    behaviour, matching "loss recovered" in direction (0 = perfect fidelity).
    """

    def splice_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        # transformer_lens's hook_points.full_hook calls hook(module_output, hook=self);
        # the parameter must be named exactly `hook` to match that keyword, so it can't
        # take the leading-underscore unused-arg convention ruff's ARG rule expects.
        shape = activation.shape
        flat = activation.reshape(-1, shape[-1])
        return reconstruct_fn(flat).reshape(shape)

    kls: list[float] = []
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size].to(device)
        clean_logits = model(batch)
        spliced_logits = model.run_with_hooks(
            batch, fwd_hooks=[(hook_name, splice_hook)]
        )
        clean_logp = F.log_softmax(clean_logits, dim=-1)
        spliced_logp = F.log_softmax(spliced_logits, dim=-1)
        # log_target=True with input=spliced_logp, target=clean_logp gives, per element,
        # exp(target) * (target - input) = KL(clean || spliced) once summed over vocab.
        per_token_kl = F.kl_div(
            spliced_logp, clean_logp, log_target=True, reduction="none"
        ).sum(dim=-1)
        kls.append(per_token_kl.mean().item())
    return float(sum(kls) / len(kls))
