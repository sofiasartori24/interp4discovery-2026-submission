"""Reseed-noise floor (Chanin 2026): how much of a ground-truth quantity's spread is
real signal between SAEs, versus noise from the probe's own random train/test split.

Only ``occupancy_balacc_*`` and ``legal_move_f1_*`` are recorded across seeds, because
they are fit with a train/test split (see ``pipeline/population.py``). Everything else is
recorded at ``seeds[0]`` only and reports as NaN here rather than a false zero.

Two caveats this module cannot fix, both of which belong in the paper rather than in the
numbers it returns:

1. ``causal_controllability_f1_*`` is **not** deterministic, despite being recorded at a
   single seed. ``groundtruth.causal_controllability.select_indices`` subsamples the rows
   it scores with ``default_rng(seed)``, so its stability across subsamples is simply
   unmeasured. Only ``next_move_fidelity_kl`` and the proxy metrics are genuinely
   deterministic given the SAE and the games.
2. The floor measures the variance of the probe's train/test split and nothing else. With
   a game-grouped split it is the honest version of that quantity; with the row-level
   split it is smaller still, because near-duplicate boards straddle the boundary. Either
   way it is the smallest variance component available and should not be reported as
   general reliability.
"""

from __future__ import annotations

from typing import cast

import pandas as pd


def reseed_noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Per ground-truth quantity name: within-SAE (across-seed) std vs between-SAE std.

    ``within_seed_std`` averages, over SAEs, each SAE's own standard deviation across its
    seeds, the noise floor a single-seed measurement would be subject to. ``between_sae_std``
    is the standard deviation, across SAEs, of each SAE's mean-over-seeds value, the actual
    signal a correlation or factor analysis is trying to use. ``signal_to_noise`` is their
    ratio: well below 1 means the quantity mostly reflects which random split it got, not a
    real difference between SAEs, and any correlation built on it should not be trusted.
    """
    gt = df[df["quantity_kind"] == "ground_truth"]
    per_sae = cast(
        pd.DataFrame, gt.groupby(["name", "sae_id"])["value"].agg(["mean", "std"])
    ).rename(columns={"mean": "sae_mean", "std": "sae_seed_std"})
    summary = cast(
        pd.DataFrame,
        per_sae.groupby("name").agg(
            within_seed_std=("sae_seed_std", "mean"),
            between_sae_std=("sae_mean", "std"),
        ),
    )
    summary["signal_to_noise"] = summary["between_sae_std"] / summary["within_seed_std"]
    return summary.sort_values("signal_to_noise")
