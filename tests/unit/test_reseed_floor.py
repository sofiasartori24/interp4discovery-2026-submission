"""Unit tests for validity/reseed_floor.py, on synthetic multi-seed long-format data."""

from __future__ import annotations

import pandas as pd

from boardsae.validity.reseed_floor import reseed_noise_floor

_COMMON = {
    "architecture": "StandardTrainer",
    "width": 4096,
    "l0": 100.0,
    "model_name": "m",
    "layer": 5,
}


def _record(
    sae_id: str, name: str, seed: int, value: float, quantity_kind: str = "ground_truth"
) -> dict:
    return {
        "sae_id": sae_id,
        "name": name,
        "seed": seed,
        "value": value,
        "quantity_kind": quantity_kind,
        **_COMMON,
    }


def test_low_within_sae_noise_and_high_between_sae_spread_gives_high_snr() -> None:
    # "good_metric" barely moves across seeds within an SAE, but differs a lot between
    # SAEs: real signal, low noise.
    rows = []
    for sae_id, base in [("sae_a", 0.9), ("sae_b", 0.5), ("sae_c", 0.1)]:
        for seed, jitter in enumerate([0.0, 0.001, -0.001]):
            rows.append(_record(sae_id, "good_metric", seed, base + jitter))
    df = pd.DataFrame(rows)
    summary = reseed_noise_floor(df)
    assert summary.loc["good_metric", "signal_to_noise"] > 50


def test_high_within_sae_noise_and_low_between_sae_spread_gives_low_snr() -> None:
    # "bad_metric" barely differs between SAEs, but swings a lot across seeds within
    # each SAE: mostly reseed noise, not a real discriminator.
    rows = []
    for sae_id in ["sae_a", "sae_b", "sae_c"]:
        for seed, jitter in enumerate([0.4, -0.4, 0.0]):
            rows.append(_record(sae_id, "bad_metric", seed, 0.5 + jitter))
    df = pd.DataFrame(rows)
    summary = reseed_noise_floor(df)
    assert summary.loc["bad_metric", "signal_to_noise"] < 1


def test_single_seed_quantity_reports_nan_not_a_false_zero() -> None:
    # Deterministic quantities (e.g. next_move_fidelity_kl) are only ever recorded at
    # seeds[0]; with one seed per SAE there is no within-SAE spread to measure.
    rows = [
        _record(sae_id, "next_move_fidelity_kl", 0, v)
        for sae_id, v in [("sae_a", 0.01), ("sae_b", 0.5)]
    ]
    df = pd.DataFrame(rows)
    summary = reseed_noise_floor(df)
    assert pd.isna(summary.loc["next_move_fidelity_kl", "within_seed_std"])
    assert pd.isna(summary.loc["next_move_fidelity_kl", "signal_to_noise"])


def test_proxy_metrics_are_excluded() -> None:
    rows = [_record("sae_a", "l0", 0, 100.0, quantity_kind="proxy_metric")]
    df = pd.DataFrame(rows)
    summary = reseed_noise_floor(df)
    assert "l0" not in summary.index
