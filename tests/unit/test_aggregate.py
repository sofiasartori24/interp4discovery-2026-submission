"""Unit tests for discovery/aggregate.py: pure pandas, no model or GPU needed."""

from __future__ import annotations

import pandas as pd

from boardsae.discovery.aggregate import ground_truth_composites, proxy_metric_table

_COMMON = {
    "architecture": "StandardTrainer",
    "width": 4096,
    "l0": 100.0,
    "model_name": "m",
    "layer": 5,
    "seed": 0,
}


def _record(sae_id: str, quantity_kind: str, name: str, value: float) -> dict:
    return {
        "sae_id": sae_id,
        "quantity_kind": quantity_kind,
        "name": name,
        "value": value,
        **_COMMON,
    }


def test_ground_truth_composites_average_within_region_not_across_regions() -> None:
    df = pd.DataFrame(
        [
            _record("sae_a", "ground_truth", "occupancy_balacc_r0c0", 0.9),  # corner
            _record("sae_a", "ground_truth", "occupancy_balacc_r7c7", 0.7),  # corner
            _record("sae_a", "ground_truth", "occupancy_balacc_r3c3", 0.5),  # centre
            _record("sae_a", "ground_truth", "occupancy_balacc_r0c3", 0.6),  # edge
            _record("sae_a", "ground_truth", "next_move_fidelity_kl", 0.01),
        ]
    )
    composites = ground_truth_composites(df)
    assert (
        composites.loc["sae_a", "occupancy_balacc_corner_mean"] == 0.8
    )  # (0.9 + 0.7) / 2
    assert composites.loc["sae_a", "occupancy_balacc_centre_mean"] == 0.5
    assert composites.loc["sae_a", "occupancy_balacc_edge_mean"] == 0.6
    assert composites.loc["sae_a", "next_move_fidelity_kl"] == 0.01


def test_legal_move_has_no_centre_column_when_no_centre_records_exist() -> None:
    # Mirrors the real population run: centre squares are permanently excluded from
    # legal-move recovery (they are occupied from move zero and stay occupied), so no
    # legal_move_f1_r3c3-style record is ever produced for them.
    df = pd.DataFrame(
        [
            _record("sae_a", "ground_truth", "legal_move_f1_r0c0", 0.8),
            _record("sae_a", "ground_truth", "legal_move_f1_r0c3", 0.85),
        ]
    )
    composites = ground_truth_composites(df)
    assert "legal_move_f1_centre_mean" not in composites.columns
    assert composites.loc["sae_a", "legal_move_f1_corner_mean"] == 0.8
    assert composites.loc["sae_a", "legal_move_f1_edge_mean"] == 0.85


def test_causal_controllability_gets_a_centre_composite_unlike_legal_move() -> None:
    # Unlike legal_move_f1, centre is a valid (the richest) edit target for
    # causal-controllability, so it should get all three region composites.
    df = pd.DataFrame(
        [
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_sae_r0c0",
                0.9,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_sae_r7c7",
                0.7,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_sae_r3c3",
                0.5,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_sae_r0c3",
                0.6,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_logreg_sae_r0c0",
                0.4,
            ),
        ]
    )
    composites = ground_truth_composites(df)
    assert (
        composites.loc["sae_a", "causal_controllability_f1_meandiff_sae_corner_mean"]
        == 0.8
    )
    assert (
        composites.loc["sae_a", "causal_controllability_f1_meandiff_sae_centre_mean"]
        == 0.5
    )
    assert (
        composites.loc["sae_a", "causal_controllability_f1_meandiff_sae_edge_mean"]
        == 0.6
    )
    assert (
        composites.loc["sae_a", "causal_controllability_f1_logreg_sae_corner_mean"]
        == 0.4
    )


def test_causal_controllability_null_random_and_oracle_variants_are_excluded() -> None:
    # Only the real SAE-space edit belongs in the ground-truth composites: oracle is
    # constant across every SAE (would zero out its own variance in the correlation
    # matrix), and null/random are validity controls, not a quality axis.
    df = pd.DataFrame(
        [
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_sae_r0c0",
                0.9,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_null_sae_r0c0",
                0.1,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_random_sae_r0c0",
                0.2,
            ),
            _record(
                "sae_a",
                "ground_truth",
                "causal_controllability_f1_meandiff_oracle_r0c0",
                0.95,
            ),
        ]
    )
    composites = ground_truth_composites(df)
    assert (
        composites.loc["sae_a", "causal_controllability_f1_meandiff_sae_corner_mean"]
        == 0.9
    )


def test_composites_have_one_row_per_sae() -> None:
    df = pd.DataFrame(
        [
            _record("sae_a", "ground_truth", "occupancy_balacc_r0c0", 0.9),
            _record("sae_b", "ground_truth", "occupancy_balacc_r0c0", 0.6),
        ]
    )
    composites = ground_truth_composites(df)
    assert sorted(composites.index) == ["sae_a", "sae_b"]


def test_proxy_metric_table_pivots_one_column_per_metric() -> None:
    df = pd.DataFrame(
        [
            _record("sae_a", "proxy_metric", "l0", 120.0),
            _record("sae_a", "proxy_metric", "fvu", 0.02),
            _record("sae_b", "proxy_metric", "l0", 300.0),
            _record("sae_b", "proxy_metric", "fvu", 0.05),
        ]
    )
    table = proxy_metric_table(df)
    assert sorted(table.columns) == ["fvu", "l0"]
    assert table.loc["sae_a", "l0"] == 120.0
    assert table.loc["sae_b", "fvu"] == 0.05
