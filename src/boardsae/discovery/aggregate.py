"""Aggregate long-format records into one row per SAE, ready for axis discovery.

The population pipeline records 16 occupancy squares, 12 legal-move squares, and 16
causal-controllability squares (for each of 2 direction types) separately, which is far
more variables than the ~32-80 SAEs typically sampled, too many for a stable factor
analysis (rule of thumb wants the sample size well above the variable count, not below
it). Composites collapse each family to one score per board region instead of one per
square.

Regions are not interchangeable: the population run showed occupancy at centre squares
floors at ~0.5 (they are occupied from move zero and never go empty again, so it is
really a 2-class problem there) while every other square floors at ~0.333 (true 3-class
chance), and legal-move recovery is undefined at centre squares altogether (see
pipeline/population.py). Averaging across regions would blur a structural difference
into a single number; averaging within a region does not. Causal controllability does
not share legal-move's centre exclusion (centre is a valid, in fact the richest, edit
target there, see pipeline/population.py's SCORED_SQUARES/CENTRE_SQUARES split), so it
gets all three region composites, matching occupancy's shape rather than legal-move's.

Only the SAE-space *real-edit* causal-controllability values
(`causal_controllability_f1_{meandiff,logreg}_sae_*`) are aggregated here, not the
oracle ceiling (constant across every SAE by construction, since it doesn't depend on
any SAE -- a zero-variance column breaks the correlation matrix parallel_analysis/EFA
need) and not the null/random controls (those validate the quantity itself, in
validity/reseed_floor.py's spirit; they are not a quality axis to discover structure
over).
"""

from __future__ import annotations

import re
from typing import cast

import pandas as pd

CORNERS: frozenset[tuple[int, int]] = frozenset({(0, 0), (0, 7), (7, 0), (7, 7)})
CENTRE: frozenset[tuple[int, int]] = frozenset({(3, 3), (3, 4), (4, 3), (4, 4)})
# Every other square in pipeline.population.SAMPLED_SQUARES is an edge midpoint.

_SQUARE_SUFFIX = re.compile(r"r(\d)c(\d)$")


def _square_from_name(name: str) -> tuple[int, int] | None:
    """Extract the (row, col) suffix from a record name like 'occupancy_balacc_r3c4'."""
    match = _SQUARE_SUFFIX.search(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _region(square: tuple[int, int]) -> str:
    if square in CORNERS:
        return "corner"
    if square in CENTRE:
        return "centre"
    return "edge"


def ground_truth_composites(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SAE: region-level means of the square-level ground-truth families,
    plus next_move_fidelity_kl unchanged (it is already a single per-SAE value).

    A composite column is only created where at least one square in that region
    actually has a record; legal-move recovery has no centre entries (see module
    docstring), so no `legal_move_f1_centre_mean` column appears, rather than a column
    of all-NaN.
    """
    gt = df[df["quantity_kind"] == "ground_truth"].copy()
    gt["square"] = cast(pd.Series, gt["name"]).map(_square_from_name)
    gt["region"] = cast(pd.Series, gt["square"]).map(
        lambda sq: _region(sq) if sq is not None else None
    )

    rows: list[dict[str, float | str]] = []
    for sae_id, group in gt.groupby("sae_id"):
        row: dict[str, float | str] = {"sae_id": str(sae_id)}
        for prefix in (
            "occupancy_balacc",
            "legal_move_f1",
            "causal_controllability_f1_meandiff_sae",
            "causal_controllability_f1_logreg_sae",
        ):
            family = group[cast(pd.Series, group["name"]).str.startswith(prefix)]
            for region in ("corner", "edge", "centre"):
                subset = family[family["region"] == region]
                if len(subset):
                    row[f"{prefix}_{region}_mean"] = cast(float, subset["value"].mean())
        kl = group.loc[group["name"] == "next_move_fidelity_kl", "value"]
        if len(kl):
            row["next_move_fidelity_kl"] = cast(float, kl.iloc[0])
        rows.append(row)

    return pd.DataFrame(rows).set_index("sae_id")


def proxy_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SAE, one column per proxy metric (l0, l1, fvu, dead_feature_rate)."""
    proxy = df[df["quantity_kind"] == "proxy_metric"]
    return proxy.pivot(index="sae_id", columns="name", values="value")


def sae_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per SAE: architecture and width, for grouping/plotting composites later."""
    return cast(pd.DataFrame, df.groupby("sae_id")[["architecture", "width"]].first())
