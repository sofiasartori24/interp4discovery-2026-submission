"""Does the 80-SAE ranking change between the row-level (leaking) and game-grouped
probe train/test split?

The occupancy and legal-move probes are the two ground-truth quantities the single
discovered factor is built from. Both were originally fit with a row-level
``train_test_split`` over flattened (game, position) rows, which lets a test board's
near-duplicate twin sit in the training set (see
``validity/occupancy_probe.py``'s ``_split`` docstring). This script does not decide
whether that leak matters -- it runs the same analysis primitives already used to
produce the paper's numbers over two parquets that differ only in the split convention,
and reports what changes.

Run with two parquet paths (defaults to the ungrouped paper run and the grouped
rerun):

    python analysis/compare_split_conventions.py results/records.parquet \\
        results/records_grouped_split.parquet

Wherever the paper's own analysis leaves a convention undecided (raw-first vs
rank-first residualization; component vs common-factor parallel analysis), both are
reported here too, exactly as ``reproduce_paper.py`` already does for one parquet --
this script does not resolve that choice either, only doubles it over two datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Same directory as this script, so a plain top-level import resolves it (sys.path[0] is
# always the invoked script's own directory, independent of cwd).
from reproduce_paper import (
    architecture_dummies,
    parallel_analysis_common_factor,
    parallel_analysis_residualized_floor,
    residualize,
)

from boardsae.discovery.aggregate import (
    ground_truth_composites,
    proxy_metric_table,
    sae_metadata,
)
from boardsae.discovery.axes import factor_scores, parallel_analysis
from boardsae.validity.matrix import validity_matrix_partial
from boardsae.validity.reseed_floor import reseed_noise_floor

DEFAULT_UNGROUPED = Path("results/records.parquet")
DEFAULT_GROUPED = Path("results/records_grouped_split.parquet")


def _header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


class SplitConvention:
    """Every quantity this script needs, derived once from one records parquet.

    Both residualization conventions (raw-first, rank-first) are computed up front
    since neither is picked over the other anywhere below.
    """

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.df = pd.read_parquet(path)
        self.composites = ground_truth_composites(self.df)
        self.proxy = proxy_metric_table(self.df).loc[self.composites.index]
        self.meta = sae_metadata(self.df).loc[self.composites.index]
        dummies = architecture_dummies(self.meta)
        self.design = np.column_stack([dummies.to_numpy(), np.ones(len(dummies))])
        self.base_controls = pd.concat(
            [self.meta[["width"]].astype(float), dummies], axis=1
        )
        self.residual_raw_first = residualize(self.composites, self.design)
        self.residual_rank_first = residualize(self.composites.rank(), self.design)
        self.factor_raw_first: pd.Series = cast(
            pd.Series, factor_scores(self.residual_raw_first, 1)["factor_1"]
        )
        self.factor_rank_first: pd.Series = cast(
            pd.Series, factor_scores(self.residual_rank_first, 1)["factor_1"]
        )


def _aligned(a: pd.Series, b: pd.Series, label: str) -> pd.DataFrame:
    """Inner-join two per-sae_id series, warning if either side loses rows."""
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    if len(aligned) != len(a) or len(aligned) != len(b):
        print(
            f"  WARNING: {label} join is not on the same sae_ids -- "
            f"a has {len(a)}, b has {len(b)}, joined has {len(aligned)}."
        )
    return aligned


def _rank_gate(a: pd.Series, b: pd.Series, label: str) -> None:
    """Spearman rho and Kendall tau between two factor-score vectors."""
    aligned = _aligned(a, b, label)
    rho, rho_p = spearmanr(aligned["a"], aligned["b"])
    tau, tau_p = kendalltau(aligned["a"], aligned["b"])
    print(
        f"  {label:28s} n={len(aligned):3d}  "
        f"Spearman rho={rho:+.4f} (p={rho_p:.2e})  "
        f"Kendall tau={tau:+.4f} (p={tau_p:.2e})"
    )


def _rank_moves(a: pd.Series, b: pd.Series, label: str) -> None:
    """SAEs whose rank position (1..n) differs by more than 5 / 10 between a and b."""
    aligned = _aligned(a, b, label)
    delta = (aligned["a"].rank() - aligned["b"].rank()).abs()
    for threshold in (5, 10):
        moved = delta[delta > threshold].sort_values(ascending=False)
        print(f"  {label} moved >{threshold} rank positions: {len(moved)}/{len(delta)}")
        if len(moved):
            print(f"    {moved.round(1).to_dict()}")


def main(ungrouped_path: Path, grouped_path: Path) -> None:
    ungrouped = SplitConvention("ungrouped (row-level, leaking)", ungrouped_path)
    grouped = SplitConvention("grouped (game-level)", grouped_path)

    residualizations = (
        (ungrouped.factor_raw_first, grouped.factor_raw_first, "raw-first"),
        (ungrouped.factor_rank_first, grouped.factor_rank_first, "rank-first"),
    )

    _header("Gate number: factor-score ranking, ungrouped vs grouped split")
    for factor_a, factor_b, label in residualizations:
        _rank_gate(factor_a, factor_b, label)

    _header("Same, per architecture")
    for architecture, index in ungrouped.meta.groupby("architecture").groups.items():
        print(f"\n architecture = {architecture}")
        for factor_a, factor_b, label in residualizations:
            _rank_gate(factor_a.loc[index], factor_b.loc[index], label)

    _header("Rank movement")
    for factor_a, factor_b, label in residualizations:
        _rank_moves(factor_a, factor_b, label)

    # ------------------------------------------------------- PARALLEL ANALYSIS
    for conv in (ungrouped, grouped):
        _header(f"Parallel analysis -- {conv.name}")
        print("naive, component variant:")
        print(parallel_analysis(conv.composites).head(4).to_string(index=False))
        print("\nnaive, common-factor variant:")
        print(
            parallel_analysis_common_factor(conv.composites)
            .head(4)
            .to_string(index=False)
        )
        print("\nresidualized (raw-first), unresidualized floor:")
        print(parallel_analysis(conv.residual_raw_first).head(3).to_string(index=False))
        print("\nresidualized (rank-first), unresidualized floor:")
        print(
            parallel_analysis(conv.residual_rank_first).head(3).to_string(index=False)
        )
        print("\nresidualized (raw-first), residualized floor:")
        print(
            parallel_analysis_residualized_floor(conv.composites, conv.design)
            .head(3)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------ TABLE 1
    for conv in (ungrouped, grouped):
        _header(f"Table 1 -- {conv.name} -- current control set")
        print(
            validity_matrix_partial(
                conv.proxy, conv.factor_raw_first, conv.base_controls
            )
            .round(4)
            .to_string()
        )
        _header(
            f"Table 1 -- {conv.name} -- uniform control set (width + architecture only)"
        )
        print(
            validity_matrix_partial(
                conv.proxy,
                conv.factor_raw_first,
                conv.base_controls,
                l0_as_control_for=(),
            )
            .round(4)
            .to_string()
        )

    # -------------------------------------------------------------- RESEED FLOOR
    _header("Reseed-noise floor")
    for conv in (ungrouped, grouped):
        floor = reseed_noise_floor(conv.df)
        for prefix in ("occupancy_balacc", "legal_move_f1"):
            subset = cast(pd.DataFrame, floor[floor.index.str.startswith(prefix)])
            print(
                f"  {conv.name:28s} {prefix:18s} SNR "
                f"{subset['signal_to_noise'].min():.2f}-{subset['signal_to_noise'].max():.2f}"
                f"  median={subset['signal_to_noise'].median():.2f}"
            )

    # ------------------------------------------------------------- PROBE LEVEL
    _header("Mean balanced accuracy / F1")
    for conv in (ungrouped, grouped):
        occ = conv.df[conv.df["name"].str.startswith("occupancy_balacc")]["value"]
        legal_move = conv.df[conv.df["name"].str.startswith("legal_move_f1")]["value"]
        print(
            f"  {conv.name:28s} mean balanced_accuracy={occ.mean():.4f} (n={len(occ)})  "
            f"mean legal_move_f1={legal_move.mean():.4f} (n={len(legal_move)})"
        )


if __name__ == "__main__":
    ungrouped_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_UNGROUPED
    grouped_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GROUPED
    main(ungrouped_arg, grouped_arg)
