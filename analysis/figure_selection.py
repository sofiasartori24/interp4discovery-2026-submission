"""Figure 1: does ranking by a proxy metric select the dictionaries ground truth ranks best?

One panel per proxy metric, each plotting every SAE's rank under that metric against its
rank under ground-truth quality. Points on the diagonal are dictionaries the metric places
correctly. The shaded band marks the ten dictionaries ground truth ranks best, which is the
set a practitioner choosing an SAE for a discovery task would want to land in; filled
points are the ones the metric's own top ten actually recovers.

Ranking is against the *unresidualized* ground-truth factor on purpose. Residualizing by
architecture answers "which dictionary is good for its architecture", which is not a
selection anyone makes; the practitioner picks from the whole population.

Run: python analysis/figure_selection.py [records.parquet] [output.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boardsae.discovery.aggregate import (
    ground_truth_composites,
    proxy_metric_table,
)
from boardsae.discovery.axes import factor_scores

# (column, higher_is_better, display name). Higher L0 and L1 track higher ground-truth
# quality in this population; FVU and dead-feature rate run the other way.
METRICS: tuple[tuple[str, bool, str], ...] = (
    ("l0", True, r"$L_0$"),
    ("fvu", False, "FVU"),
    ("l1", True, r"$L_1$"),
    ("dead_feature_rate", False, "Dead-feature rate"),
)
TOP_K = 10


def ground_truth_ranking(composites: pd.DataFrame) -> pd.Series:
    """Per-SAE ground-truth quality score, oriented so that higher is better.

    The single factor's sign is arbitrary in EFA, so it is fixed here against mean
    occupancy, which is unambiguously higher-is-better.
    """
    factor = cast(pd.Series, factor_scores(composites, 1)["factor_1"])
    occupancy = composites[
        [c for c in composites.columns if c.startswith("occupancy")]
    ].mean(axis=1)
    if cast(float, spearmanr(factor, occupancy)[0]) < 0:
        factor = -factor
    return factor


def main(
    records_path: Path = Path("results/records_grouped_split.parquet"),
    output_path: Path = Path("figures/selection.pdf"),
) -> None:
    df = pd.read_parquet(records_path)
    composites = ground_truth_composites(df)
    proxy = proxy_metric_table(df).loc[composites.index]

    quality = ground_truth_ranking(composites)
    truth_rank = quality.rank(ascending=False)
    truth_top = set(quality.nlargest(TOP_K).index)
    n = len(truth_rank)

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.5), sharex=True, sharey=True)
    for ax, (column, higher_better, label) in zip(axes, METRICS, strict=True):
        metric_rank = proxy[column].rank(ascending=not higher_better)
        metric_top = set(
            (
                proxy[column].nlargest(TOP_K)
                if higher_better
                else proxy[column].nsmallest(TOP_K)
            ).index
        )
        recovered = truth_top & metric_top

        ax.axhspan(0.5, TOP_K + 0.5, color="0.88", zorder=0)
        ax.plot([0, n], [0, n], color="0.6", linewidth=0.8, zorder=1)

        other = [i for i in composites.index if i not in truth_top]
        ax.scatter(
            metric_rank[other],
            truth_rank[other],
            s=12,
            facecolors="none",
            edgecolors="0.45",
            linewidths=0.6,
            zorder=2,
        )
        missed = [i for i in truth_top if i not in recovered]
        ax.scatter(
            metric_rank[missed],
            truth_rank[missed],
            s=34,
            marker="X",
            color="#b2182b",
            zorder=4,
        )
        ax.scatter(
            metric_rank[sorted(recovered)],
            truth_rank[sorted(recovered)],
            s=28,
            color="#2166ac",
            zorder=3,
        )

        rho = spearmanr(metric_rank, truth_rank)[0]
        ax.set_title(
            f"{label}\n" rf"$\rho={rho:.2f}$, {len(recovered)}/{TOP_K} recovered",
            fontsize=9,
        )
        ax.set_xlabel("rank by metric", fontsize=8)
        ax.tick_params(labelsize=7)
        # Both axes run 1 (best) to n (worst) in the usual direction, so agreement reads
        # as the ascending diagonal rather than as an apparent negative slope.
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, n + 1)

    axes[0].set_ylabel("rank by ground truth", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"wrote {output_path}")
    print(f"ground-truth top {TOP_K}: {sorted(truth_top)}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("results/records_grouped_split.parquet"),
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/selection.pdf"),
    )
