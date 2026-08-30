"""Figure 1, plus the screening baselines quoted in Results.

Left panel: board recovery per dictionary, sorted, showing the collapsed/healthy split.
Right panel: shortfall from the population ceiling when only each metric's top k may be
chosen from, against random draws.

Run: python analysis/figure_screening.py [records.parquet] [output.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boardsae.discovery.aggregate import ground_truth_composites, proxy_metric_table

# (column, higher_is_better, display name, colour). L0 and L1 go with better board
# recovery here; FVU and dead-feature rate run the other way.
METRICS: tuple[tuple[str, bool, str, str], ...] = (
    ("l0", True, r"$L_0$", "#2166ac"),
    ("fvu", False, "FVU", "#b2182b"),
    ("l1", True, r"$L_1$", "#4d9221"),
    ("dead_feature_rate", False, "Dead-feature rate", "#c51b7d"),
)
MAX_K = 10
N_DRAWS = 20_000


def collapsed_mask(proxy: pd.DataFrame) -> pd.Series:
    """The rule used throughout the paper: a dictionary that is dead or barely fires."""
    return (proxy["dead_feature_rate"] > 0.9) | (proxy["l0"] < 5)


def board_recovery(composites: pd.DataFrame) -> pd.Series:
    """Mean occupancy balanced accuracy: the quantity the regret curve is denominated in."""
    return composites[
        [c for c in composites.columns if c.startswith("occupancy")]
    ].mean(axis=1)


def two_group_range(values: pd.Series, collapsed: pd.Series) -> pd.Series:
    """Collapsed-vs-healthy min/max for any per-SAE quantity aligned to ``collapsed``.

    Shared by Table 1's screening ranges and Appendix C's two-group-split paragraph.
    """
    return pd.Series(
        {
            "collapsed_min": values[collapsed].min(),
            "collapsed_max": values[collapsed].max(),
            "healthy_min": values[~collapsed].min(),
            "healthy_max": values[~collapsed].max(),
        }
    )


def regret_curve(quality: pd.Series, ordering: pd.Series) -> list[float]:
    """Shortfall from the population ceiling when only the first k of ``ordering`` are
    available to pick from, for k = 1..MAX_K."""
    ceiling = quality.max()
    ranked = quality.loc[ordering.index]
    return [ceiling - ranked.iloc[:k].max() for k in range(1, MAX_K + 1)]


def random_regret_curve(quality: pd.Series, seed: int = 0) -> list[float]:
    """Expected shortfall from drawing k dictionaries uniformly at random, by simulation."""
    rng = np.random.default_rng(seed)
    values = quality.to_numpy()
    ceiling = values.max()
    out = []
    for k in range(1, MAX_K + 1):
        draws = values[rng.integers(0, len(values), size=(N_DRAWS, k))].max(axis=1)
        out.append(ceiling - draws.mean())
    return out


def main(
    records_path: Path = Path("results/records_grouped_split.parquet"),
    output_path: Path = Path("figures/screening.pdf"),
) -> None:
    df = pd.read_parquet(records_path)
    composites = ground_truth_composites(df)
    proxy = proxy_metric_table(df).loc[composites.index]
    quality = board_recovery(composites)
    collapsed = collapsed_mask(proxy)

    # Random draws restricted to the healthy dictionaries: the screened baseline the
    # Results paragraph quotes at k=1 and k=3.
    screened = random_regret_curve(cast(pd.Series, quality[~collapsed]))
    unscreened = random_regret_curve(quality)
    print(
        f"random draw, all {len(quality)}      : k=1 {unscreened[0]:.4f}  "
        f"k=3 {unscreened[2]:.4f}  k=5 {unscreened[4]:.4f}"
    )
    print(
        f"random draw, {int((~collapsed).sum())} healthy : k=1 {screened[0]:.4f}  "
        f"k=3 {screened[2]:.4f}  k=5 {screened[4]:.4f}"
    )
    print(
        f"screening removes {1 - screened[0] / unscreened[0]:.0%} of the k=1 cost "
        f"({unscreened[0]:.4f} -> {screened[0]:.4f})"
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 2.4))

    # ---- left: the two populations
    order = quality.sort_values(ascending=False)
    x = np.arange(1, len(order) + 1)
    is_collapsed = collapsed.loc[order.index].to_numpy()
    left.scatter(
        x[~is_collapsed],
        order[~is_collapsed],
        s=16,
        color="#2166ac",
        label=f"healthy ({int((~is_collapsed).sum())})",
    )
    left.scatter(
        x[is_collapsed],
        order[is_collapsed],
        s=28,
        marker="X",
        color="#b2182b",
        label=f"collapsed ({int(is_collapsed.sum())})",
    )
    healthy_quality = quality[~collapsed]
    left.axhspan(
        healthy_quality.min(),
        healthy_quality.max(),
        color="0.88",
        zorder=0,
    )
    left.annotate(
        f"healthy band:\n{healthy_quality.max() - healthy_quality.min():.3f} wide",
        xy=(46, healthy_quality.min()),
        xytext=(28, 0.70),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.4"},
    )
    left.set_xlabel("dictionary, sorted by board recovery", fontsize=8)
    left.set_ylabel("occupancy balanced acc.", fontsize=8)
    left.set_title("The population is two populations", fontsize=9)
    left.tick_params(labelsize=7)
    left.legend(fontsize=7, loc="lower left", frameon=False)

    # ---- right: what ranking inside the healthy band buys
    ks = np.arange(1, MAX_K + 1)
    for column, higher_better, label, colour in METRICS:
        ordering = proxy[column].sort_values(ascending=not higher_better)
        right.plot(
            ks,
            regret_curve(quality, ordering),
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=colour,
            label=label,
        )
    right.plot(
        ks,
        random_regret_curve(quality),
        linestyle="--",
        linewidth=1.2,
        color="0.45",
        label="random draw",
    )
    right.set_xlabel("dictionaries considered ($k$)", fontsize=8)
    right.set_ylabel("shortfall from best available", fontsize=8)
    right.set_title("What the ranking is worth", fontsize=9)
    right.set_xticks(ks)
    right.tick_params(labelsize=7)
    right.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"wrote {output_path}")
    board_range = two_group_range(quality, collapsed)
    print(
        f"healthy band: {board_range['healthy_min']:.4f} to "
        f"{board_range['healthy_max']:.4f}"
    )
    print(
        f"collapsed: {board_range['collapsed_min']:.4f} to "
        f"{board_range['collapsed_max']:.4f}"
    )
    step = board_range["healthy_min"] - board_range["collapsed_max"]
    band = board_range["healthy_max"] - board_range["healthy_min"]
    print(f"step (healthy min - collapsed max): {step:.4f} (~{step * 100:.0f} points)")
    print(f"band width (healthy max - healthy min): {band:.4f}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("results/records_grouped_split.parquet"),
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/screening.pdf"),
    )
