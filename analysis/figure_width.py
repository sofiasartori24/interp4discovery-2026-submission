"""Figure 2: ground truth against dictionary width in the healthy band.

Left panel: board recovery. Right panel: next-move KL, which involves no probe. One marker
per healthy dictionary, horizontal bars at the median of each width. The numbers behind
the figure are printed by ``reproduce_paper.py``, section "Ground truth against
dictionary width, healthy band".

Run: python analysis/figure_width.py [records.parquet] [output.pdf]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from figure_screening import board_recovery, collapsed_mask

from boardsae.discovery.aggregate import (
    ground_truth_composites,
    proxy_metric_table,
    sae_metadata,
)

# Shape distinguishes architecture and grey level backs it up, so the figure stays
# readable in greyscale print.
ARCHITECTURES: tuple[tuple[str, str, str, str, float], ...] = (
    ("StandardTrainer", "Standard", "o", "0.15", -0.3),
    ("PAnnealTrainer", "p-anneal", "s", "0.45", -0.1),
    ("GatedSAETrainer", "Gated", "^", "0.30", 0.1),
    ("GatedAnnealTrainer", "Gated-p-anneal", "D", "0.60", 0.3),
)
WIDTHS = (4096, 8192)


def main(
    records_path: Path = Path("results/records_grouped_split.parquet"),
    output_path: Path = Path("figures/width_sensitivity.pdf"),
) -> None:
    df = pd.read_parquet(records_path)
    composites = ground_truth_composites(df)
    proxy = proxy_metric_table(df).loc[composites.index]
    meta = sae_metadata(df).loc[composites.index]
    healthy = ~collapsed_mask(proxy)
    table = pd.DataFrame(
        {
            "width": meta["width"],
            "architecture": meta["architecture"],
            "recovery": board_recovery(composites),
            "kl": composites["next_move_fidelity_kl"],
        }
    )[healthy]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    panels = (
        (axes[0], "recovery", "board recovery (balanced acc.)"),
        (axes[1], "kl", "next-move KL (lower is better)"),
    )
    for ax, column, ylabel in panels:
        for architecture, label, marker, grey, offset in ARCHITECTURES:
            group = cast(pd.DataFrame, table[table["architecture"] == architecture])
            x = (
                np.where(group["width"] == WIDTHS[0], 0, 1)
                + offset
                + rng.uniform(-0.04, 0.04, len(group))
            )
            ax.scatter(
                x,
                group[column],
                marker=marker,
                s=14,
                facecolor="none",
                edgecolor=grey,
                linewidth=0.8,
                label=label,
            )
        for position, width in enumerate(WIDTHS):
            median = table.loc[table["width"] == width, column].median()
            ax.hlines(
                median, position - 0.42, position + 0.42, color="0.0", linewidth=1.2
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"{w} (n={int((table['width'] == w).sum())})" for w in WIDTHS], fontsize=8
        )
        ax.set_xlabel("dictionary width", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"saved {output_path}")


if __name__ == "__main__":
    records_arg = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("results/records_grouped_split.parquet")
    )
    output_arg = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("figures/width_sensitivity.pdf")
    )
    main(records_arg, output_arg)
