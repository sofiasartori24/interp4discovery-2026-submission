"""Reproduce every number reported in the paper from ``results/records_grouped_split.parquet``.

Complements ``src/boardsae``, which provides the analysis primitives (parallel analysis,
partial correlation, reseed floor) but does not itself compute the paper's reported
values -- the architecture residualization step above all.

Run with ``python analysis/reproduce_paper.py``. Output is organized into one section per
paper claim, tagged with the ``.tex`` section the number appears in, so a claim can be
traced to the code that produced it.

Two design choices:

1. Where the paper's number depends on an analysis choice that is not forced by the
   data, this script computes and prints *both* variants rather than silently picking
   one. That applies to the residualization convention (raw-first vs rank-first,
   Sec. RESIDUALIZATION) and to the parallel-analysis variant (component vs
   common-factor, Sec. AXES). Choosing between them is a call for the paper to make and
   state, not for a script to make by default.
2. Nothing here re-fits a probe or touches the model. Everything is read off the
   already-computed records, so the script is CPU-only and takes seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Top-level import: compare_split_conventions imports from here, so importing from it
# instead would be circular.
from figure_screening import (
    METRICS,
    board_recovery,
    collapsed_mask,
    random_regret_curve,
    regret_curve,
    two_group_range,
)

from boardsae.discovery.aggregate import (
    ground_truth_composites,
    proxy_metric_table,
    sae_metadata,
)
from boardsae.discovery.axes import (
    factor_scores,
    fit_factors,
    loadings,
    parallel_analysis,
)
from boardsae.validity.matrix import (
    partial_correlation,
    validity_matrix_partial,
)
from boardsae.validity.reseed_floor import reseed_noise_floor

DEFAULT_RECORDS_PATH = Path("results/records_grouped_split.parquet")

# Documentation string only; the executable rule lives in figure_screening.collapsed_mask.
COLLAPSED_RULE = "dead_feature_rate > 0.9 or l0 < 5"


def _header(title: str, tex_ref: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n  -> {tex_ref}\n{'=' * 78}")


def architecture_dummies(meta: pd.DataFrame) -> pd.DataFrame:
    """One-hot architecture identity with the first level dropped, plus nothing else.

    ``drop_first=True`` is what makes the dummy set plus an intercept full rank; without
    it the design matrix is singular and the residuals are only defined up to the
    dropped level. Which level gets dropped does not change the residuals.
    """
    return pd.get_dummies(meta["architecture"], drop_first=True).astype(float)


def residualize(frame: pd.DataFrame, design: np.ndarray) -> pd.DataFrame:
    """Regress every column of ``frame`` on ``design`` and return the residuals.

    ``design`` must already carry its own intercept column; this function does not add
    one, so the caller stays responsible for a full-rank design.
    """
    out = frame.copy()
    for column in frame.columns:
        y = frame[column].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        out[column] = y - design @ beta
    return out


def parallel_analysis_common_factor(
    data: pd.DataFrame, n_iter: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Horn's parallel analysis on the *reduced* correlation matrix.

    ``discovery.axes.parallel_analysis`` takes eigenvalues of the correlation matrix
    with unities on the diagonal, which is the principal-components criterion. The
    common-factor variant replaces the diagonal with squared multiple correlations, and
    is the variant that matches the maximum-likelihood common-factor model the count is
    then handed to. The two disagree on this data, which is exactly why both are
    reported: a factor count that flips with the variant is a specification-dependent
    result and should be stated as one.
    """
    ranked = data.rank()
    n, p = ranked.shape

    def reduced(corr: np.ndarray) -> np.ndarray:
        out = corr.copy()
        # SMC_i = 1 - 1/R^-1_ii is the standard estimate of variable i's communality.
        smc = 1.0 - 1.0 / np.diag(np.linalg.inv(corr))
        np.fill_diagonal(out, smc)
        return out

    real = np.linalg.eigvalsh(reduced(ranked.corr().to_numpy()))[::-1]
    rng = np.random.default_rng(seed)
    random = np.empty((n_iter, p))
    for i in range(n_iter):
        draw = rng.normal(size=(n, p))
        random[i] = np.linalg.eigvalsh(reduced(np.corrcoef(draw, rowvar=False)))[::-1]

    return pd.DataFrame(
        {
            "factor": range(1, p + 1),
            "real_eigenvalue": real,
            "random_95th_pct": np.percentile(random, 95, axis=0),
        }
    ).assign(supported=lambda d: d["real_eigenvalue"] > d["random_95th_pct"])


def parallel_analysis_residualized_floor(
    data: pd.DataFrame, design: np.ndarray, n_iter: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Parallel analysis whose random floor is residualized on the same design.

    Residualizing the real composites on k control columns spends k degrees of freedom
    per variable, which shrinks the eigenvalues a same-shaped *unresidualized* random
    draw would produce. Comparing residualized real eigenvalues against an
    unresidualized floor is therefore anti-conservative. This puts the null through the
    same projection the real data went through, so the two are comparable.
    """
    ranked = residualize(data, design).rank()
    n, p = ranked.shape
    real = np.linalg.eigvalsh(ranked.corr().to_numpy())[::-1]

    rng = np.random.default_rng(seed)
    random = np.empty((n_iter, p))
    for i in range(n_iter):
        draw = pd.DataFrame(rng.normal(size=(n, p)))
        random[i] = np.linalg.eigvalsh(
            residualize(draw, design).rank().corr().to_numpy()
        )[::-1]

    return pd.DataFrame(
        {
            "factor": range(1, p + 1),
            "real_eigenvalue": real,
            "random_95th_pct": np.percentile(random, 95, axis=0),
        }
    ).assign(supported=lambda d: d["real_eigenvalue"] > d["random_95th_pct"])


def causal_controllability_per_sae(df: pd.DataFrame) -> pd.DataFrame:
    """Per-SAE real / null / random causal-controllability F1, averaged over estimators.

    The paper reports pooled means across conditions. Per-SAE is what the "the real edit
    reliably beats the controls" claim is actually about: a pooled mean can sit well
    above the null while individual dictionaries fail, and here some do.
    """
    causal = df[df["name"].str.startswith("causal_controllability_f1")]

    def condition(pattern: str) -> pd.Series:
        name_column = cast(pd.Series, causal["name"])
        subset = cast(pd.DataFrame, causal[name_column.str.contains(pattern)])
        return cast(pd.Series, subset.groupby("sae_id")["value"].mean())

    return pd.DataFrame(
        {
            "real": pd.concat(
                [condition("meandiff_sae_r"), condition("logreg_sae_r")], axis=1
            ).mean(axis=1),
            "null": pd.concat(
                [condition("meandiff_null_sae"), condition("logreg_null_sae")], axis=1
            ).mean(axis=1),
            "random": pd.concat(
                [condition("meandiff_random_sae"), condition("logreg_random_sae")],
                axis=1,
            ).mean(axis=1),
        }
    )


def main(
    records_path: Path = DEFAULT_RECORDS_PATH,
    compare_path: Path | None = Path("results/records.parquet"),
) -> None:
    print(f"records: {records_path}")
    df = pd.read_parquet(records_path)
    composites = ground_truth_composites(df)
    proxy = proxy_metric_table(df).loc[composites.index]
    meta = sae_metadata(df).loc[composites.index]
    dummies = architecture_dummies(meta)
    design = np.column_stack([dummies.to_numpy(), np.ones(len(dummies))])

    collapsed = collapsed_mask(proxy)
    healthy = ~collapsed
    print(f"n SAEs = {len(composites)}, n composites = {composites.shape[1]}")
    print(f"collapsed ({COLLAPSED_RULE}) = {int(collapsed.sum())}")
    print(f"architectures: {meta['architecture'].value_counts().to_dict()}")

    # ---------------------------------------------------------------- AXES
    _header("Naive parallel analysis, component variant", "Sec. Results, para 1")
    print(parallel_analysis(composites).head(4).to_string(index=False))

    _header("Naive parallel analysis, dropping collapsed", "Sec. Results, para 1")
    print(
        parallel_analysis(cast(pd.DataFrame, composites[~collapsed]))
        .head(4)
        .to_string(index=False)
    )

    _header(
        "Naive parallel analysis, COMMON-FACTOR variant",
        "App. B, 'Parallel-analysis variants, in full'",
    )
    print(parallel_analysis_common_factor(composites).head(5).to_string(index=False))

    _header("Naive two-factor loadings, all twelve composites", "Sec. Results, para 1")
    naive_loadings = loadings(composites, 2)
    print(naive_loadings.round(3).to_string())
    factor_two = naive_loadings["factor_2"].abs()
    causal_and_kl = [
        c
        for c in composites.columns
        if c.startswith("causal_controllability") or c == "next_move_fidelity_kl"
    ]
    print(
        f"\nfactor_2 |loading| over the 7 causal/fidelity composites: "
        f"{factor_two[causal_and_kl].min():.3f} to {factor_two[causal_and_kl].max():.3f}"
    )
    phi = cast(np.ndarray, fit_factors(composites, 2).phi_)
    print(f"oblimin inter-factor correlation phi = {float(phi[0, 1]):.3f}")

    _header(
        "Composite correlation matrix (Spearman, KL sign-flipped)",
        "not currently in the paper -- the positive-manifold check",
    )
    signed = composites.copy()
    signed["next_move_fidelity_kl"] = -signed["next_move_fidelity_kl"]
    corr = signed.rank().corr()
    off_diagonal = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    print(corr.round(2).to_string())
    print(
        f"\noff-diagonal: n={len(off_diagonal)} min={off_diagonal.min():.3f} "
        f"median={np.median(off_diagonal):.3f} "
        f"none negative={bool((off_diagonal >= 0).all())}"
    )
    named_pair = corr.loc[
        "occupancy_balacc_centre_mean",
        "causal_controllability_f1_logreg_sae_centre_mean",
    ]
    print(
        f"named pair (occupancy centre x causal-controllability centre, logreg): "
        f"{named_pair:.4f}"
    )

    # ------------------------------------------------------- RESIDUALIZATION
    _header(
        "Architecture residualization: BOTH conventions",
        "Sec. Results para 2 reports raw-first; Appendix B describes rank-first",
    )
    raw_first = parallel_analysis(residualize(composites, design))
    rank_first = parallel_analysis(residualize(composites.rank(), design))
    print("raw-first (residualize raw values, let parallel_analysis rank once):")
    print(raw_first.head(2).to_string(index=False))
    print("\nrank-first (rank, then residualize; parallel_analysis ranks again):")
    print(rank_first.head(2).to_string(index=False))
    print(
        "\n  Both conventions agree on a single factor. The paper reports raw-first "
        "throughout (App. C, 'Factor structure')."
    )

    _header(
        "Architecture residualization with a residualized null floor",
        "not currently in the paper -- residualized null floor for the 8.49 figure",
    )
    print(
        parallel_analysis_residualized_floor(composites, design)
        .head(3)
        .to_string(index=False)
    )

    _header(
        "Loadings on the single architecture-residualized factor", "Sec. Results para 2"
    )
    residual_composites = residualize(composites, design)
    single = loadings(residual_composites, 1)
    print(single.round(3).to_string())
    print(
        f"\n|loading| range: {single.abs().min().item():.3f} to "
        f"{single.abs().max().item():.3f}"
    )

    _header(
        "Within-architecture correlation of the two naive factors",
        "Sec. Results para 2",
    )
    naive_scores = factor_scores(composites, 2)
    for architecture, index in meta.groupby("architecture").groups.items():
        subset = naive_scores.loc[index]
        rho, p_value = spearmanr(subset["factor_1"], subset["factor_2"])
        print(
            f"  {architecture:22s} n={len(subset):3d}  rho={rho:+.3f}  p={p_value:.4f}"
        )

    # ------------------------------------------------------ METRIC VALIDITY
    residual_factor = cast(pd.Series, factor_scores(residual_composites, 1)["factor_1"])
    base_controls = pd.concat([meta[["width"]].astype(float), dummies], axis=1)
    healthy_factor = cast(
        pd.Series,
        factor_scores(
            residualize(
                cast(pd.DataFrame, composites[healthy]), design[healthy.to_numpy()]
            ),
            1,
        )["factor_1"],
    )

    _header(
        "Table sensitivity / Table validity's uniform-control column (NOT Table 1)",
        "Sec. Results para 3 (Reported n=80); App. B Table tab:validity's "
        "'rho (width, architecture)' column for L0/L1",
    )
    print(
        "  n=80, with L0 added to FVU's and dead-feature-rate's controls. This is\n"
        "  tab:sensitivity's 'Reported (n=80)' column, not main-text Table 1, which\n"
        "  uses the 71 healthy dictionaries and never adds L0."
    )
    table = validity_matrix_partial(proxy, residual_factor, base_controls)
    print(table.round(4).to_string())

    _header(
        "Table 1 (tab:screening): screening ranges + ranking, 71 healthy dictionaries",
        "Sec. Results, Table 1",
    )
    screening_table = validity_matrix_partial(
        proxy[healthy],
        healthy_factor,
        cast(pd.DataFrame, base_controls[healthy]),
        l0_as_control_for=(),
    )
    for column, higher_better, label, _colour in METRICS:
        rng = two_group_range(proxy[column], collapsed)
        direction_collapsed = "<=" if higher_better else ">="
        direction_healthy = ">=" if higher_better else "<="
        print(
            f"  {label:20s} collapsed {direction_collapsed} "
            f"{rng['collapsed_max' if higher_better else 'collapsed_min']:.4f}   "
            f"healthy {direction_healthy} "
            f"{rng['healthy_min' if higher_better else 'healthy_max']:.4f}   "
            f"partial rho={screening_table.loc[column, 'rho']:+.4f}"
        )
    quality = board_recovery(composites)
    for column, higher_better, label, _colour in METRICS:
        ordering = proxy[column].sort_values(ascending=not higher_better)
        curve = regret_curve(quality, ordering)
        print(f"  {label:20s} shortfall top1={curve[0]:.4f}  top3={curve[2]:.4f}")
    random_curve = random_regret_curve(quality)
    print(
        f"  random, 1 draw={random_curve[0]:.4f}  best of 3={random_curve[2]:.4f}  "
        f"best of 5={random_curve[4]:.4f}"
    )
    screen_then_random = random_regret_curve(cast(pd.Series, quality[healthy]))
    print(
        f"  screen healthy, then best of 3 random survivors={screen_then_random[2]:.4f}"
    )
    band_range = two_group_range(quality, collapsed)
    step = band_range["healthy_min"] - band_range["collapsed_max"]
    band = band_range["healthy_max"] - band_range["healthy_min"]
    print(f"  step (healthy min - collapsed max)={step:.4f} (~{step * 100:.0f} points)")
    print(f"  healthy band width={band:.4f}")

    _header(
        "Appendix D: the two-group split in every ground-truth quantity",
        "App. B (App. D in the main-text numbering), 'two-group split' paragraph",
    )
    legal_move = composites[
        [c for c in composites.columns if c.startswith("legal_move_f1")]
    ].mean(axis=1)
    causal_controllability = composites[
        [c for c in composites.columns if c.startswith("causal_controllability")]
    ].mean(axis=1)
    next_move_kl = cast(pd.Series, composites["next_move_fidelity_kl"])
    for name, values, higher_better in (
        ("legal-move F1", legal_move, True),
        ("causal controllability", causal_controllability, True),
        ("next-move KL", next_move_kl, False),
    ):
        rng = two_group_range(values, collapsed)
        # Next-move KL is lower-is-better, so its step runs the other way.
        quantity_step = (
            rng["healthy_min"] - rng["collapsed_max"]
            if higher_better
            else rng["collapsed_min"] - rng["healthy_max"]
        )
        quantity_band = rng["healthy_max"] - rng["healthy_min"]
        print(
            f"  {name:24s} collapsed {rng['collapsed_min']:.4f}-{rng['collapsed_max']:.4f}"
            f"   healthy {rng['healthy_min']:.4f}-{rng['healthy_max']:.4f}"
            f"   step={quantity_step:.4f}  band={quantity_band:.4f}"
        )

    _header(
        "Specification sensitivity, ALL FOUR metrics",
        "App. B, Table tab:sensitivity",
    )
    specifications = {
        "reported (Spearman, n=80)": validity_matrix_partial(
            proxy, residual_factor, base_controls
        )["rho"],
        "Pearson, n=80": validity_matrix_partial(
            proxy, residual_factor, base_controls, method="pearson"
        )["rho"],
        # Two defensible readings of "restrict to the healthy dictionaries": keep the
        # factor fit on all 80 and drop rows, or refit it on the 71.
        f"n={int(healthy.sum())}, factor fixed": validity_matrix_partial(
            proxy[healthy],
            cast(pd.Series, residual_factor[healthy]),
            cast(pd.DataFrame, base_controls[healthy]),
        )["rho"],
        f"n={int(healthy.sum())}, factor refit": validity_matrix_partial(
            proxy[healthy], healthy_factor, cast(pd.DataFrame, base_controls[healthy])
        )["rho"],
        "rank-first residualization": validity_matrix_partial(
            proxy,
            cast(
                pd.Series,
                factor_scores(residualize(composites.rank(), design), 1)["factor_1"],
            ),
            base_controls,
        )["rho"],
    }
    print(pd.DataFrame(specifications).round(3).to_string())
    print(
        "\n  Only L0 and L1 keep sign and magnitude across all five. FVU and\n"
        "  dead-feature rate both flip sign; FVU's swing is wider."
    )

    _header(
        "Collinearity between each metric and its own L0 control",
        "Sec. Results para 3, Limitations -- the paper flags this for L1 only",
    )
    for metric in ("l1", "fvu", "dead_feature_rate"):
        collinearity = spearmanr(proxy[metric], proxy["l0"])[0]
        without = partial_correlation(proxy[metric], residual_factor, base_controls)[0]
        with_l0 = partial_correlation(
            proxy[metric], residual_factor, base_controls.assign(l0=proxy["l0"])
        )[0]
        flag = "  <-- SIGN FLIP" if np.sign(without) != np.sign(with_l0) else ""
        print(
            f"  {metric:18s} rho(metric, L0)={collinearity:+.3f}   "
            f"partial without L0={without:+.3f}   with L0={with_l0:+.3f}{flag}"
        )
    print(
        "\n  Only tab:validity's right-hand column adds L0 to FVU's and dead-feature\n"
        "  rate's controls; Table 1 never does. The sign that matches FVU's declared\n"
        "  direction is the uncontrolled one."
    )

    # ----------------------------------------------------------- RELIABILITY
    _header("Reseed-noise floor", "App. C, Table 3")
    floor = reseed_noise_floor(df)
    for prefix in ("occupancy_balacc", "legal_move_f1"):
        subset = floor[floor.index.str.startswith(prefix)]
        print(
            f"  {prefix:18s} within-seed SD {subset['within_seed_std'].min():.4f}-"
            f"{subset['within_seed_std'].max():.4f}  SNR "
            f"{subset['signal_to_noise'].min():.1f}-{subset['signal_to_noise'].max():.1f}"
        )
    print(
        "\n  This floor varies only the probe's split seed, so it is the smallest\n"
        "  available variance component; the SNR is not general reliability."
    )

    _header("Causal-controllability controls, per SAE", "Sec. Method / Appendix B")
    causal = causal_controllability_per_sae(df)
    beats_null = int((causal["real"] > causal["null"]).sum())
    beats_random = int((causal["real"] > causal["random"]).sum())
    print(
        f"  real  mean={causal['real'].mean():.3f}  "
        f"range={causal['real'].min():.3f}-{causal['real'].max():.3f}"
    )
    print(f"  null  mean={causal['null'].mean():.3f}  SD={causal['null'].std():.3f}")
    print(
        f"  beats null: {beats_null}/{len(causal)}   beats random: {beats_random}/{len(causal)}"
    )
    print(f"  mean gap over null: {(causal['real'] - causal['null']).mean():+.3f}")
    occupancy = composites[
        [c for c in composites.columns if c.startswith("occupancy")]
    ].mean(axis=1)
    print(
        f"  rho(real, occupancy)={spearmanr(causal['real'].loc[occupancy.index], occupancy)[0]:.3f}"
        f"  rho(real - null, occupancy)="
        f"{spearmanr((causal['real'] - causal['null']).loc[occupancy.index], occupancy)[0]:.3f}"
    )

    _header("Seeds actually recorded per quantity family", "Sec. Results, last para")
    for prefix in (
        "occupancy_balacc",
        "legal_move_f1",
        "causal_controllability_f1_meandiff_sae_r",
        "next_move_fidelity_kl",
    ):
        seed_column = cast(pd.Series, df[df["name"].str.startswith(prefix)]["seed"])
        seeds = sorted(seed_column.unique().tolist())
        print(f"  {prefix:44s} seeds={seeds}")
    print(
        "\n  Causal controllability's real edit exists at one seed only, and its row\n"
        "  subsample is drawn by select_indices(seed=0)."
    )

    # ------------------------------------------------------------ POPULATION
    _header("Population table", "App. A, Table 2")
    batch = np.where(
        pd.Series(composites.index, index=composites.index).str.contains("-v2"),
        "batch 2 (-v2)",
        "batch 1",
    )
    population = pd.DataFrame(
        {
            "architecture": meta["architecture"],
            "batch": batch,
            "width": meta["width"],
            "l0": proxy["l0"],
        }
    )
    print(
        population.groupby(["architecture", "batch"])
        .agg(
            n=("width", "size"),
            width_min=("width", "min"),
            width_max=("width", "max"),
            l0_min=("l0", "min"),
            l0_max=("l0", "max"),
        )
        .round(1)
        .to_string()
    )
    print(
        f"\n  collapsed dictionaries by architecture/batch: "
        f"{population[collapsed].groupby(['architecture', 'batch']).size().to_dict()}"
    )

    _header(
        "Is L0's association an artifact of probe input dimensionality?",
        "App. C, 'Is $L_0$'s association an artifact of probe input dimensionality?'",
    )
    # Live latents is the probe's effective input width: directions that ever fire.
    live_latents = (
        meta["width"].astype(float) * (1.0 - proxy["dead_feature_rate"])
    ).rename("live_latents")
    recovery = board_recovery(composites)
    print(
        f"  rho(live latents, occupancy)      = "
        f"{live_latents.corr(recovery, method='spearman'):+.4f}"
    )
    print(
        f"  rho(live latents, factor)         = "
        f"{live_latents.corr(residual_factor, method='spearman'):+.4f}"
    )
    l0_plain = partial_correlation(proxy["l0"], residual_factor, base_controls)
    l0_live = partial_correlation(
        proxy["l0"],
        residual_factor,
        pd.concat([base_controls, live_latents], axis=1),
    )
    print(
        f"  partial rho(L0, factor) | width+arch            = "
        f"{l0_plain[0]:+.4f}  (p={l0_plain[1]:.2e}, dof={l0_plain[2]})"
    )
    print(
        f"  partial rho(L0, factor) | width+arch+live latents = "
        f"{l0_live[0]:+.4f}  (p={l0_live[1]:.2e}, dof={l0_live[2]})"
    )

    _header(
        "Does FVU just measure reconstruction? Variance shares on the healthy subset",
        "App. C, 'Is the association with ground truth just reconstruction quality?'",
    )
    # The paper quotes rank variance (Spearman), per the Method's convention; Pearson is
    # printed too because the two differ enough here to matter.
    legal_mean = composites[
        [c for c in composites.columns if c.startswith("legal_move")]
    ].mean(axis=1)
    healthy_frame = pd.DataFrame(
        {
            "fvu": proxy.loc[healthy, "fvu"],
            "occupancy": recovery[healthy],
            "legal_move_f1": legal_mean[healthy],
        }
    )
    fvu_healthy = cast(pd.Series, healthy_frame["fvu"])
    for target in ("occupancy", "legal_move_f1"):
        other = cast(pd.Series, healthy_frame[target])
        rho = fvu_healthy.corr(other, method="spearman")
        r = fvu_healthy.corr(other, method="pearson")
        print(
            f"  FVU vs {target:14s} n={len(healthy_frame)}  "
            f"rank variance (rho^2) = {rho**2:.1%}   Pearson r^2 = {r**2:.1%}"
        )

    _header(
        "Causal controllability: pooled oracle means",
        "App. B, 'Causal controllability' paragraph",
    )
    # The paper quotes no oracle mean, so the oracle rows are context; the SAE rows are
    # the ones Appendix B reports.
    gt_rows = cast(pd.DataFrame, df[df["quantity_kind"] == "ground_truth"])
    for label, prefix, exclude in (
        ("oracle, real edit  ", "causal_controllability_f1_meandiff_oracle", ()),
        ("oracle, null edit  ", "causal_controllability_f1_meandiff_null_oracle", ()),
        ("oracle, random dir ", "causal_controllability_f1_meandiff_random_oracle", ()),
        ("SAE,    real edit  ", "causal_controllability_f1_meandiff_sae", ()),
        ("SAE,    null edit  ", "causal_controllability_f1_meandiff_null_sae", ()),
    ):
        subset = cast(pd.DataFrame, gt_rows[gt_rows["name"].str.startswith(prefix)])
        for token in exclude:
            subset = cast(pd.DataFrame, subset[~subset["name"].str.contains(token)])
        pooled = float(cast(pd.Series, subset.groupby("sae_id")["value"].mean()).mean())
        print(
            f"  {label} pooled mean = {pooled:.4f} "
            f"over {subset['name'].nunique()} squares"
        )

    _header(
        "Resolution of the healthy band against the reseed floor",
        "Sec. Limitations, para 2",
    )
    # Both quantities are on the board-wide composite, the scale the shortfall is quoted
    # in. The per-square figures in tab:reseed are larger: averaging the sixteen squares
    # reduces the split noise.
    occ_rows = cast(
        pd.DataFrame, gt_rows[gt_rows["name"].str.startswith("occupancy_balacc")]
    )
    per_seed_grouped = cast(
        pd.Series, occ_rows.groupby(["sae_id", "seed"])["value"].mean()
    )
    per_seed = cast(pd.DataFrame, per_seed_grouped.unstack())
    across_seed_sd = per_seed.std(axis=1, ddof=1)
    gaps = cast(pd.Series, recovery[healthy]).sort_values().diff().dropna()
    median_sd = float(across_seed_sd.median())
    print(f"  median adjacent gap in board recovery (71 healthy) = {gaps.median():.4f}")
    print(f"  median across-seed SD of board recovery            = {median_sd:.4f}")
    print(
        f"  adjacent pairs within one such SD                  = "
        f"{(gaps < median_sd).mean():.1%}"
    )

    _header(
        "What the region average hides: band and shortfall per denomination",
        "App. C, 'What the region average hides, and where the collapsed dictionaries "
        "come from'",
    )
    # Centre squares are a two-class problem, so the three regions have different chance
    # levels and the composite compresses the spread.
    centre_floor = (1 / 3 + 1 / 3 + 1 / 2) / 3
    print(f"  chance floor of the composite = {centre_floor:.4f}")
    at_floor = int(
        (cast(pd.Series, recovery[collapsed]) - centre_floor).abs().lt(0.001).sum()
    )
    print(
        f"  collapsed dictionaries within 0.001 of that floor = "
        f"{at_floor} of {int(collapsed.sum())}"
    )
    denominations: dict[str, pd.Series] = {"board recovery (composite)": recovery}
    for column in composites.columns:
        if column.startswith("occupancy"):
            denominations[column] = cast(pd.Series, composites[column])
    denominations["legal-move F1 (mean)"] = legal_mean
    print()
    for label, quality in denominations.items():
        band = quality[healthy].max() - quality[healthy].min()
        ceiling = quality.max()
        top1 = {}
        for metric, higher_is_better, metric_label, _colour in METRICS:
            values = proxy[metric]
            pick = values.idxmax() if higher_is_better else values.idxmin()
            top1[metric_label] = ceiling - quality.loc[pick]
        shown = "  ".join(f"{k}={v:.4f}" for k, v in top1.items())
        print(f"  {label:34s} healthy band={band:.4f}")
        print(f"    shortfall of each metric's top pick: {shown}")

    _header(
        "Top-10 overlap: how many of the ten best each metric misses",
        "Sec. Introduction, para 3",
    )
    true_top10 = set(recovery.nlargest(10).index)
    for metric, higher_is_better, label, _colour in METRICS:
        values = proxy[metric]
        picked = set(
            (values.nlargest(10) if higher_is_better else values.nsmallest(10)).index
        )
        hits = len(picked & true_top10)
        print(f"  {label:20s} hits {hits:2d} of the 10 best, misses {10 - hits:2d}")

    # ------------------------------------------------------- CAMERA-READY ADDITIONS
    _header(
        "Shortfall recomputed within each reseed",
        "Sec. Results, 'Inside the healthy band'; App. C, 'Shortfall per reseed'",
    )
    # Table 1 is denominated in board recovery averaged over seeds; this asks whether the
    # metric-versus-random comparison survives inside a single probe split.
    occupancy_rows = cast(
        pd.DataFrame, gt_rows[gt_rows["name"].str.startswith("occupancy_balacc")]
    )
    proxy_rows = cast(pd.DataFrame, df[df["quantity_kind"] != "ground_truth"])
    for seed in sorted(occupancy_rows["seed"].unique()):
        seed_rows = cast(pd.DataFrame, gt_rows[gt_rows["seed"] == seed])
        seed_composites = ground_truth_composites(
            pd.concat([proxy_rows, seed_rows])
        ).loc[composites.index]
        seed_recovery = board_recovery(seed_composites)
        shown = []
        for metric, higher_is_better, label, _colour in METRICS:
            ordering = proxy[metric].sort_values(ascending=not higher_is_better)
            curve = regret_curve(seed_recovery, ordering)
            shown.append(f"{label} top1={curve[0]:.4f} top3={curve[2]:.4f}")
        all_random = random_regret_curve(seed_recovery)
        healthy_random = random_regret_curve(cast(pd.Series, seed_recovery[healthy]))
        seed_band = seed_recovery[healthy].max() - seed_recovery[healthy].min()
        print(f"  seed {seed}: " + "  ".join(shown))
        print(
            f"          random best of 3: all 80={all_random[2]:.4f}  "
            f"healthy={healthy_random[2]:.4f}  healthy band={seed_band:.4f}  "
            f"ceiling={seed_recovery.idxmax()}"
        )

    _header(
        "A criterion without a probe: shortfall in next-move KL",
        "Sec. Results, last para",
    )
    # Lower KL is better, so the sign is flipped to reuse the higher-is-better shortfall.
    negated_kl = -cast(pd.Series, composites["next_move_fidelity_kl"])
    for metric, higher_is_better, label, _colour in METRICS:
        ordering = proxy[metric].sort_values(ascending=not higher_is_better)
        curve = regret_curve(negated_kl, ordering)
        print(f"  {label:20s} KL shortfall top1={curve[0]:.5f}  top3={curve[2]:.5f}")
    kl_all = random_regret_curve(negated_kl)
    kl_healthy = random_regret_curve(cast(pd.Series, negated_kl[healthy]))
    print(
        f"  random, 1 draw: all 80={kl_all[0]:.4f}  healthy={kl_healthy[0]:.5f}   "
        f"(simulated, as for Table 1)"
    )

    _header(
        "Choosing within a metric's top three without ground truth",
        "Sec. Discussion, para 2",
    )
    # The top-k shortfall assumes the best of the k can be identified. Without a
    # downstream check, the relevant figure is what any one of the k could cost.
    recovery_ceiling = recovery.max()
    for metric, higher_is_better, label, _colour in METRICS:
        ordering = proxy[metric].sort_values(ascending=not higher_is_better)
        top_three = recovery.loc[ordering.index[:3]]
        print(
            f"  {label:20s} best={recovery_ceiling - top_three.max():.4f}  "
            f"mean={recovery_ceiling - top_three.mean():.4f}  "
            f"worst={recovery_ceiling - top_three.min():.4f}"
        )

    _header(
        "Ground truth against dictionary width, healthy band",
        "Sec. Limitations, para 3; App. C, 'Ground truth against dictionary width'",
    )
    width = meta["width"].astype(float)
    healthy_table = pd.DataFrame(
        {
            "width": width[healthy],
            "architecture": meta.loc[healthy, "architecture"],
            "l0": proxy.loc[healthy, "l0"],
            "fvu": proxy.loc[healthy, "fvu"],
            "recovery": recovery[healthy],
            "kl": composites.loc[healthy, "next_move_fidelity_kl"],
        }
    )
    print(
        healthy_table.groupby("width")[["recovery", "kl", "l0", "fvu"]]
        .agg(["count", "median"])
        .round(4)
        .to_string()
    )
    by_architecture = healthy_table.groupby(["architecture", "width"])[
        "recovery"
    ].mean()
    for architecture, group in healthy_table.groupby("architecture"):
        rho, p = spearmanr(group["width"], group["recovery"])
        narrow, wide = by_architecture.loc[architecture].sort_index()
        print(
            f"  {architecture:20s} n={len(group):2d}  rho(width, recovery)={rho:+.3f} "
            f"(p={p:.3f})  mean(8192) - mean(4096)={wide - narrow:+.4f}"
        )
    architecture_controls = pd.get_dummies(
        healthy_table["architecture"], drop_first=True
    ).astype(float)
    for target in ("recovery", "kl"):
        outcome = cast(pd.Series, healthy_table[target])
        healthy_width = cast(pd.Series, healthy_table["width"])
        arch_only = partial_correlation(healthy_width, outcome, architecture_controls)
        arch_and_l0 = partial_correlation(
            healthy_width,
            outcome,
            architecture_controls.assign(l0=healthy_table["l0"]),
        )
        print(
            f"  partial rho(width, {target:8s}) | arch    = {arch_only[0]:+.3f} "
            f"(p={arch_only[1]:.2e})"
        )
        print(
            f"  partial rho(width, {target:8s}) | arch+L0 = {arch_and_l0[0]:+.3f} "
            f"(p={arch_and_l0[1]:.2e})"
        )

    # -------------------------------------------------- SPLIT-CONVENTION COMPARISON
    if (
        compare_path is not None
        and compare_path != records_path
        and compare_path.exists()
    ):
        _header(
            "Split-convention comparison: band/step under the row-level split",
            "App. B, 'Both split conventions' paragraph -- headline numbers only; the "
            "full diagnostic (rank movement, per-architecture agreement, reseed "
            "floors under both splits) lives in compare_split_conventions.py",
        )
        other_df = pd.read_parquet(compare_path)
        other_composites = ground_truth_composites(other_df)
        other_proxy = proxy_metric_table(other_df).loc[other_composites.index]
        other_collapsed = collapsed_mask(other_proxy)
        other_quality = board_recovery(other_composites)
        other_range = two_group_range(other_quality, other_collapsed)
        other_step = other_range["healthy_min"] - other_range["collapsed_max"]
        other_band = other_range["healthy_max"] - other_range["healthy_min"]
        print(
            f"  row-level split ({compare_path}): band={other_band:.4f}  step={other_step:.4f}"
        )
        print(f"  grouped split ({records_path}): band={band:.4f}  step={step:.4f}")
        print(
            f"  band widens from {other_band:.4f} to {band:.4f} ({band - other_band:+.4f}), "
            f"step narrows from {other_step:.4f} to {step:.4f} ({step - other_step:+.4f}) "
            "(both move against the paper's conclusion)"
        )
    elif compare_path is not None and compare_path == records_path:
        print(
            f"\n(skipping split-convention comparison: compare_path == records_path == "
            f"{records_path})"
        )
    elif compare_path is not None:
        print(f"\n(skipping split-convention comparison: {compare_path} not found)")


if __name__ == "__main__":
    # argv[2] == "" skips the split-convention comparison.
    records_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RECORDS_PATH
    if len(sys.argv) > 2:
        compare_arg = Path(sys.argv[2]) if sys.argv[2] else None
        main(records_arg, compare_arg)
    else:
        main(records_arg)
