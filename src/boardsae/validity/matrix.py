"""Metric-validity audit: each proxy metric against the discovered ground-truth axes.

Correlates each proxy metric under audit (L0, L1, FVU, dead-feature rate) against each
ground-truth axis discovered in ``discovery/axes.py``.

This was designed as a multitrait-multimethod matrix (Campbell & Fiske 1959), and is not
one. MTMM needs at least two traits crossed with at least two methods, and all four of
its criteria are comparisons *between* blocks of that grid. With a single supported
factor there is no heterotrait block, so discriminant validity is undefined and method
variance is unidentifiable; what this module produces is a loading pattern against one
criterion. Naming it convergent validity would also be circular, since the factor is
extracted from the same composites the metrics are scored against.

Spearman throughout, matching the rank-transform
already used for the factor analysis itself (see discovery/axes.py's docstring for why:
L0 and FVU are not remotely normal and span orders of magnitude across the population).

The population mixes four SAE architectures and a wide range of dictionary widths, both
of which independently shift proxy-metric scale (see pipeline/population.py and the
2026-08-17 architecture check in the paper log). A zero-order correlation between a
proxy metric and a discovered axis can therefore reflect real convergent validity, or it
can reflect the two variables happening to differ across the same architectures, without
telling those apart. ``validity_matrix_partial`` controls for the confounds declared by
the caller (width, and architecture identity as dummy-coded covariates; L0 additionally
for FVU and dead-feature-rate) before reporting the association.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from scipy.stats import t as _t_dist


def partial_correlation(
    x: pd.Series,
    y: pd.Series,
    controls: pd.DataFrame,
    method: str = "spearman",
) -> tuple[float, float, int]:
    """Partial correlation of ``x`` and ``y`` given ``controls``, by residualization.

    Standard definition (e.g. Cohen & Cohen 1983): regress ``x`` and ``y`` each on
    ``controls`` plus an intercept, and correlate the residuals. For ``method="spearman"``
    this is done on rank-transformed ``x``, ``y``, and every column of ``controls``
    first, which is the textbook generalization of Spearman partial correlation to more
    than one control (equivalent to the single-control recursive formula when there is
    only one). ``method="pearson"`` skips the rank transform.

    Cross-checked during development against ``pingouin.partial_corr`` (Vallat, 2018,
    JOSS 3(31):1026) on this project's own data: agreement to three decimal places on
    every (metric, factor, control-set) combination tried. Reimplemented here rather
    than imported so the one non-trivial step (residualization, not just a correlation
    call) is something this codebase owns and tests directly, matching the project's
    reimplement-don't-import convention for method code.

    Returns ``(rho, p_value, dof)``. ``dof = n - len(controls.columns) - 2``, the
    degrees of freedom for the t-test on the residual correlation; a negative or zero
    ``dof`` (too many controls for too few SAEs) raises rather than silently returning
    a meaningless p-value.
    """
    n = len(x)
    if len(y) != n or len(controls) != n:
        raise ValueError("x, y, and controls must have the same number of rows.")
    n_controls = controls.shape[1]
    dof = n - n_controls - 2
    if dof < 1:
        raise ValueError(
            f"n={n} with {n_controls} controls leaves dof={dof}; need at least 1."
        )

    def _prep(s: pd.Series) -> np.ndarray:
        arr = s.to_numpy(dtype=float)
        return rankdata(arr) if method == "spearman" else arr

    design = np.column_stack(
        [_prep(cast(pd.Series, controls[c])) for c in controls.columns]
    )
    design = np.column_stack([design, np.ones(n)])

    x_r, y_r = _prep(x), _prep(y)
    beta_x, *_ = np.linalg.lstsq(design, x_r, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y_r, rcond=None)
    resid_x = x_r - design @ beta_x
    resid_y = y_r - design @ beta_y

    # Pearson on the residuals in both cases: the Spearman variant differs only in that
    # `_prep` already rank-transformed the inputs above.
    rho = cast(float, pearsonr(resid_x, resid_y)[0])
    # Significance from the residual correlation's own t-statistic, not pearsonr's
    # p-value, because pearsonr does not know how many degrees of freedom the prior
    # regression already spent (see the dof computation above).
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * np.sqrt(dof / (1 - rho**2))
        p_value = float(2 * (1 - _t_dist.cdf(abs(t_stat), dof)))
    return float(rho), p_value, dof


def validity_matrix_partial(
    proxy: pd.DataFrame,
    factor: pd.Series,
    base_controls: pd.DataFrame,
    l0_as_control_for: tuple[str, ...] = ("fvu", "dead_feature_rate"),
    l0_column: str = "l0",
    method: str = "spearman",
) -> pd.DataFrame:
    """Partial-correlation validity matrix: each proxy metric against one discovered
    factor, controlling for ``base_controls`` (typically dictionary width and
    architecture dummies) plus, for the metrics named in ``l0_as_control_for``, L0
    itself.

    L0 is deliberately excluded from its own row's controls, and from L1's: L0 against
    itself is degenerate, and L0 and L1 are highly collinear in this population
    (Spearman rho=0.73), so controlling L1 for L0 produces sign-unstable estimates
    (checked directly: +0.65 without L0 in the controls, -0.71 with it) rather than a
    meaningful "L1 net of L0" reading. For the same reason, the sparsity rows (L0, L1)
    are controlled by width and, here, architecture only.
    """
    rows = []
    for metric in proxy.columns:
        controls = base_controls.copy()
        if metric in l0_as_control_for and metric != l0_column:
            controls = controls.assign(**{l0_column: proxy[l0_column]})
        rho, p_value, dof = partial_correlation(
            cast(pd.Series, proxy[metric]), factor, controls, method=method
        )
        rows.append(
            {
                "metric": metric,
                "rho": rho,
                "p_value": p_value,
                "dof": dof,
                "controls": list(controls.columns),
            }
        )
    return pd.DataFrame(rows).set_index("metric")
