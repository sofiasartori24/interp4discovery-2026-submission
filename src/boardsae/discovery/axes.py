"""Axis discovery: how many dissociable dimensions do the ground-truth composites
actually support?

Two steps, deliberately kept separate: Horn's parallel analysis decides *how many*
factors are statistically supported at this sample size, before exploratory factor
analysis (EFA) is used to see *what* those factors look like. Fitting EFA with an
arbitrary factor count first and rationalising it after is how spurious axes get
reported; parallel analysis is the check against that.

Every function ranks its input before correlating. The composites mix bounded probe
scores (occupancy balanced accuracy, legal-move F1) with next_move_fidelity_kl, which is
not bounded and not remotely normal, so Pearson correlation (what FactorAnalyzer uses on
raw values) is the wrong tool. Ranking first makes every downstream correlation and
eigenvalue Spearman-equivalent instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer


def _rank(data: pd.DataFrame) -> pd.DataFrame:
    return data.rank()


def parallel_analysis(
    data: pd.DataFrame, n_iter: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Horn's parallel analysis over the rank-transformed correlation matrix.

    Compares the real eigenvalues of ``data``'s correlation matrix against the
    distribution of eigenvalues produced by uncorrelated random data of the same shape.
    A factor is "supported" only where the real eigenvalue exceeds the 95th percentile
    of its random counterpart; below that, the apparent structure is what pure noise
    would produce anyway at this sample size, and should not be reported as a discovered
    axis.
    """
    ranked = _rank(data)
    n, p = ranked.shape
    real_eigs = np.linalg.eigvalsh(ranked.corr().to_numpy())[::-1]

    rng = np.random.default_rng(seed)
    random_eigs = np.empty((n_iter, p))
    for i in range(n_iter):
        random_data = rng.normal(size=(n, p))
        random_eigs[i] = np.linalg.eigvalsh(np.corrcoef(random_data, rowvar=False))[
            ::-1
        ]

    return pd.DataFrame(
        {
            "factor": range(1, p + 1),
            "real_eigenvalue": real_eigs,
            "random_95th_pct": np.percentile(random_eigs, 95, axis=0),
        }
    ).assign(supported=lambda d: d["real_eigenvalue"] > d["random_95th_pct"])


def n_supported_factors(data: pd.DataFrame, n_iter: int = 1000, seed: int = 0) -> int:
    """Count of leading factors whose real eigenvalue beats the parallel-analysis floor.

    Counts only a leading contiguous run: parallel analysis is meant to find where the
    real eigenvalues stop beating the noise floor, not to flag an isolated later factor
    that happens to clear it while an earlier one does not.
    """
    result = parallel_analysis(data, n_iter=n_iter, seed=seed)
    supported = result["supported"].tolist()
    count = 0
    for flag in supported:
        if not flag:
            break
        count += 1
    return count


def fit_factors(data: pd.DataFrame, n_factors: int) -> FactorAnalyzer:
    """Fit EFA with ``n_factors`` over the rank-transformed composites."""
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}.")
    ranked = _rank(data)
    rotation = "oblimin" if n_factors > 1 else None
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation,  # pyright: ignore[reportArgumentType]
        method="ml",
    )
    fa.fit(ranked.to_numpy())
    return fa


def loadings(data: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    """Factor loading matrix: how strongly each ground-truth composite loads on each axis."""
    fa = fit_factors(data, n_factors)
    columns = [f"factor_{i + 1}" for i in range(n_factors)]
    return pd.DataFrame(fa.loadings_, index=data.columns, columns=columns)


def factor_scores(data: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    """Per-SAE score on each discovered axis, for correlating proxy metrics against."""
    fa = fit_factors(data, n_factors)
    ranked = _rank(data)
    scores = fa.transform(ranked.to_numpy())
    columns = [f"factor_{i + 1}" for i in range(n_factors)]
    return pd.DataFrame(scores, index=data.index, columns=columns)
