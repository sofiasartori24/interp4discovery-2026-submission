"""Unit tests for discovery/axes.py, on synthetic data with a known factor structure.

Six variables are generated from exactly two independent latent factors (three
variables each), plus enough samples that the two real factors clearly clear the
parallel-analysis noise floor while the remaining four (pure noise dimensions of a
6-variable correlation matrix) do not. This is the one place in the analysis code where
getting the statistics wrong would be silent, a broken parallel_analysis could report
any number of "supported" factors without erroring, so this checks the known-correct
answer rather than just that the function runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from boardsae.discovery.axes import factor_scores, loadings, n_supported_factors


def _two_factor_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor_1 = rng.normal(size=n)
    factor_2 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=(n, 6))
    columns = {
        "v1": factor_1 + noise[:, 0],
        "v2": factor_1 + noise[:, 1],
        "v3": factor_1 + noise[:, 2],
        "v4": factor_2 + noise[:, 3],
        "v5": factor_2 + noise[:, 4],
        "v6": factor_2 + noise[:, 5],
    }
    return pd.DataFrame(columns, index=[f"sae_{i}" for i in range(n)])


def test_parallel_analysis_recovers_the_known_number_of_factors() -> None:
    data = _two_factor_data()
    assert n_supported_factors(data, n_iter=200, seed=0) == 2


def test_loadings_group_variables_by_their_generating_factor() -> None:
    data = _two_factor_data()
    loading_matrix = loadings(data, n_factors=2)
    # Whichever column factor_1 lands in, all of v1-v3 should load onto it and v4-v6
    # onto the other, we don't assume factor ordering matches variable ordering.
    v123_factor = loading_matrix.loc[["v1", "v2", "v3"]].abs().mean().idxmax()
    v456_factor = loading_matrix.loc[["v4", "v5", "v6"]].abs().mean().idxmax()
    assert v123_factor != v456_factor
    assert loading_matrix.loc[["v1", "v2", "v3"], v123_factor].abs().min() > 0.5
    assert loading_matrix.loc[["v4", "v5", "v6"], v456_factor].abs().min() > 0.5


def test_factor_scores_have_one_row_per_input_row() -> None:
    data = _two_factor_data(n=50)
    scores = factor_scores(data, n_factors=2)
    assert list(scores.index) == list(data.index)
    assert scores.shape == (50, 2)
