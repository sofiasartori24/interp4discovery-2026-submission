"""Unit tests for validity/matrix.py's partial-correlation additions.

Two kinds of check: agreement with a known closed-form case (the single-control
recursive partial-correlation formula), and agreement with `pingouin` (Vallat 2018,
JOSS), an independent, citable implementation, on synthetic data with multiple
controls. Together these are what "reimplemented, not imported, and checked against a
primary source" means for this function in practice.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from boardsae.validity.matrix import partial_correlation, validity_matrix_partial


def test_partial_correlation_matches_single_control_recursive_formula() -> None:
    """With exactly one control, partial correlation has a closed form:
    r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2)).
    Checked here on Pearson (method="pearson" skips the rank transform, so this
    compares like with like) against `partial_correlation`'s residualization approach.
    """
    rng = np.random.default_rng(0)
    n = 60
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(size=n)
    y = 0.4 * z + rng.normal(size=n)

    r_xy = cast(float, pearsonr(x, y)[0])
    r_xz = cast(float, pearsonr(x, z)[0])
    r_yz = cast(float, pearsonr(y, z)[0])
    expected = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    rho, _, dof = partial_correlation(
        pd.Series(x), pd.Series(y), pd.DataFrame({"z": z}), method="pearson"
    )
    assert rho == pytest.approx(expected, abs=1e-9)
    assert dof == n - 1 - 2


def test_partial_correlation_matches_pingouin_with_two_controls() -> None:
    """Cross-check against pingouin.partial_corr (an independent, citable
    implementation) with two controls and the Spearman rank transform, the exact
    configuration used for the paper's validity matrix (width + architecture)."""
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(1)
    n = 80
    c1 = rng.normal(size=n)
    c2 = rng.integers(0, 2, size=n).astype(float)
    x = 0.5 * c1 + 0.3 * c2 + rng.normal(size=n)
    y = 0.2 * c1 - 0.4 * c2 + rng.normal(size=n)
    data = pd.DataFrame({"x": x, "y": y, "c1": c1, "c2": c2})

    rho, p_value, _ = partial_correlation(
        cast(pd.Series, data["x"]),
        cast(pd.Series, data["y"]),
        cast(pd.DataFrame, data[["c1", "c2"]]),
        method="spearman",
    )
    ref = pg.partial_corr(
        data=data, x="x", y="y", covar=["c1", "c2"], method="spearman"
    )
    assert rho == pytest.approx(ref["r"].to_numpy()[0], abs=1e-3)
    assert p_value == pytest.approx(ref["p_val"].to_numpy()[0], abs=1e-3)


def test_partial_correlation_rejects_too_many_controls_for_n() -> None:
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([2.0, 4.0, 5.0])
    controls = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="dof"):
        partial_correlation(x, y, controls)


def test_validity_matrix_partial_excludes_l0_from_its_own_row_and_from_l1() -> None:
    """L0 controlling L0 is degenerate, and L0/L1 are collinear enough here that using
    L0 to control L1 flips its sign (see the module docstring); neither row should get
    L0 added to its controls, only fvu and dead_feature_rate should."""
    rng = np.random.default_rng(2)
    n = 40
    proxy = pd.DataFrame(
        {
            "l0": rng.normal(size=n),
            "l1": rng.normal(size=n),
            "fvu": rng.normal(size=n),
            "dead_feature_rate": rng.normal(size=n),
        }
    )
    factor = pd.Series(rng.normal(size=n))
    base_controls = pd.DataFrame({"width": rng.normal(size=n)})

    result = validity_matrix_partial(proxy, factor, base_controls)

    assert "l0" not in result.loc["l0", "controls"]
    assert "l0" not in result.loc["l1", "controls"]
    assert "l0" in result.loc["fvu", "controls"]
    assert "l0" in result.loc["dead_feature_rate", "controls"]
    assert set(result.index) == {"l0", "l1", "fvu", "dead_feature_rate"}
