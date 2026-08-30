"""Unit tests for the legal-move probe scoring, on synthetic imbalanced data (offline)."""

from __future__ import annotations

import numpy as np
import pytest

from boardsae.validity.legal_move_probe import LegalMoveNotEstimable, probe_legal_move


def _imbalanced_separable_data(
    n: int = 400, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """~15% positive rate, mirroring how rarely a given square is a legal move."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.15).astype(int)
    signal = labels + rng.normal(0, 0.01, n)
    noise = rng.normal(0, 1, n)
    return np.stack([signal, noise], axis=1), labels


def test_separable_imbalanced_data_scores_high_f1() -> None:
    activations, labels = _imbalanced_separable_data()
    scores = probe_legal_move(activations, labels, seed=0)
    assert scores.f1 > 0.9


def test_always_illegal_baseline_is_reported_and_low() -> None:
    activations, labels = _imbalanced_separable_data()
    scores = probe_legal_move(activations, labels, seed=0)
    assert scores.always_illegal_baseline_f1 == 0.0
    assert scores.f1 > scores.always_illegal_baseline_f1


def test_single_class_labels_raise_a_clear_error_instead_of_crashing_sklearn() -> None:
    # Regression test: a square that is never legal in a small sample (a corner, most
    # likely) used to reach sklearn's own confusing "needs at least 2 classes" error.
    activations = np.random.default_rng(0).normal(size=(100, 4))
    always_illegal_labels = np.zeros(100, dtype=int)
    with pytest.raises(LegalMoveNotEstimable):
        probe_legal_move(activations, always_illegal_labels, seed=0)
