"""Unit tests for the probe scoring and its negative control, on synthetic data (offline)."""

from __future__ import annotations

import numpy as np

from boardsae.validity.occupancy_probe import probe_occupancy


def _separable_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    signal = labels + rng.normal(0, 0.01, n)  # first feature encodes the label
    noise = rng.normal(0, 1, n)
    return np.stack([signal, noise], axis=1), labels


def test_separable_data_scores_high() -> None:
    activations, labels = _separable_data()
    scores = probe_occupancy(activations, labels, seed=0)
    assert scores.balanced_accuracy > 0.95


def test_shuffled_labels_collapse_to_chance() -> None:
    activations, labels = _separable_data()
    scores = probe_occupancy(activations, labels, seed=0, shuffle_labels=True)
    assert scores.balanced_accuracy < 0.65  # no signal survives the permutation
