"""The grouped train/test split must never put two positions from one game on
opposite sides, and must actually change the score when the leak was doing work."""

from __future__ import annotations

import numpy as np
import pytest

from boardsae.validity.legal_move_probe import LegalMoveNotEstimable, probe_legal_move
from boardsae.validity.occupancy_probe import _split, probe_occupancy


def test_grouped_split_keeps_every_game_on_one_side() -> None:
    rng = np.random.default_rng(0)
    n_games, length = 40, 10
    groups = np.arange(n_games * length) // length
    activations = rng.normal(size=(n_games * length, 6))
    labels = rng.integers(0, 3, size=n_games * length)

    x_train, x_test, _, _ = _split(activations, labels, seed=0, groups=groups)

    # Recover which game each returned row belongs to by matching rows back to the
    # original array, rather than assuming _split preserved order.
    def games_of(rows: np.ndarray) -> set[int]:
        matches = (activations[:, None, :] == rows[None, :, :]).all(axis=2)
        return set(groups[matches.any(axis=1)].tolist())

    train_games = games_of(x_train)
    test_games = games_of(x_test)
    assert train_games & test_games == set()
    assert train_games | test_games == set(range(n_games))


def test_ungrouped_split_leaks_games_across_the_boundary() -> None:
    """The control the grouped test is meaningful against: without groups, the same
    game appears on both sides, which is the leak the paper's original run had."""
    rng = np.random.default_rng(0)
    n_games, length = 40, 10
    groups = np.arange(n_games * length) // length
    activations = rng.normal(size=(n_games * length, 6))
    labels = rng.integers(0, 3, size=n_games * length)

    x_train, x_test, _, _ = _split(activations, labels, seed=0, groups=None)

    def games_of(rows: np.ndarray) -> set[int]:
        matches = (activations[:, None, :] == rows[None, :, :]).all(axis=2)
        return set(groups[matches.any(axis=1)].tolist())

    assert games_of(x_train) & games_of(x_test) != set()


def test_grouped_probe_scores_lower_when_the_signal_is_game_specific() -> None:
    """A probe that can only succeed by memorising per-game idiosyncrasies must score
    near chance under the grouped split and well above it under the leaking one.

    Each game gets its own feature coordinate and its own randomly assigned label, so
    game identity is linearly readable but carries no information that transfers to a
    game the probe has not seen. Row-level splitting lets the probe recognise a game it
    already trained on; grouped splitting leaves it guessing.
    """
    rng = np.random.default_rng(0)
    n_games, length = 60, 10
    groups = np.arange(n_games * length) // length
    activations = (
        np.eye(n_games)[groups] + rng.normal(size=(n_games * length, n_games)) * 0.01
    )
    labels = rng.integers(0, 3, size=n_games)[groups]

    leaking = probe_occupancy(activations, labels, seed=0, groups=None)
    grouped = probe_occupancy(activations, labels, seed=0, groups=groups)

    assert leaking.balanced_accuracy > 0.9
    assert grouped.balanced_accuracy < 0.6


def test_grouped_legal_move_probe_raises_when_the_test_side_has_no_positive() -> None:
    n_games, length = 20, 5
    groups = np.arange(n_games * length) // length
    activations = np.random.default_rng(0).normal(size=(n_games * length, 4))
    labels = np.zeros(n_games * length, dtype=int)
    # Every legal move lives in game 0, so any split that holds game 0 out leaves the
    # training side with no positive class, and vice versa.
    labels[:length] = 1

    with pytest.raises(LegalMoveNotEstimable):
        probe_legal_move(activations, labels, seed=0, groups=groups)
