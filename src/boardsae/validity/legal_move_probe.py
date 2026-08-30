"""Oracle/SAE read-out: a linear probe from features to legal-move structure.

Unlike occupancy (three-class: empty/yours/mine), legal-move recovery is a binary
per-square question (is this square a legal move right now) and is typically
imbalanced, since few squares are legal at any position, so F1 rather than accuracy is
the score, matching the metric named in the paper draft.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LegalMoveScores(BaseModel):
    """Held-out F1 for legal-move recovery, plus the always-illegal baseline for context."""

    f1: float
    always_illegal_baseline_f1: float


class LegalMoveNotEstimable(ValueError):
    """Raised when a square's legal-move labels contain only one class in this sample.

    Happens for squares that are rarely or never legal within the sampled games
    (corners, especially with a small n_games), regardless of how large the sample is
    in principle: F1 is undefined with zero positive examples, and there is no signal
    for a probe to fit. Callers should skip recording this square for this run rather
    than treat a degenerate score as a real measurement.
    """


def probe_legal_move(
    activations: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> LegalMoveScores:
    """Fit a logistic probe (activations -> is-legal-move) and score it on held-out data.

    ``class_weight="balanced"`` compensates for the legal/illegal imbalance so the probe
    is not rewarded for always predicting illegal.

    ``groups`` assigns each row to a game so the split never separates positions from
    the same game (see ``validity.occupancy_probe._split`` for why row-level splitting
    leaks here). Grouping and stratification cannot both be enforced by a single
    shuffle-split, so with ``groups`` the class balance of the test side is checked after
    the fact rather than imposed: a test side with no legal move leaves F1 undefined and
    raises, the same outcome the all-one-class guard below produces.
    """
    if len(np.unique(labels)) < 2:
        raise LegalMoveNotEstimable(
            f"labels contain only class {labels[0]!r} in this sample; "
            "increase n_games or skip this square for this run."
        )
    activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
    if groups is None:
        stratify = labels if labels.sum() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            activations, labels, test_size=0.2, random_state=seed, stratify=stratify
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(activations, labels, groups=groups))
        x_train, x_test = activations[train_idx], activations[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        if y_train.sum() == 0 or y_test.sum() == 0:
            raise LegalMoveNotEstimable(
                "the grouped split left no legal move on one side for this square; "
                "increase n_games or skip this square for this seed."
            )
    probe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    probe.fit(x_train, y_train)
    pred = probe.predict(x_test)
    always_illegal = np.zeros_like(y_test)
    # sklearn accepts int 0 for zero_division at runtime; its stub only declares the
    # str literals ("warn"/"raise"), so pyright flags both calls below as a mismatch.
    return LegalMoveScores(
        f1=float(f1_score(y_test, pred, zero_division=0)),  # pyright: ignore[reportArgumentType]
        always_illegal_baseline_f1=float(
            f1_score(y_test, always_illegal, zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
    )
