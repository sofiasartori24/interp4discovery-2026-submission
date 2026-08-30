"""Oracle read-out: a linear probe from model activations to board occupancy.

The SAE-free baseline (Nanda-style linear probe). It measures how well the raw residual
stream encodes a square's occupancy, and is the reference the SAE-based read-out is
compared against. Reports balanced accuracy and the majority-class baseline so a high
raw accuracy on an imbalanced square is not mistaken for real signal, and supports a
shuffled-label negative control.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ProbeScores(BaseModel):
    """Held-out probe scores plus the majority-class baseline for context."""

    accuracy: float
    balanced_accuracy: float
    majority_baseline: float


def _majority_baseline(labels: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class."""
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def _split(
    activations: np.ndarray,
    y: np.ndarray,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Held-out split, grouped by ``groups`` when one is supplied.

    Rows here are (game, position) pairs, and consecutive positions within one game are
    near-duplicate boards: one disc placed, a few flipped. Splitting rows at random
    therefore puts a test board's near-twin in the training set, and the resulting score
    measures interpolation within known games rather than read-out on unseen ones.
    Passing ``groups`` (the game index per row) keeps every position of a game on the
    same side of the split, which is the honest version of the same measurement.

    ``groups=None`` reproduces the ungrouped row-level split the first population run
    used, kept so the two can be compared on identical records rather than only in
    principle.
    """
    if groups is None:
        # train_test_split's stub returns a widened list/array union regardless of input
        # type; narrow back to ndarray (the runtime values already are).
        x_train, x_test, y_train, y_test = train_test_split(
            activations, y, test_size=0.2, random_state=seed
        )
        return (
            np.asarray(x_train),
            np.asarray(x_test),
            np.asarray(y_train),
            np.asarray(y_test),
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(activations, y, groups=groups))
    return (
        activations[train_idx],
        activations[test_idx],
        y[train_idx],
        y[test_idx],
    )


def probe_occupancy(
    activations: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    shuffle_labels: bool = False,
    groups: np.ndarray | None = None,
) -> ProbeScores:
    """Fit a logistic probe (activations -> occupancy) and score it on held-out data.

    ``shuffle_labels`` is the negative control: with labels permuted there is no signal
    to learn, so balanced accuracy must fall to chance.

    ``groups`` assigns each row to a game so the train/test split never separates
    positions from the same game; see ``_split`` for why that matters.
    """
    # A degenerate model (e.g. random weights, used as a negative control) can emit
    # non-finite activations; zeroing them keeps the read-out well defined and lets it
    # collapse to chance. On a trained model the activations are finite, so this is a no-op.
    activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
    y = labels
    if shuffle_labels:
        y = np.random.default_rng(seed).permutation(labels)
    x_train, x_test, y_train, y_test = _split(activations, y, seed, groups)
    # train_test_split's stub returns a widened list/array union regardless of input
    # type; narrow back to ndarray for _majority_baseline (the runtime value already is one).
    y_test = np.asarray(y_test)
    # Scaling puts raw activations and SAE features (very different magnitudes) on a common
    # footing so the lbfgs solver converges consistently across both read-out inputs.
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    probe.fit(x_train, y_train)
    pred = probe.predict(x_test)
    return ProbeScores(
        accuracy=float(accuracy_score(y_test, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        majority_baseline=_majority_baseline(y_test),
    )
