"""Metric-independent ground-truth legal-move structure for OthelloGPT."""

from __future__ import annotations

import torch

from boardsae import paths

paths.ensure_on_path()

# Imported after sys.path setup; mirrors groundtruth/othello_boards.py's convention.
from circuits.othello_engine_utils import to_string  # noqa: E402
from circuits.othello_utils import games_batch_to_valid_moves_BLRRC  # noqa: E402


def true_legal_moves(games: list[list[int]]) -> torch.Tensor:
    """Ground-truth legal-move structure, single-channel one-hot (B, L, 8, 8, 1): 1 if
    the square is a legal move at that position in the game, derived from the game's
    rules via the same engine the sanity suite already validates against, not from any
    SAE metric.

    Unlike occupancy (three mutually exclusive classes), a square is independently
    legal or not, so the last dimension is a single binary channel rather than a
    one-hot triple.
    """
    return games_batch_to_valid_moves_BLRRC(to_string(games))
