"""Metric-independent ground-truth boards for OthelloGPT, via SAE_BoardGameEval."""

from __future__ import annotations

import torch

from boardsae import paths

paths.ensure_on_path()

# Imported after sys.path setup; circuits/__init__.py is empty so this is light.
from circuits.othello_engine_utils import to_string  # noqa: E402
from circuits.othello_utils import (  # noqa: E402
    games_batch_to_state_stack_BLRRC,
    games_batch_to_state_stack_mine_yours_BLRRC,
    hf_othello_dataset_to_generator,
)


def sample_games(n_games: int, seq_len: int = 59) -> list[list[int]]:
    """Sample equal-length game token-sequences from the taufeeque/othellogpt dataset.

    These are model vocab tokens (fed to the model as-is); the ground-truth board
    computation maps them to board positions separately (see true_occupancy_boards).
    """
    gen = hf_othello_dataset_to_generator("taufeeque/othellogpt")
    games: list[list[int]] = []
    for game in gen:
        game = list(game)
        if len(game) >= seq_len:
            games.append(game[:seq_len])
        if len(games) >= n_games:
            break
    return games


def true_occupancy_boards(games: list[list[int]]) -> torch.Tensor:
    """Ground-truth occupancy in the absolute basis, one-hot (B, L, 8, 8, 3), index order
    white / empty / black (verified against circuits.othello_utils.board_state_to_RRC's
    actual index assignment, not just described from memory: index 1 is empty at every
    position, confirmed by checking position 0 of an always-initially-empty square).

    Note: the absolute (black/white) basis is not linearly decodable from OthelloGPT
    (Li et al. 2023); use the mine/yours basis for linear read-out.
    """
    return games_batch_to_state_stack_BLRRC(to_string(games))


def true_mine_yours_boards(games: list[list[int]]) -> torch.Tensor:
    """Ground-truth occupancy in the mine/yours basis, one-hot (B, L, 8, 8, 3), index
    order yours / empty / mine (same index-1-is-empty convention as
    true_occupancy_boards, both go through circuits.othello_utils.board_state_to_RRC).

    This player-relative basis is the one that is linearly decodable from OthelloGPT
    (Nanda et al. 2023), so it is the right target for the linear read-out oracle.
    """
    return games_batch_to_state_stack_mine_yours_BLRRC(to_string(games))
