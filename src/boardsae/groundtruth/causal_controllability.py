"""Causal-controllability ground truth: does editing a Mine-vs-Yours direction, spliced
through the model's own forward pass, causally steer its own legal-move behaviour to
match the rule-mandated counterfactual (Li et al. 2023; Nanda et al. 2023)? The new
piece this module adds beyond either paper is running that same intervention through an
SAE's encode/decode bottleneck instead of directly on the activation, comparing the
result against an SAE-free oracle ceiling that should reproduce Nanda's own effect.

Kept SAE-agnostic (no SAE import), mirroring groundtruth/fidelity.py's reconstruct_fn
callback style: callers pass a `to_activation` callback that is the identity for the
oracle quantity and the SAE's decode for the SAE quantity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from boardsae import paths

paths.ensure_on_path()

# Imported after sys.path setup; mirrors groundtruth/othello_boards.py's convention.
from circuits.othello_engine_utils import (  # noqa: E402
    OthelloBoardState,
    stoi,
    to_string,
)

# Uniform-prior baseline over the 60 move-square tokens (d_vocab=61: 60 squares + pass).
# A square with no information would get 1/60 of the model's output probability.
# OthelloGPT is trained via cross-entropy on the single actual next move, which drives
# illegal-square probability toward ~0 and lets legal squares collectively absorb the
# rest (Li et al. 2023); checked directly on clean, unedited data (50 games, centre
# excluded) this threshold recovers the true legal-move set at F1=0.9989, so it is a
# tuning-free, chance-level cutoff, not a fitted one.
UNIFORM_LEGAL_THRESHOLD = 1.0 / 60.0

# Edit magnitude, applied to both direction types after both are normalized to the
# mean-difference vector's own norm (see `normalized_direction`). Adding exactly 1x
# (mean(Mine) - mean(Yours)) to a Yours-labelled row lands it exactly at the Mine mean,
# by construction; 2x overshoots that same distance again, landing solidly past the
# opposite class's mean rather than merely at its boundary.
EDIT_SCALE = 2.0

# Reproducible cap on how many eligible rows are scored per square: the per-row
# counterfactual legal-move computation is a pure-Python rule-engine call (not
# vectorized), so this bounds that cost independently of n_games, the same kind of
# declared sampling trade-off as pipeline/population.py's SAMPLED_SQUARES. 500 rows
# across 12 scored squares (see eligible_rows) already gives ~6,000 pooled (row, square)
# pairs per F1, ample for a stable pooled score.
DEFAULT_MAX_ROWS = 500


class DirectionNotEstimable(ValueError):
    """Raised when a square's Mine/Yours labels contain only one class in this sample."""


@dataclass(frozen=True)
class MineYoursRows:
    """Eligible (non-empty) rows at one square, ready for direction-fitting and editing."""

    values: np.ndarray  # (N, D) SAE features or raw activations
    is_mine: np.ndarray  # (N,) bool: True = Mine, False = Yours
    game_idx: np.ndarray  # (N,) int: which game each row came from
    position_idx: np.ndarray  # (N,) int: which position within that game


def select_indices(
    occ_labels: np.ndarray,
    max_rows: int = DEFAULT_MAX_ROWS,
    candidate_indices: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """The flat-index selection `eligible_rows` uses, factored out so it (and anything
    derived from it that doesn't depend on `values`, like the counterfactual ground
    truth) can be computed once and shared across every space (oracle, every SAE)
    instead of redone identically, and wastefully, for each. This selection is
    deterministic given the same inputs and ``seed``, so sharing it changes nothing
    about what gets scored, only how many times it gets recomputed.

    ``candidate_indices``, when given, replaces "every non-empty row" as the pool to
    subsample from -- pass `informative_indices`' result here so the rows scored for
    causal controllability are the ones where there is actually something to detect
    (see that function's docstring for why this matters, not just for cost).

    ``seed`` picks which subsample gets drawn when the candidate pool exceeds
    ``max_rows`` (the common case for every square but the four corners, whose raw
    informative counts already sit below the default cap -- see
    project_boardsae_informative_rows). Defaults to 0, matching every existing caller;
    pass a different value to check whether a downstream result (e.g. a discovered
    factor) is a property of the quantity or an artifact of which particular subsample
    happened to get drawn.
    """
    eligible = (
        candidate_indices
        if candidate_indices is not None
        else np.flatnonzero(occ_labels != 1)
    )
    if len(eligible) > max_rows:
        eligible = np.sort(
            np.random.default_rng(seed).choice(eligible, max_rows, replace=False)
        )
    return eligible


def rows_from_indices(
    values: np.ndarray, occ_labels: np.ndarray, length: int, flat_indices: np.ndarray
) -> MineYoursRows:
    """Build `MineYoursRows` from an already-selected set of flat indices (see
    `select_indices`), for when the same row selection is shared across multiple
    `values` arrays (oracle vs. every SAE) rather than recomputed for each.

    ``occ_labels`` follows othello_boards.true_mine_yours_boards' actual one-hot index
    order (verified directly against board_state_to_RRC, not the docstring's prose,
    which lists them in the wrong order): index 0 = one polarity ("yours" here), index 1
    = empty, index 2 = the other polarity ("mine" here). ``values``/``occ_labels`` are
    flattened (n_games*length,) the same row-major way as everywhere else in
    pipeline/population.py (index = game_idx*length + position_idx), so game/position
    are recovered by div/mod rather than threaded through separately.
    """
    return MineYoursRows(
        values=values[flat_indices],
        is_mine=occ_labels[flat_indices] == 2,
        game_idx=flat_indices // length,
        position_idx=flat_indices % length,
    )


def eligible_rows(
    values: np.ndarray,
    occ_labels: np.ndarray,
    length: int,
    max_rows: int = DEFAULT_MAX_ROWS,
    candidate_indices: np.ndarray | None = None,
    seed: int = 0,
) -> MineYoursRows:
    """Non-empty (Mine or Yours) rows at one square, subsampled down to ``max_rows``.
    See `select_indices` and `rows_from_indices`, which this composes; call those
    directly instead when the same row selection will be reused across several
    ``values`` arrays (see their docstrings for why that matters for more than cost).
    """
    flat_indices = select_indices(occ_labels, max_rows, candidate_indices, seed)
    return rows_from_indices(values, occ_labels, length, flat_indices)


def _require_both_classes(rows: MineYoursRows) -> None:
    if rows.is_mine.sum() == 0 or (~rows.is_mine).sum() == 0:
        raise DirectionNotEstimable(
            "both Mine and Yours rows are required to compute a direction; "
            "increase n_games or skip this square for this run."
        )


def mean_diff_direction(rows: MineYoursRows) -> np.ndarray:
    """Mine mean minus Yours mean: the 1x edit distance from one class mean to the other."""
    _require_both_classes(rows)
    return rows.values[rows.is_mine].mean(axis=0) - rows.values[~rows.is_mine].mean(
        axis=0
    )


def logreg_direction(rows: MineYoursRows, seed: int = 0) -> np.ndarray:
    """Unit-normalized weight vector of a 2-class Mine-vs-Yours logistic regression.

    Fit without feature scaling (unlike occupancy_probe/legal_move_probe): only the
    *direction* of the coefficient is used (normalized_direction rescales it to match
    mean_diff_direction's norm before editing), and a StandardScaler would entangle that
    direction with the scaler's own per-feature rescaling, which the mean-diff direction
    never goes through.
    """
    _require_both_classes(rows)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(rows.values, rows.is_mine.astype(int))
    direction = clf.coef_[0]
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise DirectionNotEstimable(
            "logistic regression found no separating direction."
        )
    return direction / norm


def normalized_direction(direction: np.ndarray, reference_norm: float) -> np.ndarray:
    """Rescale ``direction`` to have norm ``reference_norm``, preserving its own bearing."""
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise DirectionNotEstimable("direction has zero norm.")
    return direction / norm * reference_norm


def random_direction(dim: int, reference_norm: float, seed: int) -> np.ndarray:
    """Random control direction of matching norm; must not reliably reproduce the effect."""
    vec = np.random.default_rng(seed).normal(size=dim)
    return vec / np.linalg.norm(vec) * reference_norm


def apply_edit(rows: MineYoursRows, direction: np.ndarray, scale: float) -> np.ndarray:
    """Invert each row toward the opposite label: Mine rows subtract, Yours rows add."""
    sign = np.where(rows.is_mine, -1.0, 1.0)
    return rows.values + scale * sign[:, None] * direction[None, :]


def replay_boards_and_turns(
    games: list[list[int]], length: int
) -> tuple[np.ndarray, np.ndarray]:
    """One engine pass per game: absolute board state and next-mover colour at every
    position, shared across every square (unlike the occupancy/legal-move ground truth,
    which are square-specific one-hot slices, the counterfactual needs the raw signed
    state so one square's owner can be flipped directly).

    Returns (states, next_hand_color): states is (n_games, length, 8, 8) int8 with
    -1/0/1 for white/empty/black; next_hand_color is (n_games, length) int8, matching
    OthelloBoardState's own convention (1=black, -1=white) for whoever moves next.
    """
    n = len(games)
    states = np.zeros((n, length, 8, 8), dtype=np.int8)
    next_hand_color = np.zeros((n, length), dtype=np.int8)
    # to_string is called batch-wise (list[list[int]]), matching every other call site
    # in groundtruth/ (othello_boards.py, legal_moves.py): it is an unannotated,
    # recursive, isinstance-branching function, and pyright's inferred return type for
    # it is only stable when every call site shares the same input shape.
    for g, positions in enumerate(to_string([game[:length] for game in games])):
        board = OthelloBoardState()
        for p, move in enumerate(positions):
            board.umpire(move)
            states[g, p] = board.state
            next_hand_color[g, p] = board.next_hand_color
    return states, next_hand_color


def _rule_mandated_legal_moves(
    states: np.ndarray,
    next_hand_color: np.ndarray,
    game_idx: np.ndarray,
    position_idx: np.ndarray,
    scored_squares: tuple[tuple[int, int], ...],
    flip_square: tuple[int, int] | None,
) -> np.ndarray:
    """Rule-mandated legal moves restricted to ``scored_squares``, one row per
    (game_idx[i], position_idx[i]); flips ``flip_square``'s owner first when given
    (the counterfactual), or reads the board as-is when ``flip_square`` is None (the
    clean reference `clean_legal_moves` and `counterfactual_legal_moves` both need).

    Entered via direct state assignment rather than ``.umpire()``: the counterfactual is
    generally unreachable by any real move sequence (an out-of-distribution board, per Li
    et al. 2023's own methodology), so there is no move history to replay it from.
    """
    out = np.zeros((len(game_idx), len(scored_squares)), dtype=np.int8)
    for i in range(len(game_idx)):
        g, p = int(game_idx[i]), int(position_idx[i])
        state = states[g, p].astype(float).copy()
        if flip_square is not None:
            fr, fc = flip_square
            state[fr, fc] *= -1
        board = OthelloBoardState()
        board.state = state
        board.next_hand_color = int(next_hand_color[g, p])
        valid = set(board.get_valid_moves())
        for j, (sr, sc) in enumerate(scored_squares):
            out[i, j] = int((sr * 8 + sc) in valid)
    return out


def clean_legal_moves(
    states: np.ndarray,
    next_hand_color: np.ndarray,
    game_idx: np.ndarray,
    position_idx: np.ndarray,
    scored_squares: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Rule-mandated legal moves on the untouched board, restricted to ``scored_squares``.
    The reference `informative_indices` filters against (see its docstring for why)."""
    return _rule_mandated_legal_moves(
        states,
        next_hand_color,
        game_idx,
        position_idx,
        scored_squares,
        flip_square=None,
    )


def counterfactual_legal_moves(
    states: np.ndarray,
    next_hand_color: np.ndarray,
    square: tuple[int, int],
    game_idx: np.ndarray,
    position_idx: np.ndarray,
    scored_squares: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Rule-mandated legal moves on the counterfactual board (``square``'s owner
    flipped), restricted to ``scored_squares``, one row per (game_idx[i], position_idx[i])."""
    return _rule_mandated_legal_moves(
        states,
        next_hand_color,
        game_idx,
        position_idx,
        scored_squares,
        flip_square=square,
    )


def informative_indices(
    states: np.ndarray,
    next_hand_color: np.ndarray,
    square: tuple[int, int],
    occ_labels: np.ndarray,
    length: int,
    scored_squares: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Flat (non-empty) indices at ``square`` where flipping its owner actually changes
    the rule-mandated legal-move status of at least one scored square.

    Checked directly (50 games, target=centre square): only ~9% of non-empty rows are
    informative by this definition, because flipping one square rarely changes legality
    at a *different, often distant* scored square. Pooling F1 over the other ~91% -- rows
    where the counterfactual and the clean board agree on every scored square, so every
    condition (real edit, null edit, random direction) scores the same trivially-correct
    F1 there -- dilutes a real, strong effect (measured: F1=0.97 real vs 0.74 null on the
    informative subset alone, vs both landing around 0.96-0.97 once averaged over every
    row) into what looks like no effect at all. Restricting to informative rows is what
    makes this quantity actually measure controllability rather than mostly measuring
    "predicts its own unedited next move," which every condition does equally well.
    """
    eligible = np.flatnonzero(occ_labels != 1)
    game_idx = eligible // length
    position_idx = eligible % length
    clean = clean_legal_moves(
        states, next_hand_color, game_idx, position_idx, scored_squares
    )
    counterfactual = counterfactual_legal_moves(
        states, next_hand_color, square, game_idx, position_idx, scored_squares
    )
    return eligible[(clean != counterfactual).any(axis=1)]


def _token_id_for_square(square: tuple[int, int]) -> int:
    r, c = square
    pos = r * 8 + c
    if pos not in stoi:
        raise ValueError(
            f"square {square} has no vocab token; centre squares are never played "
            "and cannot be a scored square (they can still be an edit target)."
        )
    return stoi[pos]


@torch.no_grad()
def splice_and_read_legal_moves(
    model: HookedTransformer,
    hook_name: str,
    device: str,
    tokens: torch.Tensor,
    game_idx: np.ndarray,
    position_idx: np.ndarray,
    edited_activations: torch.Tensor,
    scored_squares: tuple[tuple[int, int], ...],
    batch_size: int = 64,
) -> np.ndarray:
    """Splice ``edited_activations[i]`` into the model at (game_idx[i], position_idx[i])
    for every i, reusing exactly the hook mechanism groundtruth/fidelity.py's
    next_move_kl already uses (a `fwd_hooks` entry passed to model.run_with_hooks);
    unlike that full-batch reconstruction splice, this hook only overwrites the specific
    edited positions, leaving every other token's activation untouched.

    Returns a (len(game_idx), len(scored_squares)) 0/1 array: whether the model's own
    post-edit output puts more than the uniform-prior baseline's probability on each
    scored square as the next move (UNIFORM_LEGAL_THRESHOLD, see module docstring).
    """
    token_ids = torch.as_tensor([_token_id_for_square(sq) for sq in scored_squares])
    n = len(game_idx)
    out = np.zeros((n, len(scored_squares)), dtype=np.int8)

    for start in range(0, tokens.shape[0], batch_size):
        end = start + batch_size
        batch_mask = (game_idx >= start) & (game_idx < end)
        if not batch_mask.any():
            continue
        batch = tokens[start:end].to(device)
        local_rows = torch.as_tensor(game_idx[batch_mask] - start, dtype=torch.long)
        local_pos = torch.as_tensor(position_idx[batch_mask], dtype=torch.long)
        local_values = edited_activations[torch.as_tensor(batch_mask)].to(device)

        def splice_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            # Same naming requirement as fidelity.py's splice_hook: transformer_lens
            # calls hook(module_output, hook=self), so the parameter must be named
            # exactly `hook`.
            edited = activation.clone()
            edited[local_rows, local_pos, :] = local_values.to(activation.dtype)
            return edited

        logits = model.run_with_hooks(batch, fwd_hooks=[(hook_name, splice_hook)])
        probs = logits.softmax(dim=-1)
        row_probs = probs[local_rows, local_pos][:, token_ids]
        pred = (row_probs > UNIFORM_LEGAL_THRESHOLD).cpu().numpy().astype(np.int8)
        out[np.flatnonzero(batch_mask)] = pred

    return out


def causal_controllability_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pooled F1 over every (row, scored-square) pair, matching legal_move_f1's pattern
    of scoring a predicted binary set against ground truth (pooled here rather than
    fit-and-score on held-out rows, since there is no fitting step on the model side)."""
    return float(
        f1_score(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0)  # pyright: ignore[reportArgumentType]
    )


def causal_controllability_f1_for_direction(
    rows: MineYoursRows,
    direction: np.ndarray,
    scale: float,
    to_activation: Callable[[np.ndarray], torch.Tensor],
    true_legal_moves: np.ndarray,
    scored_squares: tuple[tuple[int, int], ...],
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    device: str,
    batch_size: int = 64,
) -> float:
    """One F1 number: edit ``rows`` by ``direction``*``scale``, map to activation space
    via ``to_activation`` (identity for the oracle quantity, sae.decode for the SAE
    quantity), splice into the model's own forward pass, and score its post-edit output
    against ``true_legal_moves`` (the rule-mandated legal moves on the counterfactual,
    ``target_square``'s owner flipped, board -- from `counterfactual_legal_moves`).

    ``true_legal_moves`` is a parameter rather than computed here because it depends
    only on ``rows.game_idx``/``rows.position_idx`` and the target square, never on the
    edit itself: every one of the ~10 conditions (real, null, 3-seeded-random, times two
    direction types) scored per square shares the exact same value, so the caller should
    compute it once per square -- and, since it is also SAE-independent, share that one
    computation across the oracle quantity and every SAE too, rather than have this
    function recompute an identical rule-engine pass on every call.
    """
    edited = apply_edit(rows, direction, scale)
    edited_activations = to_activation(edited)
    pred = splice_and_read_legal_moves(
        model,
        hook_name,
        device,
        tokens,
        rows.game_idx,
        rows.position_idx,
        edited_activations,
        scored_squares,
        batch_size,
    )
    return causal_controllability_f1(true_legal_moves, pred)
