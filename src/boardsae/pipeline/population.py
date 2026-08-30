"""Population run: loop the full quantity/metric set over a stratified sample of the
released SAE sweep.

Occupancy and legal-move recovery across a stratified square sample, next-move
fidelity, causal controllability, and the full proxy-metric set. The model, games,
boards, legal moves, and cached activations are computed once and shared across every
SAE in the loop; only load / encode / decode repeats per SAE.

This is compute-heavy: run ``make test`` first, then a small ``n_saes_per_architecture``
(e.g. 1 or 2) as a smoke test before a full population pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from transformer_lens import HookedTransformer

from boardsae.config import RunConfig
from boardsae.groundtruth.causal_controllability import (
    EDIT_SCALE,
    DirectionNotEstimable,
    causal_controllability_f1_for_direction,
    counterfactual_legal_moves,
    informative_indices,
    logreg_direction,
    mean_diff_direction,
    normalized_direction,
    random_direction,
    replay_boards_and_turns,
    rows_from_indices,
    select_indices,
)
from boardsae.groundtruth.fidelity import next_move_kl
from boardsae.groundtruth.legal_moves import true_legal_moves
from boardsae.groundtruth.othello_boards import sample_games, true_mine_yours_boards
from boardsae.io.results import QuantityKind, Record, append_records
from boardsae.metrics.reconstruction import (
    dead_feature_rate,
    fraction_variance_unexplained,
    l1,
)
from boardsae.metrics.sparsity import l0
from boardsae.models.othello import (
    cache_resid_activations,
    load_othello_gpt,
    resolve_device,
)
from boardsae.sae.encode import decode_features, encode_features
from boardsae.sae.load import load_sae, resolve_group_path
from boardsae.sae.population import PopulationEntry, stratified_population
from boardsae.validity.legal_move_probe import LegalMoveNotEstimable, probe_legal_move
from boardsae.validity.occupancy_probe import probe_occupancy

# Corners, edge midpoints, and centre: covers the board's structurally distinct regions
# instead of all 64 squares (64 probes x ~30 SAEs is on the order of 2000 regressions;
# this cuts it to 16 x 30). A declared sampling choice, not an incidental shortcut.
SAMPLED_SQUARES: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 7),
    (7, 0),
    (7, 7),  # corners
    (0, 3),
    (0, 4),
    (3, 0),
    (4, 0),
    (3, 7),
    (4, 7),
    (7, 3),
    (7, 4),  # edge midpoints
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),  # centre
)

CENTRE_SQUARES: frozenset[tuple[int, int]] = frozenset({(3, 3), (3, 4), (4, 3), (4, 4)})

# Causal-controllability *edits* every SAMPLED_SQUARES entry including centre (centre
# squares are always Mine-or-Yours from move zero, the richest, not the worst, squares
# for an ownership-flip edit) but only *scores* legal-move predictions on the non-centre
# subset, for the same reason legal_move_f1 already excludes centre there: it is never a
# legal move, so the rule-mandated label is degenerate.
SCORED_SQUARES: tuple[tuple[int, int], ...] = tuple(
    sq for sq in SAMPLED_SQUARES if sq not in CENTRE_SQUARES
)


def _causal_controllability_scores(
    values: np.ndarray,
    occ_labels: np.ndarray,
    length: int,
    flat_indices: np.ndarray,
    true_legal_moves: np.ndarray,
    model: HookedTransformer,
    tokens: torch.Tensor,
    hook_name: str,
    device: str,
    to_activation: Callable[[np.ndarray], torch.Tensor],
    seeds: tuple[int, ...],
) -> dict[str, float | dict[int, float]]:
    """Every causal-controllability number for one square, one space (oracle or SAE).

    Returns a flat dict: {"meandiff": f1, "logreg": f1, "meandiff_null": f1,
    "logreg_null": f1, "meandiff_random": {seed: f1, ...}, "logreg_random": {seed: f1,
    ...}}. Raises DirectionNotEstimable (caller's responsibility to catch and skip) if
    the square has only one class in this sample.

    ``flat_indices`` (from `select_indices` over `informative_indices`) and
    ``true_legal_moves`` (from `counterfactual_legal_moves` on those same indices) are
    both computed once per square and shared across the oracle quantity and every SAE:
    neither depends on the edit or on which SAE (if any) is being scored, so recomputing
    them per call would just repeat the same rule-engine work ~10 times per square per
    SAE for no reason (see causal_controllability_f1_for_direction's docstring).
    """
    rows = rows_from_indices(values, occ_labels, length, flat_indices)
    meandiff = mean_diff_direction(rows)
    ref_norm = float(np.linalg.norm(meandiff))
    logreg = normalized_direction(logreg_direction(rows, seed=seeds[0]), ref_norm)

    def f1_for(direction: np.ndarray, scale: float) -> float:
        return causal_controllability_f1_for_direction(
            rows,
            direction,
            scale,
            to_activation,
            true_legal_moves,
            SCORED_SQUARES,
            model,
            tokens,
            hook_name,
            device,
        )

    scores: dict[str, float | dict[int, float]] = {
        "meandiff": f1_for(meandiff, EDIT_SCALE),
        "logreg": f1_for(logreg, EDIT_SCALE),
        # Null edit (scale=0): confirms an edit is actually necessary to match the
        # flipped counterfactual -- the direction argument is irrelevant at scale=0.
        "meandiff_null": f1_for(meandiff, 0.0),
        "logreg_null": f1_for(logreg, 0.0),
        "meandiff_random": {},
        "logreg_random": {},
    }
    dim = values.shape[1]
    for seed in seeds:
        # Independent random directions per slot (offset seed) so the two random-control
        # columns aren't literally identical by construction.
        scores["meandiff_random"][seed] = f1_for(  # type: ignore[index]
            random_direction(dim, ref_norm, seed), EDIT_SCALE
        )
        scores["logreg_random"][seed] = f1_for(  # type: ignore[index]
            random_direction(dim, ref_norm, seed + 1000), EDIT_SCALE
        )
    return scores


def run(config: RunConfig) -> None:
    """Run the full quantity/metric set over a stratified sample of the SAE population."""
    device = resolve_device(config.device)
    model = load_othello_gpt(device)

    games = sample_games(config.n_games)
    tokens = torch.tensor(games)
    boards = true_mine_yours_boards(games)  # (B, L, 8, 8, 3): yours / empty / mine
    legal = true_legal_moves(games)  # (B, L, 8, 8, 1): 1 if legal at that position
    acts_bld = cache_resid_activations(
        model, tokens, config.hook_name, device, config.batch_size
    )
    n, length = acts_bld.shape[0], acts_bld.shape[1]
    acts = acts_bld.reshape(n * length, -1).float()

    # Causal-controllability's counterfactual ground truth needs the raw signed board
    # state and whose turn it is, not the one-hot mine/yours/legal tensors above; shared
    # across every square (the square being edited doesn't change the underlying replay)
    # and every SAE (it depends only on the games, not on any SAE).
    states, next_hand_color = replay_boards_and_turns(games, length)

    # Oracle causal-controllability, and the (row selection, counterfactual ground
    # truth) pair each square scores against, are all SAE-independent -- computed once
    # here rather than once per SAE, and (within a square) once total rather than once
    # per condition (see causal_controllability_f1_for_direction's docstring for why
    # that redundant recomputation mattered: ~10x the rule-engine work per square per
    # SAE for an identical result every time).
    oracle_scores: dict[tuple[int, int], dict[str, float | dict[int, float]]] = {}
    flat_indices_by_square: dict[tuple[int, int], np.ndarray] = {}
    true_legal_moves_by_square: dict[tuple[int, int], np.ndarray] = {}
    acts_np = acts.numpy()
    for r, c in SAMPLED_SQUARES:
        occ_labels = (
            boards[:, :length, r, c, :].argmax(dim=-1).reshape(n * length).numpy()
        )
        informative = informative_indices(
            states, next_hand_color, (r, c), occ_labels, length, SCORED_SQUARES
        )
        flat_indices = select_indices(occ_labels, candidate_indices=informative)
        counterfactual_truth = counterfactual_legal_moves(
            states,
            next_hand_color,
            (r, c),
            flat_indices // length,
            flat_indices % length,
            SCORED_SQUARES,
        )
        flat_indices_by_square[(r, c)] = flat_indices
        true_legal_moves_by_square[(r, c)] = counterfactual_truth
        try:
            oracle_scores[(r, c)] = _causal_controllability_scores(
                acts_np,
                occ_labels,
                length,
                flat_indices,
                counterfactual_truth,
                model,
                tokens,
                config.hook_name,
                device,
                lambda x: torch.as_tensor(x, dtype=torch.float32),
                config.seeds,
            )
        except DirectionNotEstimable as exc:
            print(f"  skipping oracle causal_controllability at ({r},{c}): {exc}")

    release_dir = resolve_group_path(config.population_release_dir)
    population = stratified_population(release_dir, config.n_saes_per_architecture)
    architectures = {entry.trainer_class for entry in population}
    print(
        f"Population: {len(population)} SAEs sampled across {len(architectures)} architectures."
    )

    for entry in population:
        _run_one_sae(
            entry,
            config,
            model,
            tokens,
            acts,
            boards,
            legal,
            n,
            length,
            device,
            oracle_scores,
            flat_indices_by_square,
            true_legal_moves_by_square,
        )


def _record_causal_controllability(
    records: list[Record],
    make_record: Callable[[QuantityKind, str, float, int | None], Record],
    square: tuple[int, int],
    space: str,
    scores: dict[str, float | dict[int, float]],
) -> None:
    """Append every causal-controllability Record for one square and one space (``sae``
    or ``oracle``) from a `_causal_controllability_scores` result: the real edit, the
    null-edit control, and the random-direction control (once per seed)."""
    r, c = square
    for direction_type in ("meandiff", "logreg"):
        records.append(
            make_record(
                QuantityKind.GROUND_TRUTH,
                f"causal_controllability_f1_{direction_type}_{space}_r{r}c{c}",
                cast(float, scores[direction_type]),
                None,
            )
        )
        records.append(
            make_record(
                QuantityKind.GROUND_TRUTH,
                f"causal_controllability_f1_{direction_type}_null_{space}_r{r}c{c}",
                cast(float, scores[f"{direction_type}_null"]),
                None,
            )
        )
        random_scores = scores[f"{direction_type}_random"]
        assert isinstance(random_scores, dict)
        for seed, value in random_scores.items():
            records.append(
                make_record(
                    QuantityKind.GROUND_TRUTH,
                    f"causal_controllability_f1_{direction_type}_random_{space}_r{r}c{c}",
                    value,
                    seed,
                )
            )


def _run_one_sae(
    entry: PopulationEntry,
    config: RunConfig,
    model: HookedTransformer,
    tokens: torch.Tensor,
    acts: torch.Tensor,
    boards: torch.Tensor,
    legal: torch.Tensor,
    n: int,
    length: int,
    device: str,
    oracle_scores: dict[tuple[int, int], dict[str, float | dict[int, float]]],
    flat_indices_by_square: dict[tuple[int, int], np.ndarray],
    true_legal_moves_by_square: dict[tuple[int, int], np.ndarray],
) -> None:
    """Evaluate one SAE against every quantity and metric, and append its records."""
    # Rows are flattened row-major as game_idx*length + position_idx everywhere in this
    # module, so the game each row came from is just its index // length. This is what
    # the probes group their train/test split by (see occupancy_probe._split).
    row_games = (
        np.arange(n * length) // length if config.group_probe_split_by_game else None
    )
    sae = load_sae(entry.path, device)
    features = encode_features(sae.module, acts, device)
    reconstruction = decode_features(sae.module, features, device)
    sparsity = l0(features)

    def _record(
        kind: QuantityKind, name: str, value: float, seed: int | None = None
    ) -> Record:
        # Only the probe-based ground-truth quantities (occupancy, legal-move) have any
        # randomness (train_test_split inside the probe); proxy metrics and next-move KL
        # are deterministic given the SAE and games, so they default to seeds[0] rather
        # than being recomputed per seed for no reason.
        return Record(
            sae_id=sae.sae_id,
            architecture=sae.architecture,
            width=sae.width,
            l0=sparsity,
            model_name=config.model_name,
            layer=config.layer,
            seed=config.seeds[0] if seed is None else seed,
            quantity_kind=kind,
            name=name,
            value=value,
        )

    records: list[Record] = [
        _record(QuantityKind.PROXY_METRIC, "l0", sparsity),
        _record(QuantityKind.PROXY_METRIC, "l1", l1(features)),
        _record(
            QuantityKind.PROXY_METRIC,
            "fvu",
            fraction_variance_unexplained(acts, reconstruction),
        ),
        _record(
            QuantityKind.PROXY_METRIC, "dead_feature_rate", dead_feature_rate(features)
        ),
    ]

    for r, c in SAMPLED_SQUARES:
        occ_labels = (
            boards[:, :length, r, c, :].argmax(dim=-1).reshape(n * length).numpy()
        )
        if len(set(occ_labels.tolist())) < 2:
            # Same degenerate case as legal-move below, in principle: a square that is
            # e.g. always empty within a short, small-n_games sample. Not yet observed,
            # but cheap to guard symmetrically rather than wait for it to happen.
            print(
                f"  skipping occupancy at ({r},{c}) for {sae.sae_id}: only one class in this sample"
            )
        else:
            # Refit per seed: each seed only changes the probe's train/test split, the
            # cached features above are reused as-is, so this is a cheap refit, not a
            # rerun of the model or the SAE. The spread across seeds is the reseed-noise
            # floor (Chanin 2026): how much of a "real" correlation is just probe luck.
            for seed in config.seeds:
                occ_scores = probe_occupancy(
                    features.numpy(), occ_labels, seed=seed, groups=row_games
                )
                records.append(
                    _record(
                        QuantityKind.GROUND_TRUTH,
                        f"occupancy_balacc_r{r}c{c}",
                        occ_scores.balanced_accuracy,
                        seed=seed,
                    )
                )

        legal_labels = legal[:, :length, r, c, 0].reshape(n * length).numpy()
        for seed in config.seeds:
            try:
                legal_scores = probe_legal_move(
                    features.numpy(), legal_labels, seed=seed, groups=row_games
                )
            except LegalMoveNotEstimable as exc:
                print(
                    f"  skipping legal_move_f1 at ({r},{c}) for {sae.sae_id} seed={seed}: {exc}"
                )
            else:
                records.append(
                    _record(
                        QuantityKind.GROUND_TRUTH,
                        f"legal_move_f1_r{r}c{c}",
                        legal_scores.f1,
                        seed=seed,
                    )
                )

        try:
            sae_causal_scores = _causal_controllability_scores(
                features.numpy(),
                occ_labels,
                length,
                flat_indices_by_square[(r, c)],
                true_legal_moves_by_square[(r, c)],
                model,
                tokens,
                config.hook_name,
                device,
                lambda x: sae.module.decode(
                    torch.as_tensor(x, dtype=torch.float32).to(device)
                ).cpu(),
                config.seeds,
            )
        except DirectionNotEstimable as exc:
            print(
                f"  skipping SAE causal_controllability at ({r},{c}) for {sae.sae_id}: {exc}"
            )
        else:
            _record_causal_controllability(
                records, _record, (r, c), "sae", sae_causal_scores
            )
            if (r, c) in oracle_scores:
                _record_causal_controllability(
                    records, _record, (r, c), "oracle", oracle_scores[(r, c)]
                )

    def reconstruct_fn(x: torch.Tensor) -> torch.Tensor:
        # x arrives on `device` from inside the splice hook; stay on-device throughout,
        # since the hook must hand the forward pass back a same-device tensor.
        return sae.module.decode(sae.module.encode(x))

    kl = next_move_kl(
        model, reconstruct_fn, tokens, config.hook_name, device, config.batch_size
    )
    records.append(_record(QuantityKind.GROUND_TRUTH, "next_move_fidelity_kl", kl))

    append_records(records, config.results_path)
    print(
        f"{sae.sae_id} ({sae.architecture}, width={sae.width}, L0={sparsity:.1f}): "
        f"fvu={records[2].value:.4f} kl={kl:.4f} -> {config.results_path}"
    )


if __name__ == "__main__":
    run(RunConfig())
