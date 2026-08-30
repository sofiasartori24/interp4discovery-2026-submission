"""Enumerate and sample the released SAE population from the sweep directory tree.

Pure path/JSON logic, no torch: this stays independently unit-testable (no GPU, no
network) so the sampling strategy can be checked without loading a single checkpoint.
Running the full sweep (240+ checkpoints) is unnecessary and slow; a stratified sample
spread across architecture x sparsity keeps the validity-matrix analysis well-powered
without the extra compute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The four architecture-group directories in the 2024-05-23 Othello release. Width is
# deliberately not part of this listing: PAnnealTrainer/GatedAnnealTrainer log dict_size
# in config.json, but StandardTrainer/GatedSAETrainer do not (see sae/load.py), so width
# is only known once a checkpoint is actually loaded. Sampling by architecture x
# sparsity avoids depending on a field that half the population doesn't have.
ARCHITECTURE_GROUPS: tuple[str, ...] = (
    "othello-trained_model-layer_5-standard",
    "othello-trained_model-layer_5-gated",
    "othello-trained_model-layer_5-p_anneal",
    "othello-trained_model-layer_5-gated_anneal",
)


@dataclass(frozen=True)
class PopulationEntry:
    """One trainer checkpoint plus the config fields the sampler needs."""

    path: Path
    trainer_class: str
    sparsity_param: float  # l1_penalty or sparsity_penalty, whichever the trainer logs


def _sparsity_param(trainer: dict) -> float | None:
    """Return whichever sparsity-target field this trainer class logs."""
    for key in ("l1_penalty", "sparsity_penalty"):
        value = trainer.get(key)
        if value is not None:
            return float(value)
    return None


def list_population(group_dir: Path) -> list[PopulationEntry]:
    """List every trainer checkpoint under one architecture-group directory.

    Entries with no recognizable sparsity field are skipped rather than raising: a
    released sweep can carry partial exports, and a silent skip here is safer than
    crashing a population run over one malformed directory.
    """
    entries: list[PopulationEntry] = []
    for config_path in sorted(group_dir.rglob("config.json")):
        trainer = json.loads(config_path.read_text())["trainer"]
        sparsity = _sparsity_param(trainer)
        if sparsity is None:
            continue
        entries.append(
            PopulationEntry(
                path=config_path.parent,
                trainer_class=trainer.get("trainer_class", "unknown"),
                sparsity_param=sparsity,
            )
        )
    return entries


def evenly_spaced_sample(
    entries: list[PopulationEntry], n: int
) -> list[PopulationEntry]:
    """Pick ``n`` entries spread evenly across the sorted sparsity range.

    Covers low, mid, and high L0 rather than clustering wherever the release happens to
    have more checkpoints. Both range endpoints are included when n > 1.
    """
    if len(entries) <= n:
        return entries
    ordered = sorted(entries, key=lambda e: e.sparsity_param)
    if n <= 1:
        return [ordered[0]]
    step = (len(ordered) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [ordered[i] for i in indices]


def stratified_population(
    release_dir: Path, n_per_architecture: int = 8
) -> list[PopulationEntry]:
    """Stratified sample across all four released architecture groups.

    Each group is sampled independently and evenly across its own sparsity range, then
    concatenated. The reported population run uses 20 per architecture, for 80 SAEs; the
    default of 8 (32 SAEs) is the cheaper smoke-test size, not the reported one.

    Note that this is a deliberate spread along the sparsity hyperparameter, not a random
    sample of the release. The paper has to declare that, because it is what puts the
    collapsed dictionaries in the population as the low extreme rather than by chance.
    """
    sample: list[PopulationEntry] = []
    for group_name in ARCHITECTURE_GROUPS:
        group_dir = release_dir / group_name
        if not group_dir.exists():
            continue
        sample.extend(
            evenly_spaced_sample(list_population(group_dir), n_per_architecture)
        )
    return sample
