"""Unit tests for population sampling (pure path/JSON logic, offline, no torch)."""

from __future__ import annotations

import json
from pathlib import Path

from boardsae.sae.population import (
    PopulationEntry,
    evenly_spaced_sample,
    list_population,
    stratified_population,
)


def _write_trainer(
    group_dir: Path, name: str, trainer_class: str, sparsity: float
) -> None:
    trainer_dir = group_dir / name
    trainer_dir.mkdir(parents=True)
    (trainer_dir / "config.json").write_text(
        json.dumps(
            {"trainer": {"trainer_class": trainer_class, "l1_penalty": sparsity}}
        )
    )


def test_list_population_skips_configs_without_a_sparsity_field(tmp_path: Path) -> None:
    group = tmp_path / "standard"
    _write_trainer(group, "trainer0", "StandardTrainer", 0.1)
    (group / "trainer1").mkdir(parents=True)
    (group / "trainer1" / "config.json").write_text(json.dumps({"trainer": {}}))

    entries = list_population(group)
    assert len(entries) == 1
    assert entries[0].sparsity_param == 0.1


def test_evenly_spaced_sample_covers_both_endpoints() -> None:
    entries = [
        PopulationEntry(Path(f"t{i}"), "StandardTrainer", float(i)) for i in range(20)
    ]
    sample = evenly_spaced_sample(entries, n=4)
    sparsities = sorted(e.sparsity_param for e in sample)
    assert sparsities[0] == 0.0
    assert sparsities[-1] == 19.0
    assert len(sample) == 4


def test_evenly_spaced_sample_returns_everything_when_population_is_small() -> None:
    entries = [PopulationEntry(Path("t0"), "StandardTrainer", 0.1)]
    assert evenly_spaced_sample(entries, n=8) == entries


def test_stratified_population_samples_every_present_architecture_group(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    for i in range(10):
        _write_trainer(
            release / "othello-trained_model-layer_5-standard",
            f"trainer{i}",
            "StandardTrainer",
            i * 0.01,
        )
    for i in range(10):
        _write_trainer(
            release / "othello-trained_model-layer_5-p_anneal",
            f"trainer{i}",
            "PAnnealTrainer",
            i * 0.01,
        )
    # gated / gated_anneal directories absent: a partial release should not raise.

    sample = stratified_population(release, n_per_architecture=3)
    trainer_classes = {e.trainer_class for e in sample}
    assert trainer_classes == {"StandardTrainer", "PAnnealTrainer"}
    assert len(sample) == 6
