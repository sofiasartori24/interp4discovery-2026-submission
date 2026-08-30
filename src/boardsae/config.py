"""Run configuration (pydantic)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Configuration for a single evaluation run over a set of SAEs.

    Seeds drive the reliability / reseed-noise-floor analysis: every quantity is
    recomputed per seed so construct-separation claims can be checked against noise.
    """

    model_name: str = Field(
        default="Baidicoot/Othello-GPT-Transformer-Lens",
        description="transformer_lens model id for OthelloGPT",
    )
    # Layer 5 is where the SAE_BoardGameEval Othello SAEs are trained (config.json:
    # trainer.layer=5, io="out" == blocks.5.hook_resid_post), so the oracle read-out and
    # the SAE read-out share the same activation point.
    layer: int = Field(default=5, description="Residual-stream layer to read")
    hook_name: str = Field(
        default="blocks.5.hook_resid_post",
        description="transformer_lens hook to cache activations from",
    )
    sae_group_path: Path = Field(
        default=Path("autoencoders/testing_othello"),
        description="SAE group dir (relative paths resolve against the SAE_BoardGameEval clone)",
    )
    population_release_dir: Path = Field(
        default=Path("autoencoders/othello-trained_model-layer_5-2024-05-23"),
        description="Released SAE sweep root, for the population run (same path rules as sae_group_path)",
    )
    n_saes_per_architecture: int = Field(
        default=8,
        description="Stratified sample size per architecture group; 4 groups x 8 = 32 SAEs by default",
    )
    seeds: tuple[int, ...] = Field(
        default=(0, 1, 2),
        description="Seeds recomputed per quantity for the reseed-noise floor",
    )
    batch_size: int = Field(
        default=64, description="Batch size for activation collection"
    )
    n_games: int = Field(default=1000, description="Number of evaluation games")
    group_probe_split_by_game: bool = Field(
        default=True,
        description=(
            "Split the occupancy/legal-move probes by game rather than by row. Rows are "
            "(game, position) pairs and consecutive positions within a game are "
            "near-duplicate boards, so a row-level split leaks. False reproduces the "
            "original ungrouped runs for comparison."
        ),
    )
    device: str = Field(
        default="auto", description="Torch device ('auto' -> cuda/mps/cpu)"
    )
    results_path: Path = Field(
        default=Path("results/records.parquet"),
        description="Tidy results store (parquet)",
    )

    # protected_namespaces=() suppresses pydantic's warning on the `model_name` field.
    model_config = {"frozen": True, "protected_namespaces": ()}
