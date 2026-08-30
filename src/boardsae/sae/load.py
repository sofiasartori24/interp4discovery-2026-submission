"""Load pretrained board-game SAEs from a SAE_BoardGameEval trainer directory.

The SAEs ship as (config.json, ae.pt) pairs under the SAE_BoardGameEval clone. Their
``dictionary`` module is self-contained (torch + einops), but importing it through the
package would trigger ``dictionary_learning.__init__``, which pulls in nnsight via the
training buffer. ``dictionary.py`` is loaded by file path instead, keeping this
project's dependency tree at the transformer_lens stack.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn

from boardsae.paths import board_game_eval_path

# dict_class values implemented in dictionary.py (no relative imports). TopK variants
# live under trainers/ with relative imports and are not wired yet.
_SUPPORTED = {
    "AutoEncoder",
    "GatedAutoEncoder",
    "AutoEncoderNew",
    "JumpReluAutoEncoder",
}

# The released Othello sweep's Standard/Gated trainer configs omit dict_class entirely
# (only PAnnealTrainer/GatedAnnealTrainer log it); inferred here from trainer_class so
# the full population loads, not just the two trainers that happen to log it.
_DICT_CLASS_BY_TRAINER_CLASS = {
    "StandardTrainer": "AutoEncoder",
    "PAnnealTrainer": "AutoEncoder",
    "GatedSAETrainer": "GatedAutoEncoder",
    "GatedAnnealTrainer": "GatedAutoEncoder",
}


@dataclass(frozen=True)
class LoadedSAE:
    """A pretrained SAE plus the metadata the results schema needs."""

    module: nn.Module
    sae_id: str
    architecture: str
    width: int
    activation_dim: int
    layer: int


def resolve_group_path(path: Path) -> Path:
    """Resolve a (possibly relative) SAE group path against the SAE_BoardGameEval clone."""
    return path if path.is_absolute() else board_game_eval_path() / path


def find_sae_dirs(group_path: Path) -> list[Path]:
    """Return every trainer directory (one holding an ae.pt) under ``group_path``, sorted."""
    return sorted(p.parent for p in group_path.rglob("ae.pt"))


@contextlib.contextmanager
def _load_checkpoints_to_cpu() -> Iterator[None]:
    """Force ``torch.load`` onto CPU within the block.

    SAE_BoardGameEval checkpoints were saved from cuda:0, and their ``from_pretrained``
    calls ``torch.load`` without a ``map_location``, which errors on a CPU/MPS host.
    Storages are mapped to CPU on load; ``from_pretrained`` then moves the module to the
    target device.
    """
    original = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        # The checkpoints are trusted local files; setting this explicitly silences the
        # weights_only FutureWarning without changing behaviour.
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = cpu_load
    try:
        yield
    finally:
        torch.load = original


def _dictionary_module() -> ModuleType:
    """Import SAE_BoardGameEval's dictionary.py by path, bypassing its package __init__."""
    dict_py = (
        board_game_eval_path() / "circuits" / "dictionary_learning" / "dictionary.py"
    )
    if not dict_py.exists():
        raise FileNotFoundError(
            f"dictionary.py not found at {dict_py}. "
            "Run `git submodule update --init` in resources/SAE_BoardGameEval."
        )
    spec = importlib.util.spec_from_file_location("boardsae_vendor_dictionary", dict_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {dict_py}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_dict_class(trainer: dict) -> str:
    """Return dict_class, inferring it from trainer_class when config.json omits it."""
    if "dict_class" in trainer:
        return trainer["dict_class"]
    trainer_class = trainer.get("trainer_class")
    if trainer_class in _DICT_CLASS_BY_TRAINER_CLASS:
        return _DICT_CLASS_BY_TRAINER_CLASS[trainer_class]
    raise KeyError(
        f"config.json has no dict_class and trainer_class {trainer_class!r} is not in "
        f"_DICT_CLASS_BY_TRAINER_CLASS; add it there."
    )


def load_sae(base_path: Path, device: str) -> LoadedSAE:
    """Load the SAE in ``base_path`` (a trainer dir with config.json and ae.pt)."""
    config = json.loads((base_path / "config.json").read_text())
    trainer = config["trainer"]
    dict_class = _resolve_dict_class(trainer)
    if dict_class not in _SUPPORTED:
        raise NotImplementedError(
            f"dict_class {dict_class!r} is not wired yet (its loader lives outside "
            f"dictionary.py). Supported: {sorted(_SUPPORTED)}."
        )
    cls = getattr(_dictionary_module(), dict_class)
    with _load_checkpoints_to_cpu():
        module = cls.from_pretrained(str(base_path / "ae.pt"), device=device)
    module.eval()
    # width/activation_dim come from the loaded module, not config.json: dict_class's
    # from_pretrained infers both from the checkpoint's own tensor shapes
    # (state_dict["encoder.weight"].shape), which is correct even for the Standard/Gated
    # configs above that never log dict_size/activation_dim at all.
    # base_path.name alone is not unique across the population: each architecture
    # group numbers its trainer folders independently, so e.g. "trainer0" exists once
    # per architecture and would collide across groups. Prefixing with trainer_class
    # makes the id unique everywhere it's actually used (the population loop).
    return LoadedSAE(
        module=module,
        sae_id=f"{trainer.get('trainer_class', dict_class)}/{base_path.name}",
        architecture=trainer.get("trainer_class", dict_class),
        width=int(module.dict_size),
        activation_dim=int(module.activation_dim),
        layer=int(trainer["layer"]),
    )
