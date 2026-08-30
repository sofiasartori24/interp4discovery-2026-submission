"""Locate the SAE_BoardGameEval clone and put it on sys.path.

Reuses SAE_BoardGameEval's board utilities (the Othello engine and ground-truth board
states) without adding its heavy dependency tree to this project. Only the pure
``circuits.othello_utils`` / ``circuits.othello_engine_utils`` modules are imported,
which need just ``datasets`` plus the standard scientific stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def board_game_eval_path() -> Path:
    """Return the SAE_BoardGameEval repo path (env override, else the sibling clone)."""
    env = os.environ.get("BOARDGAMEEVAL_PATH")
    if env:
        return Path(env).expanduser().resolve()
    # This repo lives at <root>/experiments/<this-experiment>/; the clone is at <root>/resources/.
    project_root = Path(__file__).resolve().parents[4]
    return project_root / "resources" / "SAE_BoardGameEval"


def ensure_on_path() -> None:
    """Add the SAE_BoardGameEval clone to sys.path (idempotent)."""
    path = board_game_eval_path()
    if not path.exists():
        raise FileNotFoundError(
            f"SAE_BoardGameEval not found at {path}. "
            "Clone it into resources/ or set BOARDGAMEEVAL_PATH."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
