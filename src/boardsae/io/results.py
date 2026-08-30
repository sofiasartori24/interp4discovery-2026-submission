"""Tidy results schema and parquet IO — the analysis backbone.

Every measurement is one row (SAE metadata, quantity kind, name, value). Ground-truth
quantities and proxy metrics share the schema, so axis discovery and metric-validity
analyses are plain pandas over a single table.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class QuantityKind(str, Enum):
    """A quantity is a metric-independent ground-truth anchor or a proxy metric under audit."""

    GROUND_TRUTH = "ground_truth"
    PROXY_METRIC = "proxy_metric"


class Record(BaseModel):
    """One measurement of one quantity on one SAE, for one seed.

    ``name`` identifies the quantity, e.g. ``occupancy_f1`` (ground truth) or
    ``loss_recovered`` (proxy metric).
    """

    sae_id: str
    architecture: str
    width: int
    l0: float | None
    model_name: str
    layer: int
    seed: int
    quantity_kind: QuantityKind
    name: str
    value: float

    model_config = {"protected_namespaces": ()}


def records_to_frame(records: list[Record]) -> pd.DataFrame:
    """Convert records to a DataFrame with enum values flattened to strings."""
    return pd.DataFrame(
        [{**r.model_dump(), "quantity_kind": r.quantity_kind.value} for r in records]
    )


def append_records(records: list[Record], path: Path) -> None:
    """Append records to the parquet store, creating it (and parents) if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = records_to_frame(records)
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    frame.to_parquet(path, index=False)


def load_records(path: Path) -> pd.DataFrame:
    """Load the full tidy results table."""
    return pd.read_parquet(path)
