"""Unit test: the tidy results schema round-trips through parquet."""

from __future__ import annotations

from pathlib import Path

from boardsae.io.results import QuantityKind, Record, append_records, load_records


def _record(name: str, value: float) -> Record:
    return Record(
        sae_id="sae-0",
        architecture="standard",
        width=4096,
        l0=42.0,
        model_name="othello-gpt",
        layer=6,
        seed=0,
        quantity_kind=QuantityKind.PROXY_METRIC,
        name=name,
        value=value,
    )


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    append_records([_record("loss_recovered", 0.9)], path)
    append_records([_record("l0", 42.0)], path)

    df = load_records(path)
    assert len(df) == 2
    assert set(df["name"]) == {"loss_recovered", "l0"}
    assert df["quantity_kind"].iloc[0] == QuantityKind.PROXY_METRIC.value
