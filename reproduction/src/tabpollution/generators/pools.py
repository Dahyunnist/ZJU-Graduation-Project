"""Independent synthetic-pool construction and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tabpollution.generators.base import stable_synth_ids
from tabpollution.utils import sha256_file, write_json


POOL_NAMES = ("S_detector_train", "S_detector_val", "S_final_test", "S_downstream_mix")
POOL_PROVENANCE = (
    "synth_row_id",
    "dataset_id",
    "generator_name",
    "generator_seed",
    "sample_seed",
    "pool_name",
)


def add_pool_provenance(
    records: pd.DataFrame,
    dataset_id: str,
    generator_name: str,
    generator_seed: int,
    sample_seed: int,
    pool_name: str,
) -> pd.DataFrame:
    if pool_name not in POOL_NAMES:
        raise ValueError(f"Unknown synthetic pool: {pool_name}")
    result = records.copy()
    result.insert(
        0,
        "synth_row_id",
        stable_synth_ids(result, dataset_id, generator_name, generator_seed, sample_seed, pool_name),
    )
    result["dataset_id"] = dataset_id
    result["generator_name"] = generator_name
    result["generator_seed"] = generator_seed
    result["sample_seed"] = sample_seed
    result["pool_name"] = pool_name
    return result


def pool_content_hash(records: pd.DataFrame, feature_columns: list[str]) -> str:
    canonical = records.loc[:, feature_columns].astype("string").fillna("<NA>")
    text = canonical.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_pools(pools: dict[str, pd.DataFrame], expected_columns: list[str]) -> dict[str, Any]:
    if set(pools) != set(POOL_NAMES):
        raise ValueError(f"Expected exactly four pools: {POOL_NAMES}")
    seen_ids: set[str] = set()
    summary: dict[str, Any] = {}
    for name in POOL_NAMES:
        frame = pools[name]
        ids = set(frame["synth_row_id"].astype(str))
        overlap = seen_ids & ids
        if overlap:
            raise ValueError(f"Synthetic row ID overlap in {name}: {list(overlap)[:3]}")
        seen_ids |= ids
        if frame.loc[:, expected_columns].columns.tolist() != expected_columns:
            raise ValueError(f"Schema/order mismatch in {name}")
        if set(frame["pool_name"].astype(str)) != {name}:
            raise ValueError(f"Incorrect pool_name provenance in {name}")
        summary[name] = {"rows": len(frame), "id_unique": bool(frame["synth_row_id"].is_unique)}
    summary["all_synth_ids_unique"] = len(seen_ids) == sum(len(frame) for frame in pools.values())
    return summary


def write_pool(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
    source_run_id: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_name = str(frame["pool_name"].iloc[0])
    path = output_dir / f"{pool_name}.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    payload = {
        "pool_name": pool_name,
        "rows": len(frame),
        "sample_seed": int(frame["sample_seed"].iloc[0]),
        "file": path.name,
        "file_sha256": sha256_file(path),
        "content_sha256": pool_content_hash(frame, feature_columns),
        "feature_columns": feature_columns,
        "provenance_columns": list(POOL_PROVENANCE),
        "source_run_id": source_run_id,
    }
    write_json(payload, output_dir / f"{pool_name}.json")
    return payload

