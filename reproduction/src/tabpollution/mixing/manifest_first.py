"""Recipe-first contamination construction and on-demand materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from tabpollution.mixing.contamination import CONDITIONS, half_up_count
from tabpollution.utils import sha256_file, write_json


def _members_hash(members: pd.DataFrame) -> str:
    text = members.sort_values("position").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contamination_members(
    real_base: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
    condition: str,
    proportion: float,
    mix_seed: int,
    real_extra_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown contamination condition: {condition}")
    count = half_up_count(len(real_base), proportion)
    real = real_base.sample(frac=1, random_state=mix_seed).reset_index(drop=True)
    synthetic = synthetic_pool.sample(
        n=count, replace=count > len(synthetic_pool), random_state=mix_seed + 1
    ).reset_index(drop=True)
    parts: list[tuple[str, pd.Series]]
    if condition == "real_only":
        parts = [("real", real["row_id"].astype(str))]
    elif condition == "real_append":
        if real_extra_pool is None:
            raise ValueError("real_append requires explicit real_extra_pool")
        extra = real_extra_pool.sample(
            n=count, replace=count > len(real_extra_pool), random_state=mix_seed + 2
        ).reset_index(drop=True)
        parts = [
            ("real", real["row_id"].astype(str)),
            ("real_bootstrap", extra["row_id"].astype(str)),
        ]
    elif condition == "synthetic_append":
        parts = [
            ("real", real["row_id"].astype(str)),
            ("synthetic", synthetic["synth_row_id"].astype(str)),
        ]
    else:
        parts = [
            ("real", real.iloc[count:]["row_id"].astype(str)),
            ("synthetic", synthetic["synth_row_id"].astype(str)),
        ]
    rows = []
    for source_type, ids in parts:
        rows.extend({"source_type": source_type, "record_id": value} for value in ids.tolist())
    members = pd.DataFrame(rows)
    members["position"] = range(len(members))
    return members.loc[:, ["position", "source_type", "record_id"]]


def contamination_recipe(
    real_base: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
    condition: str,
    proportion: float,
    mix_seed: int,
    *,
    dataset: str,
    generator: str,
    real_extra_pool: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    members = contamination_members(
        real_base, synthetic_pool, condition, proportion, mix_seed, real_extra_pool
    )
    synthetic_count = int((members["source_type"] == "synthetic").sum())
    recipe = {
        "format": "manifest-first-v1",
        "dataset": dataset,
        "generator": generator,
        "condition": condition,
        "effective_condition": (
            "real_append_bootstrap_control" if condition == "real_append" else condition
        ),
        "p_requested": proportion,
        "base_n": len(real_base),
        "rows": len(members),
        "synthetic_count": synthetic_count,
        "actual_synthetic_proportion": synthetic_count / len(members) if len(members) else 0.0,
        "mix_seed": mix_seed,
        "rounding": "half_up=floor(p*N+0.5)",
        "members_sha256": _members_hash(members),
        "materialized_by_default": False,
        "reconstruction": {
            "real_id_column": "row_id",
            "synthetic_id_column": "synth_row_id",
            "real_sampling_seed": mix_seed,
            "synthetic_sampling_seed": mix_seed + 1,
            "real_extra_sampling_seed": mix_seed + 2 if condition == "real_append" else None,
        },
    }
    return recipe, members


def rebuild_contamination(
    members: pd.DataFrame,
    real_pool: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
) -> pd.DataFrame:
    real_by_id = real_pool.set_index(real_pool["row_id"].astype(str), drop=False)
    synth_by_id = synthetic_pool.set_index(synthetic_pool["synth_row_id"].astype(str), drop=False)
    rows = []
    for member in members.sort_values("position").itertuples(index=False):
        source = synth_by_id if member.source_type == "synthetic" else real_by_id
        row = source.loc[str(member.record_id)].to_dict()
        row.update(
            {
                "position": int(member.position),
                "source_type": member.source_type,
                "record_id": str(member.record_id),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_recipe(
    recipe: dict[str, Any],
    members: pd.DataFrame,
    output: str | Path,
    *,
    materialize: bool = False,
    real_pool: pd.DataFrame | None = None,
    synthetic_pool: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output = Path(output)
    write_json(recipe, output)
    result = {"manifest": str(output), "manifest_only": not materialize, **recipe}
    if materialize:
        if real_pool is None or synthetic_pool is None:
            raise ValueError("Materialization requires real and synthetic pools")
        frame = rebuild_contamination(members, real_pool, synthetic_pool)
        data_path = output.with_suffix(".csv")
        frame.to_csv(data_path, index=False, lineterminator="\n")
        result["materialized_file"] = str(data_path)
        result["materialized_sha256"] = sha256_file(data_path)
    return result
