"""Manifest-first quantification bags and deterministic reconstruction."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from tabpollution.mixing.contamination import half_up_count


def build_bag_members(
    real_pool: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
    bag_id: str,
    proportion: float,
    bag_size: int,
    mix_seed: int,
) -> pd.DataFrame:
    synthetic_count = half_up_count(bag_size, proportion)
    real_count = bag_size - synthetic_count
    if real_count > len(real_pool) or synthetic_count > len(synthetic_pool):
        raise ValueError("Bag requires more rows than available for within-bag sampling")
    real_ids = real_pool.sample(n=real_count, replace=False, random_state=mix_seed)["row_id"].astype(str)
    synthetic_ids = synthetic_pool.sample(
        n=synthetic_count, replace=False, random_state=mix_seed + 1
    )["synth_row_id"].astype(str)
    members = pd.DataFrame(
        {
            "bag_id": [bag_id] * bag_size,
            "source_type": ["real"] * real_count + ["synthetic"] * synthetic_count,
            "record_id": [*real_ids, *synthetic_ids],
            "position": range(bag_size),
        }
    )
    members = members.sample(frac=1, random_state=mix_seed + 2).reset_index(drop=True)
    members["position"] = range(bag_size)
    if members["record_id"].duplicated().any():
        raise ValueError("A bag contains duplicate record IDs")
    return members


def members_hash(members: pd.DataFrame) -> str:
    text = members.sort_values("position").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bag_summary(
    members: pd.DataFrame,
    dataset: str,
    generator: str,
    stage: str,
    requested_p: float,
    mix_seed: int,
) -> dict[str, Any]:
    synthetic_count = int((members["source_type"] == "synthetic").sum())
    return {
        "bag_id": str(members["bag_id"].iloc[0]),
        "dataset": dataset,
        "generator": generator,
        "stage": stage,
        "requested_proportion": requested_p,
        "actual_proportion": synthetic_count / len(members),
        "bag_size": len(members),
        "real_count": len(members) - synthetic_count,
        "synthetic_count": synthetic_count,
        "mix_seed": mix_seed,
        "sampling": "without_replacement_within_bag;reuse_allowed_between_bags",
        "members_sha256": members_hash(members),
    }


def rebuild_bag(
    members: pd.DataFrame,
    real_pool: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
) -> pd.DataFrame:
    real_by_id = real_pool.set_index(real_pool["row_id"].astype(str))
    synth_by_id = synthetic_pool.set_index(synthetic_pool["synth_row_id"].astype(str))
    rows = []
    for member in members.sort_values("position").itertuples(index=False):
        if member.source_type == "real":
            row = real_by_id.loc[member.record_id].to_dict()
        else:
            row = synth_by_id.loc[member.record_id].to_dict()
        row.update(
            {
                "bag_id": member.bag_id,
                "source_type": member.source_type,
                "record_id": member.record_id,
                "position": member.position,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

