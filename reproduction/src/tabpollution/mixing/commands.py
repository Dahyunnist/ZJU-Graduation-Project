"""Project-aware CLI workflows for manifest-first C3 operations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from tabpollution.mixing.bags import bag_summary, build_bag_members, members_hash, rebuild_bag
from tabpollution.mixing.manifest_first import contamination_recipe, write_recipe
from tabpollution.mixing.smoke import _partitioned_adult, _pool, _successful_c2_run
from tabpollution.utils import sha256_file, write_json


def build_smoke_contamination_recipe(
    project_root: str | Path,
    generator_name: str,
    condition: str,
    proportion: float,
    output: str | Path,
    materialize: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    c2 = _successful_c2_run(root, generator_name, 42)
    real = _partitioned_adult(root, 2026)
    real_source = real.loc[real["split"] == "R_source_train"].copy()
    synthetic = _pool(c2, "S_downstream_mix")
    rate_index = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0].index(float(proportion))
    condition_index = ["real_only", "real_append", "synthetic_append", "synthetic_replace"].index(condition)
    mix_seed = 42 * 10000 + rate_index * 100 + condition_index
    recipe, members = contamination_recipe(
        real_source,
        synthetic,
        condition,
        proportion,
        mix_seed,
        dataset="adult",
        generator=generator_name,
        real_extra_pool=real_source,
    )
    return write_recipe(
        recipe,
        members,
        root / output,
        materialize=materialize,
        real_pool=real_source,
        synthetic_pool=synthetic,
    )


def build_smoke_bag_manifest(
    project_root: str | Path,
    generator_name: str,
    stage: str,
    proportion: float,
    bag_index: int,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    c2 = _successful_c2_run(root, generator_name, 42)
    real = _partitioned_adult(root, 2026)
    if stage == "calibration":
        real_pool = real.loc[real["split"] == "R_detector_val"].copy()
        synth_pool = _pool(c2, "S_detector_val")
        stage_index = 0
    elif stage == "test":
        real_pool = real.loc[real["split"] == "R_final_test"].copy()
        synth_pool = _pool(c2, "S_final_test")
        stage_index = 1
    else:
        raise ValueError("stage must be calibration or test")
    rates = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    rate_index = rates.index(float(proportion))
    mix_seed = 42 * 100000 + stage_index * 10000 + rate_index * 100 + bag_index
    bag_id = (
        f"bag:adult:{generator_name}:s42:{stage}:"
        f"p{int(proportion * 100):03d}:b{bag_index:02d}"
    )
    members = build_bag_members(real_pool, synth_pool, bag_id, proportion, 1000, mix_seed)
    output_path = root / output
    members_path = output_path.with_suffix(".members.csv")
    members_path.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(members_path, index=False, lineterminator="\n")
    summary = bag_summary(members, "adult", generator_name, stage, proportion, mix_seed)
    payload = {
        "format": "bag-manifest-first-v1",
        "run_type": "smoke_fixture",
        "members_file": members_path.name,
        "members_file_sha256": sha256_file(members_path),
        "materialized_features": False,
        **summary,
    }
    write_json(payload, output_path)
    return payload


def rebuild_smoke_bag(
    project_root: str | Path,
    generator_name: str,
    manifest_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_file = root / manifest_path
    import json

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    members = pd.read_csv(manifest_file.with_name(manifest["members_file"]), dtype={"record_id": "string"})
    c2 = _successful_c2_run(root, generator_name, 42)
    real = _partitioned_adult(root, 2026)
    if manifest["stage"] == "calibration":
        real_pool = real.loc[real["split"] == "R_detector_val"].copy()
        synth_pool = _pool(c2, "S_detector_val")
    else:
        real_pool = real.loc[real["split"] == "R_final_test"].copy()
        synth_pool = _pool(c2, "S_final_test")
    rebuilt = rebuild_bag(members, real_pool, synth_pool)
    content = rebuilt.astype("string").fillna("<NA>").to_csv(index=False, lineterminator="\n")
    return {
        "bag_id": manifest["bag_id"],
        "rows": len(rebuilt),
        "source_counts": rebuilt["source_type"].value_counts().to_dict(),
        "actual_proportion": float((rebuilt["source_type"] == "synthetic").mean()),
        "members_sha256": members_hash(members),
        "expected_members_sha256": manifest["members_sha256"],
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
