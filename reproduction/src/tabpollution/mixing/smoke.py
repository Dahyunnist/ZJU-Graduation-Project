"""C3 smoke artifact construction from completed C2 Adult generator runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tabpollution.data.loaders import load_processed_dataset
from tabpollution.data.registry import load_dataset_spec
from tabpollution.generators.pools import POOL_NAMES
from tabpollution.mixing.bags import bag_summary, build_bag_members, members_hash, rebuild_bag
from tabpollution.mixing.contamination import CONDITIONS, build_contamination, contamination_summary
from tabpollution.mixing.protocols import validate_protocol
from tabpollution.utils import sha256_file, write_json


def _partitioned_adult(project_root: Path, split_seed: int) -> pd.DataFrame:
    spec = load_dataset_spec(project_root / "configs" / "datasets" / "adult.yaml")
    data = load_processed_dataset(spec, project_root / "data" / "processed" / "adult" / "adult_clean.csv")
    assignment = pd.read_csv(
        project_root / "data" / "splits" / "benchmark_v1" / "adult" / f"seed_{split_seed}.csv",
        dtype={"row_id": "string", "target": "string"},
    )
    split_by_id = assignment.set_index("row_id")["split"]
    data["split"] = data["row_id"].map(split_by_id)
    return data


def _pool(run_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(
        run_dir / "pools" / f"{name}.csv",
        dtype={"synth_row_id": "string", "generator_name": "string", "pool_name": "string"},
    )


def _successful_c2_run(project_root: Path, generator_name: str, generator_seed: int) -> Path:
    prefix = f"c2-smoke-adult-{generator_name.lower()}-s{generator_seed}"
    candidates = sorted((project_root / "runs").glob(f"{prefix}*"))
    successful = [path for path in candidates if (path / "smoke_summary.json").exists()]
    if not successful:
        raise FileNotFoundError(f"No successful C2 smoke run found for {generator_name}")
    return successful[-1]


def run_c3_smoke(project_root: str | Path, generator_name: str) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config = yaml.safe_load((project_root / "configs" / "smoke_c2_c3.yaml").read_text(encoding="utf-8"))
    generator_seed = int(config["generator_seed"])
    split_seed = int(config["split_seed"])
    c2_run_dir = _successful_c2_run(project_root, generator_name, generator_seed)
    c2_run_id = c2_run_dir.name
    run_id = f"c3-smoke-adult-{generator_name.lower()}-s{generator_seed}"
    run_dir = project_root / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"C3 smoke run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    real = _partitioned_adult(project_root, split_seed)
    downstream_real = real.loc[real["split"] == "R_source_train"].copy()
    downstream_synth = _pool(c2_run_dir, "S_downstream_mix")
    formal_config = yaml.safe_load(
        (project_root / "configs" / "benchmark_v1.yaml").read_text(encoding="utf-8")
    )
    rates = [float(rate) for rate in formal_config["pollution_rates"]]

    contamination_entries = []
    contamination_dir = run_dir / "contamination"
    contamination_dir.mkdir()
    for rate_index, rate in enumerate(rates):
        for condition_index, condition in enumerate(CONDITIONS):
            mix_seed = generator_seed * 10000 + rate_index * 100 + condition_index
            mixed = build_contamination(
                downstream_real,
                downstream_synth,
                condition,
                rate,
                mix_seed,
                real_extra_pool=downstream_real,
            )
            name = f"{condition}_p{int(rate * 100):03d}"
            path = contamination_dir / f"{name}.csv"
            mixed.to_csv(path, index=False, lineterminator="\n")
            summary = contamination_summary(mixed, len(downstream_real))
            summary.update(
                {
                    "name": name,
                    "file": path.name,
                    "file_sha256": sha256_file(path),
                    "mix_seed": mix_seed,
                    "generator": generator_name,
                }
            )
            contamination_entries.append(summary)
    write_json(contamination_entries, run_dir / "contamination_manifest.json")

    bag_config = config["bag_smoke"]
    bag_size = int(bag_config["bag_size"])
    stages = {
        "calibration": (
            real.loc[real["split"] == "R_detector_val"].copy(),
            _pool(c2_run_dir, "S_detector_val"),
            int(bag_config["calibration_bags"]),
        ),
        "test": (
            real.loc[real["split"] == "R_final_test"].copy(),
            _pool(c2_run_dir, "S_final_test"),
            int(bag_config["test_bags"]),
        ),
    }
    bag_entries = []
    all_members = []
    for stage_index, (stage, (real_pool, synth_pool, bag_count)) in enumerate(stages.items()):
        for rate_index, rate in enumerate(rates):
            for bag_index in range(bag_count):
                mix_seed = generator_seed * 100000 + stage_index * 10000 + rate_index * 100 + bag_index
                bag_id = (
                    f"bag:adult:{generator_name}:s{generator_seed}:{stage}:"
                    f"p{int(rate * 100):03d}:b{bag_index:02d}"
                )
                members = build_bag_members(real_pool, synth_pool, bag_id, rate, bag_size, mix_seed)
                all_members.append(members)
                bag_entries.append(
                    bag_summary(members, "adult", generator_name, stage, rate, mix_seed)
                )
    members_frame = pd.concat(all_members, ignore_index=True)
    members_path = run_dir / "bag_members.csv"
    members_frame.to_csv(members_path, index=False, lineterminator="\n")
    write_json(
        {
            "run_id": run_id,
            "run_type": "smoke",
            "source_c2_run_id": c2_run_id,
            "bag_size": bag_size,
            "calibration_bags_per_rate": int(bag_config["calibration_bags"]),
            "test_bags_per_rate": int(bag_config["test_bags"]),
            "bags": bag_entries,
            "members_file": members_path.name,
            "members_file_sha256": sha256_file(members_path),
        },
        run_dir / "bags_manifest.json",
    )

    calibration_ids = set(
        members_frame.loc[
            members_frame["bag_id"].str.contains(":calibration:"), "record_id"
        ].astype(str)
    )
    test_ids = set(
        members_frame.loc[members_frame["bag_id"].str.contains(":test:"), "record_id"].astype(str)
    )
    if calibration_ids & test_ids:
        raise ValueError("Calibration/test bag record leakage")

    sample_entry = bag_entries[0]
    sample_members = members_frame.loc[members_frame["bag_id"] == sample_entry["bag_id"]]
    sample_stage = sample_entry["stage"]
    sample_real, sample_synth, _ = stages[sample_stage]
    rebuilt = rebuild_bag(sample_members, sample_real, sample_synth)
    rebuild_summary = {
        "bag_id": sample_entry["bag_id"],
        "rows": len(rebuilt),
        "members_sha256": members_hash(sample_members),
        "expected_members_sha256": sample_entry["members_sha256"],
        "source_counts": rebuilt["source_type"].value_counts().to_dict(),
    }
    write_json(rebuild_summary, run_dir / "bag_rebuild_example.json")

    protocol_examples = {
        "P1": {
            "train_tables": ["adult"], "test_tables": ["adult"],
            "train_generators": [generator_name], "test_generators": [generator_name],
            "train_records": ["a"], "test_records": ["b"],
            "train_domains": ["social"], "test_domains": ["social"],
        },
        "P2": {
            "train_tables": ["adult"], "test_tables": ["adult"],
            "train_generators": [generator_name], "test_generators": ["heldout_generator"],
            "train_records": ["a"], "test_records": ["b"],
            "train_domains": ["social"], "test_domains": ["social"],
        },
        "P3": {
            "train_tables": ["adult"], "test_tables": ["credit"],
            "train_generators": [generator_name], "test_generators": [generator_name],
            "train_records": ["a"], "test_records": ["b"],
            "train_domains": ["social"], "test_domains": ["finance"],
        },
        "P4": {
            "train_tables": ["adult"], "test_tables": ["credit"],
            "train_generators": [generator_name], "test_generators": ["heldout_generator"],
            "train_records": ["a"], "test_records": ["b"],
            "train_domains": ["social"], "test_domains": ["finance"],
        },
        "P5": {
            "train_tables": ["adult"], "test_tables": ["credit"],
            "train_generators": [generator_name], "test_generators": [generator_name],
            "train_records": ["a"], "test_records": ["b"],
            "train_domains": ["social"], "test_domains": ["finance"],
        },
    }
    protocol_results = {key: validate_protocol(key, value) for key, value in protocol_examples.items()}
    write_json(protocol_results, run_dir / "protocol_validation.json")
    result = {
        "run_id": run_id,
        "run_type": "smoke",
        "status": "smoke_passed",
        "generator": generator_name,
        "source_c2_run_id": c2_run_id,
        "contamination_artifacts": len(contamination_entries),
        "bag_count": len(bag_entries),
        "bag_member_rows": len(members_frame),
        "calibration_test_ids_disjoint": True,
        "rebuild_example": rebuild_summary,
        "protocol_validation": protocol_results,
    }
    write_json(result, run_dir / "smoke_summary.json")
    return result


def inspect_bag(project_root: str | Path, generator_name: str, bag_id: str) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    candidates = sorted((project_root / "runs").glob(f"c3-smoke-adult-{generator_name.lower()}-s42*"))
    run_dir = [path for path in candidates if (path / "bags_manifest.json").exists()][-1]
    members = pd.read_csv(run_dir / "bag_members.csv", dtype={"record_id": "string"})
    selected = members.loc[members["bag_id"] == bag_id]
    if selected.empty:
        raise KeyError(f"Unknown bag_id: {bag_id}")
    return {
        "bag_id": bag_id,
        "rows": len(selected),
        "source_counts": selected["source_type"].value_counts().to_dict(),
        "members_sha256": members_hash(selected),
    }
