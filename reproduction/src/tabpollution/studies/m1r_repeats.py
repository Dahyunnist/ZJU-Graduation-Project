"""M1R independent generator-training repeat orchestration.

This module reuses E3 pool building and E3B detection without changing either
historical protocol. Generator instances, not split seeds, are the repeat unit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tabpollution.studies import e3_pools
from tabpollution.studies.e3b_detectability import run as run_e3b
from tabpollution.utils import sha256_file


EXPECTED_TABLES = ["adult", "abalone"]
EXPECTED_GENERATORS = ["CTGAN", "TVAE"]
EXPECTED_SPLIT_SEEDS = [2026, 2027, 2028]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "study_id", "tables", "generators", "split_seeds", "minimum_complete_instances",
        "historical_instance", "new_instances", "generator_budget", "e3b_protocol",
        "replication_rules",
    }
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("M1R config fields do not match the frozen protocol")
    if cfg["study_id"] != "m1r-independent-repeats-v1":
        raise ValueError("M1R study identity changed")
    if cfg["tables"] != EXPECTED_TABLES or cfg["generators"] != EXPECTED_GENERATORS:
        raise ValueError("M1R tables or generators changed")
    if cfg["split_seeds"] != EXPECTED_SPLIT_SEEDS:
        raise ValueError("M1R split seeds changed")
    if int(cfg["minimum_complete_instances"]) != 3 or len(cfg["new_instances"]) != 2:
        raise ValueError("M1R requires one historical and two new instances")
    historical_required = {"instance_id", "pool_dir", "checkpoint_dir", "historical_e3b_output"}
    new_required = {"instance_id", "pool_config", "pool_dir", "checkpoint_dir"}
    if set(cfg["historical_instance"]) != historical_required:
        raise ValueError("M1R historical instance fields changed")
    if any(set(instance) != new_required for instance in cfg["new_instances"]):
        raise ValueError("M1R new instance fields changed")
    instance_ids = [cfg["historical_instance"]["instance_id"]] + [
        instance["instance_id"] for instance in cfg["new_instances"]
    ]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("M1R instance IDs must be unique")
    budget = cfg["generator_budget"]
    if budget != {
        "implementation": "sdv-1.37.3", "epochs": 300, "batch_size": 500,
        "sample_size": "match_fixed_real_evaluation_rows",
    }:
        raise ValueError("M1R generator budget changed")
    protocol = cfg["e3b_protocol"]
    expected_protocol = {
        "primary_detector": "c2st_lr", "capacity_detector": "c2st_xgb",
        "model_direction": "high_score_is_synthetic", "train_split": "detector_train",
        "calibration_split": "detector_val", "evaluation_split": "final_test",
        "target_fpr": 0.05, "bootstrap_replicates": 1000,
        "bootstrap_confidence": 0.95, "transfer_min_auroc_gain": 0.10,
        "near_random_auroc": 0.60,
    }
    if protocol != expected_protocol:
        raise ValueError("M1R E3B protocol changed")
    rules = cfg["replication_rules"]
    if rules != {
        "adult_required_branch": "E3C-Transfer",
        "abalone_required_branch": "E3C-Capacity",
        "require_all_complete_instances": True,
        "mixed_branch_result": "Conditional-Heterogeneity",
        "incomplete_result": "Inconclusive",
    }:
        raise ValueError("M1R replication rules changed")
    return cfg


def _expected_generator_config_hash(generator: str) -> str:
    settings = {
        "CTGAN": {"epochs": 300, "batch_size": 500, "cuda": True, "verbose": True},
        "TVAE": {"epochs": 300, "batch_size": 500, "cuda": True},
    }[generator]
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_legacy_base(base_root: Path) -> dict[str, Any]:
    manifest_path = base_root / "pool_build_manifest.json"
    reasons: list[str] = []
    if not manifest_path.is_file():
        reasons.append("missing pool_build_manifest.json")
        return {"candidate": "legacy_governance", "eligible": False, "reasons": reasons}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = [
        row for row in manifest.get("runs", [])
        if row.get("table_id") in EXPECTED_TABLES and row.get("generator") in EXPECTED_GENERATORS
    ]
    if len(runs) != 4:
        reasons.append("Adult/Abalone x CTGAN/TVAE matrix is incomplete")
    if any(not row.get("model_sha256") for row in runs):
        reasons.append("model SHA-256 is absent")
    if any(row.get("synthetic_rows") is None for row in runs):
        reasons.append("synthetic row count is absent")
    if "registry_sha256" not in manifest:
        reasons.append("registry is not hash-bound to manifest")
    if "plan" not in manifest:
        reasons.append("independent-instance training plan is absent")
    generator_seeds = {int(row["generator_seed"]) for row in runs if row.get("generator_seed") is not None}
    if generator_seeds & set(EXPECTED_SPLIT_SEEDS):
        reasons.append("generator and real-split seed namespaces reuse a numeric seed")
    return {"candidate": "legacy_governance", "eligible": not reasons, "reasons": reasons}


def audit_replica_pool(pool_root: Path, checkpoint_root: Path, base_root: Path,
                       expected_instance_id: str) -> dict[str, Any]:
    reasons: list[str] = []
    manifest_path, registry_path = pool_root / "replica_manifest.json", pool_root / "pool_registry.csv"
    if not manifest_path.is_file() or not registry_path.is_file():
        return {
            "candidate": expected_instance_id, "eligible": False,
            "reasons": ["replica manifest or registry is missing"],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_path)
    if manifest.get("status") != "complete":
        reasons.append("manifest status is not complete")
    if manifest.get("plan", {}).get("replicate_id") != expected_instance_id:
        reasons.append("replicate identity mismatch")
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        reasons.append("registry hash mismatch")
    required = {(table, generator) for table in EXPECTED_TABLES for generator in EXPECTED_GENERATORS}
    if set(zip(registry.table_id, registry.generator)) != required:
        reasons.append("registry matrix is incomplete")
    runs = {(row.get("table_id"), row.get("generator")): row for row in manifest.get("runs", [])}
    if set(runs) != required:
        reasons.append("manifest run matrix is incomplete")
    source_hashes: dict[str, str] = {}
    real_hashes: dict[str, str] = {}
    pool_hashes: dict[str, str] = {}
    generator_seeds: dict[str, int] = {}
    for table, generator in sorted(required):
        row = runs.get((table, generator))
        if row is None:
            continue
        source_path = base_root / "source_train" / f"{table}.csv"
        registry_row = registry.loc[(registry.table_id == table) & (registry.generator == generator)]
        if len(registry_row) != 1:
            continue
        real_path = (pool_root / str(registry_row.real_path.iloc[0])).resolve()
        synthetic_path = (pool_root / str(registry_row.synthetic_path.iloc[0])).resolve()
        model_path = checkpoint_root / table / f"{generator}.pkl"
        for field in (
            "status", "generator_seed", "sample_seed", "source_train_sha256", "real_sha256",
            "generator_config_sha256", "model_sha256", "synthetic_sha256", "fit_seconds",
            "synthetic_rows", "provenance",
        ):
            if row.get(field) is None:
                reasons.append(f"{table}/{generator} missing {field}")
        if row.get("status") != "complete":
            reasons.append(f"{table}/{generator} is incomplete")
        if row.get("generator_config_sha256") != _expected_generator_config_hash(generator):
            reasons.append(f"{table}/{generator} budget hash mismatch")
        if not source_path.is_file() or row.get("source_train_sha256") != sha256_file(source_path):
            reasons.append(f"{table}/{generator} source hash mismatch")
        if not real_path.is_file() or row.get("real_sha256") != sha256_file(real_path):
            reasons.append(f"{table}/{generator} real hash mismatch")
        if not synthetic_path.is_file() or row.get("synthetic_sha256") != sha256_file(synthetic_path):
            reasons.append(f"{table}/{generator} synthetic hash mismatch")
            continue
        if not model_path.is_file() or row.get("model_sha256") != sha256_file(model_path):
            reasons.append(f"{table}/{generator} model hash mismatch")
        source = pd.read_csv(source_path, nrows=1)
        real = pd.read_csv(real_path)
        synthetic = pd.read_csv(synthetic_path)
        if list(source.columns) != list(real.columns) or list(real.columns) != list(synthetic.columns):
            reasons.append(f"{table}/{generator} schema mismatch")
        if len(synthetic) != len(real) or int(row.get("synthetic_rows", -1)) != len(real):
            reasons.append(f"{table}/{generator} sample size mismatch")
        source_hashes[table] = row.get("source_train_sha256")
        real_hashes[table] = row.get("real_sha256")
        pool_hashes[f"{table}/{generator}"] = row.get("synthetic_sha256")
        generator_seeds[f"{table}/{generator}"] = int(row.get("generator_seed"))
    return {
        "candidate": expected_instance_id,
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "manifest_sha256": sha256_file(manifest_path),
        "registry_sha256": sha256_file(registry_path),
        "source_hashes": source_hashes,
        "real_hashes": real_hashes,
        "pool_hashes": pool_hashes,
        "generator_seeds": generator_seeds,
    }


def planned_instances(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [cfg["historical_instance"], *cfg["new_instances"]]


def preflight(config_path: Path, repository_root: Path, base_root: Path,
              dataset_root: Path, checkpoint_root: Path, results_root: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    audits = [audit_legacy_base(base_root)]
    plans = []
    seed_owners: dict[int, str] = {}
    for instance in planned_instances(cfg):
        pool_root = dataset_root / instance["pool_dir"]
        models = checkpoint_root / instance["checkpoint_dir"]
        audit = audit_replica_pool(pool_root, models, base_root, instance["instance_id"])
        audits.append(audit)
        if instance is cfg["historical_instance"]:
            historical_output = results_root / instance["historical_e3b_output"]
            if not historical_output.is_dir():
                raise FileNotFoundError(f"Historical E3B output is missing: {historical_output}")
            continue
        pool_config = repository_root / "reproduction" / instance["pool_config"]
        plan = e3_pools.preflight(pool_config, base_root, pool_root, models)
        if plan["replicate_id"] != instance["instance_id"]:
            raise ValueError("M1R pool config identity mismatch")
        plans.append(plan)
        for run in plan["runs"]:
            seed = int(run["generator_seed"])
            if seed in seed_owners:
                raise ValueError(f"Generator seed {seed} reused by two instances")
            seed_owners[seed] = instance["instance_id"]
    historical = audits[1]
    for seed in historical.get("generator_seeds", {}).values():
        if seed in seed_owners:
            raise ValueError(f"Historical/new generator seed collision: {seed}")
        seed_owners[seed] = historical["candidate"]
    pool_hashes: dict[str, str] = {}
    for audit in audits[1:]:
        for key, digest in audit.get("pool_hashes", {}).items():
            if digest in pool_hashes:
                raise ValueError(f"Synthetic pool hash reused by {pool_hashes[digest]} and {audit['candidate']}/{key}")
            pool_hashes[digest] = f"{audit['candidate']}/{key}"
    return {
        "status": "preflight_pass",
        "study_id": cfg["study_id"],
        "config_sha256": sha256_file(config_path),
        "legacy_base_audit": audits[0],
        "instance_audits": audits[1:],
        "new_instance_plans": plans,
        "registered_generator_seeds": seed_owners,
    }


def train_new_instances(config_path: Path, repository_root: Path, base_root: Path,
                        dataset_root: Path, checkpoint_root: Path) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    results = []
    for instance in cfg["new_instances"]:
        pool_config = repository_root / "reproduction" / instance["pool_config"]
        result = e3_pools.build(
            pool_config, base_root, dataset_root / instance["pool_dir"],
            checkpoint_root / instance["checkpoint_dir"], resume=True,
        )
        results.append({"instance_id": instance["instance_id"], "status": result["status"]})
    return results


def make_e3b_binding(cfg: dict[str, Any], instance_id: str, pool_root: Path,
                     destination: Path) -> Path:
    manifest_path, registry_path = pool_root / "replica_manifest.json", pool_root / "pool_registry.csv"
    protocol = cfg["e3b_protocol"]
    binding = {
        "study_id": f"m1r-e3b-{instance_id}",
        "tables": cfg["tables"], "generators": cfg["generators"], "seeds": cfg["split_seeds"],
        **protocol,
        "expected_replicate_id": instance_id,
        "expected_replica_manifest_sha256": sha256_file(manifest_path),
        "expected_registry_sha256": sha256_file(registry_path),
    }
    _atomic_yaml(destination, binding)
    return destination


def run_new_detections(config_path: Path, base_root: Path, dataset_root: Path,
                       checkpoint_root: Path, output_root: Path) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    results = []
    for instance in cfg["new_instances"]:
        pool_root = dataset_root / instance["pool_dir"]
        audit = audit_replica_pool(
            pool_root, checkpoint_root / instance["checkpoint_dir"], base_root,
            instance["instance_id"],
        )
        if not audit["eligible"]:
            raise ValueError(f"M1R pool is not eligible: {instance['instance_id']} {audit['reasons']}")
        binding = make_e3b_binding(
            cfg, instance["instance_id"], pool_root,
            output_root / "bindings" / f"{instance['instance_id']}.yaml",
        )
        result = run_e3b(
            binding, pool_root / "pool_registry.csv", pool_root / "replica_manifest.json",
            output_root / "detection" / instance["instance_id"],
        )
        results.append({"instance_id": instance["instance_id"], **result})
    _atomic_json(output_root / "detection_summary.json", {"status": "complete", "instances": results})
    return results


def decide_cross_instance(decisions: pd.DataFrame, minimum_instances: int = 3) -> dict[str, Any]:
    expected = {"adult": "E3C-Transfer", "abalone": "E3C-Capacity"}
    table_results: dict[str, str] = {}
    for table, required_branch in expected.items():
        rows = decisions.loc[decisions.table == table]
        branches = rows.branch.astype(str).tolist()
        if len(rows) < minimum_instances:
            result = "Inconclusive"
        elif len(set(branches)) != 1:
            result = "Conditional-Heterogeneity"
        elif branches[0] == required_branch:
            result = required_branch
        else:
            result = "Conditional-Heterogeneity"
        table_results[table] = result
    if "Inconclusive" in table_results.values():
        overall = "Inconclusive"
    elif "Conditional-Heterogeneity" in table_results.values():
        overall = "Conditional-Heterogeneity"
    else:
        overall = "Stable-Split-Branches"
    return {"table_results": table_results, "overall_result": overall}

