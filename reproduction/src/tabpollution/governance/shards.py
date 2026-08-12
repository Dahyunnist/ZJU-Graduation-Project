"""Resumable seed/protocol/detector execution for governance experiments.

The base configuration remains the sole research contract.  Shards are derived
only after that complete configuration has passed strict validation, so formal
runs cannot bypass the P1--P4, seed, bag, valuation, or method requirements.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import pandas as pd

from tabpollution.governance.config import GovernanceConfig, load_governance_config
from tabpollution.governance.metrics import finite_or_none
from tabpollution.governance.pipeline import (
    _write_findings,
    _write_statistical_inference,
    run_governance_benchmark,
)
from tabpollution.utils import read_json, sha256_file, write_json


DEEP_DETECTORS = {
    "flat_transformer", "table_transformer", "column_positional_ablation",
    "datum_transformer", "datum_ta",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serializable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _write_json_atomic(value: Any, path: Path) -> None:
    """Publish a marker atomically so interruption cannot create false completion."""
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    write_json(value, temporary)
    temporary.replace(path)


def _valid_detector(protocol: str, detector: str, config: GovernanceConfig) -> bool:
    if detector.startswith("c2st") and protocol in {"P3", "P4"}:
        return False
    if detector == "datum_ta" and len(config.protocols[protocol].train_tables) < 2:
        return False
    return True


def build_shard_plan(config_path: str | Path) -> dict[str, Any]:
    """Build and persist the deterministic shard plan for a validated base run."""
    path = Path(config_path).resolve()
    config = load_governance_config(path)
    config_hash = sha256_file(path)
    shards: list[dict[str, Any]] = []
    for seed in config.seeds:
        for protocol in config.protocols:
            valid = [name for name in config.detectors if _valid_detector(protocol, name, config)]
            if not valid:
                raise ValueError(f"No valid detector remains for {protocol}")
            valuation_owner = valid[0]
            for detector in valid:
                shards.append({
                    "shard_id": f"s{seed}__{protocol}__{detector}",
                    "seed": seed,
                    "protocol": protocol,
                    "detector": detector,
                    "valuation_owner": bool(config.valuation_enabled and detector == valuation_owner),
                })
    plan = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "run_type": config.run_type,
        "base_config_path": str(path),
        "base_config_sha256": config_hash,
        "output_dir": str(config.output_dir),
        "created_at": _utc_now(),
        "shard_count": len(shards),
        "shards": shards,
    }
    _write_json_atomic(plan, config.output_dir / "shard_plan.json")
    return plan


def shard_queue(
    config_path: str | Path,
    *,
    seed: int,
    resource_class: str,
) -> dict[str, Any]:
    """Return a disjoint pending queue for a seed and CPU/GPU worker.

    Queue selection changes only execution order.  The immutable shard metadata
    and the base research contract remain untouched.
    """
    if resource_class not in {"cpu", "gpu"}:
        raise ValueError("resource_class must be cpu or gpu")
    plan = build_shard_plan(config_path)
    config = load_governance_config(config_path)
    selected: list[str] = []
    completed: list[str] = []
    for shard in plan["shards"]:
        if int(shard["seed"]) != int(seed):
            continue
        shard_resource = "gpu" if shard["detector"] in DEEP_DETECTORS else "cpu"
        if shard_resource != resource_class:
            continue
        if _completion(config, shard, plan["base_config_sha256"]):
            completed.append(shard["shard_id"])
        else:
            selected.append(shard["shard_id"])
    return {
        "experiment_id": config.experiment_id,
        "seed": int(seed),
        "resource_class": resource_class,
        "completed_shards": completed,
        "pending_shards": selected,
        "pending_count": len(selected),
    }


def _completion_path(config: GovernanceConfig, shard_id: str) -> Path:
    return config.output_dir / "shards" / shard_id / "COMPLETE.json"


def _completion(config: GovernanceConfig, shard: dict[str, Any], config_hash: str) -> dict[str, Any] | None:
    marker = _completion_path(config, shard["shard_id"])
    if not marker.is_file():
        return None
    value = read_json(marker)
    if value.get("base_config_sha256") != config_hash:
        # The stability revision adds deep-training fields to the same frozen
        # research contract.  Classical shards are reusable when every field
        # recorded by the legacy attempt still matches; deep shards must run
        # again under the new optimizer and validation-selection contract.
        if shard["detector"] in DEEP_DETECTORS:
            return None
        attempt_value = Path(value.get("attempt_dir", ""))
        shard_config_path = attempt_value / "shard_config.json"
        if not shard_config_path.is_file():
            raise ValueError(f"Cannot validate legacy completion marker for {shard['shard_id']}")
        old_resolved = read_json(shard_config_path).get("resolved_config", {})
        expected = _serializable(asdict(_shard_config(config, shard, attempt_value)))
        if any(
            expected.get(key) != old_value
            for key, old_value in old_resolved.items()
            if not key.startswith("deep_")
        ):
            raise ValueError(
                f"Stale completion marker for {shard['shard_id']}; use a new output_dir after changing config"
            )
        value = {**value, "legacy_compatible": True}
    if value.get("shard") != shard:
        raise ValueError(f"Completion marker metadata mismatch for {shard['shard_id']}")
    attempt = Path(value["attempt_dir"])
    if not (attempt / "summary.json").is_file():
        raise ValueError(f"Completion marker points to an incomplete attempt: {attempt}")
    return value


def _shard_config(
    base: GovernanceConfig,
    shard: dict[str, Any],
    output_dir: Path,
) -> GovernanceConfig:
    valuation_enabled = bool(shard["valuation_owner"])
    return replace(
        base,
        seeds=(int(shard["seed"]),),
        protocols={shard["protocol"]: base.protocols[shard["protocol"]]},
        detectors=(shard["detector"],),
        valuation_enabled=valuation_enabled,
        valuation_methods=base.valuation_methods if valuation_enabled else (),
        valuation_bags_per_rate=base.valuation_bags_per_rate if valuation_enabled else 0,
        output_dir=output_dir,
    )


def run_governance_shard(
    config_path: str | Path,
    shard_id: str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one immutable shard and publish its completion marker last."""
    plan = build_shard_plan(config_path)
    base = load_governance_config(config_path)
    matches = [row for row in plan["shards"] if row["shard_id"] == shard_id]
    if not matches:
        raise ValueError(f"Unknown shard_id: {shard_id}")
    shard = matches[0]
    complete = _completion(base, shard, plan["base_config_sha256"])
    if complete:
        if resume:
            return {"status": "skipped_complete", **complete}
        raise FileExistsError(f"Shard is already complete: {shard_id}")

    shard_root = base.output_dir / "shards" / shard_id
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    attempt_dir = shard_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    derived = _shard_config(base, shard, attempt_dir)
    write_json({
        "base_config_sha256": plan["base_config_sha256"],
        "shard": shard,
        "resolved_config": _serializable(asdict(derived)),
    }, attempt_dir / "shard_config.json")
    started = time.perf_counter()
    try:
        summary = run_governance_benchmark(derived)
    except Exception as exc:
        write_json({
            "status": "failed",
            "failed_at": _utc_now(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }, attempt_dir / "FAILED.json")
        raise
    marker = {
        "status": "complete",
        "completed_at": _utc_now(),
        "seconds": time.perf_counter() - started,
        "base_config_sha256": plan["base_config_sha256"],
        "shard": shard,
        "attempt_dir": str(attempt_dir.resolve()),
        "summary": summary,
    }
    _write_json_atomic(marker, _completion_path(base, shard_id))
    return marker


def shard_status(config_path: str | Path) -> dict[str, Any]:
    plan = build_shard_plan(config_path)
    config = load_governance_config(config_path)
    completed: list[str] = []
    pending: list[str] = []
    for shard in plan["shards"]:
        if _completion(config, shard, plan["base_config_sha256"]):
            completed.append(shard["shard_id"])
        else:
            pending.append(shard["shard_id"])
    return {
        "experiment_id": config.experiment_id,
        "status": "complete" if not pending else "partial",
        "shard_count": len(plan["shards"]),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "completed_shards": completed,
        "pending_shards": pending,
    }


def _read_frames(attempts: list[Path], filename: str) -> pd.DataFrame:
    frames = [pd.read_csv(path / filename) for path in attempts]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_governance_shards(config_path: str | Path) -> dict[str, Any]:
    """Aggregate only a complete, hash-consistent shard set."""
    started = time.perf_counter()
    plan = build_shard_plan(config_path)
    config = load_governance_config(config_path)
    markers: list[dict[str, Any]] = []
    pending: list[str] = []
    for shard in plan["shards"]:
        marker = _completion(config, shard, plan["base_config_sha256"])
        if marker is None:
            pending.append(shard["shard_id"])
        else:
            markers.append(marker)
    if pending:
        return {
            "experiment_id": config.experiment_id,
            "status": "partial",
            "completed_count": len(markers),
            "pending_count": len(pending),
            "pending_shards": pending,
        }

    attempts = [Path(row["attempt_dir"]) for row in markers]
    evidence = _read_frames(attempts, "governance_evidence.csv")
    artifacts_all = _read_frames(attempts, "format_artifact_audit.csv")
    valuation_attempts = [
        Path(row["attempt_dir"]) for row in markers if row["shard"]["valuation_owner"]
    ]
    valuation = _read_frames(valuation_attempts, "record_valuation.csv")
    if valuation.empty and not len(valuation.columns):
        valuation = pd.DataFrame(columns=[
            "seed", "protocol", "test_table", "test_generator", "contamination_mode",
            "true_prevalence", "bag_index", "valuation_method", "valuation_row_index",
            "record_id", "source_label", "task_value", "oob_coverage",
        ])
    if evidence.empty or artifacts_all.empty:
        raise ValueError("Completed shards did not contain governance evidence")

    artifact_keys = ["seed", "protocol"]
    inconsistent = []
    for keys, group in artifacts_all.groupby(artifact_keys, dropna=False):
        if len(group.drop_duplicates()) != 1:
            inconsistent.append(keys)
    if inconsistent:
        raise ValueError(f"Artifact audits differ across detector shards: {inconsistent}")
    artifacts = artifacts_all.drop_duplicates().sort_values(artifact_keys).reset_index(drop=True)

    protocol_manifests: dict[str, Any] = {}
    for attempt in attempts:
        for key, value in read_json(attempt / "protocol_manifests.json").items():
            if key in protocol_manifests and protocol_manifests[key] != value:
                raise ValueError(f"Protocol manifest differs across shards: {key}")
            protocol_manifests[key] = value

    output = config.output_dir
    evidence.to_csv(output / "governance_evidence.csv", index=False)
    artifacts.to_csv(output / "format_artifact_audit.csv", index=False)
    valuation.to_csv(output / "record_valuation.csv", index=False)
    write_json(protocol_manifests, output / "protocol_manifests.json")
    write_json(_serializable(asdict(config)), output / "resolved_config.json")
    first_registry = attempts[0] / "method_registry_snapshot.json"
    write_json(read_json(first_registry), output / "method_registry_snapshot.json")
    diagnostics: dict[str, Any] = {}
    for attempt in attempts:
        path = attempt / "detector_diagnostics.json"
        if not path.is_file():
            continue
        for key, value in read_json(path).items():
            if key in diagnostics and diagnostics[key] != value:
                raise ValueError(f"Detector diagnostic differs across shards: {key}")
            diagnostics[key] = value
    write_json(diagnostics, output / "detector_diagnostics.json")
    _write_findings(evidence, artifacts, valuation, output, config.primary_quantifier)
    _write_statistical_inference(evidence, output, config.primary_quantifier)
    write_json({
        "base_config_sha256": plan["base_config_sha256"],
        "shard_count": len(markers),
        "completed_shards": [row["shard"]["shard_id"] for row in markers],
        "attempts": [row["attempt_dir"] for row in markers],
    }, output / "completed_shards_manifest.json")
    quantifier_failures = evidence["quantifier_status"].astype(str).ne("ok")
    primary_failures = quantifier_failures & evidence["quantifier"].eq(config.primary_quantifier)
    failed_combinations = (
        evidence.loc[quantifier_failures, ["protocol", "detector", "quantifier", "quantifier_status"]]
        .drop_duplicates().sort_values(["protocol", "detector", "quantifier"])
        .to_dict(orient="records")
    )
    analysis_ready = not bool(primary_failures.any())
    summary = {
        "experiment_id": config.experiment_id,
        "run_type": config.run_type,
        "status": "complete" if analysis_ready else "complete_with_method_failures",
        "analysis_ready": analysis_ready,
        "formal_inclusion": config.run_type == "formal" and analysis_ready,
        "created_at": _utc_now(),
        "aggregation_seconds": time.perf_counter() - started,
        "shard_count": len(markers),
        "evidence_rows": len(evidence),
        "protocols": sorted(evidence["protocol"].unique().tolist()),
        "detectors": sorted(evidence["detector"].unique().tolist()),
        "calibration_policies": sorted(evidence["calibration_policy"].unique().tolist()),
        "quantifiers": sorted(evidence["quantifier"].unique().tolist()),
        "max_absolute_error_decomposition_residual": finite_or_none(
            evidence["error_decomposition_residual"].abs().max()
        ),
        "low_prevalence_rows": int(evidence["true_prevalence"].isin([.05, .10]).sum()),
        "artifact_gate_failures": int((~artifacts["artifact_gate_passed"].astype(bool)).sum()),
        "valuation_rows": len(valuation),
        "quantifier_failure_rows": int(quantifier_failures.sum()),
        "primary_quantifier_failure_rows": int(primary_failures.sum()),
        "failed_quantifier_combinations": failed_combinations,
        "mean_prevalence_mae": finite_or_none(evidence["prevalence_absolute_error"].mean()),
        "mean_decision_regret": finite_or_none(evidence["decision_regret"].mean()),
        "outputs": {
            "evidence": str(output / "governance_evidence.csv"),
            "findings": [str(path) for path in sorted(output.glob("finding_*.csv"))],
            "valuation": str(output / "record_valuation.csv"),
            "statistical_summary": str(output / "statistical_summary_ci95.csv"),
            "paired_tests": str(output / "paired_detector_tests.csv"),
            "shards": str(output / "completed_shards_manifest.json"),
            "detector_diagnostics": str(output / "detector_diagnostics.json"),
        },
    }
    write_json(summary, output / "summary.json")
    return summary


def run_governance_sharded(
    config_path: str | Path,
    *,
    resume: bool = True,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Run pending shards in plan order, checkpointing after each shard."""
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be positive when provided")
    plan = build_shard_plan(config_path)
    config = load_governance_config(config_path)
    executed: list[str] = []
    for shard in plan["shards"]:
        if _completion(config, shard, plan["base_config_sha256"]):
            if not resume:
                raise FileExistsError(f"Shard is already complete: {shard['shard_id']}")
            continue
        run_governance_shard(config_path, shard["shard_id"], resume=resume)
        executed.append(shard["shard_id"])
        if max_shards is not None and len(executed) >= max_shards:
            break
    status = shard_status(config_path)
    status["executed_shards"] = executed
    if status["pending_count"] == 0:
        status["aggregate"] = aggregate_governance_shards(config_path)
    return status
