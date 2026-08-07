"""Adult seed-2026 formal-configuration pilot orchestration."""

from __future__ import annotations

import contextlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from tabpollution.generators.base import audit_generator_access, derive_pool_seeds
from tabpollution.generators.pools import POOL_NAMES, add_pool_provenance, validate_pools, write_pool
from tabpollution.generators.preflight import ensure_disk_space, generator_preflight
from tabpollution.generators.quality import format_diagnostics, overlap_diagnostics, sdmetrics_quality, utility_smoke
from tabpollution.generators.quality_gate import evaluate_quality_gate
from tabpollution.generators.sdv_adapter import SDVGenerator, create_generator
from tabpollution.generators.smoke import _load_partitioned_adult, _pool_sizes
from tabpollution.runs import artifact_manifest, mark_complete, set_status, validate_artifact_manifest
from tabpollution.utils import sha256_file, write_json


PILOT_REQUIRED = {
    "config_resolved.yaml",
    "environment.json",
    "access_audit.json",
    "stdout.log",
    "stderr.log",
    "timing.json",
    "metadata.json",
    "generator_provenance.json",
    "pools_manifest.json",
    "quality.json",
    "reload_validation.json",
    "run_manifest.json",
    "status.json",
    "model.pkl",
}


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("run_type") != "pilot":
        raise ValueError("Pilot config must have run_type=pilot")
    if config.get("dataset") != "adult" or int(config.get("split_seed", -1)) != 2026:
        raise ValueError("This pilot is frozen to Adult split_seed=2026")
    if int(config.get("generator_seed", -1)) != 2026:
        raise ValueError("Pilot generator_seed must be 2026")
    if config.get("fit_scope") != "full_R_source_train":
        raise ValueError("Pilot may not reduce the formal fit scope")
    for name in ("CTGAN", "TVAE"):
        if int(config["generators"][name].get("epochs", 0)) != 300:
            raise ValueError(f"{name} pilot must retain 300 epochs")
    return config


def _new_run_dir(project_root: Path, generator_name: str, generator_seed: int) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"c2-pilot-adult-{generator_name.lower()}-s{generator_seed}-{stamp}"
    run_dir = project_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _run_manifest(run_id: str, generator_name: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": "pilot",
        "dataset": "adult",
        "split_seed": 2026,
        "generator_seed": 2026,
        "generator": generator_name,
        "fit_scope": "full_R_source_train",
        "formal_configuration": True,
        "eligible_for_formal_aggregation": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_source": "configs/pilot_c2.yaml",
    }


def create_gpu_blocked_run(
    project_root: str | Path,
    generator_name: str,
    config_path: str | Path = "configs/pilot_c2.yaml",
) -> dict[str, Any]:
    if generator_name not in {"CTGAN", "TVAE"}:
        raise ValueError("GPU blocked records are only valid for CTGAN/TVAE")
    root = Path(project_root).resolve()
    config = load_pilot_config(root / config_path)
    preflight = generator_preflight(root)
    if preflight["hard_gpu_passed"]:
        raise RuntimeError("GPU preflight passed; blocked_by_gpu would be false")
    run_id, run_dir = _new_run_dir(root, generator_name, int(config["generator_seed"]))
    (run_dir / "stdout.log").write_text("Training not started: GPU preflight failed.\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("blocked_by_gpu\n", encoding="utf-8")
    resolved = {
        **_run_manifest(run_id, generator_name, config),
        "generator_config": config["generators"][generator_name],
        "status": "blocked_by_gpu",
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    write_json(preflight, run_dir / "environment.json")
    write_json({"fit_called": False, "reason": "GPU hard preflight failed"}, run_dir / "timing.json")
    write_json(_run_manifest(run_id, generator_name, config), run_dir / "run_manifest.json")
    set_status(run_dir, "blocked_by_gpu", {"fit_called": False})
    artifact_manifest(run_dir, run_id)
    return {
        "run_id": run_id,
        "generator": generator_name,
        "status": "blocked_by_gpu",
        "fit_called": False,
        "hard_gpu_passed": False,
    }


def run_adult_generator_pilot(
    project_root: str | Path,
    generator_name: str,
    config_path: str | Path = "configs/pilot_c2.yaml",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_pilot_config(root / config_path)
    if generator_name not in config["generators"]:
        raise ValueError(f"Unknown pilot generator: {generator_name}")
    preflight = generator_preflight(root)
    if generator_name in {"CTGAN", "TVAE"} and not preflight["hard_gpu_passed"]:
        return create_gpu_blocked_run(root, generator_name, config_path)
    ensure_disk_space(root, int(config["minimum_free_disk_bytes"]))
    run_id, run_dir = _new_run_dir(root, generator_name, int(config["generator_seed"]))
    set_status(run_dir, "preflight_passed")
    manifest = _run_manifest(run_id, generator_name, config)
    write_json(manifest, run_dir / "run_manifest.json")
    write_json(preflight, run_dir / "environment.json")
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.touch()
    stderr_path.touch()
    total_started = perf_counter()
    try:
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                set_status(run_dir, "running")
                spec, partitioned = _load_partitioned_adult(root, int(config["split_seed"]))
                source = partitioned.loc[partitioned["split"] == "R_source_train"].copy()
                audit = audit_generator_access(source, set(source["row_id"].astype(str)))
                audit_payload = {**audit.__dict__, "actual_partitions": list(audit.actual_partitions)}
                write_json(audit_payload, run_dir / "access_audit.json")
                generator_config = dict(config["generators"][generator_name])
                resolved = {
                    **manifest,
                    "generator_config": generator_config,
                    "pool_seed_offsets": config["pool_seed_offsets"],
                    "pool_redundancy": config["pool_redundancy"],
                }
                (run_dir / "config_resolved.yaml").write_text(
                    yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True), encoding="utf-8"
                )
                generator = create_generator(generator_name, generator_config)
                metadata = generator.build_metadata(source)
                write_json(metadata.to_dict(), run_dir / "metadata.json")
                generator.fit(source, metadata, int(config["generator_seed"]))
                model_path = run_dir / "model.pkl"
                generator.save(model_path)
                pool_seeds = derive_pool_seeds(int(config["generator_seed"]), config["pool_seed_offsets"])
                sizes = _pool_sizes(partitioned, float(config["pool_redundancy"]))
                feature_and_target = [*spec.feature_columns, spec.target_column]
                pools: dict[str, pd.DataFrame] = {}
                pool_manifests = []
                for pool_name in POOL_NAMES:
                    sample = generator.sample(sizes[pool_name], pool_seeds[pool_name], pool_name)
                    pool = add_pool_provenance(
                        sample,
                        "adult",
                        generator_name,
                        int(config["generator_seed"]),
                        pool_seeds[pool_name],
                        pool_name,
                    )
                    pools[pool_name] = pool
                    pool_manifests.append(write_pool(pool, feature_and_target, run_dir / "pools", run_id))
                pool_validation = validate_pools(pools, feature_and_target)
                write_json(
                    {"pools": pool_manifests, "validation": pool_validation},
                    run_dir / "pools_manifest.json",
                )
                source_model = source.loc[:, feature_and_target]
                final_real = partitioned.loc[
                    partitioned["split"] == "R_final_test", feature_and_target
                ]
                quality_sample = pools["S_final_test"].loc[:, feature_and_target]
                quality_n = min(len(source_model), len(quality_sample), 10000)
                utility_n = min(len(source_model), len(pools["S_downstream_mix"]))
                quality = {
                    "sdmetrics": sdmetrics_quality(source_model.head(quality_n), quality_sample.head(quality_n)),
                    "overlap": overlap_diagnostics(source_model, pools, feature_and_target),
                    "format": format_diagnostics(pools["S_final_test"], spec),
                    "real_target_distribution": source_model[spec.target_column].astype(str).value_counts(normalize=True).to_dict(),
                    "synthetic_target_distribution": pools["S_final_test"][spec.target_column].astype(str).value_counts(normalize=True).to_dict(),
                    "TRTR": utility_smoke(source_model.head(utility_n), final_real, spec),
                    "TSTR": utility_smoke(
                        pools["S_downstream_mix"].loc[:, feature_and_target].head(utility_n),
                        final_real,
                        spec,
                    ),
                }
                write_json(quality, run_dir / "quality.json")
                loaded = SDVGenerator.load(generator_name, model_path, generator_config)
                reloaded_sample = loaded.sample(10, 9090, "pilot_reload_validation")
                reload_validation = {
                    "model_reload_ok": True,
                    "schema_order_ok": reloaded_sample.columns.tolist() == feature_and_target,
                    "sample_rows": len(reloaded_sample),
                    "sample_seed": 9090,
                }
                write_json(reload_validation, run_dir / "reload_validation.json")
                write_json(generator.get_provenance(), run_dir / "generator_provenance.json")
        timing = {
            "total_seconds": perf_counter() - total_started,
            "fit_seconds": generator.fit_seconds,
            "sample_times": generator.sample_times,
        }
        write_json(timing, run_dir / "timing.json")
        required_present = PILOT_REQUIRED.issubset({path.name for path in run_dir.iterdir()})
        smoke_candidates = sorted(
            (root / "runs").glob(f"c2-smoke-adult-{generator_name.lower()}-s42*")
        )
        smoke_success = [path for path in smoke_candidates if (path / "pools_manifest.json").exists()]
        cross_run_content_distinct = True
        if smoke_success:
            smoke_manifest = json.loads(
                (smoke_success[-1] / "pools_manifest.json").read_text(encoding="utf-8")
            )
            smoke_hashes = {
                item["pool_name"]: item["content_sha256"] for item in smoke_manifest["pools"]
            }
            pilot_hashes = {
                item["pool_name"]: item["content_sha256"] for item in pool_manifests
            }
            cross_run_content_distinct = smoke_hashes != pilot_hashes
        gate = evaluate_quality_gate(
            access_audit=audit_payload,
            pool_validation=pool_validation,
            quality=quality,
            real_target_values=set(source_model[spec.target_column].dropna().astype(str)),
            reload_validation=reload_validation,
            required_artifacts_present=required_present,
            cross_run_content_distinct=cross_run_content_distinct,
        )
        write_json(gate, run_dir / "quality_gate.json")
        status = "pilot_passed" if gate["passed"] else "quality_blocked"
        set_status(run_dir, status, {"quality_gate": gate})
        artifact_manifest(run_dir, run_id)
        validation = validate_artifact_manifest(run_dir)
        if status == "pilot_passed":
            mark_complete(run_dir)
        result = {
            "run_id": run_id,
            "run_type": "pilot",
            "status": status,
            "generator": generator_name,
            "fit_rows": len(source),
            "fit_scope": "full_R_source_train",
            "generator_seed": int(config["generator_seed"]),
            "pool_seeds": pool_seeds,
            "pool_sizes": sizes,
            "model_sha256": sha256_file(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "quality_gate": gate,
            "quality": quality,
            "timing": timing,
            "artifact_validation": validation,
        }
        write_json(result, run_dir / "pilot_summary.json")
        # Refresh after summary, then validate the final inventory.
        artifact_manifest(run_dir, run_id)
        result["artifact_validation"] = validate_artifact_manifest(run_dir)
        return result
    except Exception as exc:
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        write_json(
            {"total_seconds": perf_counter() - total_started, "failed": True},
            run_dir / "timing.json",
        )
        set_status(run_dir, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        artifact_manifest(run_dir, run_id)
        raise


def validate_pilot_run(project_root: str | Path, run_id: str) -> dict[str, Any]:
    run_dir = Path(project_root).resolve() / "runs" / run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"]
    artifacts = validate_artifact_manifest(run_dir)
    return {
        "run_id": run_id,
        "run_type": manifest["run_type"],
        "status": status,
        "complete_marker": (run_dir / "COMPLETE").exists(),
        "artifacts": artifacts,
    }
