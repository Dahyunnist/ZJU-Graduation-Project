"""Adult C2 smoke orchestration with strict split access and artifact manifests."""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tabpollution.data.loaders import load_processed_dataset
from tabpollution.data.registry import load_dataset_spec
from tabpollution.generators.base import audit_generator_access, derive_pool_seeds
from tabpollution.generators.pools import POOL_NAMES, add_pool_provenance, validate_pools, write_pool
from tabpollution.generators.quality import (
    format_diagnostics,
    overlap_diagnostics,
    sdmetrics_quality,
    utility_smoke,
)
from tabpollution.generators.sdv_adapter import create_generator
from tabpollution.generators.sdv_adapter import SDVGenerator
from tabpollution.utils import sha256_file, write_json


def _load_smoke_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_partitioned_adult(project_root: Path, split_seed: int) -> tuple[Any, pd.DataFrame]:
    spec = load_dataset_spec(project_root / "configs" / "datasets" / "adult.yaml")
    data = load_processed_dataset(spec, project_root / "data" / "processed" / "adult" / "adult_clean.csv")
    assignment = pd.read_csv(
        project_root / "data" / "splits" / "benchmark_v1" / "adult" / f"seed_{split_seed}.csv",
        dtype={"row_id": "string", "target": "string"},
    )
    split_by_id = assignment.set_index("row_id")["split"]
    result = data.copy()
    result["split"] = result["row_id"].map(split_by_id)
    if result["split"].isna().any():
        raise ValueError("Some Adult rows are absent from frozen split manifest")
    return spec, result


def _pool_sizes(partitioned: pd.DataFrame, redundancy: float) -> dict[str, int]:
    counts = partitioned["split"].value_counts()
    return {
        "S_detector_train": math.ceil(int(counts["R_detector_train"]) * redundancy),
        "S_detector_val": max(1000, math.ceil(int(counts["R_detector_val"]) * redundancy)),
        "S_final_test": max(1000, math.ceil(int(counts["R_final_test"]) * redundancy)),
        "S_downstream_mix": math.ceil(int(counts["R_source_train"]) * redundancy),
    }


def _environment() -> dict[str, Any]:
    import sdmetrics
    import sdv
    import sklearn
    import torch

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "sdv": sdv.__version__,
        "sdmetrics": sdmetrics.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }


def run_adult_generator_smoke(
    project_root: str | Path,
    generator_name: str,
    smoke_config_path: str | Path = "configs/smoke_c2_c3.yaml",
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config = _load_smoke_config(project_root / smoke_config_path)
    split_seed = int(config["split_seed"])
    generator_seed = int(config["generator_seed"])
    base_run_id = f"c2-smoke-adult-{generator_name.lower()}-s{generator_seed}"
    run_id = base_run_id
    run_dir = project_root / "runs" / run_id
    attempt = 1
    while run_dir.exists():
        attempt += 1
        run_id = f"{base_run_id}-a{attempt}"
        run_dir = project_root / "runs" / run_id
    run_dir.mkdir(parents=True)

    spec, partitioned = _load_partitioned_adult(project_root, split_seed)
    source = partitioned.loc[partitioned["split"] == "R_source_train"].copy()
    allowed_ids = set(source["row_id"].astype(str))
    fit_scope = config["fit_scope"][generator_name]
    fit_records = source
    if fit_scope == "smoke_subset_3000":
        fit_records = source.sample(
            n=int(config["fit_rows"][generator_name]), random_state=generator_seed
        ).sort_values("row_id")
    audit = audit_generator_access(fit_records, allowed_ids)
    generator_config = dict(config["generators"][generator_name])
    generator = create_generator(generator_name, generator_config)
    metadata = generator.build_metadata(fit_records)
    generator.fit(fit_records, metadata, generator_seed)
    model_path = run_dir / "model.pkl"
    generator.save(model_path)

    pool_seeds = derive_pool_seeds(generator_seed, config["pool_seed_offsets"])
    sizes = _pool_sizes(partitioned, float(config["pool_redundancy"]))
    pools: dict[str, pd.DataFrame] = {}
    pool_manifests = []
    feature_and_target = [*spec.feature_columns, spec.target_column]
    for pool_name in POOL_NAMES:
        sampled = generator.sample(sizes[pool_name], pool_seeds[pool_name], pool_name)
        pool = add_pool_provenance(
            sampled, "adult", generator_name, generator_seed, pool_seeds[pool_name], pool_name
        )
        pools[pool_name] = pool
        pool_manifests.append(
            write_pool(pool, feature_and_target, run_dir / "pools", run_id)
        )
    pool_validation = validate_pools(pools, feature_and_target)

    source_model = source.loc[:, feature_and_target]
    final_real = partitioned.loc[partitioned["split"] == "R_final_test", feature_and_target]
    quality_sample = pools["S_final_test"].loc[:, feature_and_target]
    quality_n = min(len(source_model), len(quality_sample), 10000)
    quality = {
        "sdmetrics": sdmetrics_quality(source_model.head(quality_n), quality_sample.head(quality_n)),
        "overlap": overlap_diagnostics(source_model, pools, feature_and_target),
        "format": format_diagnostics(pools["S_final_test"], spec),
    }
    utility_n = min(len(source_model), len(pools["S_downstream_mix"]))
    quality["TRTR"] = utility_smoke(source_model.head(utility_n), final_real, spec)
    quality["TSTR"] = utility_smoke(
        pools["S_downstream_mix"].loc[:, feature_and_target].head(utility_n), final_real, spec
    )

    environment = _environment()
    resolved = {
        "run_id": run_id,
        "run_type": "smoke",
        "attempt": attempt,
        "dataset": "adult",
        "split_seed": split_seed,
        "generator_seed": generator_seed,
        "generator": generator_name,
        "fit_scope": fit_scope,
        "fit_rows": len(fit_records),
        "generator_config": generator_config,
        "pool_seeds": pool_seeds,
        "pool_sizes": sizes,
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    write_json(audit.__dict__, run_dir / "access_audit.json")
    write_json(environment, run_dir / "environment.json")
    write_json(generator.get_provenance(), run_dir / "generator_provenance.json")
    write_json({"pools": pool_manifests, "validation": pool_validation}, run_dir / "pools_manifest.json")
    write_json(quality, run_dir / "quality_smoke.json")
    result = {
        **resolved,
        "status": "smoke_passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_file": "model.pkl",
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "access_audit": audit.__dict__,
        "environment": environment,
        "generator_provenance": generator.get_provenance(),
        "pool_manifests": pool_manifests,
        "pool_validation": pool_validation,
        "quality": quality,
    }
    write_json(result, run_dir / "smoke_summary.json")
    return result


def validate_saved_adult_smokes(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    spec, partitioned = _load_partitioned_adult(project_root, 2026)
    source = partitioned.loc[partitioned["split"] == "R_source_train"].copy()
    metadata = SDVGenerator.build_metadata(source)
    feature_and_target = [*spec.feature_columns, spec.target_column]
    results = {}
    for generator_name in ("GaussianCopula", "CTGAN", "TVAE"):
        prefix = f"c2-smoke-adult-{generator_name.lower()}-s42"
        candidates = sorted((project_root / "runs").glob(f"{prefix}*"))
        successful = [path for path in candidates if (path / "smoke_summary.json").exists()]
        if not successful:
            raise FileNotFoundError(f"No successful smoke run for {generator_name}")
        run_dir = successful[-1]
        summary = json.loads((run_dir / "smoke_summary.json").read_text(encoding="utf-8"))
        loaded = SDVGenerator.load(generator_name, run_dir / "model.pkl", summary["generator_config"])
        sampled = loaded.sample(10, 9090, "reload_validation")
        schema_ok = sampled.columns.tolist() == feature_and_target
        if not schema_ok:
            raise ValueError(f"Reloaded {generator_name} schema mismatch")
        metadata_path = run_dir / "metadata.json"
        write_json(metadata.to_dict(), metadata_path)
        result = {
            "run_id": run_dir.name,
            "model_reload_ok": True,
            "sample_after_load_rows": len(sampled),
            "sample_after_load_seed": 9090,
            "schema_order_ok": schema_ok,
            "metadata_file": metadata_path.name,
            "metadata_sha256": sha256_file(metadata_path),
        }
        write_json(result, run_dir / "reload_validation.json")
        results[generator_name] = result
    return results
