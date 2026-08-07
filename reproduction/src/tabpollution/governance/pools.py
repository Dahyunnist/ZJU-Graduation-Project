"""Build registry-backed real/synthetic pools without changing benchmark code."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import gc
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from tabpollution.generators.sdv_adapter import create_generator
from tabpollution.pipeline import prepare_benchmark
from tabpollution.utils import read_json, sha256_file, write_json


ABALONE_URL = "https://archive.ics.uci.edu/static/public/1/abalone.zip"
ABALONE_COLUMNS = [
    "sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
    "viscera_weight", "shell_weight", "rings",
]


@dataclass(frozen=True)
class PoolDatasetSpec:
    table_id: str
    domain: str
    source_path: Path
    source_target: str
    target_column: str
    target_transform: str
    target_threshold: float | None = None


@dataclass(frozen=True)
class PoolBuildConfig:
    build_id: str
    seed: int
    sample_size: int | str
    source_train_fraction: float
    datasets: tuple[PoolDatasetSpec, ...]
    generators: dict[str, dict[str, Any]]
    output_dir: Path
    checkpoint_dir: Path


def _download_atomic(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tabpollution-benchmark/0.1 (academic dataset preparation)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    return destination


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _parse_abalone_archive(archive_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name.lower() == "abalone.data"]
        if len(members) != 1:
            raise ValueError(f"Expected one abalone.data member in {archive_path}, found {members}")
        payload = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(payload), names=ABALONE_COLUMNS, header=None)
    if len(frame) != 4177 or frame.isna().any().any():
        raise ValueError(f"Unexpected Abalone data shape or missing values: {frame.shape}")
    fingerprints = frame.astype(str).agg("|".join, axis=1)
    frame.insert(0, "row_id", [
        f"abalone:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}:{index:04d}"
        for index, value in enumerate(fingerprints)
    ])
    return frame


def prepare_governance_sources(config_path: str | Path) -> dict[str, Any]:
    """Prepare the three official real tables required by the governance benchmark."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_pool_build_config(config_path)
    expected = {spec.table_id for spec in config.datasets}
    if expected != {"adult", "credit", "abalone"}:
        raise ValueError("Source preparation currently requires exactly adult, credit, and abalone")

    benchmark_summary = prepare_benchmark(project_root / "configs" / "benchmark_v1.yaml")
    raw_path = _download_atomic(ABALONE_URL, project_root / "data" / "raw" / "abalone" / "abalone.zip")
    abalone = _parse_abalone_archive(raw_path)
    processed_path = project_root / "data" / "processed" / "abalone" / "abalone_clean.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = processed_path.with_suffix(".csv.part")
    abalone.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(processed_path)
    summary = {
        "status": "complete",
        "official_sources": {
            "adult": benchmark_summary["adult"],
            "credit": benchmark_summary["credit"],
            "abalone": {
                "source_url": ABALONE_URL,
                "license": "CC BY 4.0",
                "raw_path": str(raw_path),
                "raw_sha256": sha256_file(raw_path),
                "processed_path": str(processed_path),
                "processed_sha256": sha256_file(processed_path),
                "rows": len(abalone),
                "features": len(ABALONE_COLUMNS) - 1,
                "target": "rings",
            },
        },
    }
    write_json(summary, project_root / "manifests" / config.build_id / "source_prepare_summary.json")
    return summary


def load_pool_build_config(path: str | Path) -> PoolBuildConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    required = {
        "build_id", "seed", "sample_size", "source_train_fraction", "datasets",
        "generators", "output_dir", "checkpoint_dir",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(f"Pool build config must contain exactly {sorted(required)}")
    sample_size = raw["sample_size"]
    if sample_size != "match_real" and (not isinstance(sample_size, int) or sample_size <= 0):
        raise ValueError("sample_size must be match_real or a positive integer")
    source_train_fraction = float(raw["source_train_fraction"])
    if not 0.1 <= source_train_fraction <= 0.8:
        raise ValueError("source_train_fraction must be within [0.1, 0.8]")
    datasets = []
    for table_id, value in raw["datasets"].items():
        fields = {"domain", "source_path", "source_target", "target_column", "target_transform"}
        if not isinstance(value, dict) or not fields.issubset(value) or set(value) - (fields | {"target_threshold"}):
            raise ValueError(f"Dataset {table_id} has invalid fields")
        transform = str(value["target_transform"])
        if transform not in {"identity", "median_binary", "threshold_binary"}:
            raise ValueError(f"Unsupported target transform for {table_id}: {transform}")
        threshold = value.get("target_threshold")
        if transform == "threshold_binary" and not isinstance(threshold, (int, float)):
            raise ValueError(f"Dataset {table_id} requires numeric target_threshold")
        if transform != "threshold_binary" and threshold is not None:
            raise ValueError(f"Dataset {table_id} cannot set target_threshold for {transform}")
        datasets.append(PoolDatasetSpec(
            table_id=str(table_id), domain=str(value["domain"]),
            source_path=(path.parent / str(value["source_path"])).resolve(),
            source_target=str(value["source_target"]),
            target_column=str(value["target_column"]), target_transform=transform,
            target_threshold=None if threshold is None else float(threshold),
        ))
    if not datasets:
        raise ValueError("At least one pool dataset is required")
    generators = raw["generators"]
    if not isinstance(generators, dict) or not generators:
        raise ValueError("generators must be a non-empty mapping")
    unknown = sorted(set(generators) - {"GaussianCopula", "CTGAN", "TVAE"})
    if unknown:
        raise ValueError(f"Unsupported generators: {unknown}")
    if any(not isinstance(value, dict) for value in generators.values()):
        raise ValueError("Every generator configuration must be a mapping")
    return PoolBuildConfig(
        build_id=str(raw["build_id"]), seed=int(raw["seed"]), sample_size=sample_size,
        source_train_fraction=source_train_fraction,
        datasets=tuple(datasets), generators={str(k): dict(v) for k, v in generators.items()},
        output_dir=(path.parent / str(raw["output_dir"])).resolve(),
        checkpoint_dir=(path.parent / str(raw["checkpoint_dir"])).resolve(),
    )


def _canonical_real(spec: PoolDatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(spec.source_path)
    if spec.source_target not in frame:
        raise ValueError(f"Target {spec.source_target!r} absent from {spec.source_path}")
    frame = frame.drop(columns=[column for column in ("row_id", "split") if column in frame]).copy()
    transform: dict[str, Any] = {"name": spec.target_transform}
    if spec.target_transform == "identity":
        if spec.target_column != spec.source_target:
            frame = frame.rename(columns={spec.source_target: spec.target_column})
    else:
        numeric = pd.to_numeric(frame[spec.source_target], errors="raise")
        threshold = (
            float(numeric.median())
            if spec.target_transform == "median_binary"
            else float(spec.target_threshold)
        )
        frame[spec.target_column] = (numeric > threshold).astype(int)
        if spec.target_column != spec.source_target:
            frame = frame.drop(columns=[spec.source_target])
        transform["threshold"] = threshold
        transform["rule"] = f"{spec.source_target} > {threshold:g}"
    if frame[spec.target_column].nunique(dropna=True) != 2:
        raise ValueError(f"Canonical target for {spec.table_id} is not binary")
    return frame, transform


def preflight_pool_build(config_or_path: PoolBuildConfig | str | Path) -> dict[str, Any]:
    config = config_or_path if isinstance(config_or_path, PoolBuildConfig) else load_pool_build_config(config_or_path)
    datasets = []
    passed = True
    for spec in config.datasets:
        row: dict[str, Any] = {"table_id": spec.table_id, "source_path": str(spec.source_path)}
        try:
            canonical, transform = _canonical_real(spec)
            row.update(passed=True, rows=len(canonical), columns=len(canonical.columns), target_transform=transform)
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            passed = False
            row.update(passed=False, error=str(exc))
        datasets.append(row)
    try:
        sdv_version = importlib_metadata.version("sdv")
    except importlib_metadata.PackageNotFoundError:
        sdv_version = None
        passed = False
    return {
        "build_id": config.build_id,
        "source_train_fraction": config.source_train_fraction,
        "passed": passed,
        "sdv_version": sdv_version,
        "datasets": datasets,
        "generators": sorted(config.generators),
        "output_dir": str(config.output_dir),
        "checkpoint_dir": str(config.checkpoint_dir),
    }


def build_governance_pools(config_or_path: PoolBuildConfig | str | Path, *, resume: bool = False) -> dict[str, Any]:
    config = config_or_path if isinstance(config_or_path, PoolBuildConfig) else load_pool_build_config(config_or_path)
    preflight = preflight_pool_build(config)
    if not preflight["passed"]:
        raise RuntimeError(f"Pool build preflight failed: {preflight}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "pool_build_manifest.json"
    registry_path = config.output_dir / "pool_registry.csv"
    if registry_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite completed registry: {registry_path}")
    previous_runs: dict[tuple[str, str], dict[str, Any]] = {}
    if resume and manifest_path.exists():
        previous = read_json(manifest_path)
        previous_runs = {
            (str(row["table_id"]), str(row["generator"])): row
            for row in previous.get("runs", [])
        }
    started = time.perf_counter()
    registry_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for dataset_index, spec in enumerate(config.datasets):
        canonical, transform = _canonical_real(spec)
        source_train, real = train_test_split(
            canonical,
            train_size=config.source_train_fraction,
            random_state=config.seed + dataset_index * 1009,
            shuffle=True,
            stratify=canonical[spec.target_column],
        )
        source_train = source_train.reset_index(drop=True)
        real = real.reset_index(drop=True)
        source_train_path = config.output_dir / "source_train" / f"{spec.table_id}.csv"
        real_path = config.output_dir / "real" / f"{spec.table_id}.csv"
        expected_source_sha = _frame_sha256(source_train)
        expected_real_sha = _frame_sha256(real)
        if resume and source_train_path.exists() and sha256_file(source_train_path) != expected_source_sha:
            raise RuntimeError(f"Frozen source split changed for {spec.table_id}; use a new build_id")
        if resume and real_path.exists() and sha256_file(real_path) != expected_real_sha:
            raise RuntimeError(f"Frozen evaluation pool changed for {spec.table_id}; use a new build_id")
        if not source_train_path.exists() or not resume:
            _write_frame_atomic(source_train, source_train_path)
        if not real_path.exists() or not resume:
            _write_frame_atomic(real, real_path)
        for generator_index, (generator_name, generator_config) in enumerate(config.generators.items()):
            synthetic_path = config.output_dir / "synthetic" / spec.table_id / f"{generator_name}.csv"
            model_path = config.checkpoint_dir / spec.table_id / f"{generator_name}.pkl"
            generator_seed = config.seed + dataset_index * 1009 + generator_index * 101
            sample_seed = generator_seed + 1_000_003
            generator_config_sha = hashlib.sha256(
                json.dumps(generator_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if synthetic_path.exists() and model_path.exists() and resume:
                previous_run = previous_runs.get((spec.table_id, generator_name))
                expected_resume = {
                    "generator_seed": generator_seed,
                    "sample_seed": sample_seed,
                    "source_train_sha256": expected_source_sha,
                    "generator_config_sha256": generator_config_sha,
                }
                if previous_run is None or any(previous_run.get(key) != value for key, value in expected_resume.items()):
                    raise RuntimeError(
                        f"Cannot safely resume stale {spec.table_id}/{generator_name}; use a new build_id"
                    )
                status = "reused_complete"
                synthetic = pd.read_csv(synthetic_path)
                provenance: dict[str, Any] = {"reused": True}
                fit_seconds = 0.0
            else:
                generator = create_generator(generator_name, generator_config)
                metadata = generator.build_metadata(source_train)
                fit_started = time.perf_counter()
                generator.fit(source_train, metadata, generator_seed)
                fit_seconds = time.perf_counter() - fit_started
                n = len(real) if config.sample_size == "match_real" else int(config.sample_size)
                synthetic = generator.sample(n, sample_seed, "governance_formal")
                if set(synthetic.columns) != set(real.columns):
                    raise ValueError(f"Generated schema mismatch for {spec.table_id}/{generator_name}")
                synthetic = synthetic[real.columns]
                synthetic_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _write_frame_atomic(synthetic, synthetic_path)
                temporary_model = model_path.with_suffix(model_path.suffix + ".partial")
                generator.save(temporary_model)
                temporary_model.replace(model_path)
                provenance = generator.get_provenance()
                del generator
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                status = "complete"
            registry_rows.append({
                "table_id": spec.table_id, "domain": spec.domain,
                "target_column": spec.target_column, "generator": generator_name,
                "real_path": real_path.relative_to(config.output_dir).as_posix(),
                "synthetic_path": synthetic_path.relative_to(config.output_dir).as_posix(),
            })
            runs.append({
                "table_id": spec.table_id, "generator": generator_name, "status": status,
                "generator_seed": generator_seed, "sample_seed": sample_seed,
                "rows": len(synthetic), "fit_seconds": fit_seconds,
                "source_train_rows": len(source_train), "evaluation_real_rows": len(real),
                "source_train_sha256": sha256_file(source_train_path),
                "real_sha256": sha256_file(real_path),
                "generator_config_sha256": generator_config_sha,
                "synthetic_sha256": sha256_file(synthetic_path),
                "model_path": str(model_path), "target_transform": transform,
                "provenance": provenance,
            })
            write_json({"build_id": config.build_id, "status": "running", "runs": runs}, manifest_path)
    registry = pd.DataFrame(registry_rows).sort_values(["table_id", "generator"])
    temporary_registry = registry_path.with_suffix(".csv.partial")
    registry.to_csv(temporary_registry, index=False)
    temporary_registry.replace(registry_path)
    summary = {
        "build_id": config.build_id, "status": "complete",
        "seconds": time.perf_counter() - started,
        "registry_path": str(registry_path), "registry_rows": len(registry),
        "runs": runs,
    }
    write_json(summary, manifest_path)
    return summary
