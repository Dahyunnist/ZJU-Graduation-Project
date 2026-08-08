"""C1 dataset preparation and validation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from tabpollution.config import load_benchmark_config, write_resolved_config
from tabpollution.data.cards import build_card_payload, write_dataset_card
from tabpollution.data.cleaning import clean_dataset
from tabpollution.data.loaders import ensure_raw_file, load_processed_dataset, load_raw_dataset
from tabpollution.data.registry import load_dataset_spec
from tabpollution.data.splits import (
    build_split_assignment,
    validate_split_assignment,
    write_split_artifacts,
)
from tabpollution.utils import read_json, sha256_file, write_json


def prepare_benchmark(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_benchmark_config(config_path)
    manifest_root = project_root / "manifests" / config["benchmark_id"]
    write_resolved_config(config, manifest_root / "config_resolved.yaml")
    registry_path = manifest_root / "datasets.json"
    existing_registry = read_json(registry_path) if registry_path.is_file() else {}
    existing_entries = {
        item["dataset_id"]: item
        for item in existing_registry.get("datasets", [])
        if isinstance(item, dict) and "dataset_id" in item
    }

    registry_entries: list[dict[str, Any]] = []
    prepared: dict[str, Any] = {}
    for dataset_id in config["datasets"]:
        spec_path = project_root / "configs" / "datasets" / f"{dataset_id}.yaml"
        spec = load_dataset_spec(spec_path)
        raw_path = ensure_raw_file(spec, project_root / "data" / "raw" / dataset_id)
        raw_sha = sha256_file(raw_path)
        existing_entry = existing_entries.get(dataset_id, {})
        retrieval_date = (
            existing_entry.get("retrieval_date")
            if existing_entry.get("raw_sha256") == raw_sha and existing_entry.get("retrieval_date")
            else date.today().isoformat()
        )
        raw = load_raw_dataset(spec, raw_path)
        cleaned, cleaning_report = clean_dataset(raw, spec)

        processed_dir = project_root / "data" / "processed" / dataset_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / f"{dataset_id}_clean.csv"
        cleaned.to_csv(processed_path, index=False, lineterminator="\n")
        processed_sha = sha256_file(processed_path)

        split_summaries = []
        for seed in config["formal_seeds"]:
            assignment = build_split_assignment(cleaned, spec.target_column, seed)
            summary = write_split_artifacts(
                dataset_id, seed, assignment, project_root / "data" / "splits" / config["benchmark_id"]
            )
            split_summaries.append(summary)

        card = build_card_payload(
            spec=spec,
            frame=cleaned,
            cleaning=cleaning_report,
            raw_sha256=raw_sha,
            processed_sha256=processed_sha,
            retrieval_date=retrieval_date,
            splits=split_summaries,
        )
        write_dataset_card(card, project_root / "reports" / "data_cards")

        entry = {
            "dataset_id": dataset_id,
            "canonical_name": spec.canonical_name,
            "source_page": spec.source_page,
            "download_url": spec.download_url,
            "doi": spec.doi,
            "license": spec.license,
            "retrieval_date": retrieval_date,
            "raw_path": raw_path.relative_to(project_root).as_posix(),
            "raw_sha256": raw_sha,
            "processed_path": processed_path.relative_to(project_root).as_posix(),
            "processed_sha256": processed_sha,
            "task_type": spec.task_type,
            "rows": len(cleaned),
            "features": len(spec.feature_columns),
            "target": spec.target_column,
            "numeric_columns": list(spec.numeric_columns),
            "categorical_columns": list(spec.categorical_columns),
            "missing_markers": list(spec.missing_markers),
            "cleaning": asdict(cleaning_report),
        }
        registry_entries.append(entry)
        prepared[dataset_id] = {
            "cleaning": asdict(cleaning_report),
            "raw_sha256": raw_sha,
            "processed_sha256": processed_sha,
            "split_hashes": {str(item.seed): item.assignment_sha256 for item in split_summaries},
        }

    generated_at = (
        existing_registry.get("generated_at")
        if existing_registry.get("datasets") == registry_entries and existing_registry.get("generated_at")
        else date.today().isoformat()
    )
    write_json({
        "benchmark_id": config["benchmark_id"],
        "generated_at": generated_at,
        "datasets": registry_entries,
    }, registry_path)
    write_json(prepared, manifest_root / "prepare_summary.json")
    return prepared


def validate_prepared_benchmark(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_benchmark_config(config_path)
    registry_path = project_root / "manifests" / config["benchmark_id"] / "datasets.json"
    registry = read_json(registry_path)
    entries = {item["dataset_id"]: item for item in registry["datasets"]}
    results: dict[str, Any] = {}
    for dataset_id in config["datasets"]:
        entry = entries[dataset_id]
        processed_path = project_root / entry["processed_path"]
        if sha256_file(processed_path) != entry["processed_sha256"]:
            raise ValueError(f"Processed checksum mismatch: {dataset_id}")
        spec = load_dataset_spec(project_root / "configs" / "datasets" / f"{dataset_id}.yaml")
        frame = load_processed_dataset(spec, processed_path)
        expected_ids = set(frame["row_id"].astype(str))
        seed_results = {}
        for seed in config["formal_seeds"]:
            split_path = (
                project_root
                / "data"
                / "splits"
                / config["benchmark_id"]
                / dataset_id
                / f"seed_{seed}.csv"
            )
            assignment = pd.read_csv(split_path, dtype={"row_id": "string", "target": "string"})
            validate_split_assignment(assignment, expected_ids)
            counts = assignment["split"].value_counts().to_dict()
            seed_results[str(seed)] = {key: int(value) for key, value in counts.items()}
        results[dataset_id] = {
            "rows": len(frame),
            "row_id_unique": bool(frame["row_id"].is_unique),
            "seeds": seed_results,
        }
    return results
