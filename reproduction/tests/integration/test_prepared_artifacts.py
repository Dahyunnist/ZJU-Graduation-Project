from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tabpollution.config import SPLIT_NAMES
from tabpollution.data.splits import assignment_content_hash, validate_split_assignment
from tabpollution.utils import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("dataset_id", ["adult", "credit"])
def test_real_dataset_prepare_to_card_integration(dataset_id: str) -> None:
    registry_path = PROJECT_ROOT / "manifests" / "benchmark_v1" / "datasets.json"
    assert registry_path.exists(), "Run data prepare before the real-data integration tests"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = next(item for item in registry["datasets"] if item["dataset_id"] == dataset_id)
    processed = PROJECT_ROOT / entry["processed_path"]
    assert sha256_file(processed) == entry["processed_sha256"]
    frame = pd.read_csv(processed, dtype={"row_id": "string"})
    assert frame["row_id"].is_unique
    assert (PROJECT_ROOT / "reports" / "data_cards" / f"dataset_card_{dataset_id}.md").exists()
    assert (PROJECT_ROOT / "reports" / "data_cards" / f"dataset_card_{dataset_id}.json").exists()

    expected_ids = set(frame["row_id"].astype(str))
    hashes = []
    for seed in (2026, 2027, 2028, 2029, 2030):
        split_path = PROJECT_ROOT / "data" / "splits" / "benchmark_v1" / dataset_id / f"seed_{seed}.csv"
        assignment = pd.read_csv(split_path, dtype={"row_id": "string", "target": "string"})
        validate_split_assignment(assignment, expected_ids)
        assert set(assignment["split"]) == set(SPLIT_NAMES)
        hashes.append(assignment_content_hash(assignment))
    assert len(hashes) == 5


@pytest.mark.parametrize("dataset_id", ["adult", "credit"])
def test_saved_split_hash_matches_manifest(dataset_id: str) -> None:
    for seed in (2026, 2027, 2028, 2029, 2030):
        root = PROJECT_ROOT / "data" / "splits" / "benchmark_v1" / dataset_id
        assignment = pd.read_csv(root / f"seed_{seed}.csv", dtype={"row_id": "string", "target": "string"})
        metadata = json.loads((root / f"seed_{seed}.json").read_text(encoding="utf-8"))
        assert assignment_content_hash(assignment) == metadata["assignment_sha256"]

