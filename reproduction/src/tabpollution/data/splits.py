"""Frozen, deterministic and stratified Track A splits."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from tabpollution.config import SPLIT_NAMES
from tabpollution.utils import sha256_file, write_json


@dataclass(frozen=True)
class SplitSummary:
    dataset_id: str
    seed: int
    strategy: str
    rows: int
    counts: dict[str, int]
    proportions: dict[str, float]
    target_distribution: dict[str, dict[str, float]]
    assignment_sha256: str


def build_split_assignment(
    frame: pd.DataFrame,
    target_column: str,
    seed: int,
) -> pd.DataFrame:
    if not frame["row_id"].is_unique:
        raise ValueError("row_id must be unique before splitting")
    ids = frame["row_id"].astype(str).to_numpy()
    labels = frame[target_column].astype(str).to_numpy()

    remain_85, final_15 = train_test_split(
        ids, test_size=0.15, random_state=seed, stratify=labels
    )
    label_by_id = dict(zip(ids, labels, strict=True))
    remain_labels = [label_by_id[row_id] for row_id in remain_85]
    remain_75, val_10 = train_test_split(
        remain_85,
        test_size=10 / 85,
        random_state=seed + 1,
        stratify=remain_labels,
    )
    remain_75_labels = [label_by_id[row_id] for row_id in remain_75]
    source_60, detector_15 = train_test_split(
        remain_75,
        test_size=0.20,
        random_state=seed + 2,
        stratify=remain_75_labels,
    )
    split_ids = {
        "R_source_train": source_60,
        "R_detector_train": detector_15,
        "R_detector_val": val_10,
        "R_final_test": final_15,
    }
    rows: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        for row_id in split_ids[split_name]:
            rows.append(
                {"row_id": str(row_id), "split": split_name, "target": label_by_id[str(row_id)]}
            )
    assignment = pd.DataFrame(rows).sort_values("row_id", kind="stable").reset_index(drop=True)
    validate_split_assignment(assignment, set(ids))
    return assignment


def validate_split_assignment(assignment: pd.DataFrame, expected_ids: set[str]) -> None:
    required = {"row_id", "split", "target"}
    if set(assignment.columns) != required:
        raise ValueError(f"Split assignment columns must be {sorted(required)}")
    if assignment["row_id"].duplicated().any():
        raise ValueError("A row_id occurs in more than one split")
    unknown_splits = sorted(set(assignment["split"]) - set(SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(f"Unknown split labels: {unknown_splits}")
    actual_ids = set(assignment["row_id"].astype(str))
    missing = expected_ids - actual_ids
    unknown = actual_ids - expected_ids
    if missing or unknown:
        raise ValueError(f"Split coverage mismatch; missing={len(missing)}, unknown={len(unknown)}")
    if set(assignment["split"]) != set(SPLIT_NAMES):
        raise ValueError("All four frozen partitions must be present")


def assignment_content_hash(assignment: pd.DataFrame) -> str:
    canonical = assignment.sort_values("row_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def summarize_split(dataset_id: str, seed: int, assignment: pd.DataFrame) -> SplitSummary:
    counts = {name: int((assignment["split"] == name).sum()) for name in SPLIT_NAMES}
    total = len(assignment)
    proportions = {name: counts[name] / total for name in SPLIT_NAMES}
    target_distribution: dict[str, dict[str, float]] = {}
    for name in SPLIT_NAMES:
        values = assignment.loc[assignment["split"] == name, "target"].value_counts(normalize=True)
        target_distribution[name] = {str(key): float(value) for key, value in values.items()}
    return SplitSummary(
        dataset_id=dataset_id,
        seed=seed,
        strategy="sequential_stratified_sklearn(seed, seed+1, seed+2)",
        rows=total,
        counts=counts,
        proportions=proportions,
        target_distribution=target_distribution,
        assignment_sha256=assignment_content_hash(assignment),
    )


def write_split_artifacts(
    dataset_id: str,
    seed: int,
    assignment: pd.DataFrame,
    split_dir: str | Path,
) -> SplitSummary:
    split_dir = Path(split_dir) / dataset_id
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = split_dir / f"seed_{seed}.csv"
    assignment.to_csv(csv_path, index=False, lineterminator="\n")
    summary = summarize_split(dataset_id, seed, assignment)
    payload = asdict(summary)
    payload["file_sha256"] = sha256_file(csv_path)
    write_json(payload, split_dir / f"seed_{seed}.json")
    return summary

