"""Deterministic cleaning and content-addressed row identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tabpollution.data.registry import DatasetSpec, validate_dataframe_schema


@dataclass(frozen=True)
class CleaningReport:
    raw_rows: int
    cleaned_rows: int
    exact_duplicate_rows_removed: int
    raw_columns: int
    cleaned_columns: int


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value).strip()


def content_fingerprints(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    fingerprints: list[str] = []
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        canonical = [_canonical_value(value) for value in values]
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        fingerprints.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return fingerprints


def add_stable_row_ids(frame: pd.DataFrame, dataset_id: str, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    fingerprints = pd.Series(content_fingerprints(result, columns), index=result.index)
    occurrence = fingerprints.groupby(fingerprints, sort=False).cumcount()
    result.insert(
        0,
        "row_id",
        [
            f"{dataset_id}:{fingerprint[:24]}:{int(ordinal):04d}"
            for fingerprint, ordinal in zip(fingerprints, occurrence, strict=True)
        ],
    )
    if not result["row_id"].is_unique:
        raise AssertionError("Stable row_id construction produced a collision")
    return result


def clean_dataset(raw: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, CleaningReport]:
    validate_dataframe_schema(raw.columns.tolist(), spec)
    data = raw.loc[:, [*spec.feature_columns, spec.target_column]].copy()
    for column in spec.numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    for column in spec.categorical_columns:
        data[column] = data[column].astype("string").str.strip()
    data[spec.target_column] = data[spec.target_column].astype("string").str.strip().str.rstrip(".")

    all_columns = [*spec.feature_columns, spec.target_column]
    raw_rows = len(data)
    duplicate_mask = data.duplicated(subset=all_columns, keep="first")
    duplicates = int(duplicate_mask.sum())
    data = data.loc[~duplicate_mask].copy()
    data = add_stable_row_ids(data, spec.dataset_id, all_columns)
    data = data.sort_values("row_id", kind="stable").reset_index(drop=True)
    validate_dataframe_schema(data.columns.tolist(), spec)
    report = CleaningReport(
        raw_rows=raw_rows,
        cleaned_rows=len(data),
        exact_duplicate_rows_removed=duplicates,
        raw_columns=len(raw.columns),
        cleaned_columns=len(data.columns),
    )
    return data, report


def feature_frame(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    missing = sorted(set(spec.feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return frame.loc[:, list(spec.feature_columns)].copy()

