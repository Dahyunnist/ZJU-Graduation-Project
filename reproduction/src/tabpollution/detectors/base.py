"""Shared detector contract and leakage-safe record helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROVENANCE_COLUMNS = {
    "row_id", "synth_row_id", "record_id", "mix_row_id", "source_row_id",
    "split", "pool", "pool_name", "generator", "generator_name",
    "generator_seed", "sample_seed", "dataset_id", "table_id", "source_label",
    "run_id", "condition", "proportion", "p", "mix_seed", "source_type",
    "source_synth_row_id",
}


def feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    """Return model features and fail if no useful feature remains."""
    cols = [c for c in records.columns if c not in PROVENANCE_COLUMNS]
    if not cols:
        raise ValueError("No feature columns remain after provenance removal")
    return records.loc[:, cols].copy()


def normalize_scalar(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    return str(value).strip()


def serialize_record(row: pd.Series, *, shuffle: bool = False, seed: int = 0) -> str:
    parts = [f"{c}:{normalize_scalar(row[c])}" for c in row.index if c not in PROVENANCE_COLUMNS]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(parts)
    return " | ".join(parts)


class Detector(ABC):
    @abstractmethod
    def fit(self, train_records: pd.DataFrame, train_labels: np.ndarray,
            val_records: pd.DataFrame | None = None, val_labels: np.ndarray | None = None,
            **context: Any) -> "Detector": ...

    @abstractmethod
    def predict_score(self, records: pd.DataFrame, **context: Any) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Detector": ...

    @abstractmethod
    def get_provenance(self) -> dict[str, Any]: ...
