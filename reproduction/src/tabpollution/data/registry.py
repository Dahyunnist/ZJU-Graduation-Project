"""Dataset registry schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabpollution.config import ConfigError, load_yaml


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    canonical_name: str
    uci_id: int
    doi: str
    source_page: str
    download_url: str
    raw_filename: str
    license: str
    task_type: str
    target_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    missing_markers: tuple[str, ...]
    source_id_column: str | None = None

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.categorical_columns


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    raw: dict[str, Any] = load_yaml(path)
    required = {
        "dataset_id",
        "canonical_name",
        "uci_id",
        "doi",
        "source_page",
        "download_url",
        "raw_filename",
        "license",
        "task_type",
        "target_column",
        "numeric_columns",
        "categorical_columns",
        "missing_markers",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"Dataset registry is missing fields {missing}: {path}")
    numeric = tuple(raw["numeric_columns"])
    categorical = tuple(raw["categorical_columns"])
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ConfigError(f"Columns cannot be both numeric and categorical: {overlap}")
    if raw["target_column"] in set(numeric) | set(categorical):
        raise ConfigError("target_column must not be included in feature columns")
    return DatasetSpec(
        dataset_id=str(raw["dataset_id"]),
        canonical_name=str(raw["canonical_name"]),
        uci_id=int(raw["uci_id"]),
        doi=str(raw["doi"]),
        source_page=str(raw["source_page"]),
        download_url=str(raw["download_url"]),
        raw_filename=str(raw["raw_filename"]),
        license=str(raw["license"]),
        task_type=str(raw["task_type"]),
        target_column=str(raw["target_column"]),
        numeric_columns=numeric,
        categorical_columns=categorical,
        missing_markers=tuple(str(x) for x in raw["missing_markers"]),
        source_id_column=raw.get("source_id_column"),
    )


def validate_dataframe_schema(columns: list[str], spec: DatasetSpec) -> None:
    expected = set(spec.feature_columns) | {spec.target_column}
    actual = set(columns)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected - {"row_id"})
    if missing or unknown:
        raise ConfigError(
            f"Schema mismatch for {spec.dataset_id}; missing={missing}, unknown={unknown}"
        )

