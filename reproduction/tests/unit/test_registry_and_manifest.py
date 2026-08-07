from __future__ import annotations

from copy import deepcopy

import pytest

from tabpollution.config import ConfigError
from tabpollution.data.registry import validate_dataframe_schema
from tabpollution.manifests.schema import ManifestError, validate_run_manifest, write_run_manifest


def _manifest() -> dict:
    return {
        "run_id": "c1-test-20260715",
        "created_at": "2026-07-15T00:00:00+08:00",
        "code_state": {"git": "unavailable"},
        "config_snapshot": "configs/benchmark_v1.yaml",
        "data_checksums": {},
        "dataset": "adult",
        "protocol": "C1-data-prepare",
        "algorithm": "none",
        "generator": "none",
        "seed": 2026,
        "input_artifacts": [],
        "output_artifacts": [],
        "environment": {},
    }


def test_unknown_schema_column_is_rejected(toy_spec) -> None:
    with pytest.raises(ConfigError, match="unknown"):
        validate_dataframe_schema(["number", "category", "target", "leak"], toy_spec)


def test_manifest_missing_required_field_fails() -> None:
    manifest = _manifest()
    manifest.pop("dataset")
    with pytest.raises(ManifestError, match="missing fields"):
        validate_run_manifest(manifest)


def test_manifest_cannot_silently_overwrite(tmp_path) -> None:
    manifest = _manifest()
    first = write_run_manifest(manifest, tmp_path)
    assert first.exists()
    with pytest.raises(FileExistsError):
        write_run_manifest(manifest, tmp_path)

