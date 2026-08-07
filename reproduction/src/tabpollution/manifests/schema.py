"""Strict run manifest validation with no-silent-overwrite persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REQUIRED_RUN_FIELDS = {
    "run_id",
    "created_at",
    "code_state",
    "config_snapshot",
    "data_checksums",
    "dataset",
    "protocol",
    "algorithm",
    "generator",
    "seed",
    "input_artifacts",
    "output_artifacts",
    "environment",
}


class ManifestError(ValueError):
    pass


def validate_run_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_RUN_FIELDS - set(manifest))
    if missing:
        raise ManifestError(f"Run manifest is missing fields: {missing}")
    unknown = sorted(set(manifest) - REQUIRED_RUN_FIELDS)
    if unknown:
        raise ManifestError(f"Run manifest contains unknown fields: {unknown}")
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", run_id):
        raise ManifestError("run_id has an invalid format")
    seed = manifest["seed"]
    valid_seed = isinstance(seed, int) or (
        isinstance(seed, list)
        and bool(seed)
        and all(isinstance(item, int) for item in seed)
        and len(seed) == len(set(seed))
    )
    if not valid_seed:
        raise ManifestError("seed must be an integer or a non-empty unique integer list")
    for key in ("data_checksums", "input_artifacts", "output_artifacts", "environment"):
        if not isinstance(manifest[key], (dict, list)):
            raise ManifestError(f"{key} must be a mapping or list")
    return manifest


def write_run_manifest(manifest: dict[str, Any], runs_dir: str | Path) -> Path:
    validate_run_manifest(manifest)
    run_dir = Path(runs_dir) / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "run_manifest.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path
