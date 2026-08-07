"""Run lifecycle, artifact hashing and formal-only aggregation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tabpollution.utils import sha256_file, write_json


TERMINAL_STATUSES = {
    "pilot_passed",
    "formal_passed",
    "quality_blocked",
    "blocked_by_gpu",
    "failed",
}


def set_status(run_dir: str | Path, status: str, detail: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "detail": detail or {},
    }
    write_json(payload, Path(run_dir) / "status.json")


def artifact_manifest(run_dir: str | Path, source_run_id: str) -> dict[str, Any]:
    root = Path(run_dir)
    artifacts = []
    excluded = {"artifacts_manifest.json", "COMPLETE"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        entry: dict[str, Any] = {
            "path": relative,
            "type": path.suffix.lstrip(".") or "file",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_run_id": source_run_id,
        }
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as handle:
                entry["rows"] = max(sum(1 for _ in handle) - 1, 0)
        artifacts.append(entry)
    payload = {"run_id": source_run_id, "artifacts": artifacts}
    write_json(payload, root / "artifacts_manifest.json")
    return payload


def validate_artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    payload = json.loads((root / "artifacts_manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for item in payload["artifacts"]:
        path = root / item["path"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            mismatches.append(item["path"])
    if mismatches:
        raise ValueError(f"Artifact manifest mismatch: {mismatches}")
    return {"passed": True, "artifact_count": len(payload["artifacts"])}


def mark_complete(run_dir: str | Path) -> None:
    path = Path(run_dir) / "COMPLETE"
    path.write_text("complete\n", encoding="utf-8", newline="\n")


def aggregate_formal_runs(runs_dir: str | Path) -> dict[str, Any]:
    included = []
    excluded: dict[str, str] = {}
    for run_dir in sorted(path for path in Path(runs_dir).iterdir() if path.is_dir()):
        manifest_path = run_dir / "run_manifest.json"
        status_path = run_dir / "status.json"
        if not manifest_path.exists() or not status_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))["status"]
        if (
            manifest.get("run_type") == "formal"
            and status == "formal_passed"
            and (run_dir / "COMPLETE").exists()
        ):
            included.append(run_dir.name)
        else:
            excluded[run_dir.name] = f"run_type={manifest.get('run_type')};status={status};complete={(run_dir / 'COMPLETE').exists()}"
    return {"included": included, "excluded": excluded}
