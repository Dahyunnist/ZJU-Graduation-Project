"""Opt-in, allowlisted local table intake; never downloads or reads real data by default."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


class IntakeError(ValueError):
    """Raised when an intake request is not explicitly allowlisted."""


def load_catalog(path: str | Path) -> dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("default_run_mode") != "NO-RUN":
        raise IntakeError("catalog must declare default_run_mode=NO-RUN")
    records = catalog.get("records")
    if not isinstance(records, list) or len(records) != 15:
        raise IntakeError("v2 catalog must contain exactly 15 records")
    ids = [row.get("dataset_id") for row in records]
    if len(set(ids)) != len(ids):
        raise IntakeError("dataset IDs must be unique")
    return catalog


def status_report(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return metadata-only readiness; does not open source files."""
    return {
        "catalog_id": catalog.get("catalog_id"),
        "run_mode": "NO-RUN",
        "source_data_read": False,
        "summary": {
            "prepared": sum(r.get("status") == "prepared" for r in catalog["records"]),
            "blocked": sum(r.get("status") == "blocked" for r in catalog["records"]),
        },
        "records": [
            {"dataset_id": r["dataset_id"], "status": r["status"],
             "blocked_reason": r.get("blocked_reason", "")}
            for r in catalog["records"]
        ],
    }


def _record(catalog: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [r for r in catalog["records"] if r["dataset_id"] == dataset_id]
    if len(matches) != 1:
        raise IntakeError(f"dataset is not allowlisted: {dataset_id}")
    return matches[0]


def _safe_member_name(name: str) -> str:
    if not name or "\\" in name or name.startswith("/") or name.startswith("!"):
        raise IntakeError(f"unsafe archive member path: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts) or re.match(r"^[A-Za-z]:", name):
        raise IntakeError(f"unsafe archive member path: {name!r}")
    return name


def _validate_zip_members(zf: zipfile.ZipFile, max_member_bytes: int) -> dict[str, zipfile.ZipInfo]:
    names: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    infos = zf.infolist()
    if len(infos) > 20_000:
        raise IntakeError("archive entry-count limit exceeded")
    for info in infos:
        raw_name = info.filename.rstrip("/")
        if not raw_name:
            continue
        safe = _safe_member_name(raw_name)
        key = safe.casefold()
        if key in names:
            raise IntakeError(f"duplicate archive member path: {safe}")
        names[key] = info
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise IntakeError(f"symlink archive member is forbidden: {info.filename}")
        if info.flag_bits & 1:
            raise IntakeError(f"encrypted archive member is forbidden: {info.filename}")
        if info.file_size > max_member_bytes:
            raise IntakeError(f"archive member exceeds size limit: {info.filename}")
        total_size += info.file_size
        if total_size > max_member_bytes * 4:
            raise IntakeError("archive cumulative decompressed-size limit exceeded")
    return names


def _read_archive_member(
    archive: Path, member: str, max_member_bytes: int, *, approved_nested_sha256: str | None = None,
) -> tuple[bytes, dict[str, str] | None]:
    member_parts = member.split("!")
    if len(member_parts) == 1:
        outer_member = None
        allowed = _safe_member_name(member_parts[0])
        inner_allowed = None
    elif len(member_parts) == 2:
        outer_member = _safe_member_name(member_parts[0])
        inner_allowed = _safe_member_name(member_parts[1])
        allowed = member
        if not approved_nested_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", approved_nested_sha256):
            raise IntakeError("an explicitly approved nested-archive SHA-256 is required")
    else:
        raise IntakeError("only one nested ZIP layer is supported")
    try:
        with zipfile.ZipFile(archive) as zf:
            names = _validate_zip_members(zf, max_member_bytes)
            if outer_member is None:
                info = names.get(allowed.casefold())
                if info is None:
                    raise IntakeError(f"member is not present in archive allowlist: {allowed}")
                with zf.open(info, "r") as stream:
                    payload = stream.read(max_member_bytes + 1)
                if len(payload) > max_member_bytes:
                    raise IntakeError("decompressed archive member exceeds size limit")
                return payload, None
            nested_info = names.get(outer_member.casefold())
            if nested_info is None or nested_info.is_dir():
                raise IntakeError(f"nested ZIP member is not present in archive: {outer_member}")
            if not outer_member.lower().endswith(".zip"):
                raise IntakeError("nested member container must have .zip extension")
            with zf.open(nested_info, "r") as stream:
                nested_payload = stream.read(max_member_bytes + 1)
            if len(nested_payload) > max_member_bytes:
                raise IntakeError("nested ZIP member exceeds size limit")
            nested_digest = hashlib.sha256(nested_payload).hexdigest()
            if nested_digest.lower() != approved_nested_sha256.lower():
                raise IntakeError("nested ZIP SHA-256 does not match approved digest")
            with zipfile.ZipFile(io.BytesIO(nested_payload)) as nested_zip:
                nested_names = _validate_zip_members(nested_zip, max_member_bytes)
                target_info = nested_names.get(inner_allowed.casefold())
                if target_info is None:
                    raise IntakeError(f"member is not present in nested archive allowlist: {inner_allowed}")
                with nested_zip.open(target_info, "r") as stream:
                    payload = stream.read(max_member_bytes + 1)
            if len(payload) > max_member_bytes:
                raise IntakeError("decompressed nested member exceeds size limit")
            return payload, {"nested_member": outer_member, "nested_archive_sha256": nested_digest}
    except zipfile.BadZipFile as exc:
        raise IntakeError("source file is not a valid ZIP archive") from exc


def _parse_payload(record: dict[str, Any], payload: bytes) -> pd.DataFrame:
    fmt = record.get("source_format")
    if fmt == "csv":
        text = payload.decode(record.get("encoding", "utf-8-sig"), errors="strict")
        header = 0 if record.get("header") is True else None
        names = None if header == 0 else record["source_column_order"]
        frame = pd.read_csv(
            io.StringIO(text), sep=record.get("delimiter") or ",", header=header,
            names=names, dtype=object, keep_default_na=False, on_bad_lines="error",
            comment=record.get("comment_char"),
        )
    elif fmt == "excel":
        try:
            frame = pd.read_excel(
                io.BytesIO(payload), sheet_name=record.get("sheet_name") or 0,
                header=record.get("header", 0), dtype=object,
            )
        except ImportError as exc:
            raise IntakeError("required Excel engine is unavailable") from exc
        frame.columns = [str(c).strip() for c in frame.columns]
    elif fmt == "arff":
        from scipy.io import arff
        try:
            data, _ = arff.loadarff(io.StringIO(payload.decode("utf-8", errors="strict")))
        except (ValueError, OSError, UnicodeDecodeError, TypeError) as exc:
            raise IntakeError(f"invalid ARFF source member: {exc}") from exc
        frame = pd.DataFrame(data)
        for col in frame.select_dtypes(include=["object"]).columns:
            frame[col] = frame[col].map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    else:
        raise IntakeError(f"unsupported source format: {fmt!r}")

    expected = record.get("source_column_order")
    actual = [str(c).strip() for c in frame.columns]
    if actual != expected:
        raise IntakeError(f"schema/order mismatch for {record['dataset_id']}: expected={expected!r}; actual={actual!r}")
    source_renames = record.get("source_column_renames", record.get("rename_columns", {})) or {}
    if not isinstance(source_renames, dict) or not set(source_renames) <= set(expected):
        raise IntakeError("catalog source-column aliases are malformed")
    if len(set(source_renames.values())) != len(source_renames.values()):
        raise IntakeError("catalog source-column aliases are not unique")
    frame = frame.rename(columns=source_renames)
    permitted = set(frame.columns) - set(record.get("exclude_columns", []))
    features = record.get("features")
    target = record.get("target")
    if not isinstance(features, list) or not target or target not in frame.columns:
        raise IntakeError("catalog feature/target mapping is incomplete")
    if set(features) & set(record.get("exclude_columns", [])):
        raise IntakeError("excluded columns may not enter the feature list")
    if set(features) | {target} != permitted:
        raise IntakeError("feature/target mapping does not cover permitted source columns")
    return frame.loc[:, [*features, target]].copy()


def read_allowlisted_table(
    catalog: dict[str, Any], dataset_id: str, archive_path: str | Path, *,
    input_root: str | Path, approved_sha256: str | None = None,
    allow_source_data_read: bool = False, approved_nested_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one exact source member only after explicit path/hash/read approval.

    This function never downloads. The default call is a hard NO-RUN refusal.
    """
    if not allow_source_data_read:
        raise IntakeError("source-data read is disabled; pass explicit allow_source_data_read=True")
    record = _record(catalog, dataset_id)
    if record.get("status") != "prepared":
        raise IntakeError(record.get("blocked_reason") or f"dataset status is {record.get('status')!r}")
    if not isinstance(record.get("member"), str):
        raise IntakeError("catalog member is not an explicit ZIP path")
    members = [part.strip() for part in re.split(r"\s+\+\s+", record["member"])]
    if any(not member for member in members) or len(members) > 2:
        raise IntakeError("source member list is malformed or exceeds the reviewed two-member adapter")
    path = Path(archive_path)
    if path.is_symlink() or not path.is_file():
        raise IntakeError("source path must be an existing regular non-symlink file")
    root = Path(input_root).resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise IntakeError("source path is outside the approved input root") from exc
    max_archive = int(record.get("archive_max_bytes", catalog["safety"]["max_archive_bytes"]))
    if resolved.stat().st_size > max_archive:
        raise IntakeError("source archive exceeds catalog size limit")
    if not approved_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", approved_sha256):
        raise IntakeError("an explicitly approved SHA-256 is required")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest.lower() != approved_sha256.lower():
        raise IntakeError("source archive SHA-256 does not match approved digest")
    member_frames, member_counts = [], {}
    nested_receipts = {}
    total_member_bytes = 0
    max_total_member_bytes = int(catalog["safety"].get(
        "max_total_member_bytes", int(catalog["safety"]["max_member_bytes"]) * 4))
    for member in members:
        member_record = dict(record)
        # Adult's official test member starts with pipe-prefixed comment lines.
        if dataset_id == "adult" and Path(member).name == "adult.test":
            member_record["comment_char"] = "|"
        payload, nested_receipt = _read_archive_member(
            resolved, member, int(catalog["safety"]["max_member_bytes"]),
            approved_nested_sha256=approved_nested_sha256,
        )
        total_member_bytes += len(payload)
        if total_member_bytes > max_total_member_bytes:
            raise IntakeError("source member cumulative size limit exceeded")
        if nested_receipt:
            nested_receipts[member] = nested_receipt
        parsed = _parse_payload(member_record, payload)
        if dataset_id == "adult" and Path(member).name == "adult.test":
            target = record["target"]
            parsed[target] = parsed[target].astype(str).str.strip().str.removesuffix(".")
        member_counts[member] = len(parsed)
        if len(members) > 1:
            parsed.insert(0, "source_partition", member)
        member_frames.append(parsed)
    frame = pd.concat(member_frames, ignore_index=True) if len(member_frames) > 1 else member_frames[0]
    return frame, {
        "dataset_id": dataset_id, "status": "prepared", "source_data_read": True,
        "archive_sha256": digest, "member": record["member"], "members": members,
        "member_row_counts": member_counts,
        "member_payload_bytes": total_member_bytes,
        "nested_members": nested_receipts,
        "row_count": int(len(frame)), "columns": list(frame.columns),
        "split_created": False, "model_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata-only NO-RUN intake catalog validator")
    parser.add_argument("catalog", type=Path)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    print(json.dumps(status_report(catalog), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
