"""Single-writer accepted-unit registry bound to inputs, roles, method and runtime."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from tabpollution.studies.broad_utility import PROTOCOL_ID

FAILURE_CATEGORIES = {
    "source_integrity_or_parse", "role_or_schema_gate", "generator_fit_or_sample",
    "downstream_fit_or_metric", "aggregate_audit", "runtime_or_resource",
}
IDENTITY_HASHES = {
    "source_sha256", "roles_assignment_sha256", "train_sample_ids_sha256",
    "evaluation_sample_ids_sha256", "generator_config_sha256", "learner_config_sha256",
    "selection_definition_sha256", "package_sha256", "runtime_sha256",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_identity(identity):
    required = {"protocol", "dataset_id", "task", "seed", *IDENTITY_HASHES}
    if not isinstance(identity, dict) or set(identity) != required:
        raise ValueError("unit identity schema differs")
    if identity["protocol"] != PROTOCOL_ID or not identity["dataset_id"] or identity["task"] not in {"binary", "multiclass", "regression"}:
        raise ValueError("unit identity protocol/task mismatch")
    if not isinstance(identity["seed"], int):
        raise ValueError("unit seed must be an integer")
    for key in IDENTITY_HASHES:
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"unit identity {key} must be lowercase SHA-256")
    return identity


def identity_sha256(identity):
    validate_identity(identity)
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def unit_key(identity):
    return identity_sha256(identity)


def build_unit_identity(*, dataset_id, task, seed, source_sha256, role_assignments,
                        train_ids, evaluation_ids, generator_config, learner_config,
                        package_sha256, runtime_sha256, selection_definition=None,
                        roles_assignment_sha256=None):
    """Fingerprint the exact source, fixed outer split, capped analysis IDs and method/runtime."""
    roles = role_assignments.copy()
    required = {"record_id", "role"}
    if not required <= set(roles.columns) or roles["record_id"].isna().any() or roles["record_id"].astype(str).duplicated().any():
        raise ValueError("role assignments require unique record_id and role")
    if roles["role"].isna().any() or not {"train", "tune", "evaluation", "final_test"} <= set(roles["role"].astype(str)):
        raise ValueError("all fixed outer roles must be present")
    train_ids = [str(value) for value in train_ids]
    evaluation_ids = [str(value) for value in evaluation_ids]
    if not train_ids or not evaluation_ids or set(train_ids) & set(evaluation_ids):
        raise ValueError("capped train/evaluation identities must be nonempty and disjoint")
    role_by_id = dict(zip(roles["record_id"].astype(str), roles["role"].astype(str)))
    if any(role_by_id.get(value) != "train" for value in train_ids):
        raise ValueError("analysis training IDs must come only from the fixed train role")
    if any(role_by_id.get(value) != "evaluation" for value in evaluation_ids):
        raise ValueError("analysis evaluation IDs must come only from the fixed evaluation role")
    assignment_columns = [column for column in ("record_id", "role", "source_partition", "source_order")
                          if column in roles.columns]
    assignments = roles[assignment_columns].astype(str).sort_values("record_id").to_dict("records")
    def json_sha(value):
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    if selection_definition is None:
        selection_definition = {"train": "all fixed train-role IDs", "evaluation": "all fixed evaluation-role IDs"}
    role_sha = roles_assignment_sha256 or json_sha(assignments)
    identity = {"protocol": PROTOCOL_ID, "dataset_id": dataset_id, "task": task, "seed": int(seed),
        "source_sha256": source_sha256,
        "roles_assignment_sha256": role_sha,
        "train_sample_ids_sha256": json_sha(sorted(train_ids)),
        "evaluation_sample_ids_sha256": json_sha(sorted(evaluation_ids)),
        "generator_config_sha256": json_sha(generator_config),
        "learner_config_sha256": json_sha(learner_config),
        "selection_definition_sha256": json_sha(selection_definition),
        "package_sha256": package_sha256, "runtime_sha256": runtime_sha256}
    validate_identity(identity)
    return identity


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_accepted_artifacts(row):
    aggregate = Path(row["aggregate_path"])
    audit = Path(row["audit_receipt_path"])
    if not aggregate.is_file() or not audit.is_file():
        raise ValueError("accepted aggregate or independent audit receipt is missing")
    if _sha(aggregate) != row["aggregate_file_sha256"] or _sha(audit) != row["audit_receipt_sha256"]:
        raise ValueError("accepted aggregate or audit receipt hash changed")
    report = json.loads(aggregate.read_text(encoding="utf-8"))
    receipt = json.loads(audit.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("row_outputs") is not False:
        raise ValueError("accepted aggregate is incomplete or contains row outputs")
    if (receipt.get("status") != "pass" or receipt.get("independent") is not True
            or receipt.get("auditor_id") in {None, "", report.get("producer_id")}
            or receipt.get("aggregate_sha256") != row["aggregate_file_sha256"]
            or receipt.get("identity_sha256") != row["identity_sha256"]):
        raise ValueError("accepted result lacks a bound independent passing audit")


def read_registry(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [row.get("unit_key") for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("registry contains duplicate consumed identities")
    for row in rows:
        identity = validate_identity(row.get("identity"))
        if (row.get("protocol") != PROTOCOL_ID or row.get("identity_sha256") != identity_sha256(identity)
                or row.get("unit_key") != unit_key(identity)):
            raise ValueError("registry identity binding mismatch")
        if row.get("status") == "accepted":
            _verify_accepted_artifacts(row)
        elif row.get("status") == "failed_consumed":
            if row.get("failure_category") not in FAILURE_CATEGORIES:
                raise ValueError("consumed failure category is invalid")
        else:
            raise ValueError("unknown registry status")
    return rows


def register_unit(path, *, identity, status, aggregate_path=None, audit_receipt_path=None,
                  failure_category=None):
    """Consume one exact identity. The registry is single-writer; accepted evidence is file-bound."""
    identity = validate_identity(identity)
    digest = identity_sha256(identity)
    key = unit_key(identity)
    path = Path(path)
    if path.is_file():
        prior = next((item for item in read_registry(path) if item["unit_key"] == key), None)
        if prior and prior["status"] == "failed_consumed":
            raise ValueError("unit identity already consumed; no overwrite or retry")
    row = {"protocol": PROTOCOL_ID, "unit_key": key, "identity_sha256": digest,
           "identity": identity, "status": status}
    if status == "accepted":
        if not aggregate_path or not audit_receipt_path:
            raise ValueError("accepted unit requires aggregate and independent audit receipt files")
        row.update(aggregate_path=str(Path(aggregate_path).resolve()),
                   audit_receipt_path=str(Path(audit_receipt_path).resolve()))
        if not Path(row["aggregate_path"]).is_file() or not Path(row["audit_receipt_path"]).is_file():
            raise ValueError("accepted aggregate and audit files must already exist")
        row["aggregate_file_sha256"] = _sha(row["aggregate_path"])
        row["audit_receipt_sha256"] = _sha(row["audit_receipt_path"])
        _verify_accepted_artifacts(row)
    elif status == "failed_consumed":
        if failure_category not in FAILURE_CATEGORIES:
            raise ValueError("consumed failure needs a declared failure category")
        row["failure_category"] = failure_category
    else:
        raise ValueError("status must be accepted or failed_consumed")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("registry is single-writer; another writer or stale lock exists") from exc
    temporary = None
    try:
        existing = read_registry(path)
        prior = next((item for item in existing if item["unit_key"] == key), None)
        if prior:
            if status == "accepted" and prior == row:
                return prior
            raise ValueError("unit identity already consumed; no overwrite or retry")
        content = "".join(_canonical(item) + "\n" for item in [*existing, row])
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        os.close(lock_fd)
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        if lock_path.exists():
            lock_path.unlink()
    return row


def ledger_for_plan(plan, registry_rows, expected_identities):
    """Skip only units whose full identity exactly matches this plan's expected binding."""
    ledger = {}
    for record in registry_rows:
        identity = validate_identity(record.get("identity"))
        if (record.get("unit_key") != unit_key(identity)
                or record.get("identity_sha256") != identity_sha256(identity)):
            raise ValueError("registry row identity binding mismatch")
        if record.get("status") == "accepted":
            _verify_accepted_artifacts(record)
    for job in plan["jobs"]:
        label = f"{job['dataset']}|seed={job['seed']}"
        identity = expected_identities.get(label)
        if identity is None:
            continue
        identity = validate_identity(identity)
        if identity["dataset_id"] != job["dataset"] or identity["seed"] != job["seed"]:
            raise ValueError("expected unit identity does not match plan dataset/seed")
        if (identity["generator_config_sha256"] != plan.get("generator_config_sha256")
                or identity["learner_config_sha256"] != plan.get("learner_config_sha256")):
            raise ValueError("expected unit identity configuration differs from current plan")
        key_digest = unit_key(identity)
        record = next((item for item in registry_rows if item["unit_key"] == key_digest), None)
        if not record:
            continue
        key = json.dumps(job, sort_keys=True)
        if record["status"] == "accepted":
            ledger[key] = {"status": "accepted", "result_sha256": record["aggregate_file_sha256"]}
        else:
            ledger[key] = {"status": "failed_consumed", "failure_category": record["failure_category"]}
    return ledger


def ledger_from_registry(plan, registry_path, expected_identities):
    return ledger_for_plan(plan, read_registry(registry_path), expected_identities)
