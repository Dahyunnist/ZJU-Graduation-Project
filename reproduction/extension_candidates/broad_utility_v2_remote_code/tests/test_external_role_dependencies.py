from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import pytest
from tabpollution.data.candidate_roles import CandidateRoleError, require_external_role_dependencies, verify_release_code_manifest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_release(tmp_path: Path, *, required=("bank_marketing",), blocked=()):
    package = tmp_path / "release"
    package.mkdir()
    public = {
        "manifest_id": "broad_utility_v2_external_role_dependency_declaration",
        "schema_version": 1,
        "execution_enabled": False,
        "policy": "server_external_only_fail_closed",
        "private_manifest_env": "TEST_PRIVATE_ROLE_MANIFEST",
        "required_datasets": list(required),
        "blocked_datasets": list(blocked),
    }
    public_path = package / "external_role_dependencies.json"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    payload = package / "code.py"
    payload.write_text("pass\n", encoding="utf-8")
    files = {p.name: digest(p) for p in (public_path, payload)}
    manifest = {"manifest_type": "code_only_external_roles", "files": files}
    (package / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "code_package_manifest_file": "package_manifest.json",
        "external_role_dependencies_file": "external_role_dependencies.json",
        "external_role_dependencies_sha256": digest(public_path),
        "execution": {"enabled": False},
    }
    return package, config


def private_manifest(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "manifest_id": "broad_utility_v2_private_role_dependency_manifest",
        "schema_version": 1,
        "entries": entries,
    }), encoding="utf-8")
    return path


def test_code_release_manifest_verifies_exact_inventory(tmp_path):
    package, _ = make_release(tmp_path)
    assert verify_release_code_manifest(package)["file_count"] == 2
    (package / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(CandidateRoleError, match="inventory mismatch"):
        verify_release_code_manifest(package)


def test_missing_code_file_fails_integrity_check(tmp_path):
    package, _ = make_release(tmp_path)
    (package / "code.py").unlink()
    with pytest.raises(CandidateRoleError, match="inventory mismatch"):
        verify_release_code_manifest(package)


def test_missing_private_manifest_fails_closed(tmp_path):
    package, config = make_release(tmp_path)
    with pytest.raises(CandidateRoleError, match="server-private role manifest is missing or unsafe"):
        require_external_role_dependencies(package, config, private_manifest_path=tmp_path / "missing.json")


def test_missing_private_role_file_fails_closed(tmp_path):
    package, config = make_release(tmp_path)
    root = tmp_path / "private"
    manifest = private_manifest(root / "manifest.json", [{
        "dataset_id": "bank_marketing", "status": "available",
        "artifact_path": "artifacts/bank_marketing.csv", "sha256": "0" * 64,
        "rows": 1, "role_counts": {"train": 1, "tune": 1, "evaluation": 1, "final_test": 1},
    }])
    with pytest.raises(CandidateRoleError, match="private role artifact missing"):
        require_external_role_dependencies(package, config, private_manifest_path=manifest)


def test_blocked_dependency_cannot_carry_locator_or_digest(tmp_path):
    package, config = make_release(tmp_path, required=(), blocked=("letter_recognition",))
    manifest = private_manifest(tmp_path / "private" / "manifest.json", [{
        "dataset_id": "letter_recognition", "status": "blocked", "sha256": "0" * 64,
    }])
    with pytest.raises(CandidateRoleError, match="must not expose"):
        require_external_role_dependencies(package, config, private_manifest_path=manifest)


def test_available_private_file_is_verified_before_no_run_refusal(tmp_path):
    package, config = make_release(tmp_path)
    root = tmp_path / "private"
    artifact = root / "artifacts" / "bank_marketing.csv"
    artifact.parent.mkdir(parents=True)
    with artifact.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["record_id", "role"])
        writer.writerows([
            ["public:bank_marketing:0", "train"],
            ["public:bank_marketing:1", "tune"],
            ["public:bank_marketing:2", "evaluation"],
            ["public:bank_marketing:3", "final_test"],
        ])
    manifest = private_manifest(root / "manifest.json", [{
        "dataset_id": "bank_marketing", "status": "available",
        "artifact_path": "artifacts/bank_marketing.csv", "sha256": digest(artifact),
        "rows": 4, "role_counts": {"train": 1, "tune": 1, "evaluation": 1, "final_test": 1},
    }])
    with pytest.raises(CandidateRoleError, match="execution is not authorized"):
        require_external_role_dependencies(package, config, private_manifest_path=manifest)
