"""Acceptance checks over the saved C2-C3 Adult smoke artifacts.

These tests never retrain a synthesizer.  They validate the one-time expensive
smoke outputs and the frozen C0-C1 files that the second-round work must not
modify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
C2_RUNS = {
    "GaussianCopula": "c2-smoke-adult-gaussiancopula-s42-a2",
    "CTGAN": "c2-smoke-adult-ctgan-s42-a2",
    "TVAE": "c2-smoke-adult-tvae-s42-a2",
}
C3_RUNS = {
    name: f"c3-smoke-adult-{name.lower()}-s42" for name in C2_RUNS
}
POOL_SPEC = {
    "S_detector_train": (8051, 143),
    "S_detector_val": (5367, 244),
    "S_final_test": (8051, 345),
    "S_downstream_mix": (32201, 446),
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_saved_c2_smokes_are_complete_and_isolated():
    for generator, run_id in C2_RUNS.items():
        run_dir = ROOT / "runs" / run_id
        summary = _json(run_dir / "smoke_summary.json")
        assert summary["status"] == "smoke_passed"
        assert summary["run_type"] == "smoke"
        assert summary["split_seed"] == 2026
        assert summary["generator_seed"] == 42
        assert summary["access_audit"]["passed"] is True
        assert summary["access_audit"]["actual_partitions"] == ["R_source_train"]
        assert summary["pool_validation"]["all_synth_ids_unique"] is True
        assert summary["quality"]["format"]["schema_valid"] is True
        assert (run_dir / "model.pkl").exists()
        assert _sha256(run_dir / "model.pkl") == summary["model_sha256"]

        manifests = {entry["pool_name"]: entry for entry in summary["pool_manifests"]}
        assert set(manifests) == set(POOL_SPEC)
        all_ids: set[str] = set()
        for pool_name, (rows, sample_seed) in POOL_SPEC.items():
            manifest = manifests[pool_name]
            path = run_dir / "pools" / manifest["file"]
            assert manifest["rows"] == rows
            assert manifest["sample_seed"] == sample_seed
            assert _sha256(path) == manifest["file_sha256"]
            frame = pd.read_csv(path, usecols=["synth_row_id"])
            ids = set(frame["synth_row_id"].astype(str))
            assert len(ids) == rows
            assert all_ids.isdisjoint(ids)
            all_ids.update(ids)

        reload_result = _json(run_dir / "reload_validation.json")
        assert reload_result["model_reload_ok"] is True
        assert reload_result["schema_order_ok"] is True


def test_saved_c3_smokes_have_exact_bags_and_protocols():
    for generator, run_id in C3_RUNS.items():
        run_dir = ROOT / "runs" / run_id
        summary = _json(run_dir / "smoke_summary.json")
        assert summary["status"] == "smoke_passed"
        assert summary["contamination_artifacts"] == 28
        assert summary["bag_count"] == 35
        assert summary["bag_member_rows"] == 35000
        assert summary["calibration_test_ids_disjoint"] is True
        assert all(item["passed"] for item in summary["protocol_validation"].values())
        assert summary["protocol_validation"]["P5"]["include_in_p3_macro"] is False

        contamination = _json(run_dir / "contamination_manifest.json")
        assert len(contamination) == 28
        assert {entry["rounding"] for entry in contamination} == {
            "half_up=floor(p*N+0.5)"
        }

        manifest = _json(run_dir / "bags_manifest.json")
        bags = manifest["bags"]
        assert len(bags) == 35
        members_path = run_dir / manifest["members_file"]
        assert _sha256(members_path) == manifest["members_file_sha256"]
        members = pd.read_csv(members_path, dtype={"record_id": "string"})
        assert len(members) == 35000
        for bag in bags:
            selected = members.loc[members["bag_id"] == bag["bag_id"]]
            assert len(selected) == 1000
            assert selected["record_id"].nunique() == 1000
            assert int((selected["source_type"] == "synthetic").sum()) == bag["synthetic_count"]
            assert bag["actual_proportion"] == bag["synthetic_count"] / 1000

        calibration = set(
            members.loc[members["bag_id"].str.contains(":calibration:"), "record_id"]
        )
        test = set(members.loc[members["bag_id"].str.contains(":test:"), "record_id"])
        assert calibration.isdisjoint(test)
        rebuilt = _json(run_dir / "bag_rebuild_example.json")
        assert rebuilt["members_sha256"] == rebuilt["expected_members_sha256"]


def test_c0_c1_and_legacy_hashes_are_unchanged():
    expected = {
        "data/processed/adult/adult_clean.csv": "9cf6e79f08b62089624828b3b3c60c64950fea93948f5a5b2829d9194724f2d2",
        "data/processed/credit/credit_clean.csv": "ee2fa731865931ec6c4c54d7c16e00f1051b9163a0bfeb7aa318e795302a4d29",
        "run_minimal_loop.py": "495d59d81c40202a762dec2d001cdc6397e19e2dacdeab4cfc43d006a0974b6f",
        "run_adult_baseline.py": "f3d08bae0393f48ae8f9a460b763b34b939234ba6ab6a5650b027ef381196349",
        "outputs/adult_baseline/baseline_summary.json": "bf7b05721ef78424fe05ec4d279c9830162a32f9cf1900c9a8d94aa1b76422ef",
        "outputs/adult_baseline_reuse/baseline_summary.json": "abf98860c93608db72e8b9cdd1fa8cd072ecb20ca5a67b76b0e0af93d26b594f",
        "outputs/week0/detector_report.txt": "09aea371b4af89836c531ecd10d4388ecb68f2ce55ba2081433e511915e56bd8",
        "outputs/week0_smoke/detector_report.txt": "3224c765373ace07f19c4e8857cf3c25237b1525a6fc90c61729d2799f8ded21",
    }
    for relative, digest in expected.items():
        assert _sha256(ROOT / relative) == digest


def test_c2_c3_success_summaries_are_unchanged():
    expected = {
        "runs/c2-smoke-adult-gaussiancopula-s42-a2/smoke_summary.json": "0e266f4d1dedee94cdbcb851941d8bee3eec08dc7a24ea78b783d7485c728eb1",
        "runs/c2-smoke-adult-ctgan-s42-a2/smoke_summary.json": "444e244318850e1ca9b80dee85a3f396ed6755fbf6c2787fa97a02d95747a855",
        "runs/c2-smoke-adult-tvae-s42-a2/smoke_summary.json": "db11046593c58efe9c6a6f7d8a6d5345a003c3a051f8b3b8578dc38d310823f3",
        "runs/c3-smoke-adult-gaussiancopula-s42/smoke_summary.json": "caae3aa2780bc439c9c2c5cd60ba7cd4cda02dfbf971bd1f7f27a7009460c151",
        "runs/c3-smoke-adult-ctgan-s42/smoke_summary.json": "926f23122173bb33025325a0b7cfae3155b8abc12f36d6232ce46eb4df7da680",
        "runs/c3-smoke-adult-tvae-s42/smoke_summary.json": "15f76f6932a18ad8884aa97f1da4f2bf1bf04d254552a0a81c264737f2c822b5",
    }
    for relative, digest in expected.items():
        assert _sha256(ROOT / relative) == digest


def test_gaussian_seed_2026_pilot_saved_artifacts_pass():
    candidates = sorted((ROOT / "runs").glob("c2-pilot-adult-gaussiancopula-s2026-*"))
    successful = [path for path in candidates if (path / "COMPLETE").exists()]
    assert successful
    run = successful[-1]
    summary = _json(run / "pilot_summary.json")
    status = _json(run / "status.json")
    assert summary["run_type"] == "pilot"
    assert summary["status"] == "pilot_passed"
    assert summary["fit_rows"] == 29273
    assert summary["quality_gate"]["passed"] is True
    assert summary["quality_gate"]["hard_failures"] == []
    assert summary["pool_sizes"] == {
        "S_detector_train": 8051,
        "S_detector_val": 5367,
        "S_downstream_mix": 32201,
        "S_final_test": 8051,
    }
    assert status["status"] == "pilot_passed"
    assert (run / "stdout.log").exists() and (run / "stderr.log").exists()


def test_gpu_blocked_pilots_did_not_fit_or_complete():
    for generator in ("ctgan", "tvae"):
        candidates = sorted((ROOT / "runs").glob(f"c2-pilot-adult-{generator}-s2026-*"))
        assert candidates
        run = candidates[-1]
        status = _json(run / "status.json")
        timing = _json(run / "timing.json")
        assert status["status"] == "blocked_by_gpu"
        assert timing["fit_called"] is False
        assert not (run / "COMPLETE").exists()


def test_manifest_first_fixture_matches_existing_smoke_members():
    recipe = _json(ROOT / "reports" / "manifest_first_contamination_fixture.json")
    assert recipe["materialized_by_default"] is False
    assert not (ROOT / "reports" / "manifest_first_contamination_fixture.csv").exists()
    old = pd.read_csv(
        ROOT
        / "runs"
        / "c3-smoke-adult-gaussiancopula-s42"
        / "contamination"
        / "synthetic_replace_p025.csv",
        dtype={"source_row_id": "string", "source_synth_row_id": "string"},
        low_memory=False,
    )
    compact = pd.DataFrame(
        {
            "position": range(len(old)),
            "source_type": old["source_type"],
            "record_id": old["source_synth_row_id"].where(
                old["source_type"] == "synthetic", old["source_row_id"]
            ),
        }
    )
    text = compact.to_csv(index=False, lineterminator="\n")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == recipe["members_sha256"]


def test_manifest_first_bag_fixture_reuses_exact_smoke_members():
    manifest = _json(ROOT / "reports" / "manifest_first_bag_fixture.json")
    assert manifest["materialized_features"] is False
    assert manifest["members_sha256"] == "3fdf3497c3bd31cff5f07241f2e6e85ea76666333b7bcdd59245e5da1c5d477c"
