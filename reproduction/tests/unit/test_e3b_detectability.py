import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tabpollution.studies.e3b_detectability import _score_frame, preflight, run
from tabpollution.utils import sha256_file


def _fixture(tmp_path: Path):
    rng = np.random.default_rng(41)
    pool = tmp_path / "pool"
    (pool / "real").mkdir(parents=True)
    rows, runs = [], []
    for table in ("adult", "abalone"):
        x = rng.normal(size=500)
        real = pd.DataFrame({"x": x, "category": np.where(x > 0, "a", "b"), "target": (x > 0).astype(int)})
        real_path = pool / "real" / f"{table}.csv"
        real.to_csv(real_path, index=False)
        for generator, offset in (("CTGAN", 1.1), ("TVAE", -1.1)):
            synthetic = real.copy()
            synthetic["x"] += offset
            synthetic_path = pool / "synthetic" / table / f"{generator}.csv"
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic.to_csv(synthetic_path, index=False)
            rows.append({
                "table_id": table, "domain": "toy", "target_column": "target",
                "generator": generator, "real_path": f"real/{table}.csv",
                "synthetic_path": f"synthetic/{table}/{generator}.csv",
            })
            runs.append({
                "table_id": table, "generator": generator, "status": "complete",
                "real_sha256": sha256_file(real_path), "synthetic_sha256": sha256_file(synthetic_path),
            })
    registry = pool / "pool_registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    manifest = pool / "replica_manifest.json"
    manifest.write_text(json.dumps({
        "status": "complete", "registry_sha256": sha256_file(registry),
        "plan": {"replicate_id": "e3-independent-pools-r3031"}, "runs": runs,
    }), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "study_id": "toy-e3b", "tables": ["adult", "abalone"],
        "generators": ["CTGAN", "TVAE"], "seeds": [2026, 2027, 2028],
        "primary_detector": "c2st_lr", "capacity_detector": "c2st_xgb",
        "model_direction": "high_score_is_synthetic", "train_split": "detector_train",
        "calibration_split": "detector_val", "evaluation_split": "final_test",
        "target_fpr": .05, "bootstrap_replicates": 20, "bootstrap_confidence": .95,
        "transfer_min_auroc_gain": .10, "near_random_auroc": .60,
        "expected_replicate_id": "e3-independent-pools-r3031",
        "expected_replica_manifest_sha256": sha256_file(manifest),
        "expected_registry_sha256": sha256_file(registry),
    }), encoding="utf-8")
    return config, registry, manifest


def _analyzer():
    path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_e3b_detectability.py"
    spec = importlib.util.spec_from_file_location("analyze_e3b_detectability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e3b_full_matrix_atomic_cases_and_independent_audit(tmp_path):
    config, registry, manifest = _fixture(tmp_path)
    assert preflight(config, registry, manifest)["lr_cases"] == 6
    output = tmp_path / "results"
    result = run(config, registry, manifest, output)
    assert result["completed_lr_cases"] == 6
    assert not list(output.glob("**/*.partial"))
    complete = sorted(output.glob("lr/*/seed*.complete/complete.json"))
    assert len(complete) == 6
    for path in complete:
        folder = path.parent
        scores = pd.read_csv(folder / "scores.csv.gz")
        assert set(zip(scores.train_generator, scores.eval_generator)) == {
            ("CTGAN", "CTGAN"), ("CTGAN", "TVAE"),
            ("TVAE", "CTGAN"), ("TVAE", "TVAE"),
        }
        assert set(scores.model_direction) == {"high_score_is_synthetic"}
        assignments = pd.read_csv(folder / "split_assignments.csv.gz")
        for _, group in assignments.groupby("generator"):
            ids = {split: set(rows.record_id) for split, rows in group.groupby("split")}
            assert not ids["detector_train"] & ids["calibration"]
            assert not ids["detector_train"] & ids["evaluation"]
            assert not ids["calibration"] & ids["evaluation"]
    audit = _analyzer().analyze(output, tmp_path / "audit")
    assert audit["audited"] and audit["complete_lr_cases"] == 6 and audit["matrix_cells"] == 24


def test_target_evaluation_labels_do_not_change_generated_scores():
    class FixedDetector:
        def predict_score(self, frame):
            return frame["x"].to_numpy(float)

    frame = pd.DataFrame({
        "x": [.1, .9], "record_id": ["a", "b"], "generator": ["real", "TVAE"],
        "source_label": [0, 1], "table_id": ["adult", "adult"],
        "source_type": ["real", "synthetic"], "schema_columns": ["x", "x"],
    })
    first = _score_frame(FixedDetector(), frame, table="adult", seed=2026,
                         detector_name="c2st_lr", train_generator="CTGAN", eval_generator="TVAE")
    changed = frame.copy()
    changed["source_label"] = 1 - changed["source_label"]
    second = _score_frame(FixedDetector(), changed, table="adult", seed=2026,
                          detector_name="c2st_lr", train_generator="CTGAN", eval_generator="TVAE")
    np.testing.assert_array_equal(first.raw_score, second.raw_score)
    assert not np.array_equal(first.source_label, second.source_label)


def test_preflight_rejects_manifest_hash_mismatch(tmp_path):
    config, registry, manifest = _fixture(tmp_path)
    values = yaml.safe_load(config.read_text(encoding="utf-8"))
    values["expected_replica_manifest_sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        preflight(config, registry, manifest)


def test_incomplete_case_never_enters_audit(tmp_path):
    config, registry, manifest = _fixture(tmp_path)
    output = tmp_path / "results"
    run(config, registry, manifest, output)
    case = output / "lr" / "adult" / "seed2026.complete"
    case.rename(output / "lr" / "adult" / "seed2026.partial")
    with pytest.raises(AssertionError, match="partial cases"):
        _analyzer().analyze(output, tmp_path / "audit")


def test_audit_recomputes_scores_and_rejects_tampering(tmp_path):
    config, registry, manifest = _fixture(tmp_path)
    output = tmp_path / "results"
    run(config, registry, manifest, output)
    score_path = output / "lr" / "adult" / "seed2026.complete" / "scores.csv.gz"
    with score_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(AssertionError, match="file hash changed"):
        _analyzer().analyze(output, tmp_path / "audit")
