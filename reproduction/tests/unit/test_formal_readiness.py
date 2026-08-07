from __future__ import annotations

import json
import sys
import types
from collections import namedtuple
from pathlib import Path

import pandas as pd
import pytest

from tabpollution.generators.pilot import create_gpu_blocked_run, load_pilot_config
from tabpollution.generators.preflight import ensure_disk_space
from tabpollution.generators.quality_gate import evaluate_quality_gate
from tabpollution.generators.sdv_adapter import load_sdv_synthesizer
from tabpollution.mixing.contamination import build_contamination
from tabpollution.mixing.manifest_first import (
    contamination_members,
    contamination_recipe,
    rebuild_contamination,
    write_recipe,
)
from tabpollution.runs import (
    aggregate_formal_runs,
    artifact_manifest,
    mark_complete,
    set_status,
    validate_artifact_manifest,
)
from tabpollution.utils import write_json


def _pilot_config(path: Path, *, run_type: str = "pilot", seed: int = 2026) -> Path:
    path.write_text(
        f"""run_type: {run_type}
dataset: adult
split_seed: 2026
generator_seed: {seed}
fit_scope: full_R_source_train
minimum_free_disk_bytes: 1
pool_redundancy: 1.1
pool_seed_offsets: {{S_detector_train: 101, S_detector_val: 202, S_final_test: 303, S_downstream_mix: 404}}
generators:
  GaussianCopula: {{enforce_min_max_values: true, enforce_rounding: true}}
  CTGAN: {{epochs: 300, enable_gpu: true}}
  TVAE: {{epochs: 300, enable_gpu: true}}
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("run_type,seed", [("smoke", 2026), ("formal", 2026), ("pilot", 42)])
def test_pilot_config_rejects_wrong_type_or_seed(tmp_path, run_type, seed):
    with pytest.raises(ValueError):
        load_pilot_config(_pilot_config(tmp_path / "pilot.yaml", run_type=run_type, seed=seed))


def test_pilot_config_rejects_reduced_epochs(tmp_path):
    path = _pilot_config(tmp_path / "pilot.yaml")
    path.write_text(path.read_text(encoding="utf-8").replace("epochs: 300", "epochs: 20", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="300 epochs"):
        load_pilot_config(path)


def _gate(targets, tstr="ok", required=True):
    return evaluate_quality_gate(
        access_audit={"passed": True, "actual_partitions": ["R_source_train"]},
        pool_validation={"all_synth_ids_unique": True},
        quality={
            "format": {"schema_valid": True, "numeric_parse_failures": 0, "target_values": targets},
            "TSTR": {"status": tstr, "auroc": 0.7},
            "TRTR": {"status": "ok", "auroc": 0.8},
            "sdmetrics": {"overall": 0.8},
            "overlap": {},
        },
        real_target_values={"a", "b"},
        reload_validation={"model_reload_ok": True, "schema_order_ok": True},
        required_artifacts_present=required,
    )


def test_quality_gate_blocks_single_class_and_tstr_failure():
    result = _gate(["a"], tstr="failed_single_class_train")
    assert result["passed"] is False
    assert "target_class_collapse" in result["hard_failures"]
    assert "tstr_unavailable" in result["hard_failures"]


def test_quality_gate_accepts_structurally_valid_two_classes():
    assert _gate(["a", "b"])["passed"] is True


def test_quality_gate_blocks_illegal_target_and_missing_artifact():
    result = _gate(["a", "b", "c"], required=False)
    assert "illegal_target_value" in result["hard_failures"]
    assert "required_artifacts" in result["hard_failures"]


def test_quality_gate_blocks_ineffective_cross_run_sampling_seed():
    result = evaluate_quality_gate(
        access_audit={"passed": True, "actual_partitions": ["R_source_train"]},
        pool_validation={"all_synth_ids_unique": True},
        quality={
            "format": {"schema_valid": True, "numeric_parse_failures": 0, "target_values": ["a", "b"]},
            "TSTR": {"status": "ok", "auroc": 0.7},
            "TRTR": {"status": "ok", "auroc": 0.8},
        },
        real_target_values={"a", "b"},
        reload_validation={"model_reload_ok": True, "schema_order_ok": True},
        required_artifacts_present=True,
        cross_run_content_distinct=False,
    )
    assert "cross_run_sampling_seed_ineffective" in result["hard_failures"]


def _frames():
    real = pd.DataFrame({"row_id": [f"r{i}" for i in range(5)], "x": range(5)})
    synth = pd.DataFrame({"synth_row_id": [f"s{i}" for i in range(8)], "x": range(10, 18)})
    return real, synth


@pytest.mark.parametrize("condition", ["real_only", "real_append", "synthetic_append", "synthetic_replace"])
def test_manifest_first_members_match_existing_constructor(condition):
    real, synth = _frames()
    members = contamination_members(real, synth, condition, 0.4, 99, real)
    existing = build_contamination(real, synth, condition, 0.4, 99, real)
    ids = existing["source_synth_row_id"].where(
        existing["source_type"] == "synthetic", existing["source_row_id"]
    ).astype(str).tolist()
    assert members["record_id"].tolist() == ids
    assert members["source_type"].tolist() == existing["source_type"].tolist()


def test_manifest_first_default_does_not_materialize_features(tmp_path):
    real, synth = _frames()
    recipe, members = contamination_recipe(
        real, synth, "synthetic_replace", 0.4, 99, dataset="toy", generator="fake"
    )
    result = write_recipe(recipe, members, tmp_path / "recipe.json")
    assert result["manifest_only"] is True
    assert not (tmp_path / "recipe.csv").exists()


def test_manifest_first_explicit_materialization_rebuilds(tmp_path):
    real, synth = _frames()
    recipe, members = contamination_recipe(
        real, synth, "synthetic_append", 0.4, 99, dataset="toy", generator="fake"
    )
    result = write_recipe(
        recipe,
        members,
        tmp_path / "recipe.json",
        materialize=True,
        real_pool=real,
        synthetic_pool=synth,
    )
    rebuilt = rebuild_contamination(members, real, synth)
    assert Path(result["materialized_file"]).exists()
    assert len(rebuilt) == 7


def test_artifact_manifest_and_complete_marker(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "stdout.log").write_text("ok\n", encoding="utf-8")
    (run / "stderr.log").write_text("", encoding="utf-8")
    set_status(run, "pilot_passed")
    artifact_manifest(run, "pilot-1")
    assert validate_artifact_manifest(run)["passed"] is True
    mark_complete(run)
    assert (run / "COMPLETE").exists()


def test_formal_aggregation_excludes_pilot_failed_and_incomplete(tmp_path):
    cases = [
        ("formal-good", "formal", "formal_passed", True),
        ("pilot", "pilot", "pilot_passed", True),
        ("failed", "formal", "failed", False),
        ("incomplete", "formal", "formal_passed", False),
    ]
    for run_id, run_type, status, complete in cases:
        run = tmp_path / run_id
        run.mkdir()
        write_json({"run_id": run_id, "run_type": run_type}, run / "run_manifest.json")
        set_status(run, status)
        if complete:
            mark_complete(run)
    result = aggregate_formal_runs(tmp_path)
    assert result["included"] == ["formal-good"]
    assert "pilot" in result["excluded"]


def test_gpu_blocked_record_never_calls_fit(tmp_path, monkeypatch):
    _pilot_config(tmp_path / "configs" / "pilot.yaml") if (tmp_path / "configs").mkdir() is None else None
    monkeypatch.setattr(
        "tabpollution.generators.pilot.generator_preflight",
        lambda root: {"hard_gpu_passed": False, "cuda_available": False},
    )
    result = create_gpu_blocked_run(tmp_path, "CTGAN", "configs/pilot.yaml")
    assert result["fit_called"] is False
    assert result["status"] == "blocked_by_gpu"
    assert not (tmp_path / "runs" / result["run_id"] / "COMPLETE").exists()


def test_disk_preflight_fails_before_training(tmp_path, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("tabpollution.generators.preflight.shutil.disk_usage", lambda path: Usage(10, 9, 1))
    with pytest.raises(RuntimeError, match="Insufficient disk"):
        ensure_disk_space(tmp_path, 2)


def test_new_sdv_loader_path(monkeypatch, tmp_path):
    fake_sdv = types.ModuleType("sdv")
    fake_sdv.__path__ = []
    fake_utils = types.ModuleType("sdv.utils")
    fake_utils.load_synthesizer = lambda path: ("new", Path(path).name)
    monkeypatch.setitem(sys.modules, "sdv", fake_sdv)
    monkeypatch.setitem(sys.modules, "sdv.utils", fake_utils)
    assert load_sdv_synthesizer("CTGAN", tmp_path / "m.pkl") == ("new", "m.pkl")


def test_legacy_sdv_loader_fallback(monkeypatch, tmp_path):
    fake_sdv = types.ModuleType("sdv")
    fake_sdv.__path__ = []
    monkeypatch.setitem(sys.modules, "sdv", fake_sdv)
    monkeypatch.delitem(sys.modules, "sdv.utils", raising=False)

    class FakeModel:
        @classmethod
        def load(cls, path):
            return ("legacy", Path(path).name)

    assert load_sdv_synthesizer("TVAE", tmp_path / "m.pkl", FakeModel) == ("legacy", "m.pkl")
