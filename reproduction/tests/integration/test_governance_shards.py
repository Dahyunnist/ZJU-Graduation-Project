from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from tabpollution.governance.shards import (
    build_shard_plan,
    run_governance_sharded,
    shard_queue,
    shard_status,
)


def _raw_config(output: Path) -> dict:
    return {
        "experiment_id": "pytest-governance-shards",
        "run_type": "smoke",
        "seeds": [2026],
        "protocols": {
            "P1": {
                "train_tables": ["table_a"], "test_tables": ["table_a"],
                "train_generators": ["marginal"], "test_generators": ["marginal"],
            },
        },
        "prevalence_rates": [0, .05, .10],
        "contamination_modes": ["replace"],
        "bags_per_rate": 1,
        "utility_bags_per_rate": 1,
        "bag_size": 40,
        "detectors": ["char3gram", "c2st_lr"],
        "quantifiers": ["pacc"],
        "primary_quantifier": "pacc",
        "valuation": {
            "enabled": True, "methods": ["knn_shapley"], "bags_per_rate": 1,
            "sample_limit": 40, "oob_estimators": 10,
        },
        "data": {"mode": "synthetic_fixture", "rows_per_table": 400},
        "thresholds": {
            "detector_fpr_target": .05, "artifact_auc_gate": .65,
            "governance_prevalence": .10, "harm_tolerance": .005,
        },
        "output_dir": str(output),
        "resources": {"device": "cpu", "max_cpu_threads": 1},
        "deep_training": {
            "dim": 24, "heads": 4, "layers": 1, "max_len": 192,
            "max_datum": 32, "max_columns": 24, "epochs": 2, "batch_size": 32,
            "learning_rate": .002, "weight_decay": .01, "gradient_clip_norm": 1.,
            "early_stopping_patience": 2, "min_epochs": 1,
        },
    }


def test_sharded_run_resumes_without_repeating_valuation(tmp_path: Path) -> None:
    config_path = tmp_path / "governance.yaml"
    config_path.write_text(yaml.safe_dump(_raw_config(tmp_path / "run")), encoding="utf-8")
    plan = build_shard_plan(config_path)
    assert plan["shard_count"] == 2
    assert sum(row["valuation_owner"] for row in plan["shards"]) == 1

    first = run_governance_sharded(config_path, resume=True, max_shards=1)
    assert first["status"] == "partial"
    assert first["completed_count"] == 1
    second = run_governance_sharded(config_path, resume=True, max_shards=1)
    assert second["status"] == "complete"
    assert second["aggregate"]["status"] == "complete"

    attempts_before = sorted((tmp_path / "run" / "shards").glob("*/attempts/*"))
    third = run_governance_sharded(config_path, resume=True, max_shards=1)
    attempts_after = sorted((tmp_path / "run" / "shards").glob("*/attempts/*"))
    assert third["executed_shards"] == []
    assert attempts_after == attempts_before
    assert shard_status(config_path)["pending_count"] == 0

    evidence = pd.read_csv(tmp_path / "run" / "governance_evidence.csv")
    assert set(evidence["detector"]) == {"char3gram", "c2st_lr"}
    valuation = pd.read_csv(tmp_path / "run" / "record_valuation.csv")
    assert set(valuation["valuation_method"]) == {"knn_shapley"}
    assert not valuation.duplicated().any()

    # Changing only deep-training hyperparameters must not invalidate completed
    # classical shards; the deep shards are the only ones that require reruns.
    revised = _raw_config(tmp_path / "run")
    revised["deep_training"]["learning_rate"] = .0002
    config_path.write_text(yaml.safe_dump(revised), encoding="utf-8")
    assert shard_status(config_path)["completed_count"] == 2


def test_formal_plan_preserves_full_contract(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/governance_formal.yaml").read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path / "formal")
    config_path = tmp_path / "formal.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    plan = build_shard_plan(config_path)
    assert plan["run_type"] == "formal"
    assert plan["shard_count"] == 110
    assert {row["protocol"] for row in plan["shards"]} == {"P1", "P2", "P3", "P4"}
    cpu = shard_queue(config_path, seed=2026, resource_class="cpu")
    gpu = shard_queue(config_path, seed=2026, resource_class="gpu")
    assert cpu["pending_count"] == 6
    assert gpu["pending_count"] == 16
    assert not set(cpu["pending_shards"]) & set(gpu["pending_shards"])
    assert all("__c2st_" in shard or "__char3gram" in shard for shard in cpu["pending_shards"])
