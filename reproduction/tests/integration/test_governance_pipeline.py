from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabpollution.governance.config import validate_governance_config
from tabpollution.governance.pipeline import run_governance_benchmark, validate_governance_setup


def test_unified_governance_pipeline_emits_linked_evidence(tmp_path: Path) -> None:
    raw = {
        "experiment_id": "pytest-governance",
        "run_type": "smoke",
        "seeds": [2026],
        "protocols": {
            "P1": {
                "train_tables": ["table_a"], "test_tables": ["table_a"],
                "train_generators": ["marginal"], "test_generators": ["marginal"],
            },
        },
        "prevalence_rates": [0, .05, .10, .25],
        "contamination_modes": ["replace"],
        "bags_per_rate": 1,
        "utility_bags_per_rate": 1,
        "bag_size": 60,
        "detectors": ["char3gram"],
        "quantifiers": ["pacc"],
        "primary_quantifier": "pacc",
        "valuation": {"enabled": True, "methods": ["knn_shapley", "data_oob"], "bags_per_rate": 1, "sample_limit": 60, "oob_estimators": 20},
        "data": {"mode": "synthetic_fixture", "rows_per_table": 450},
        "thresholds": {
            "detector_fpr_target": .05, "artifact_auc_gate": .65,
            "governance_prevalence": .10, "harm_tolerance": .005,
        },
        "output_dir": str(tmp_path / "run"),
        "resources": {"device": "cpu", "max_cpu_threads": 1},
        "deep_training": {
            "dim": 24, "heads": 4, "layers": 1, "max_len": 192,
            "max_datum": 32, "max_columns": 24, "epochs": 2, "batch_size": 32,
            "learning_rate": .002, "weight_decay": .01, "gradient_clip_norm": 1.,
            "early_stopping_patience": 2, "min_epochs": 1,
        },
    }
    config = validate_governance_config(raw)
    assert validate_governance_setup(config)["passed"]
    summary = run_governance_benchmark(config)
    assert summary["status"] == "complete"
    evidence = pd.read_csv(tmp_path / "run" / "governance_evidence.csv")
    assert set([
        "detection_auroc", "estimated_prevalence", "prevalence_absolute_error",
        "contaminated_utility_delta", "decision_error", "decision_regret",
    ]).issubset(evidence.columns)
    assert set(evidence["true_prevalence"]) == {0, .05, .10, .25}
    assert (tmp_path / "run" / "finding_3_low_prevalence.csv").is_file()
    valuation = pd.read_csv(tmp_path / "run" / "record_valuation.csv")
    assert set(valuation["valuation_method"]) == {"knn_shapley", "data_oob"}
    assert (tmp_path / "run" / "finding_6_source_task_value.csv").is_file()
