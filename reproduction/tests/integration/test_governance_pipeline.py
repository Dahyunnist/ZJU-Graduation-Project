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
        "bags_per_rate": 1,
        "utility_bags_per_rate": 1,
        "bag_size": 60,
        "detectors": ["char3gram"],
        "quantifiers": ["pacc"],
        "primary_quantifier": "pacc",
        "data": {"mode": "synthetic_fixture", "rows_per_table": 450},
        "thresholds": {
            "detector_fpr_target": .05, "artifact_auc_gate": .65,
            "governance_prevalence": .10, "harm_tolerance": .005,
        },
        "output_dir": str(tmp_path / "run"),
        "resources": {"device": "cpu", "max_cpu_threads": 1},
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
