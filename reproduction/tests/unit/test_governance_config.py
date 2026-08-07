from __future__ import annotations

from pathlib import Path

import pytest

from tabpollution.governance.config import (
    GovernanceConfigError,
    load_governance_config,
    validate_governance_config,
)


def valid_governance_config() -> dict:
    return {
        "experiment_id": "test",
        "run_type": "smoke",
        "seeds": [2026],
        "protocols": {
            "P1": {
                "train_tables": ["table_a"], "test_tables": ["table_a"],
                "train_generators": ["marginal"], "test_generators": ["marginal"],
            },
            "P4": {
                "train_tables": ["table_a"], "test_tables": ["table_c"],
                "train_generators": ["marginal"], "test_generators": ["rounded"],
            },
        },
        "prevalence_rates": [0, .05, .10, .5],
        "bags_per_rate": 1,
        "utility_bags_per_rate": 1,
        "bag_size": 80,
        "detectors": ["char3gram"],
        "quantifiers": ["pacc"],
        "primary_quantifier": "pacc",
        "data": {"mode": "synthetic_fixture", "rows_per_table": 500},
        "thresholds": {
            "detector_fpr_target": .05, "artifact_auc_gate": .65,
            "governance_prevalence": .10, "harm_tolerance": .005,
        },
        "output_dir": "runs/test",
        "resources": {"device": "cpu", "max_cpu_threads": 2},
    }


def test_governance_smoke_config_loads() -> None:
    config = load_governance_config(Path("configs/governance_smoke.yaml"))
    assert set(config.protocols) == {"P1", "P2", "P3", "P4"}
    assert .05 in config.prevalence_rates and .10 in config.prevalence_rates


def test_low_prevalence_rates_are_mandatory() -> None:
    raw = valid_governance_config()
    raw["prevalence_rates"] = [0, .25, .5]
    with pytest.raises(GovernanceConfigError, match="0.05 and 0.10"):
        validate_governance_config(raw)


def test_p4_rejects_table_or_generator_overlap() -> None:
    raw = valid_governance_config()
    raw["protocols"]["P4"]["test_generators"] = ["marginal"]
    with pytest.raises(GovernanceConfigError, match="P4 requires disjoint"):
        validate_governance_config(raw)


def test_registry_preflight_configuration_can_exist_before_data() -> None:
    config = load_governance_config(Path("configs/governance_formal.yaml"))
    assert config.data_mode == "registry"
    assert config.run_type == "formal"
    assert config.device == "cuda"
