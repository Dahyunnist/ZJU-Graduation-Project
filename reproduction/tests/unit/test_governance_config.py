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
        "contamination_modes": ["replace"],
        "bags_per_rate": 1,
        "utility_bags_per_rate": 1,
        "bag_size": 80,
        "detectors": ["char3gram"],
        "quantifiers": ["pacc"],
        "primary_quantifier": "pacc",
        "valuation": {"enabled": False, "methods": [], "bags_per_rate": 0, "sample_limit": 80, "oob_estimators": 20},
        "data": {"mode": "synthetic_fixture", "rows_per_table": 500},
        "thresholds": {
            "detector_fpr_target": .05, "artifact_auc_gate": .65,
            "governance_prevalence": .10, "harm_tolerance": .005,
        },
        "output_dir": "runs/test",
        "resources": {"device": "cpu", "max_cpu_threads": 2},
        "deep_training": {
            "dim": 24, "heads": 4, "layers": 1, "max_len": 192,
            "max_datum": 32, "max_columns": 24, "epochs": 2, "batch_size": 32,
            "learning_rate": .002, "weight_decay": .01, "gradient_clip_norm": 1.,
            "early_stopping_patience": 2, "min_epochs": 1,
        },
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


def test_pilot_and_formal_freeze_the_same_protocol_matrix() -> None:
    pilot = load_governance_config(Path("configs/governance_pilot.yaml"))
    formal = load_governance_config(Path("configs/governance_formal.yaml"))
    assert pilot.run_type == "pilot"
    assert pilot.protocols == formal.protocols
    assert pilot.detectors == formal.detectors
    assert pilot.quantifiers == formal.quantifiers
    assert pilot.seeds == (2026,)


def test_resource_probe_uses_one_formal_size_deep_detector_and_is_not_formal() -> None:
    probe = load_governance_config(Path("configs/governance_resource_probe.yaml"))
    assert probe.run_type == "pilot"
    assert set(probe.protocols) == {"P1"}
    assert probe.detectors == ("flat_transformer",)
    assert probe.bags_per_rate == 1
    assert not probe.valuation_enabled
    assert probe.device == "cuda"


def test_deep_stability_probe_uses_lower_learning_rate_and_validation_selection() -> None:
    probe = load_governance_config(Path("configs/governance_deep_stability_probe.yaml"))
    assert probe.detectors == ("flat_transformer",)
    assert probe.deep_dim == 192 and probe.deep_layers == 6
    assert probe.deep_learning_rate == .0002
    assert probe.deep_min_epochs == 4


def test_formal_contract_rejects_legacy_approximation() -> None:
    raw = valid_governance_config()
    formal = load_governance_config(Path("configs/governance_formal.yaml"))
    raw.update({
        "run_type": "formal", "seeds": list(formal.seeds), "protocols": formal.protocols,
        "contamination_modes": ["replace", "append"], "bags_per_rate": 100,
        "utility_bags_per_rate": 5,
        "detectors": ["char3gram", "flat_transformer", "table_transformer", "datum_transformer", "datum_ta"],
        "valuation": {"enabled": True, "methods": ["knn_shapley", "data_oob"], "bags_per_rate": 1,
                      "sample_limit": 1000, "oob_estimators": 80},
        "resources": {"device": "cuda", "max_cpu_threads": 2},
    })
    with pytest.raises(GovernanceConfigError, match="legacy approximation"):
        validate_governance_config(raw)


def test_formal_contract_requires_both_contamination_modes() -> None:
    raw = valid_governance_config()
    formal = load_governance_config(Path("configs/governance_formal.yaml"))
    raw.update({
        "run_type": "formal", "seeds": list(formal.seeds), "protocols": formal.protocols,
        "bags_per_rate": 100, "utility_bags_per_rate": 5,
        "detectors": ["char3gram", "flat_transformer", "datum_transformer", "datum_ta"],
        "valuation": {"enabled": True, "methods": ["knn_shapley", "data_oob"], "bags_per_rate": 1,
                      "sample_limit": 1000, "oob_estimators": 80},
        "resources": {"device": "cuda", "max_cpu_threads": 2},
    })
    with pytest.raises(GovernanceConfigError, match="replace and append"):
        validate_governance_config(raw)
