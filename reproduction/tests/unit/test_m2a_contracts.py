from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tabpollution.manifests.research_contract import (
    ResearchContractError,
    load_yaml,
    validate_contract_bundle,
    validate_study_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "m2a_contract_smoke_v1.yaml"
CONTRACTS = ROOT / "configs" / "contracts"


def _valid() -> dict:
    return load_yaml(CONFIG)


def test_contract_bundle_cpu_smoke() -> None:
    result = validate_contract_bundle(CONTRACTS, CONFIG)
    assert result["status"] == "pass"
    assert result["information_budget"] == "source_only"


def test_information_budget_conflict_is_rejected() -> None:
    config = _valid()
    config["information_budget"]["used_information"].append("target_source_labels")
    with pytest.raises(ResearchContractError, match="forbids"):
        validate_study_contract(config)


def test_split_leakage_is_rejected() -> None:
    config = _valid()
    config["splits"]["final_test"].append("r001")
    with pytest.raises(ResearchContractError, match="Split leakage"):
        validate_study_contract(config)


def test_seed_level_confusion_is_rejected() -> None:
    config = _valid()
    config["seeds"]["detector_train"] = [1101]
    with pytest.raises(ResearchContractError, match="Seed hierarchy conflict"):
        validate_study_contract(config)


@pytest.mark.parametrize(
    ("name", "semantics", "fixed"),
    [
        ("replacement_effect", "append", True),
        ("replacement_effect", "replacement", False),
        ("append_effect", "replacement", False),
        ("append_effect", "append", True),
    ],
)
def test_replacement_and_append_semantics_cannot_mix(name, semantics, fixed) -> None:
    config = _valid()
    config["estimand"] = {
        "name": name,
        "mixture_semantics": semantics,
        "total_training_size_fixed": fixed,
    }
    with pytest.raises(ResearchContractError):
        validate_study_contract(config)


def test_generator_manifest_seed_must_use_generator_level() -> None:
    config = deepcopy(_valid())
    config["generator_manifest"]["generator_train_seed"] = 3101
    with pytest.raises(ResearchContractError, match="not registered"):
        validate_study_contract(config)

