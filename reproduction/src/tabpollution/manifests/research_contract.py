"""Minimal M2-A research-contract validation.

The validator owns semantic boundaries that ordinary JSON/YAML schemas cannot
express: information use, split leakage, seed hierarchy and mixture estimands.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


class ResearchContractError(ValueError):
    """Raised when a study violates the frozen research contract."""


INFORMATION_LEVELS = {
    "source_only": {"source_real_records", "source_synthetic_labels"},
    "target_real_only": {
        "source_real_records", "source_synthetic_labels", "target_real_records",
    },
    "target_labeled_source": {
        "source_real_records", "source_synthetic_labels", "target_real_records",
        "target_source_labels",
    },
    "task_validation": {
        "source_real_records", "source_synthetic_labels", "target_real_records",
        "task_validation_labels",
    },
    "oracle": {
        "source_real_records", "source_synthetic_labels", "target_real_records",
        "target_source_labels", "task_validation_labels", "final_test_labels",
    },
}
SPLIT_ROLES = {
    "generator_train", "detector_train", "calibration", "task_validation", "final_test",
}
SEED_LEVELS = {
    "generator_train", "real_split", "detector_train", "learner_train", "bag_sampling",
}
ESTIMANDS = {
    "replacement_effect", "append_effect", "attribute_effect",
    "measurement_effect", "governance_effect", "decision_effect",
}
GENERATOR_MANIFEST_FIELDS = {
    "generator_name", "implementation_version", "code_commit", "environment", "budget",
    "input_sha256", "generator_train_seed", "model_sha256", "sample_seed", "output_sha256",
    "fit_seconds", "peak_gpu_memory_mb", "status",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResearchContractError(f"YAML root must be a mapping: {path}")
    return value


def validate_information_budget(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"level", "used_information"}:
        raise ResearchContractError("information_budget requires level and used_information")
    level = value["level"]
    if level not in INFORMATION_LEVELS:
        raise ResearchContractError(f"Unknown information budget: {level}")
    used = value["used_information"]
    if not isinstance(used, list) or len(used) != len(set(used)):
        raise ResearchContractError("used_information must be a unique list")
    forbidden = sorted(set(used) - INFORMATION_LEVELS[level])
    if forbidden:
        raise ResearchContractError(
            f"Information budget {level} forbids: {forbidden}"
        )


def validate_splits(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != SPLIT_ROLES:
        raise ResearchContractError(f"splits must contain exactly {sorted(SPLIT_ROLES)}")
    seen: dict[str, str] = {}
    for role, row_ids in value.items():
        if not isinstance(row_ids, list) or len(row_ids) != len(set(row_ids)):
            raise ResearchContractError(f"split {role} must be a unique row-id list")
        for row_id in row_ids:
            if not isinstance(row_id, str) or not row_id:
                raise ResearchContractError(f"split {role} contains invalid row id")
            if row_id in seen:
                raise ResearchContractError(
                    f"Split leakage: {row_id} appears in {seen[row_id]} and {role}"
                )
            seen[row_id] = role


def validate_seed_hierarchy(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != SEED_LEVELS:
        raise ResearchContractError(f"seeds must contain exactly {sorted(SEED_LEVELS)}")
    seen: dict[int, str] = {}
    for level, seeds in value.items():
        if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
            raise ResearchContractError(f"seed level {level} must be a nonempty unique list")
        for seed in seeds:
            if not isinstance(seed, int) or seed <= 0:
                raise ResearchContractError(f"seed level {level} contains invalid seed")
            if seed in seen:
                raise ResearchContractError(
                    f"Seed hierarchy conflict: {seed} is used by {seen[seed]} and {level}"
                )
            seen[seed] = level


def validate_estimand(value: dict[str, Any]) -> None:
    required = {"name", "mixture_semantics", "total_training_size_fixed"}
    if not isinstance(value, dict) or set(value) != required:
        raise ResearchContractError(f"estimand requires exactly {sorted(required)}")
    name, semantics = value["name"], value["mixture_semantics"]
    if name not in ESTIMANDS:
        raise ResearchContractError(f"Unknown estimand: {name}")
    if semantics not in {"replacement", "append"}:
        raise ResearchContractError("mixture_semantics must be replacement or append")
    fixed = value["total_training_size_fixed"]
    if not isinstance(fixed, bool):
        raise ResearchContractError("total_training_size_fixed must be boolean")
    if name == "replacement_effect" and (semantics != "replacement" or not fixed):
        raise ResearchContractError("replacement_effect requires replacement with fixed total size")
    if name == "append_effect" and (semantics != "append" or fixed):
        raise ResearchContractError("append_effect requires append without fixed total size")


def validate_generator_manifest(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != GENERATOR_MANIFEST_FIELDS:
        missing = sorted(GENERATOR_MANIFEST_FIELDS - set(value or {}))
        unknown = sorted(set(value or {}) - GENERATOR_MANIFEST_FIELDS)
        raise ResearchContractError(
            f"generator_manifest field mismatch; missing={missing}, unknown={unknown}"
        )
    if value["status"] not in {
        "complete", "preflight_failed", "training_failed", "sampling_failed",
        "interrupted", "invalid_output",
    }:
        raise ResearchContractError("Unknown generator status")
    if value["status"] == "complete":
        for name in ("input_sha256", "model_sha256", "output_sha256"):
            if not isinstance(value[name], str) or not SHA256.fullmatch(value[name]):
                raise ResearchContractError(f"{name} must be a lowercase SHA-256")
    if not isinstance(value["generator_train_seed"], int):
        raise ResearchContractError("generator_train_seed must be an integer")
    if value["generator_train_seed"] == value["sample_seed"]:
        raise ResearchContractError("generator and sample seeds must differ")
    if not isinstance(value["budget"], dict) or not value["budget"]:
        raise ResearchContractError("generator budget must be a nonempty mapping")


def validate_study_contract(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "study_id", "information_budget", "estimand", "splits", "seeds", "generator_manifest",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ResearchContractError(f"study contract requires exactly {sorted(required)}")
    if not isinstance(config["study_id"], str) or len(config["study_id"]) < 3:
        raise ResearchContractError("study_id is invalid")
    validate_information_budget(config["information_budget"])
    validate_estimand(config["estimand"])
    validate_splits(config["splits"])
    validate_seed_hierarchy(config["seeds"])
    validate_generator_manifest(config["generator_manifest"])
    if config["generator_manifest"]["generator_train_seed"] not in config["seeds"]["generator_train"]:
        raise ResearchContractError("generator manifest seed is not registered at generator_train level")
    return copy.deepcopy(config)


def validate_dataset_card(card: dict[str, Any], schema: dict[str, Any]) -> None:
    required = set(schema.get("required", []))
    missing = sorted(required - set(card))
    if missing:
        raise ResearchContractError(f"dataset card missing fields: {missing}")
    for section, fields in schema.get("fields", {}).items():
        value = card.get(section)
        if not isinstance(value, dict):
            raise ResearchContractError(f"dataset card section {section} must be a mapping")
        absent = sorted(set(fields) - set(value))
        if absent:
            raise ResearchContractError(f"dataset card {section} missing fields: {absent}")


def validate_contract_bundle(root: str | Path, smoke_config: str | Path) -> dict[str, Any]:
    root = Path(root)
    variable_dictionary = load_yaml(root / "variable_dictionary_v1.yaml")
    dataset_schema = load_yaml(root / "dataset_card_schema_v1.yaml")
    generator_schema = load_yaml(root / "generator_contract_v1.yaml")
    artifact_schema = load_yaml(root / "artifact_schema_v1.yaml")
    config = validate_study_contract(load_yaml(smoke_config))
    if set(variable_dictionary.get("variables", {})) != {
        "independent", "mediators", "moderators", "measurements", "actions", "outcomes",
    }:
        raise ResearchContractError("variable dictionary categories are incomplete")
    if generator_schema.get("manifest_required") != list(GENERATOR_MANIFEST_FIELDS):
        # YAML order is intentional so the executable and readable contracts stay aligned.
        if set(generator_schema.get("manifest_required", [])) != GENERATOR_MANIFEST_FIELDS:
            raise ResearchContractError("generator schema fields disagree with validator")
    required_artifacts = {
        "resolved_config", "row_predictions", "source_scores", "structural_metrics",
        "task_metrics", "governance_actions", "failure", "audit", "complete",
    }
    if set(artifact_schema.get("artifacts", {})) != required_artifacts:
        raise ResearchContractError("standard artifact schema is incomplete")
    return {
        "status": "pass",
        "study_id": config["study_id"],
        "information_budget": config["information_budget"]["level"],
        "variable_categories": sorted(variable_dictionary["variables"]),
        "dataset_schema_id": dataset_schema.get("schema_id"),
        "generator_schema_id": generator_schema.get("schema_id"),
        "artifact_schema_id": artifact_schema.get("schema_id"),
    }

