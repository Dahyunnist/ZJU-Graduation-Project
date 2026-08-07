"""Loading and strict validation for the frozen benchmark configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SPLIT_NAMES = (
    "R_source_train",
    "R_detector_train",
    "R_detector_val",
    "R_final_test",
)
ALLOWED_DATASETS = {"adult", "credit"}
ALLOWED_GENERATORS = {"GaussianCopula", "CTGAN", "TVAE"}


class ConfigError(ValueError):
    """Raised when a benchmark configuration violates the frozen schema."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration must be a mapping: {path}")
    return value


def validate_benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "benchmark_id",
        "version",
        "datasets",
        "formal_seeds",
        "smoke_seed",
        "real_splits",
        "pollution_rates",
        "quantification",
        "generators",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"Missing benchmark fields: {missing}")

    datasets = config["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ConfigError("datasets must be a non-empty list")
    unknown_datasets = sorted(set(datasets) - ALLOWED_DATASETS)
    if unknown_datasets:
        raise ConfigError(f"Unknown Track A datasets: {unknown_datasets}")
    if len(datasets) != len(set(datasets)):
        raise ConfigError("datasets contains duplicates")

    seeds = config["formal_seeds"]
    if not isinstance(seeds, list) or not seeds or not all(isinstance(x, int) for x in seeds):
        raise ConfigError("formal_seeds must be a non-empty list of integers")
    if len(seeds) != len(set(seeds)):
        raise ConfigError("formal_seeds contains duplicates")

    splits = config["real_splits"]
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ConfigError(f"real_splits must contain exactly {list(SPLIT_NAMES)}")
    if any(not isinstance(value, (int, float)) or value <= 0 or value >= 1 for value in splits.values()):
        raise ConfigError("Every real split proportion must be strictly between 0 and 1")
    if abs(sum(float(value) for value in splits.values()) - 1.0) > 1e-9:
        raise ConfigError("real_splits must sum to 1.0")

    rates = config["pollution_rates"]
    if not isinstance(rates, list) or not rates:
        raise ConfigError("pollution_rates must be a non-empty list")
    if any(not isinstance(rate, (int, float)) or rate < 0 or rate > 1 for rate in rates):
        raise ConfigError("pollution_rates must be numeric values in [0, 1]")
    if len(rates) != len(set(float(rate) for rate in rates)):
        raise ConfigError("pollution_rates contains duplicates")

    quant = config["quantification"]
    quant_fields = {"bag_size", "calibration_bags", "test_bags"}
    if not isinstance(quant, dict) or set(quant) != quant_fields:
        raise ConfigError(f"quantification must contain exactly {sorted(quant_fields)}")
    if any(not isinstance(quant[key], int) or quant[key] <= 0 for key in quant_fields):
        raise ConfigError("Quantification sizes must be positive integers")

    generators = config["generators"]
    if not isinstance(generators, list) or len(generators) != len(set(generators)):
        raise ConfigError("generators must be a unique list")
    unknown_generators = sorted(set(generators) - ALLOWED_GENERATORS)
    if unknown_generators:
        raise ConfigError(f"Unknown Track A generators: {unknown_generators}")

    return deepcopy(config)


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    return validate_benchmark_config(load_yaml(path))


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)

