from __future__ import annotations

from copy import deepcopy

import pytest

from tabpollution.config import ConfigError, validate_benchmark_config


def test_valid_config_passes(valid_config: dict) -> None:
    assert validate_benchmark_config(valid_config)["benchmark_id"] == "benchmark_v1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["real_splits"].update(R_source_train=0.50), "sum to 1.0"),
        (lambda cfg: cfg["formal_seeds"].append(2026), "duplicates"),
        (lambda cfg: cfg["datasets"].append("unknown_table"), "Unknown"),
        (lambda cfg: cfg["pollution_rates"].append(1.2), "[0, 1]"),
    ],
)
def test_invalid_config_fails(valid_config: dict, mutation, message: str) -> None:
    config = deepcopy(valid_config)
    mutation(config)
    with pytest.raises(ConfigError, match=message):
        validate_benchmark_config(config)

