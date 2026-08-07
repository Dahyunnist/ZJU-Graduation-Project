from __future__ import annotations

import pandas as pd
import pytest

from tabpollution.mixing.bags import bag_summary, build_bag_members, members_hash, rebuild_bag
from tabpollution.mixing.contamination import (
    build_contamination,
    contamination_summary,
    half_up_count,
    mixture_features,
)


@pytest.fixture
def real_pool() -> pd.DataFrame:
    return pd.DataFrame({"row_id": [f"r{i}" for i in range(1200)], "x": range(1200), "target": [i % 2 for i in range(1200)]})


@pytest.fixture
def synth_pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "synth_row_id": [f"s{i}" for i in range(1200)],
            "x": range(2000, 3200),
            "target": [i % 2 for i in range(1200)],
            "generator_name": ["fake"] * 1200,
            "pool_name": ["S_downstream_mix"] * 1200,
        }
    )


@pytest.mark.parametrize(
    ("n", "p", "expected"),
    [(101, 0, 0), (101, 0.05, 5), (101, 0.10, 10), (101, 0.25, 25), (101, 0.50, 51), (101, 0.75, 76), (101, 1, 101)],
)
def test_half_up_rounding(n: int, p: float, expected: int) -> None:
    assert half_up_count(n, p) == expected


def test_replace_keeps_size_and_boundaries(real_pool, synth_pool) -> None:
    zero = build_contamination(real_pool, synth_pool, "synthetic_replace", 0, 42)
    full = build_contamination(real_pool, synth_pool, "synthetic_replace", 1, 42)
    assert len(zero) == len(real_pool) == len(full)
    assert contamination_summary(zero, len(real_pool))["synthetic_count"] == 0
    assert contamination_summary(full, len(real_pool))["synthetic_count"] == len(real_pool)


def test_append_size_and_real_extra_requirement(real_pool, synth_pool) -> None:
    appended = build_contamination(real_pool, synth_pool, "synthetic_append", 0.25, 42)
    assert len(appended) == len(real_pool) + half_up_count(len(real_pool), 0.25)
    with pytest.raises(ValueError, match="real_extra_pool"):
        build_contamination(real_pool, synth_pool, "real_append", 0.25, 42)


def test_mixture_is_deterministic_and_metadata_excluded(real_pool, synth_pool) -> None:
    first = build_contamination(real_pool, synth_pool, "synthetic_replace", 0.1, 9)
    second = build_contamination(real_pool, synth_pool, "synthetic_replace", 0.1, 9)
    assert first["mix_row_id"].tolist() == second["mix_row_id"].tolist()
    assert mixture_features(first, ["x", "target"]).columns.tolist() == ["x", "target"]


def test_bag_has_exact_size_ratio_and_no_duplicates(real_pool, synth_pool) -> None:
    members = build_bag_members(real_pool, synth_pool, "bag1", 0.05, 1000, 42)
    summary = bag_summary(members, "toy", "fake", "test", 0.05, 42)
    assert summary["bag_size"] == 1000
    assert summary["synthetic_count"] == 50
    assert summary["actual_proportion"] == 0.05
    assert members["record_id"].is_unique


def test_bag_rebuild_is_repeatable(real_pool, synth_pool) -> None:
    members = build_bag_members(real_pool, synth_pool, "bag1", 0.1, 1000, 7)
    rebuilt = rebuild_bag(members, real_pool, synth_pool)
    assert len(rebuilt) == 1000
    assert members_hash(members) == members_hash(members.copy())


def test_calibration_and_test_pools_are_disjoint() -> None:
    calibration_real = pd.DataFrame({"row_id": [f"cr{i}" for i in range(1000)]})
    test_real = pd.DataFrame({"row_id": [f"tr{i}" for i in range(1000)]})
    calibration_synth = pd.DataFrame({"synth_row_id": [f"cs{i}" for i in range(1000)]})
    test_synth = pd.DataFrame({"synth_row_id": [f"ts{i}" for i in range(1000)]})
    cal = build_bag_members(calibration_real, calibration_synth, "c", 0.5, 1000, 1)
    test = build_bag_members(test_real, test_synth, "t", 0.5, 1000, 1)
    assert set(cal["record_id"]).isdisjoint(set(test["record_id"]))

