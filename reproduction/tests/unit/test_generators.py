from __future__ import annotations

import pandas as pd
import pytest

from tabpollution.generators.base import (
    GeneratorError,
    audit_generator_access,
    derive_pool_seeds,
    model_frame,
)
from tabpollution.generators.pools import POOL_NAMES, add_pool_provenance, validate_pools
from tabpollution.generators.sdv_adapter import create_generator


def test_generator_access_accepts_only_source_partition() -> None:
    frame = pd.DataFrame(
        {"row_id": ["a", "b"], "split": ["R_source_train", "R_source_train"], "x": [1, 2]}
    )
    audit = audit_generator_access(frame, {"a", "b"})
    assert audit.passed and audit.row_count == 2


def test_generator_access_rejects_forbidden_partition() -> None:
    frame = pd.DataFrame(
        {"row_id": ["a", "x"], "split": ["R_source_train", "R_final_test"], "x": [1, 2]}
    )
    with pytest.raises(GeneratorError, match="access violation"):
        audit_generator_access(frame, {"a", "b"})


def test_model_frame_removes_all_provenance() -> None:
    frame = pd.DataFrame(
        {
            "row_id": ["a"],
            "split": ["R_source_train"],
            "generator_name": ["fake"],
            "x": [1],
            "target": ["yes"],
        }
    )
    assert model_frame(frame).columns.tolist() == ["x", "target"]


def test_generator_registry_and_unknown_error() -> None:
    for name in ("GaussianCopula", "CTGAN", "TVAE"):
        assert create_generator(name).generator_name == name
    with pytest.raises(GeneratorError, match="Unknown generator"):
        create_generator("unknown")


def test_sample_seed_is_forwarded_to_sdv_internal_random_state() -> None:
    generator = create_generator("GaussianCopula")

    class FakeModel:
        def __init__(self):
            self.seed = None

        def _set_random_state(self, seed):
            self.seed = seed

        def sample(self, num_rows):
            return pd.DataFrame({"value": [self.seed] * num_rows})

    generator.model = FakeModel()
    first = generator.sample(2, 101, "first")
    second = generator.sample(2, 202, "second")
    assert first["value"].tolist() == [101, 101]
    assert second["value"].tolist() == [202, 202]


def test_pool_seed_derivation_is_stable_and_unique() -> None:
    offsets = {name: index * 100 + 1 for index, name in enumerate(POOL_NAMES)}
    assert derive_pool_seeds(42, offsets) == derive_pool_seeds(42, offsets)
    assert len(set(derive_pool_seeds(42, offsets).values())) == 4


def test_four_pool_ids_are_globally_disjoint_and_schema_ordered() -> None:
    pools = {}
    columns = ["x", "target"]
    for index, name in enumerate(POOL_NAMES):
        raw = pd.DataFrame({"x": [index, index + 1], "target": ["a", "b"]})
        pools[name] = add_pool_provenance(raw, "toy", "fake", 42, 100 + index, name)
    result = validate_pools(pools, columns)
    assert result["all_synth_ids_unique"]


def test_pool_validator_rejects_id_overlap() -> None:
    columns = ["x", "target"]
    pools = {
        name: add_pool_provenance(
            pd.DataFrame({"x": [index], "target": ["a"]}), "toy", "fake", 42, 100 + index, name
        )
        for index, name in enumerate(POOL_NAMES)
    }
    pools["S_detector_val"].loc[0, "synth_row_id"] = pools["S_detector_train"].loc[0, "synth_row_id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_pools(pools, columns)
