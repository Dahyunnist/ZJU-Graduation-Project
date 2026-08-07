from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabpollution.data.registry import DatasetSpec


@pytest.fixture
def toy_spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="toy",
        canonical_name="Toy",
        uci_id=0,
        doi="none",
        source_page="https://example.test/toy",
        download_url="https://example.test/toy.csv",
        raw_filename="toy.csv",
        license="test",
        task_type="binary_classification",
        target_column="target",
        numeric_columns=("number",),
        categorical_columns=("category",),
        missing_markers=("?",),
    )


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    rows = []
    for index in range(200):
        rows.append(
            {
                "row_id": f"toy:{index:04d}",
                "number": index,
                "category": "a" if index % 3 else "b",
                "target": str(index % 2),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def valid_config() -> dict:
    return {
        "benchmark_id": "benchmark_v1",
        "version": "1.0",
        "datasets": ["adult", "credit"],
        "formal_seeds": [2026, 2027, 2028, 2029, 2030],
        "smoke_seed": 42,
        "real_splits": {
            "R_source_train": 0.60,
            "R_detector_train": 0.15,
            "R_detector_val": 0.10,
            "R_final_test": 0.15,
        },
        "pollution_rates": [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1],
        "quantification": {"bag_size": 1000, "calibration_bags": 50, "test_bags": 100},
        "generators": ["GaussianCopula", "CTGAN", "TVAE"],
    }

