from __future__ import annotations

import pandas as pd

from tabpollution.data.cleaning import add_stable_row_ids, feature_frame
from tabpollution.data.loaders import load_processed_dataset


def test_reloading_same_data_produces_same_ids(toy_spec) -> None:
    frame = pd.DataFrame({"number": [1, 2], "category": ["a", "b"], "target": ["0", "1"]})
    first = add_stable_row_ids(frame.copy(), "toy", ["number", "category", "target"])
    second = add_stable_row_ids(frame.copy(), "toy", ["number", "category", "target"])
    assert first["row_id"].tolist() == second["row_id"].tolist()


def test_duplicate_records_receive_unique_deterministic_ids() -> None:
    frame = pd.DataFrame({"x": [1, 1, 2], "target": ["a", "a", "b"]})
    first = add_stable_row_ids(frame, "toy", ["x", "target"])
    second = add_stable_row_ids(frame.sample(frac=1, random_state=9), "toy", ["x", "target"])
    assert first["row_id"].is_unique
    assert sorted(first["row_id"]) == sorted(second["row_id"])


def test_metadata_and_target_do_not_enter_features(toy_frame, toy_spec) -> None:
    frame = toy_frame.assign(split="R_source_train", provenance="fixture")
    features = feature_frame(frame, toy_spec)
    assert features.columns.tolist() == ["number", "category"]
    assert not {"row_id", "split", "provenance", "target"} & set(features.columns)


def test_processed_loader_preserves_numeric_category_codes_as_strings(tmp_path, toy_spec) -> None:
    path = tmp_path / "toy.csv"
    pd.DataFrame(
        {"row_id": ["toy:1"], "number": [3.5], "category": [2], "target": [1]}
    ).to_csv(path, index=False)
    loaded = load_processed_dataset(toy_spec, path)
    assert str(loaded["category"].dtype).startswith("string")
    assert loaded.loc[0, "category"] == "2"
    assert loaded.loc[0, "target"] == "1"
