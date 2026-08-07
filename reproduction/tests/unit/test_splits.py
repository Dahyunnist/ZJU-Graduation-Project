from __future__ import annotations

import pytest

from tabpollution.config import SPLIT_NAMES
from tabpollution.data.splits import (
    assignment_content_hash,
    build_split_assignment,
    validate_split_assignment,
)


def test_same_seed_rebuilds_identical_split(toy_frame) -> None:
    first = build_split_assignment(toy_frame, "target", 2026)
    second = build_split_assignment(toy_frame, "target", 2026)
    assert assignment_content_hash(first) == assignment_content_hash(second)


def test_splits_are_disjoint_and_cover_all_rows(toy_frame) -> None:
    assignment = build_split_assignment(toy_frame, "target", 2026)
    validate_split_assignment(assignment, set(toy_frame["row_id"]))
    groups = {
        split: set(assignment.loc[assignment["split"] == split, "row_id"])
        for split in SPLIT_NAMES
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            assert groups[left].isdisjoint(groups[right])
    assert set().union(*groups.values()) == set(toy_frame["row_id"])


def test_split_sizes_follow_frozen_rule(toy_frame) -> None:
    assignment = build_split_assignment(toy_frame, "target", 2027)
    counts = assignment["split"].value_counts().to_dict()
    assert counts == {
        "R_source_train": 120,
        "R_detector_train": 30,
        "R_detector_val": 20,
        "R_final_test": 30,
    }


def test_stratification_preserves_target_rate(toy_frame) -> None:
    assignment = build_split_assignment(toy_frame, "target", 2028)
    overall = (toy_frame["target"] == "1").mean()
    for split in SPLIT_NAMES:
        rate = (assignment.loc[assignment["split"] == split, "target"] == "1").mean()
        assert abs(rate - overall) <= 0.05


def test_injected_overlap_is_rejected(toy_frame) -> None:
    assignment = build_split_assignment(toy_frame, "target", 2026)
    bad = assignment.copy()
    bad.loc[len(bad)] = bad.iloc[0]
    bad.loc[len(bad) - 1, "split"] = "R_final_test"
    with pytest.raises(ValueError, match="more than one split"):
        validate_split_assignment(bad, set(toy_frame["row_id"]))


def test_unknown_row_id_is_rejected(toy_frame) -> None:
    assignment = build_split_assignment(toy_frame, "target", 2026)
    assignment.loc[0, "row_id"] = "toy:unknown"
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_split_assignment(assignment, set(toy_frame["row_id"]))

