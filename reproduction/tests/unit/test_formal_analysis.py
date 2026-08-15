from __future__ import annotations

import numpy as np
import pandas as pd

from tabpollution.governance.formal_analysis import (
    holm_adjust,
    paired_detector_tests,
    summarize_seed_replicates,
)


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    raw = np.array([.03, .001, .02])
    adjusted = holm_adjust(raw)
    assert np.allclose(adjusted, [.04, .003, .04])
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)


def test_seed_summary_does_not_treat_bags_as_independent() -> None:
    rows = pd.DataFrame({
        "seed": [1, 1, 1, 2, 2],
        "protocol": ["P1"] * 5,
        "metric_value": [0., 0., 0., 1., 1.],
    })
    seed, ci = summarize_seed_replicates(
        rows, group_columns=["protocol"], metrics=["metric_value"],
    )
    assert len(seed) == 2
    result = ci.iloc[0]
    assert result["n_seeds"] == 2
    assert result["mean"] == .5


def test_paired_detector_tests_use_seed_level_differences() -> None:
    rows = []
    for seed in [1, 2, 3]:
        for bag in [0, 1]:
            for detector, value in [("a", .2 + seed * .01), ("b", .1 + seed * .01)]:
                rows.append({
                    "seed": seed,
                    "protocol": "P1",
                    "calibration_policy": "source_only",
                    "contamination_mode": "replace",
                    "true_prevalence": .1,
                    "test_table": "t",
                    "test_generator": "g",
                    "bag_index": bag,
                    "detector": detector,
                    "quantifier": "pacc",
                    "quantifier_status": "ok",
                    "prevalence_absolute_error": value,
                })
    result = paired_detector_tests(pd.DataFrame(rows), "pacc")
    assert len(result) == 1
    assert result.iloc[0]["n_seeds"] == 3
    assert result.iloc[0]["matched_bags"] == 6
    assert np.isclose(result.iloc[0]["mean_difference"], .1)

