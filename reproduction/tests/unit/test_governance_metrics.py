from __future__ import annotations

import numpy as np
import pandas as pd

from tabpollution.governance.artifacts import artifact_features
from tabpollution.governance.metrics import (
    analytical_positive_rate,
    detection_metrics,
    expected_calibration_error,
    select_fpr_threshold,
)
from tabpollution.quantification.methods import ScoreQuantifier
from tabpollution.detectors.base import serialize_record


def test_low_prevalence_false_positive_mass_can_dominate() -> None:
    result = analytical_positive_rate(prevalence=.05, tpr=.8, fpr=.10)
    assert result["false_positive_mass"] > result["true_positive_mass"]
    assert result["false_positive_share"] > .5


def test_detection_metrics_include_calibration_and_operating_point() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([.01, .1, .2, .3, .7, .8, .9, .99])
    threshold = select_fpr_threshold(labels, scores, .05)["threshold"]
    metrics = detection_metrics(labels, scores, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["fpr"] <= .05
    assert expected_calibration_error(labels, scores) >= 0


def test_artifact_features_have_schema_independent_columns() -> None:
    first = artifact_features(pd.DataFrame({"a": [1.234], "b": ["x"], "table_id": ["t1"]}))
    second = artifact_features(pd.DataFrame({"different": [2], "other": [None], "table_id": ["t2"]}))
    assert first.columns.tolist() == second.columns.tolist()
    assert "serialized_length" in first


def test_cross_table_serialization_uses_native_schema_only() -> None:
    row = pd.Series({
        "a": 1.0, "b": np.nan, "table_id": "table_a",
        "schema_columns": "a",
    })
    text = serialize_record(row)
    assert text == "a:1"
    assert "b:" not in text


def test_kdey_recovers_a_separated_mixture() -> None:
    rng = np.random.default_rng(2026)
    negative = np.clip(rng.normal(.15, .05, 300), 0, 1)
    positive = np.clip(rng.normal(.85, .05, 300), 0, 1)
    calibration = np.concatenate([negative, positive])
    labels = np.concatenate([np.zeros(len(negative), int), np.ones(len(positive), int)])
    test = np.concatenate([negative[:240], positive[:60]])
    estimate = ScoreQuantifier("kdey").fit(calibration, labels).predict_prevalence(test)["clipped"]
    assert abs(estimate - .20) < .05
