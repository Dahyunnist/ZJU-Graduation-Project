"""Metrics that connect detection, prevalence estimation and decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, int)
    scores = np.clip(np.asarray(scores, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores < high if high < 1 else scores <= high)
        if mask.any():
            error += mask.mean() * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(error if total else np.nan)


def select_fpr_threshold(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, float]:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.flatnonzero(fpr <= target_fpr)
    if not len(valid):
        index = int(np.argmin(fpr))
    else:
        # Maximise sensitivity under the deployment FPR constraint.
        index = int(valid[np.argmax(tpr[valid])])
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(scores.max(), np.inf))
    return {"threshold": threshold, "validation_fpr": float(fpr[index]), "validation_tpr": float(tpr[index])}


def select_negative_anchor_threshold(scores: np.ndarray, target_fpr: float) -> dict[str, float]:
    """Choose the most permissive threshold whose empirical clean-anchor FPR is controlled."""
    scores = np.asarray(scores, float)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("negative anchor scores must be non-empty and finite")
    candidates = np.concatenate(([np.nextafter(scores.max(), np.inf)], np.unique(scores)))
    valid = [(float(t), float(np.mean(scores >= t))) for t in candidates if np.mean(scores >= t) <= target_fpr]
    threshold, achieved = min(valid, key=lambda item: item[0])
    return {"threshold": threshold, "validation_fpr": achieved, "validation_tpr": float("nan")}


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, int)
    scores = np.clip(np.asarray(scores, float), 0, 1)
    predictions = scores >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": expected_calibration_error(labels, scores),
        "fpr": float(predictions[negative].mean()) if negative.any() else float("nan"),
        "tpr": float(predictions[positive].mean()) if positive.any() else float("nan"),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def prevalence_error(true_prevalence: float, estimated_prevalence: float) -> dict[str, float]:
    error = float(estimated_prevalence - true_prevalence)
    return {
        "prevalence_error": error,
        "prevalence_absolute_error": abs(error),
        "prevalence_squared_error": error * error,
    }


def analytical_positive_rate(prevalence: float, tpr: float, fpr: float) -> dict[str, float]:
    true_positive_mass = prevalence * tpr
    false_positive_mass = (1 - prevalence) * fpr
    observed_positive_rate = true_positive_mass + false_positive_mass
    return {
        "expected_positive_rate": float(observed_positive_rate),
        "true_positive_mass": float(true_positive_mass),
        "false_positive_mass": float(false_positive_mass),
        "false_positive_share": float(false_positive_mass / observed_positive_rate) if observed_positive_rate > 0 else 0.0,
    }


def finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None
