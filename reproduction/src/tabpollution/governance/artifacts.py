"""Format-artifact-only diagnostics used as a benchmark anti-cheating gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabpollution.detectors.base import record_feature_items


def _decimal_places(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value)
    if "e" in text.lower():
        return float(len(text.split("e", 1)[0].split(".", 1)[-1])) if "." in text else 0.0
    return float(len(text.rsplit(".", 1)[1])) if "." in text else 0.0


def artifact_features(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in records.iterrows():
        values = [value for _, value in record_feature_items(row)]
        numeric_values = [v for v in values if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v)]
        rendered = ["<NA>" if pd.isna(v) else str(v) for v in values]
        rows.append({
            "missing_count": sum(pd.isna(v) for v in values),
            "empty_count": sum((not pd.isna(v)) and str(v).strip() == "" for v in values),
            "serialized_length": sum(len(v) for v in rendered),
            "max_field_length": max((len(v) for v in rendered), default=0),
            "numeric_fraction": len(numeric_values) / max(1, len(values)),
            "integral_fraction": sum(float(v).is_integer() for v in numeric_values) / max(1, len(numeric_values)),
            "mean_decimal_places": np.mean([_decimal_places(v) for v in numeric_values]) if numeric_values else 0.0,
        })
    return pd.DataFrame(rows)


def artifact_only_auc(train_records: pd.DataFrame, train_labels: np.ndarray,
                      test_records: pd.DataFrame, test_labels: np.ndarray,
                      seed: int) -> dict[str, Any]:
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=400, random_state=seed),
    )
    x_train = artifact_features(train_records)
    x_test = artifact_features(test_records)
    model.fit(x_train, np.asarray(train_labels, int))
    scores = model.predict_proba(x_test)[:, 1]
    estimator = model[-1]
    contributions = {
        name: float(weight)
        for name, weight in zip(x_train.columns, estimator.coef_[0])
    }
    raw_auc = float(roc_auc_score(test_labels, scores))
    return {
        # Direction-agnostic separability: AUC near zero still means that
        # formatting variables almost perfectly separate the two sources,
        # although the learned direction reverses under shift.
        "artifact_auroc": max(raw_auc, 1 - raw_auc),
        "artifact_auroc_raw": raw_auc,
        "feature_coefficients": contributions,
        "features": list(x_train.columns),
    }
