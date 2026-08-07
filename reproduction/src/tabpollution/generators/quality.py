"""Smoke-level quality, overlap, formatting and TSTR/TRTR diagnostics."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpollution.data.registry import DatasetSpec
from tabpollution.generators.base import model_frame


def _row_hashes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    canonical = frame.loc[:, columns].astype("string").fillna("<NA>")
    return canonical.apply(
        lambda row: hashlib.sha256("\x1f".join(row.tolist()).encode("utf-8")).hexdigest(), axis=1
    )


def overlap_diagnostics(
    real_source: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, Any]:
    real_hashes = set(_row_hashes(real_source, feature_columns))
    result: dict[str, Any] = {"pools": {}, "pairwise_content_overlap": {}}
    pool_hashes: dict[str, set[str]] = {}
    for name, frame in pools.items():
        hashes = _row_hashes(frame, feature_columns)
        unique = set(hashes)
        pool_hashes[name] = unique
        result["pools"][name] = {
            "rows": len(frame),
            "duplicate_rows": int(hashes.duplicated().sum()),
            "duplicate_rate": float(hashes.duplicated().mean()),
            "real_exact_overlap_rows": int(hashes.isin(real_hashes).sum()),
            "real_exact_overlap_rate": float(hashes.isin(real_hashes).mean()),
        }
    names = sorted(pools)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result["pairwise_content_overlap"][f"{left}__{right}"] = len(
                pool_hashes[left] & pool_hashes[right]
            )
    return result


def format_diagnostics(frame: pd.DataFrame, spec: DatasetSpec) -> dict[str, Any]:
    model = model_frame(frame)
    missing_columns = sorted((set(spec.feature_columns) | {spec.target_column}) - set(model.columns))
    target_domain = set(model[spec.target_column].dropna().astype(str))
    string_spaces = 0
    for column in spec.categorical_columns:
        values = model[column].dropna().astype(str)
        string_spaces += int((values != values.str.strip()).sum())
    numeric_parse_failures = 0
    for column in spec.numeric_columns:
        numeric_parse_failures += int(pd.to_numeric(model[column], errors="coerce").isna().sum() - model[column].isna().sum())
    return {
        "schema_valid": not missing_columns,
        "missing_columns": missing_columns,
        "target_values": sorted(target_domain),
        "string_outer_space_count": string_spaces,
        "numeric_parse_failures": numeric_parse_failures,
        "valid_rate": float((numeric_parse_failures == 0 and not missing_columns)),
        "missing_rates": {column: float(model[column].isna().mean()) for column in model.columns},
    }


def sdmetrics_quality(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    from sdmetrics.reports.single_table import QualityReport
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real)
    report = QualityReport()
    report.generate(real, synthetic, metadata.to_dict(), verbose=False)
    properties = report.get_properties()
    return {
        "overall": float(report.get_score()),
        "properties": {
            str(row["Property"]): float(row["Score"]) for _, row in properties.iterrows()
        },
    }


def _classifier(spec: DatasetSpec) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                list(spec.numeric_columns),
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(spec.categorical_columns),
            ),
        ]
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=42))]
    )


def utility_smoke(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: DatasetSpec,
) -> dict[str, Any]:
    x_train = model_frame(train).loc[:, list(spec.feature_columns)].copy()
    y_train = model_frame(train)[spec.target_column].astype(str)
    x_test = model_frame(test).loc[:, list(spec.feature_columns)].copy()
    y_test = model_frame(test)[spec.target_column].astype(str)
    if y_train.nunique() < 2:
        return {
            "status": "failed_single_class_train",
            "train_classes": sorted(y_train.unique().tolist()),
            "auroc": None,
            "f1": None,
            "balanced_accuracy": None,
            "fit_and_infer_seconds": 0.0,
            "train_rows": len(train),
            "test_rows": len(test),
        }
    for column in spec.categorical_columns:
        x_train[column] = x_train[column].astype(object).where(x_train[column].notna(), np.nan)
        x_test[column] = x_test[column].astype(object).where(x_test[column].notna(), np.nan)
    for column in spec.numeric_columns:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
    model = _classifier(spec)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = model.predict(x_test)
    positive = sorted(y_test.unique())[-1]
    binary = (y_test == positive).astype(int)
    return {
        "status": "ok",
        "auroc": float(roc_auc_score(binary, probability)),
        "f1": float(f1_score(y_test, prediction, pos_label=positive)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "fit_and_infer_seconds": time.perf_counter() - started,
        "train_rows": len(train),
        "test_rows": len(test),
    }
