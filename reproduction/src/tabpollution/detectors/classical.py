"""Leakage-safe classical C2ST and character 3-gram baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import Detector, feature_frame, serialize_record


class C2STDetector(Detector):
    def __init__(self, algorithm: str = "lr", seed: int = 2026):
        if algorithm not in {"lr", "xgb"}:
            raise ValueError("algorithm must be lr or xgb")
        self.algorithm, self.seed = algorithm, seed
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.best_params: dict[str, Any] = {}

    def _pipeline(self, frame: pd.DataFrame, params: dict[str, Any]) -> Pipeline:
        numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [c for c in frame.columns if c not in numeric]
        prep = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])
        if self.algorithm == "lr":
            estimator = LogisticRegression(C=params["C"], max_iter=600, random_state=self.seed)
        else:
            from xgboost import XGBClassifier
            estimator = XGBClassifier(
                n_estimators=params.get("n_estimators", 80), max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", .08), subsample=.9,
                colsample_bytree=.9, eval_metric="logloss", random_state=self.seed,
                n_jobs=2, tree_method="hist",
            )
        return Pipeline([("preprocess", prep), ("model", estimator)])

    def fit(self, train_records: pd.DataFrame, train_labels: np.ndarray,
            val_records: pd.DataFrame | None = None, val_labels: np.ndarray | None = None,
            **context: Any) -> "C2STDetector":
        x = feature_frame(train_records)
        self.feature_columns = x.columns.tolist()
        candidates = ([{"C": c} for c in (.01, .1, 1., 10.)] if self.algorithm == "lr" else [
            {"n_estimators": 60, "max_depth": 3, "learning_rate": .1},
            {"n_estimators": 100, "max_depth": 5, "learning_rate": .06},
        ])
        best_auc, best = -1., None
        for params in candidates:
            model = self._pipeline(x, params)
            model.fit(x, train_labels)
            score = (roc_auc_score(val_labels, model.predict_proba(feature_frame(val_records)[self.feature_columns])[:, 1])
                     if val_records is not None and val_labels is not None else 0.)
            if score > best_auc:
                best_auc, best, self.best_params = score, model, params
        self.model = best
        return self

    def predict_score(self, records: pd.DataFrame, **context: Any) -> np.ndarray:
        return self.model.predict_proba(feature_frame(records)[self.feature_columns])[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "C2STDetector":
        with Path(path).open("rb") as f: return pickle.load(f)

    def get_provenance(self) -> dict[str, Any]:
        return {"implementation": "local_sklearn_xgboost", "algorithm": self.algorithm,
                "seed": self.seed, "best_params": self.best_params,
                "feature_columns": self.feature_columns}


class Char3GramDetector(Detector):
    def __init__(self, seed: int = 2026, max_features: int = 30000):
        self.seed, self.max_features = seed, max_features
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3),
                                          max_features=max_features, min_df=2)
        self.model = LogisticRegression(max_iter=500, random_state=seed)

    @staticmethod
    def texts(records: pd.DataFrame, shuffle: bool = True, seed: int = 2026) -> list[str]:
        frame = feature_frame(records)
        return [serialize_record(row, shuffle=shuffle, seed=seed + i) for i, (_, row) in enumerate(frame.iterrows())]

    def fit(self, train_records: pd.DataFrame, train_labels: np.ndarray,
            val_records: pd.DataFrame | None = None, val_labels: np.ndarray | None = None,
            **context: Any) -> "Char3GramDetector":
        mat = self.vectorizer.fit_transform(self.texts(train_records, True, self.seed))
        self.model.fit(mat, train_labels)
        return self

    def predict_score(self, records: pd.DataFrame, **context: Any) -> np.ndarray:
        shuffle = bool(context.get("shuffle", False))
        return self.model.predict_proba(self.vectorizer.transform(self.texts(records, shuffle, self.seed + 10000)))[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Char3GramDetector":
        with Path(path).open("rb") as f: return pickle.load(f)

    def get_provenance(self) -> dict[str, Any]:
        return {"implementation": "paper_aligned_self_implementation", "paper_baseline": "D1/D2 char-3gram LR",
                "vocabulary_size": len(getattr(self.vectorizer, "vocabulary_", {})), "seed": self.seed}
