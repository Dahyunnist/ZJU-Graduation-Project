"""Shared aggregate-only utility core. No file reader, downloader or real-run CLI.

Real execution must supply separately authorized, identity-checked role frames.
The public CLI only plans or exercises generated artificial records.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpollution.generators.sdv_adapter import create_generator, SDVGenerator

TASKS = ("binary", "multiclass", "regression")
LEARNERS = ("linear", "rf", "histgb", "mlp")
GENERATORS = ("Bootstrap", "GaussianCopula", "CTGAN", "TVAE")
RESERVED = {"row_id", "record_id", "split", "split_role", "source_type", "source_row_id"}
PROTOCOL_ID = "broad-utility-v1-preparation"


def stable_seed(seed, *keys):
    text = ":".join(map(str, (seed, *keys)))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def validate_frames(train, evaluation, *, target, numeric, categorical, task, classes):
    if task not in TASKS:
        raise ValueError("unsupported task")
    features = list(numeric) + list(categorical)
    if not features or len(features) != len(set(features)):
        raise ValueError("feature schema empty/overlapping")
    if target in features or set(features) & RESERVED:
        raise ValueError("target/provenance leakage in features")
    expected = {"record_id", target, *features}
    for name, frame in (("train", train), ("evaluation", evaluation)):
        if set(frame.columns) != expected or frame.columns.duplicated().any() or frame.empty:
            raise ValueError(f"{name} schema or empty role")
        if frame.record_id.isna().any() or frame.record_id.astype(str).duplicated().any():
            raise ValueError(f"{name} invalid identity")
        if frame[target].isna().any():
            raise ValueError("missing target")
        for col in numeric:
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError("infinite numeric feature")
    if set(train.record_id.astype(str)) & set(evaluation.record_id.astype(str)):
        raise ValueError("training/evaluation identity overlap")
    if task == "regression":
        if classes:
            raise ValueError("regression cannot declare classes")
        for f in (train, evaluation):
            if not np.isfinite(pd.to_numeric(f[target], errors="raise")).all():
                raise ValueError("nonfinite regression target")
    else:
        if len(classes) != len(set(classes)) or len(classes) < 2:
            raise ValueError("declare unique target classes")
        if task == "binary" and len(classes) != 2 or task == "multiclass" and len(classes) < 3:
            raise ValueError("task/class mismatch")
        if set(train[target].astype(str)) != set(classes):
            raise ValueError("training class support incomplete")
        if not set(evaluation[target].astype(str)) <= set(classes):
            raise ValueError("unknown evaluation label")
    return features


def build_model(task, learner, numeric, categorical, seed):
    branches = []
    if numeric:
        branches.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), list(numeric)))
    if categorical:
        branches.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     min_frequency=2, max_categories=64)),
        ]), list(categorical)))
    prep = ColumnTransformer(branches)
    classification = task != "regression"
    if learner == "linear":
        model = LogisticRegression(max_iter=1000, C=1, random_state=seed) if classification else Ridge(alpha=1)
    elif learner == "rf":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        model = cls(n_estimators=100, min_samples_leaf=2, n_jobs=1, random_state=seed)
    elif learner == "histgb":
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        model = cls(max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=seed)
    elif learner == "mlp":
        cls = MLPClassifier if classification else MLPRegressor
        model = cls(hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=False,
                    random_state=seed)
    else:
        raise ValueError("unknown learner")
    return Pipeline([("preprocess", prep), ("task", model)])


def _label_indices(y, classes):
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[str(v)] for v in y], dtype=int)


def metrics(y, predictions, task, classes):
    """Multiclass Brier is sum over classes; binary Brier is positive-class MSE."""
    if task == "regression":
        y = np.asarray(y, dtype=float)
        pred = np.asarray(predictions, dtype=float)
        return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "mae": float(mean_absolute_error(y, pred)),
                "r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None}
    labels = _label_indices(y, classes)
    p = np.asarray(predictions)
    if p.shape != (len(labels), len(classes)) or not np.isfinite(p).all():
        raise ValueError("invalid probability shape/values")
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("probability normalization")
    predicted = p.argmax(axis=1)
    onehot = np.eye(len(classes))[labels]
    confidence = p.max(axis=1)
    correct = predicted == labels
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else confidence <= 1)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    result = {"log_loss": float(loss_rows(y, p, task, classes).mean()),
              "brier": float(np.mean((p[:, 1] - labels) ** 2)) if task == "binary"
              else float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
              "accuracy": float(accuracy_score(labels, predicted)),
              "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
              "macro_f1": float(f1_score(labels, predicted, labels=list(range(len(classes))),
                                         average="macro", zero_division=0)),
              "ece_toplabel_10_equal_width": ece, "roc_auc": None, "average_precision": None}
    if len(np.unique(labels)) == len(classes):
        result["roc_auc"] = float(roc_auc_score(labels, p[:, 1])) if task == "binary" else float(
            roc_auc_score(labels, p, multi_class="ovr", average="macro"))
    if task == "binary" and len(np.unique(labels)) == 2:
        result["average_precision"] = float(average_precision_score(labels, p[:, 1]))
    return result


def loss_rows(y, predictions, task, classes):
    if task == "regression":
        return (np.asarray(y, dtype=float) - predictions) ** 2
    labels = _label_indices(y, classes)
    # One explicit rule for aggregate, groups, bootstrap and relative-risk denominator.
    p = np.clip(np.asarray(predictions, dtype=np.float64), np.finfo(np.float64).eps, 1.)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(labels)), labels])


def group_masks(train, evaluation, target, numeric, categorical, task, classes, group_features):
    """All target-tail/group definitions use real training records only."""
    groups = {"overall": np.ones(len(evaluation), dtype=bool)}
    if task == "regression":
        lo, hi = np.quantile(train[target].to_numpy(dtype=float), [.1, .9])
        groups["target_bottom10_train"] = evaluation[target].to_numpy(dtype=float) <= lo
        groups["target_top10_train"] = evaluation[target].to_numpy(dtype=float) >= hi
    else:
        for i, c in enumerate(classes):
            groups[f"class_{i}"] = evaluation[target].astype(str).to_numpy() == c
    definitions = {}
    for col in group_features:
        if col not in set(numeric) | set(categorical):
            raise ValueError("group feature outside schema")
        if col in numeric:
            values = pd.to_numeric(train[col], errors="raise").dropna()
            if values.empty:
                continue
            lo, hi = np.quantile(values, [.1, .9])
            definitions[col] = {"kind": "numeric_train_quantiles", "q10": float(lo), "q90": float(hi)}
            ev = evaluation[col].to_numpy(dtype=float)
            groups[f"{col}:bottom10"] = ev <= lo
            groups[f"{col}:top10"] = ev >= hi
        else:
            counts = train[col].fillna("__MISSING__").astype(str).value_counts()
            if len(counts) > 12:
                raise ValueError("group cardinality >12; predefine a coarser mapping")
            levels = sorted(counts.index.tolist())
            definitions[col] = {"kind": "categorical_train_levels", "levels": levels}
            ev = evaluation[col].fillna("__MISSING__").astype(str).to_numpy()
            for i, level in enumerate(levels):
                groups[f"{col}:level_{i}"] = ev == level
            groups[f"{col}:unseen"] = ~np.isin(ev, levels)
    return groups, definitions


def _tv(left, right):
    a = left.fillna("__MISSING__").astype(str).value_counts(normalize=True)
    b = right.fillna("__MISSING__").astype(str).value_counts(normalize=True)
    return float(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a.index) | set(b.index)) / 2)


def property_summary(real, synthetic, target, numeric, categorical, task, classes):
    """Prespecified descriptive properties; never a causal mediation estimate."""
    out = {"numeric": {}, "categorical": {}, "target": {}, "joint_numeric": {},
           "synthetic_exact_duplicate_fraction": float(synthetic.duplicated().mean()),
           "task_association_numeric": {}}
    for col in numeric:
        a = pd.to_numeric(real[col], errors="raise").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(synthetic[col], errors="raise").dropna().to_numpy(dtype=float)
        scale = float(np.subtract(*np.quantile(a, [.75, .25]))) if len(a) else None
        out["numeric"][col] = {
            "missing_rate_delta": float(synthetic[col].isna().mean() - real[col].isna().mean()),
            "w1_iqr_normalized": float(wasserstein_distance(a, b) / scale) if len(a) and len(b) and scale > 1e-12 else None,
            "normalization_status": "valid" if scale is not None and scale > 1e-12 else "empty_or_degenerate_train_iqr",
            "zero_rate_delta": float((synthetic[col] == 0).mean() - (real[col] == 0).mean()),
            "tail_outside_train_q01_q99": float(np.mean((b < np.quantile(a, .01)) | (b > np.quantile(a, .99))))
            if len(a) and len(b) else None,
        }
    for col in categorical:
        levels = set(real[col].fillna("__MISSING__").astype(str))
        out["categorical"][col] = {"tv": _tv(real[col], synthetic[col]),
                                  "unseen_fraction": float((~synthetic[col].fillna("__MISSING__").astype(str).isin(levels)).mean())}
    if task != "regression":
        out["target"]["class_prior_tv"] = _tv(real[target], synthetic[target])
        out["target"]["missing_classes"] = len(set(classes) - set(synthetic[target].astype(str)))
    else:
        out["target"]["wasserstein"] = float(wasserstein_distance(real[target], synthetic[target]))
    if len(numeric) >= 2:
        a = real[list(numeric)].corr(method="spearman").to_numpy()
        b = synthetic[list(numeric)].corr(method="spearman").to_numpy()
        idx = np.triu_indices(len(numeric), 1)
        good = np.isfinite(a[idx]) & np.isfinite(b[idx])
        out["joint_numeric"] = {"valid_pairs": int(good.sum()),
                                "mean_abs_spearman_delta": float(np.mean(abs(a[idx][good] - b[idx][good]))) if good.any() else None}
    for col in numeric:
        associations = []
        for frame in (real, synthetic):
            if task == "regression":
                value = frame[[col, target]].corr(method="spearman").iloc[0, 1]
            else:
                valid = frame[col].notna()
                values = frame.loc[valid, col].astype(float)
                total = float(np.sum((values - values.mean()) ** 2))
                between = sum(len(g) * (g[col].mean() - values.mean()) ** 2
                              for _, g in frame.loc[valid].groupby(target))
                value = between / total if total > 0 else np.nan
            associations.append(float(value) if np.isfinite(value) else None)
        a, b = associations
        out["task_association_numeric"][col] = {"real": a, "synthetic": b,
            "delta": b-a if a is not None and b is not None else None,
            "kind": "spearman_target" if task == "regression" else "class_eta_squared"}
    return out


def make_arms(train, synthetic, *, target, features, seed, replacement=(.25, .5), augmentation=(.5, 1.0), dose_rounding="half_up"):
    """Same information budget: generator fitted only on this real train role."""
    n = len(train)
    if dose_rounding not in {"half_up", "floor"}:
        raise ValueError("unsupported dose rounding")
    def dose_count(dose):
        return int(math.floor(n * dose + .5)) if dose_rounding == "half_up" else int(math.floor(n * dose))
    columns = [*features, target]
    if set(synthetic.columns) != set(columns) or len(synthetic) < n:
        raise ValueError("synthetic schema/size mismatch; strip provenance explicitly in adapter")
    real = train[columns].reset_index(drop=True)
    syn = synthetic[columns].reset_index(drop=True)
    rng = np.random.default_rng(stable_seed(seed, "arms"))
    positions = rng.permutation(n)
    # Prefixes give nested doses; no class quota repair conceals generator failure.
    syn_positions = rng.permutation(len(syn))
    boot_positions = rng.integers(0, n, size=max(n, int(math.ceil(n * max(augmentation, default=1)))))
    arms = {"TRTR": real.copy(), "TSTR": syn.iloc[syn_positions[:n]].reset_index(drop=True),
            "bootstrap_TSTR": real.iloc[boot_positions[:n]].reset_index(drop=True)}
    for dose in replacement:
        if not 0 < dose <= 1:
            raise ValueError("replacement dose outside (0,1]")
        k = dose_count(dose)
        shuffled = real.copy()
        shuffled.iloc[positions[:k]] = syn.iloc[syn_positions[:k]].to_numpy()
        boot = real.copy()
        boot.iloc[positions[:k]] = real.iloc[boot_positions[:k]].to_numpy()
        arms[f"replace_{dose:g}"] = shuffled
        arms[f"bootstrap_replace_{dose:g}"] = boot
    for dose in augmentation:
        if not 0 < dose <= 1:
            raise ValueError("augmentation dose outside (0,1]")
        k = dose_count(dose)
        arms[f"augment_{dose:g}"] = pd.concat([real, syn.iloc[syn_positions[:k]]], ignore_index=True)
        arms[f"bootstrap_augment_{dose:g}"] = pd.concat([real, real.iloc[boot_positions[:k]]], ignore_index=True)
    return arms


def paired_interval(reference, candidate, *, seed, reps=2000, task="binary"):
    if reps < 20:
        raise ValueError("at least20 bootstrap replicates")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        idx = rng.integers(0, len(reference), size=len(reference))
        a, b = reference[idx].mean(), candidate[idx].mean()
        estimates.append(np.sqrt(b) - np.sqrt(a) if task == "regression" else b - a)
    point = np.sqrt(candidate.mean()) - np.sqrt(reference.mean()) if task == "regression" else candidate.mean() - reference.mean()
    return {"delta": float(point), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "replicates": reps, "conditional_on_fixed_fits": True}


def evaluate_case(train, evaluation, synthetic, *, target, numeric, categorical, task,
                  classes=(), group_features=(), learners=LEARNERS, seed=2026,
                  bootstrap_reps=2000, replacement=(.25, .5), augmentation=(.5, 1.0),
                  dose_rounding="half_up", progress_callback=None, fitted_control_cache=None):
    features = validate_frames(train, evaluation, target=target, numeric=numeric,
                               categorical=categorical, task=task, classes=classes)
    if len(learners) != len(set(learners)) or not set(learners) <= set(LEARNERS):
        raise ValueError("invalid learner set")
    groups, definitions = group_masks(train, evaluation, target, numeric, categorical, task, classes, group_features)
    if synthetic[target].isna().any():
        raise ValueError("missing generated target")
    for col in numeric:
        if np.isinf(pd.to_numeric(synthetic[col], errors="raise").to_numpy(dtype=float)).any():
            raise ValueError("infinite generated feature")
    if task == "regression" and not np.isfinite(pd.to_numeric(synthetic[target], errors="raise")).all():
        raise ValueError("nonfinite generated target")
    arms = make_arms(train, synthetic, target=target, features=features, seed=seed,
                     replacement=replacement, augmentation=augmentation, dose_rounding=dose_rounding)
    out = {"protocol": PROTOCOL_ID, "task": task, "seed": seed,
           "group_definitions_from_train_only": definitions,
           "properties": property_summary(train[[*features, target]], synthetic, target, numeric, categorical, task, classes),
           "arm_properties": {arm: property_summary(train[[*features, target]], data, target, numeric, categorical, task, classes)
                              for arm, data in arms.items()},
           "results": [], "model_calls": 0, "row_outputs": False,
           "interpretation": "fixed-fit exploratory; no generator-population or causal claim"}
    identity = hashlib.sha256()
    for f in (train, evaluation):
        identity.update(pd.util.hash_pandas_object(f[["record_id", *features, target]], index=False).to_numpy().tobytes())
    identity.update(repr((task, classes, numeric, categorical, group_features, seed, bootstrap_reps,
                          replacement, augmentation, dose_rounding)).encode())
    cache_context = identity.hexdigest()
    for learner in learners:
        baseline_losses = None
        baseline_primary = None
        arm_losses = {}
        learner_rows = []
        for arm, data in arms.items():
            row = {"learner": learner, "arm": arm, "train_n": len(data),
                   "properties": out["arm_properties"][arm], "status": "pending"}
            control_key = (cache_context, learner, arm)
            is_control = arm == "TRTR" or arm.startswith("bootstrap_")
            cached = fitted_control_cache.get(control_key) if fitted_control_cache is not None and is_control else None
            if cached is not None:
                row = copy.deepcopy(cached["row"])
                row["properties"] = out["arm_properties"][arm]
                row["fit_reused"] = True
                primary = cached["primary"].copy()
                if arm == "TRTR":
                    baseline_losses = primary.copy()
                    baseline_primary = row["metrics"]["rmse" if task == "regression" else "log_loss"]
                arm_losses[arm] = primary
                out["results"].append(row)
                learner_rows.append(row)
                continue
            if task != "regression" and set(data[target].astype(str)) != set(classes):
                observed = set(data[target].astype(str))
                expected = set(classes)
                row.update(status="invalid_training_class_support", metrics=None,
                    class_support={"expected_class_count": len(expected),
                        "observed_class_count": len(observed),
                        "missing_class_count": len(expected - observed),
                        "unexpected_class_count": len(observed - expected)})
                out["results"].append(row)
                continue
            model = build_model(task, learner, numeric, categorical, stable_seed(seed, learner))
            y = data[target].astype(str) if task != "regression" else data[target].astype(float)
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out["model_calls"] += 1
                if progress_callback is not None:
                    progress_callback({"event": "task_fit_started", "local_fit_calls": out["model_calls"],
                                       "learner": learner, "arm": arm, "train_n": len(data)})
                model.fit(data[features], y)
            if task == "regression":
                predictions = model.predict(evaluation[features])
            else:
                probs = model.predict_proba(evaluation[features])
                order = [list(model.classes_).index(c) for c in classes]
                predictions = probs[:, order]
            primary = loss_rows(evaluation[target], predictions, task, classes)
            arm_losses[arm] = primary
            row.update(status="complete", fit_seconds=time.monotonic() - started,
                       warning_types=sorted({type(w.message).__name__ for w in caught}),
                       metrics=metrics(evaluation[target], predictions, task, classes), groups={})
            if arm == "TRTR":
                baseline_losses = primary.copy()
                baseline_primary = row["metrics"]["rmse" if task == "regression" else "log_loss"]
            if baseline_losses is None:
                raise RuntimeError("reference not first")
            for name, mask in groups.items():
                n = int(mask.sum())
                group = {"n": n, "small_group": n < 20, "primary_loss": None, "comparison": None}
                if n:
                    group["primary_loss"] = float(np.sqrt(primary[mask].mean()) if task == "regression" else primary[mask].mean())
                    group["comparison"] = paired_interval(baseline_losses[mask], primary[mask],
                                                          seed=stable_seed(seed, "bootstrap", name),
                                                          reps=bootstrap_reps, task=task)
                row["groups"][name] = group
            delta = row["groups"]["overall"]["comparison"]["delta"]
            row["relative_delta_vs_TRTR"] = delta / baseline_primary if baseline_primary > 1e-12 else None
            if task == "binary":
                yi = _label_indices(evaluation[target], classes)
                dec = predictions[:, 1] >= .5
                fp, fn = int(np.sum(dec & (yi == 0))), int(np.sum(~dec & (yi == 1)))
                row["decision_sensitivity"] = {"threshold": .5, "fp": fp, "fn": fn,
                    "cost_fn_over_fp": {str(c): float((fp + c * fn) / len(yi)) for c in (1, 2, 5, 10)},
                    "status": "prespecified_sensitivity_not_domain_utility"}
            if fitted_control_cache is not None:
                row["fit_reused"] = False
                if is_control:
                    fitted_control_cache[control_key] = {"row": copy.deepcopy(row), "primary": primary.copy()}
            out["results"].append(row)
            learner_rows.append(row)
        for row in learner_rows:
            arm = row["arm"]
            control = "bootstrap_TSTR" if arm == "TSTR" else (
                "bootstrap_" + arm if arm.startswith(("replace_", "augment_")) else None)
            if control and control in arm_losses:
                row["matched_resampling_control"] = {
                    "arm": control,
                    **paired_interval(arm_losses[control], arm_losses[arm], task=task,
                                      seed=stable_seed(seed, "bootstrap", "overall"), reps=bootstrap_reps),
                }
    return out


def evaluate_pool_matrix(train, evaluation, pools, **kwargs):
    """Reuse identical real/resampling fits across generator pools within one seed.

    Cache is in-memory and context-binds full role values/IDs and analysis settings.
    It never reuses generator predictions, generator fits, or cross-seed controls.
    """
    if not pools or set(pools)-set(GENERATORS):
        raise ValueError("declare supported generator pools")
    cache, cases = {}, {}
    for name in sorted(pools):
        cases[name] = evaluate_case(train, evaluation, pools[name], fitted_control_cache=cache, **kwargs)
    return {"protocol": PROTOCOL_ID, "cases": cases,
            "actual_downstream_model_calls": sum(c["model_calls"] for c in cases.values()),
            "reference_control_cache_entries": len(cache), "row_outputs": False}


def run_generator_matrix(train, evaluation, *, target, numeric, categorical, task,
                         classes=(), group_features=(), generator_names=GENERATORS,
                         generator_factory=None, metadata_builder=None, seed=2026,
                         learners=LEARNERS, bootstrap_reps=2000,
                         replacement=(.25, .5), augmentation=(.5, 1.0),
                         dose_rounding="half_up", progress_callback=None):
    """Execute one dataset/seed unit through generator adapters and shared utility.

    Generator training receives only declared training features and target, never record IDs,
    evaluation values, tune rows, or final-test rows. Factories/builders are injectable for
    artificial integration tests; learned generators remain lazily imported by their adapter.
    """
    if not generator_names or len(generator_names) != len(set(generator_names)) or not set(generator_names) <= set(GENERATORS):
        raise ValueError("generator names must be unique supported routes")
    features = validate_frames(train, evaluation, target=target, numeric=numeric,
                               categorical=categorical, task=task, classes=classes)
    generator_factory = generator_factory or create_generator
    metadata_builder = metadata_builder or SDVGenerator.build_metadata
    columns = [*features, target]
    generator_train = train[columns].copy().reset_index(drop=True)
    pools, provenance = {}, {}
    for name in generator_names:
        generator = generator_factory(name)
        metadata = metadata_builder(generator_train) if name != "Bootstrap" else None
        fit_seed = stable_seed(seed, name, "generator_fit")
        sample_seed = stable_seed(seed, name, "generator_sample")
        generator.fit(generator_train, metadata=metadata, seed=fit_seed)
        sample = generator.sample(len(generator_train), sample_seed=sample_seed, pool_name=f"seed{seed}")
        if "record_id" in sample.columns:
            sample = sample.drop(columns="record_id")
        if set(sample.columns) != set(columns):
            raise ValueError(f"{name} sample schema differs from declared model schema")
        pools[name] = sample[columns].copy()
        provenance[name] = generator.get_provenance()
    evaluated = evaluate_pool_matrix(
        train, evaluation, pools, target=target, numeric=numeric, categorical=categorical,
        task=task, classes=classes, group_features=group_features, learners=learners,
        seed=seed, bootstrap_reps=bootstrap_reps, replacement=replacement,
        augmentation=augmentation, dose_rounding=dose_rounding, progress_callback=progress_callback)
    return {**evaluated, "unit": {"seed": seed, "task": task,
            "generator_names": list(generator_names), "generator_fit_calls": len(generator_names),
            "generator_provenance": provenance},
            "status": "complete", "row_outputs": False}


def artificial_records(task, *, seed=17, n=160):
    from sklearn.datasets import make_classification, make_regression
    if task == "regression":
        x, y = make_regression(n_samples=n, n_features=4, noise=15, random_state=seed)
    else:
        x, y = make_classification(n_samples=n, n_features=4, n_informative=3, n_redundant=0,
                                   n_classes=2 if task == "binary" else 3, n_clusters_per_class=1,
                                   random_state=seed)
        y = y.astype(str)
    frame = pd.DataFrame(x, columns=["x0", "x1", "x2", "x3"])
    frame["category"] = np.where(frame.x0 > 0, "a", "b")
    frame["target"] = y
    frame.insert(0, "record_id", [f"artificial:{task}:{i}" for i in range(n)])
    from sklearn.model_selection import train_test_split
    train, evaluation = train_test_split(frame, test_size=.3, random_state=seed,
                                         stratify=None if task == "regression" else frame.target)
    return train.reset_index(drop=True), evaluation.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artificial-check", action="store_true")
    p.add_argument("--artificial-generator", choices=("Bootstrap", "GaussianCopula"), default="Bootstrap")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not a.artificial_check:
        print(json.dumps({"status": "NO-RUN", "protocol": PROTOCOL_ID,
                          "tasks": TASKS, "learners": LEARNERS, "generators": GENERATORS,
                          "real_data_reader": False}))
        return
    if a.output is None or a.output.exists():
        raise ValueError("new output directory required")
    a.output.mkdir(parents=True)
    from threadpoolctl import threadpool_limits
    summaries = []
    with threadpool_limits(limits=1):
        for task in TASKS:
            train, evaluation = artificial_records(task)
            cols = [c for c in train if c != "record_id"]
            gen = create_generator(a.artificial_generator)
            meta = SDVGenerator.build_metadata(train[cols]) if a.artificial_generator == "GaussianCopula" else None
            gen.fit(train[cols], metadata=meta, seed=2026)
            synthetic = gen.sample(len(train), sample_seed=2027, pool_name="artificial")
            if a.artificial_generator == "Bootstrap":
                synthetic = synthetic.drop(columns="record_id")
            report = evaluate_case(train, evaluation, synthetic, target="target", numeric=("x0", "x1", "x2", "x3"),
                categorical=("category",), task=task, classes=() if task == "regression" else tuple(sorted(train.target.unique())),
                group_features=("x0", "category"), bootstrap_reps=100, replacement=(.5,), augmentation=(1.0,))
            summaries.append(report)
    result = {"artificial_only": True, "real_data_reads": 0,
              "artificial_generator": a.artificial_generator,
              "artificial_generator_fit_calls": 3 if a.artificial_generator == "GaussianCopula" else 0,
              "research_model_calls": 0, "artificial_model_calls": sum(x["model_calls"] for x in summaries),
              "cases": summaries,
              "versions": {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scikit-learn")}}
    (a.output / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "artificial_complete", "cases": len(summaries),
                      "artificial_model_calls": result["artificial_model_calls"], "real_data_reads": 0}))


if __name__ == "__main__":
    main()
