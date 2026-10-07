"""E1B: paired learner/probability diagnostics using disjoint real calibration data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpollution.studies.e1 import _labels, _model, _seed, _feature_columns, label_permute, paired_split, sha256_file


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _fractional_reserve(real: pd.DataFrame, reserve_idx: np.ndarray, target: str, positive: str,
                        table: str, seed: int, calibration_size: int, donor_size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(reserve_idx) < calibration_size + donor_size:
        raise ValueError("Real reserve too small for disjoint calibration and donor samples")
    calibration, remaining = train_test_split(
        reserve_idx, train_size=calibration_size,
        random_state=_seed(seed, table, "calibration_500"),
        stratify=_labels(real.iloc[reserve_idx], target, positive),
    )
    donors = np.random.default_rng(_seed(seed, table, .5, "diagnostic_real_donor")).choice(
        remaining, size=donor_size, replace=False,
    )
    assert not (set(calibration) & set(donors))
    return calibration, donors


def _scores(y: np.ndarray, probability: np.ndarray, tail: np.ndarray) -> dict[str, float | int]:
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probability")
    eps = np.finfo(float).eps
    bounded = np.clip(p, eps, 1 - eps)
    losses = -y * np.log(bounded) - (1 - y) * np.log1p(-bounded)
    worst = max(1, int(np.ceil(len(y) * .01)))
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "positive_log_loss": float(log_loss(y[y == 1], p[y == 1], labels=[0, 1])),
        "negative_log_loss": float(log_loss(y[y == 0], p[y == 0], labels=[0, 1])),
        "tail_log_loss": float(log_loss(y[tail], p[tail], labels=[0, 1])),
        "clipped_1e3_log_loss": float(log_loss(y, np.clip(p, .001, .999), labels=[0, 1])),
        "wrong_endpoint_count": int((((y == 1) & (p == 0)) | ((y == 0) & (p == 1))).sum()),
        "top1pct_loss_share": float(np.sort(losses)[-worst:].sum() / losses.sum()),
    }


def _fit_calibrator(raw_validation: np.ndarray, y_validation: np.ndarray,
                    input_clip: float, C: float) -> LogisticRegression:
    if len(np.unique(y_validation)) != 2:
        raise ValueError("Calibration fold contains one class")
    x = logit(np.clip(raw_validation, input_clip, 1 - input_clip)).reshape(-1, 1)
    calibrated = LogisticRegression(C=C, solver="lbfgs", max_iter=500)
    calibrated.fit(x, y_validation)
    if calibrated.coef_[0, 0] <= 0:
        raise ValueError("Non-monotone calibration slope; cannot interpret score-rank invariance")
    return calibrated


def _apply_calibrator(calibrator: LogisticRegression, raw: np.ndarray, clip: float) -> np.ndarray:
    x = logit(np.clip(raw, clip, 1 - clip)).reshape(-1, 1)
    probability = calibrator.predict_proba(x)[:, 1]
    return np.clip(probability, clip, 1 - clip)


def run_case(config_path: Path, registry_path: Path | None, output_root: Path,
             table: str, generator: str, seed: int, *, preflight: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if {"table": table, "generator": generator} not in config["cases"] or seed not in config["seeds"]:
        raise ValueError("Case is not in frozen E1B design")
    if registry_path is None:
        registry_path = (config_path.parent / config["registry_path"]).resolve()
    registry = pd.read_csv(registry_path)
    rows = registry.loc[(registry.table_id == table) & (registry.generator == generator)]
    if len(rows) != 1:
        raise ValueError("Expected exactly one registry row")
    entry = rows.iloc[0]
    target = str(entry.target_column)
    real_path = registry_path.parent / str(entry.real_path)
    synthetic_path = registry_path.parent / str(entry.synthetic_path)
    real = pd.read_csv(real_path)
    synthetic_pool = pd.read_csv(synthetic_path)
    features = _feature_columns(real, target)
    if set(features + [target]) - set(synthetic_pool.columns):
        raise ValueError("Schema mismatch")
    core_idx, reserve_idx, test_idx, positive = paired_split(
        real, target, seed, float(config["test_fraction"]), int(config["train_size"]),
    )
    calibration_idx, donor_idx = _fractional_reserve(
        real, reserve_idx, target, positive, table, seed,
        int(config["calibration_size"]), int(config["real_donor_size"]),
    )
    assert not (set(core_idx) & set(test_idx) or set(core_idx) & set(calibration_idx)
                or set(core_idx) & set(donor_idx) or set(test_idx) & set(calibration_idx)
                or set(test_idx) & set(donor_idx))
    core, validation, test = real.iloc[core_idx].copy(), real.iloc[calibration_idx].copy(), real.iloc[test_idx].copy()
    prevalence = float(config["prevalence"])
    count = round(len(core) * prevalence)
    if count != len(donor_idx):
        raise ValueError("Donor budget must equal exact replacement count")
    positions = np.random.default_rng(_seed(seed, table, prevalence, "slots")).choice(np.arange(len(core)), count, replace=False)
    source_indices = np.random.default_rng(_seed(seed, table, generator, prevalence, "natural")).choice(
        synthetic_pool.index.to_numpy(), count, replace=False,
    )
    natural = synthetic_pool.loc[source_indices].copy()
    permuted = label_permute(natural, target, _seed(seed, table, generator, prevalence, "label_permute"))
    core = core.reset_index(drop=True)
    conditions = {"real_reference": core.copy()}
    conditions["independent_real_donor"] = core.copy()
    conditions["independent_real_donor"].iloc[positions] = real.loc[donor_idx, core.columns].to_numpy()
    conditions["natural"] = core.copy()
    conditions["natural"].iloc[positions] = natural[core.columns].to_numpy()
    conditions["label_permute"] = core.copy()
    conditions["label_permute"].iloc[positions] = permuted[core.columns].to_numpy()
    if list(conditions) != config["conditions"]:
        raise ValueError("Frozen conditions changed")
    tail_feature = str(config["tail_features"][table])
    threshold = float(pd.to_numeric(core[tail_feature], errors="coerce").quantile(config["tail_quantile"]))
    context = {
        "table": table, "generator": generator, "seed": seed, "target": target,
        "positive_label": positive, "core_ids": [int(i) for i in core_idx],
        "validation_ids": [int(i) for i in calibration_idx],
        "donor_ids": [int(i) for i in donor_idx], "test_ids": [int(i) for i in test_idx],
        "replace_positions": [int(i) for i in positions],
        "synthetic_pool_ids": [int(i) for i in source_indices],
        "tail_feature": tail_feature, "tail_threshold": threshold,
        "config_sha256": sha256_file(config_path), "registry_sha256": sha256_file(registry_path),
        "real_sha256": sha256_file(real_path), "synthetic_sha256": sha256_file(synthetic_path),
    }
    if preflight:
        return {"status": "preflight_ok", "table": table, "generator": generator, "seed": seed,
                "core_n": len(core), "validation_n": len(validation), "donor_n": len(donor_idx), "test_n": len(test),
                "positive_validation_n": int(_labels(validation, target, positive).sum())}
    final = output_root / f"{table}_{generator}_seed{seed}"
    if (final / "complete.json").exists():
        saved = json.loads((final / "complete.json").read_text(encoding="utf-8"))
        same = all(saved["context"][field] == context[field] for field in
                   ["config_sha256", "registry_sha256", "real_sha256", "synthetic_sha256"])
        if not same:
            raise ValueError("Completed shard has different data or design hashes")
        return {"status": "already_complete", "shard": final.name}
    temp = output_root / f".{final.name}.partial"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=False)
    _write_json(temp / "context.json", context)
    start = time.monotonic()
    y_valid = _labels(validation, target, positive)
    y_test = _labels(test, target, positive)
    tail = (pd.to_numeric(test[tail_feature], errors="coerce") >= threshold).fillna(False).to_numpy()
    rows_out: list[dict[str, Any]] = []
    for learner, settings in config["learners"].items():
        for leaf in settings:
            name = learner if learner == "lr" else f"rf_leaf{leaf}"
            for condition, train in conditions.items():
                y_train = _labels(train, target, positive)
                if len(np.unique(y_train)) != 2:
                    raise ValueError(f"Training became single-class: {condition}")
                model = _model(learner, train[features], _seed(seed, table, learner))
                if learner == "rf":
                    model.set_params(classifier__min_samples_leaf=int(leaf))
                train_start = time.monotonic()
                model.fit(train[features], y_train)
                p_val = model.predict_proba(validation[features])[:, 1]
                p_test = model.predict_proba(test[features])[:, 1]
                duration = time.monotonic() - train_start
                cfg = config["calibration"]
                clip = float(cfg["input_clip"])
                calibrator = _fit_calibrator(p_val, y_valid, clip, float(cfg["regularization_C"]))
                p_test_cal = _apply_calibrator(calibrator, p_test, clip)
                p_val_cal = _apply_calibrator(calibrator, p_val, clip)
                identifier = f"{table}_{generator}_{seed}_{name}_{condition}"
                save_dir = temp / "models"
                save_dir.mkdir(exist_ok=True)
                model_file = save_dir / f"{identifier}.joblib"
                joblib.dump({"model": model, "calibrator": calibrator,
                             "features": features, "positive_label": positive}, model_file, compress=3)
                pred_dir = temp / "predictions"
                pred_dir.mkdir(exist_ok=True)
                pred_file = pred_dir / f"{identifier}.csv.gz"
                pd.DataFrame({"row_id": [f"{table}:real:{int(i)}" for i in test_idx],
                              "y_true": y_test, "tail_group": tail.astype(int),
                              "p_raw": p_test, "p_calibrated": p_test_cal}).to_csv(pred_file, index=False, compression="gzip")
                valid_file = pred_dir / f"{identifier}_validation.csv.gz"
                pd.DataFrame({"row_id": [f"{table}:real:{int(i)}" for i in calibration_idx],
                              "y_true": y_valid, "p_raw": p_val, "p_calibrated": p_val_cal}).to_csv(
                                  valid_file, index=False, compression="gzip")
                for mode, probability in [("raw", p_test), ("calibrated", p_test_cal)]:
                    rows_out.append({
                        "table": table, "generator": generator, "seed": seed,
                        "model": name, "condition": condition, "mode": mode,
                        "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
                        "train_positive_rate": float(y_train.mean()), "train_seconds": round(duration, 4),
                        "calibration_slope": float(calibrator.coef_[0, 0]),
                        "calibration_intercept": float(calibrator.intercept_[0]),
                        "model_path": str(model_file.relative_to(temp)),
                        "predictions_path": str(pred_file.relative_to(temp)),
                        "validation_path": str(valid_file.relative_to(temp)),
                        **_scores(y_test, probability, tail),
                    })
    expected_fits = sum(len(settings) for settings in config["learners"].values()) * len(conditions)
    if len(rows_out) != expected_fits * 2:
        raise AssertionError("Incomplete condition matrix")
    metrics = pd.DataFrame(rows_out)
    metrics.to_csv(temp / "metrics.csv", index=False)
    summary = {"status": "complete", "context": context, "fits": expected_fits,
               "metric_rows": len(metrics), "wall_seconds": round(time.monotonic() - start, 3)}
    _write_json(temp / "complete.json", summary)
    if final.exists():
        raise FileExistsError(final)
    temp.replace(final)
    return {"status": "complete", "shard": final.name, "fits": expected_fits,
            "metric_rows": len(metrics), "wall_seconds": summary["wall_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run_case(arguments.config.resolve(),
                              arguments.registry.resolve() if arguments.registry else None,
                              arguments.output.resolve(), arguments.table,
                              arguments.generator, arguments.seed, preflight=arguments.preflight),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
