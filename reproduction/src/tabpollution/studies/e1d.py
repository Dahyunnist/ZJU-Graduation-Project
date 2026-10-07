"""E1D: small, pre-registered Credit/RF structure kill-test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from tabpollution.studies.e1 import (_feature_columns, _labels, _model, _seed,
                                     class_shuffle, paired_split, sha256_file)
from tabpollution.studies.e1b import (_apply_calibrator, _fit_calibrator,
                                      _fractional_reserve, _scores, _write_json)
from tabpollution.studies.e1c import _numeric_correlation_error


def partial_class_shuffle(frame: pd.DataFrame, target: str, features: list[str],
                          fraction: float, seed: int) -> pd.DataFrame:
    """Shuffle each selected column within a fixed fraction of rows in each class."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    result = frame.copy()
    rng = np.random.default_rng(seed)
    labels = frame[target].astype(str).to_numpy()
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        chosen = rng.choice(positions, size=int(round(len(positions) * fraction)), replace=False)
        for feature in features:
            index = result.columns.get_loc(feature)
            result.iloc[chosen, index] = rng.permutation(frame.iloc[chosen][feature].to_numpy())
    return result


def _check_marginals(original: pd.DataFrame, changed: pd.DataFrame,
                     target: str, features: list[str]) -> None:
    if not original[target].astype(str).equals(changed[target].astype(str)):
        raise AssertionError("Label positions changed")
    labels = original[target].astype(str)
    for label in labels.unique():
        subset = labels == label
        for feature in features:
            if sorted(original.loc[subset, feature].astype(str)) != sorted(
                    changed.loc[subset, feature].astype(str)):
                raise AssertionError(f"Class-conditional marginal changed: {feature}")


def run_case(config_path: Path, registry_path: Path | None, output_root: Path,
             generator: str, seed: int, *, preflight: bool = False) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    table = str(cfg["table"])
    if generator not in cfg["generators"] or seed not in cfg["seeds"] or cfg["learner"] != "rf":
        raise ValueError("Case not in frozen E1D design")
    if registry_path is None:
        registry_path = (config_path.parent / cfg["registry_path"]).resolve()
    registry = pd.read_csv(registry_path)
    entries = registry.loc[(registry.table_id == table) & (registry.generator == generator)]
    if len(entries) != 1:
        raise ValueError("Expected one registry entry")
    entry = entries.iloc[0]
    target = str(entry.target_column)
    real_path = registry_path.parent / str(entry.real_path)
    synthetic_path = registry_path.parent / str(entry.synthetic_path)
    real = pd.read_csv(real_path)
    synthetic_pool = pd.read_csv(synthetic_path)
    features = _feature_columns(real, target)
    if set(features + [target]) - set(synthetic_pool.columns):
        raise ValueError("Synthetic/real schema mismatch")
    for block in cfg["feature_blocks"].values():
        if not set(block).issubset(features):
            raise ValueError(f"Missing feature block columns: {set(block)-set(features)}")
    core_idx, reserve_idx, test_idx, positive = paired_split(
        real, target, seed, float(cfg["test_fraction"]), int(cfg["train_size"]))
    calibration_idx, donor_idx = _fractional_reserve(
        real, reserve_idx, target, positive, table, seed,
        int(cfg["calibration_size"]), int(cfg["real_donor_size"]))
    groups = [set(core_idx), set(calibration_idx), set(donor_idx), set(test_idx)]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i+1, 4)):
        raise AssertionError("Information split overlap")
    core = real.iloc[core_idx].copy().reset_index(drop=True)
    validation = real.iloc[calibration_idx].copy()
    test = real.iloc[test_idx].copy()
    count = round(len(core) * float(cfg["prevalence"]))
    if count != len(donor_idx):
        raise ValueError("Donor count does not match replacement budget")
    slots = np.random.default_rng(_seed(seed, table, cfg["prevalence"], "slots")).choice(
        np.arange(len(core)), count, replace=False)
    syn_ids = np.random.default_rng(_seed(seed, table, generator, cfg["prevalence"], "natural")).choice(
        synthetic_pool.index.to_numpy(), count, replace=False)
    natural = synthetic_pool.loc[syn_ids].copy()
    donor = real.loc[donor_idx, core.columns].copy()
    balance = list(cfg["feature_blocks"]["balances"])
    status = list(cfg["feature_blocks"]["payment_status"])
    natural_variants = {
        "natural": natural,
        "global_half": partial_class_shuffle(natural, target, features, .5,
                                             _seed(seed, table, generator, "global_half")),
        "payment_status_full": class_shuffle(natural, target, status,
                                             _seed(seed, table, generator, "payment_status_full")),
        "balances_full": class_shuffle(natural, target, balance,
                                       _seed(seed, table, generator, "balances_full")),
    }
    donor_variants = {
        "independent_real_donor": donor,
        "real_global_half": partial_class_shuffle(donor, target, features, .5,
                                                   _seed(seed, table, "real_donor", "global_half")),
        "real_payment_status_full": class_shuffle(donor, target, status,
                                                   _seed(seed, table, "real_donor", "payment_status_full")),
        "real_balances_full": class_shuffle(donor, target, balance,
                                            _seed(seed, table, "real_donor", "balances_full")),
    }
    for changed in natural_variants.values():
        _check_marginals(natural, changed, target, features)
    for changed in donor_variants.values():
        _check_marginals(donor, changed, target, features)
    sources = {**natural_variants, **donor_variants}
    if list(sources) != cfg["conditions"]:
        raise AssertionError("Frozen condition matrix changed")
    threshold = float(pd.to_numeric(core[cfg["tail_feature"]], errors="coerce").quantile(
        cfg["tail_quantile"]))
    numeric = core[features].select_dtypes(include=["number", "bool"]).columns.tolist()
    structure = {condition: {"class_numeric_corr_error":
                             _numeric_correlation_error(core, source, target, numeric),
                             "positive_rate": float(_labels(source, target, positive).mean())}
                 for condition, source in sources.items()}
    context = {"table": table, "generator": generator, "seed": seed, "target": target,
               "positive_label": positive, "core_ids": [int(i) for i in core_idx],
               "validation_ids": [int(i) for i in calibration_idx],
               "donor_ids": [int(i) for i in donor_idx], "test_ids": [int(i) for i in test_idx],
               "replace_positions": [int(i) for i in slots],
               "synthetic_pool_ids": [int(i) for i in syn_ids],
               "tail_threshold": threshold, "structure": structure,
               "config_sha256": sha256_file(config_path),
               "registry_sha256": sha256_file(registry_path),
               "real_sha256": sha256_file(real_path),
               "synthetic_sha256": sha256_file(synthetic_path)}
    if preflight:
        return {"status": "preflight_ok", "generator": generator, "seed": seed,
                "core_n": len(core), "validation_n": len(validation),
                "donor_n": len(donor_idx), "test_n": len(test),
                "positive_validation_n": int(_labels(validation, target, positive).sum())}
    final = output_root / f"{table}_{generator}_seed{seed}"
    if (final / "complete.json").exists():
        saved = json.loads((final / "complete.json").read_text(encoding="utf-8"))
        if any(saved["context"][field] != context[field] for field in
               ["config_sha256", "registry_sha256", "real_sha256", "synthetic_sha256"]):
            raise ValueError("Completed shard has changed source/config")
        return {"status": "already_complete", "shard": final.name}
    temp = output_root / f".{final.name}.partial.{os.getpid()}"
    temp.mkdir(parents=True, exist_ok=False)
    _write_json(temp / "context.json", context)
    started = time.monotonic()
    y_test = _labels(test, target, positive)
    y_validation = _labels(validation, target, positive)
    tail = (pd.to_numeric(test[cfg["tail_feature"]], errors="coerce") >= threshold).fillna(False).to_numpy()
    records = []
    for condition, source in sources.items():
        train = core.copy()
        train.iloc[slots] = source[core.columns].to_numpy()
        y_train = _labels(train, target, positive)
        model = _model("rf", train[features], _seed(seed, table, "rf"))
        start = time.monotonic()
        model.fit(train[features], y_train)
        p_validation = model.predict_proba(validation[features])[:, 1]
        p_test = model.predict_proba(test[features])[:, 1]
        duration = time.monotonic() - start
        clip = float(cfg["calibration"]["input_clip"])
        calibrator = _fit_calibrator(p_validation, y_validation, clip,
                                     float(cfg["calibration"]["regularization_C"]))
        p_test_cal = _apply_calibrator(calibrator, p_test, clip)
        identifier = f"{table}_{generator}_{seed}_rf_{condition}"
        (temp / "models").mkdir(exist_ok=True)
        (temp / "predictions").mkdir(exist_ok=True)
        model_file = temp / "models" / f"{identifier}.joblib"
        joblib.dump({"model": model, "calibrator": calibrator,
                     "features": features, "positive_label": positive}, model_file, compress=3)
        pred_file = temp / "predictions" / f"{identifier}.csv.gz"
        pd.DataFrame({"row_id": [f"{table}:real:{int(i)}" for i in test_idx],
                      "y_true": y_test, "tail_group": tail.astype(int),
                      "p_raw": p_test, "p_calibrated": p_test_cal}).to_csv(
                          pred_file, index=False, compression="gzip")
        for mode, probability in [("raw", p_test), ("calibrated", p_test_cal)]:
            records.append({"table": table, "generator": generator, "seed": seed,
                            "model": "rf", "condition": condition, "mode": mode,
                            "n_train": len(train), "n_validation": len(validation),
                            "n_test": len(test), "train_positive_rate": float(y_train.mean()),
                            "train_seconds": round(duration, 4),
                            "calibration_slope": float(calibrator.coef_[0, 0]),
                            "calibration_intercept": float(calibrator.intercept_[0]),
                            "model_path": str(model_file.relative_to(temp)),
                            "predictions_path": str(pred_file.relative_to(temp)),
                            **_scores(y_test, probability, tail)})
    if len(records) != 16:
        raise AssertionError("Incomplete E1D matrix")
    pd.DataFrame(records).to_csv(temp / "metrics.csv", index=False)
    result = {"status": "complete", "context": context, "fits": 8,
              "metric_rows": 16, "wall_seconds": round(time.monotonic()-started, 3)}
    _write_json(temp / "complete.json", result)
    if final.exists():
        raise FileExistsError(final)
    temp.replace(final)
    return {"status": "complete", "shard": final.name, "fits": 8, "metric_rows": 16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_case(args.config.resolve(),
                              args.registry.resolve() if args.registry else None,
                              args.output.resolve(), args.generator, args.seed,
                              preflight=args.preflight), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
