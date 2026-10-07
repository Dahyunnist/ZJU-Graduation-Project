"""E1C: paired Credit class-conditional structure perturbation with real-data control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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


def _numeric_correlation_error(reference: pd.DataFrame, candidate: pd.DataFrame,
                               target: str, numeric: list[str]) -> float:
    if len(numeric) < 2:
        return float("nan")
    errors = []
    upper = np.triu_indices(len(numeric), 1)
    for label in reference[target].astype(str).unique():
        real_part = reference.loc[reference[target].astype(str) == label, numeric]
        candidate_part = candidate.loc[candidate[target].astype(str) == label, numeric]
        a = real_part.corr().to_numpy()[upper]
        b = candidate_part.corr().to_numpy()[upper]
        difference = np.abs(a - b)
        errors.extend(difference[np.isfinite(difference)].tolist())
    return float(np.mean(errors)) if errors else float("nan")


def run_case(config_path: Path, registry_path: Path | None, output_root: Path,
             generator: str, seed: int, *, preflight: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    table = config["table"]
    if generator not in config["generators"] or seed not in config["seeds"]:
        raise ValueError("Case is not in the frozen E1C design")
    if registry_path is None:
        registry_path = (config_path.parent / config["registry_path"]).resolve()
    registry = pd.read_csv(registry_path)
    selected = registry.loc[(registry.table_id == table) & (registry.generator == generator)]
    if len(selected) != 1:
        raise ValueError("Expected exactly one registry entry")
    entry = selected.iloc[0]
    target = str(entry.target_column)
    real_path = registry_path.parent / str(entry.real_path)
    synthetic_path = registry_path.parent / str(entry.synthetic_path)
    real = pd.read_csv(real_path)
    synthetic_pool = pd.read_csv(synthetic_path)
    features = _feature_columns(real, target)
    if set(features + [target]) - set(synthetic_pool.columns):
        raise ValueError("Synthetic/real schema mismatch")
    core_idx, reserve_idx, test_idx, positive = paired_split(
        real, target, seed, float(config["test_fraction"]), int(config["train_size"]))
    calibration_idx, donor_idx = _fractional_reserve(
        real, reserve_idx, target, positive, table, seed,
        int(config["calibration_size"]), int(config["real_donor_size"]))
    partitions = [set(core_idx), set(calibration_idx), set(donor_idx), set(test_idx)]
    if any(partitions[i] & partitions[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError("Information split overlap")
    core = real.iloc[core_idx].copy().reset_index(drop=True)
    validation = real.iloc[calibration_idx].copy()
    test = real.iloc[test_idx].copy()
    count = round(len(core) * float(config["prevalence"]))
    if len(donor_idx) != count:
        raise ValueError("Real donor budget must equal replacement count")
    slots = np.random.default_rng(_seed(seed, table, config["prevalence"], "slots")).choice(
        np.arange(len(core)), count, replace=False)
    selected_synthetic = np.random.default_rng(_seed(seed, table, generator,
                                                      config["prevalence"], "natural")).choice(
        synthetic_pool.index.to_numpy(), count, replace=False)
    natural = synthetic_pool.loc[selected_synthetic].copy()
    synthetic_shuffle = class_shuffle(
        natural, target, features,
        _seed(seed, table, generator, config["prevalence"], "class_shuffle"))
    real_donor = real.loc[donor_idx, core.columns].copy()
    real_shuffle = class_shuffle(real_donor, target, features,
                                 _seed(seed, table, "real_donor", config["prevalence"], "class_shuffle"))
    for original, shuffled in [(natural, synthetic_shuffle), (real_donor, real_shuffle)]:
        if original[target].astype(str).value_counts().to_dict() != shuffled[target].astype(str).value_counts().to_dict():
            raise AssertionError("Target distribution changed")
        for label in original[target].astype(str).unique():
            mask = original[target].astype(str) == label
            for feature in features:
                if sorted(original.loc[mask, feature].astype(str).tolist()) != sorted(
                        shuffled.loc[mask, feature].astype(str).tolist()):
                    raise AssertionError("Within-class marginal changed")
    sources = {"natural": natural, "class_shuffle": synthetic_shuffle,
               "independent_real_donor": real_donor, "real_donor_class_shuffle": real_shuffle}
    if list(sources) != config["conditions"]:
        raise AssertionError("Condition matrix changed")
    conditions = {}
    for condition, source in sources.items():
        mixed = core.copy()
        mixed.iloc[slots] = source[core.columns].to_numpy()
        conditions[condition] = mixed
    numeric = core[features].select_dtypes(include=["number", "bool"]).columns.tolist()
    structure = {condition: {"class_numeric_corr_error":
                             _numeric_correlation_error(core, source, target, numeric),
                             "positive_rate": float(_labels(source, target, positive).mean())}
                 for condition, source in sources.items()}
    threshold = float(pd.to_numeric(core[config["tail_feature"]], errors="coerce").quantile(
        config["tail_quantile"]))
    context = {"table": table, "generator": generator, "seed": seed, "target": target,
               "positive_label": positive, "core_ids": [int(i) for i in core_idx],
               "validation_ids": [int(i) for i in calibration_idx],
               "donor_ids": [int(i) for i in donor_idx], "test_ids": [int(i) for i in test_idx],
               "replace_positions": [int(i) for i in slots],
               "synthetic_pool_ids": [int(i) for i in selected_synthetic],
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
            raise ValueError("Completed shard has different config or source data")
        return {"status": "already_complete", "shard": final.name}
    temp = output_root / f".{final.name}.partial"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=False)
    _write_json(temp / "context.json", context)
    began = time.monotonic()
    y_validation = _labels(validation, target, positive)
    y_test = _labels(test, target, positive)
    tail = (pd.to_numeric(test[config["tail_feature"]], errors="coerce") >= threshold).fillna(False).to_numpy()
    rows = []
    for learner in config["learners"]:
        for condition, train in conditions.items():
            y_train = _labels(train, target, positive)
            if len(np.unique(y_train)) != 2:
                raise ValueError(f"Training became one-class: {condition}")
            model = _model(learner, train[features], _seed(seed, table, learner))
            start = time.monotonic()
            model.fit(train[features], y_train)
            p_validation = model.predict_proba(validation[features])[:, 1]
            p_test = model.predict_proba(test[features])[:, 1]
            duration = time.monotonic() - start
            clip = float(config["calibration"]["input_clip"])
            calibrator = _fit_calibrator(p_validation, y_validation, clip,
                                         float(config["calibration"]["regularization_C"]))
            p_test_cal = _apply_calibrator(calibrator, p_test, clip)
            identifier = f"{table}_{generator}_{seed}_{learner}_{condition}"
            (temp / "models").mkdir(exist_ok=True)
            (temp / "predictions").mkdir(exist_ok=True)
            model_file = temp / "models" / f"{identifier}.joblib"
            joblib.dump({"model": model, "calibrator": calibrator, "features": features,
                         "positive_label": positive}, model_file, compress=3)
            pred_file = temp / "predictions" / f"{identifier}.csv.gz"
            pd.DataFrame({"row_id": [f"{table}:real:{int(i)}" for i in test_idx],
                          "y_true": y_test, "tail_group": tail.astype(int),
                          "p_raw": p_test, "p_calibrated": p_test_cal}).to_csv(
                              pred_file, index=False, compression="gzip")
            validation_file = temp / "predictions" / f"{identifier}_validation.csv.gz"
            pd.DataFrame({"row_id": [f"{table}:real:{int(i)}" for i in calibration_idx],
                          "y_true": y_validation, "p_raw": p_validation,
                          "p_calibrated": _apply_calibrator(calibrator, p_validation, clip)}).to_csv(
                              validation_file, index=False, compression="gzip")
            for mode, probability in [("raw", p_test), ("calibrated", p_test_cal)]:
                rows.append({"table": table, "generator": generator, "seed": seed,
                             "model": learner, "condition": condition, "mode": mode,
                             "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
                             "train_positive_rate": float(y_train.mean()),
                             "train_seconds": round(duration, 4),
                             "calibration_slope": float(calibrator.coef_[0, 0]),
                             "calibration_intercept": float(calibrator.intercept_[0]),
                             "model_path": str(model_file.relative_to(temp)),
                             "predictions_path": str(pred_file.relative_to(temp)),
                             "validation_path": str(validation_file.relative_to(temp)),
                             **_scores(y_test, probability, tail)})
    if len(rows) != 16:
        raise AssertionError("Incomplete fit matrix")
    pd.DataFrame(rows).to_csv(temp / "metrics.csv", index=False)
    result = {"status": "complete", "context": context, "fits": 8,
              "metric_rows": 16, "wall_seconds": round(time.monotonic() - began, 3)}
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
