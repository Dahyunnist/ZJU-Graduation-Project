"""E2 minimal source-calibration -> decision -> downstream-risk experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpollution.detectors.classical import C2STDetector
from tabpollution.governance.data import RegistrySource, exact_mixture
from tabpollution.governance.metrics import select_fpr_threshold, select_negative_anchor_threshold
from tabpollution.governance.pipeline import _PlattCalibrator, _collect, _split
from tabpollution.quantification.methods import ScoreQuantifier
from tabpollution.studies.e1 import _feature_columns, _labels, _model, _seed, sha256_file
from tabpollution.studies.e1b import _apply_calibrator, _fit_calibrator, _write_json


def top_k(scores: np.ndarray, count: int) -> np.ndarray:
    """Stable descending score order with a fixed index tie breaker."""
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or not 0 <= count <= len(values):
        raise ValueError("Invalid top-k request")
    return np.argsort(-values, kind="stable")[:count]


def oracle_top_k(source_labels: np.ndarray, count: int) -> np.ndarray:
    """Best *source-identity* ranking at a fixed budget, not task-optimal removal."""
    return top_k(np.asarray(source_labels, dtype=float), count)


def _task_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) != 2:
        raise ValueError("Real test must contain both classes")
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "positive_log_loss": float(log_loss(y[y == 1], p[y == 1], labels=[0, 1]))}


def _fit_task(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
              target: str, positive: str, features: list[str], seed: int,
              clip: float, C: float, destination: Path) -> dict[str, Any]:
    y_train = _labels(train, target, positive)
    if len(train) < 10 or len(np.unique(y_train)) < 2:
        return {"status": "undefined_training_class", "metrics": None}
    y_validation = _labels(validation, target, positive)
    y_test = _labels(test, target, positive)
    model = _model("lr", train[features], seed)
    model.fit(train[features], y_train)
    p_validation = model.predict_proba(validation[features])[:, 1]
    p_test = model.predict_proba(test[features])[:, 1]
    raw = _task_scores(y_test, p_test)
    task_calibrator = None
    calibrated = None
    p_calibrated = np.full(len(p_test), np.nan)
    calibration_status = "ok"
    try:
        task_calibrator = _fit_calibrator(p_validation, y_validation, clip, C)
        p_calibrated = _apply_calibrator(task_calibrator, p_test, clip)
        calibrated = _task_scores(y_test, p_calibrated)
    except ValueError as exc:
        calibration_status = f"failed:{exc}"
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump({"model": model, "task_calibrator": task_calibrator,
                 "positive_label": positive, "features": features}, destination / "model.joblib", compress=3)
    pd.DataFrame({"row_id": test.record_id.astype(str), "y_true": y_test,
                  "p_raw": p_test, "p_calibrated": p_calibrated}).to_csv(
                      destination / "predictions.csv.gz", index=False, compression="gzip")
    return {"status": "ok", "task_calibration_status": calibration_status,
            "metrics": {"raw": raw, "calibrated": calibrated},
            "prediction_path": str((destination / "predictions.csv.gz").name)}


def _policy(name: str, raw_source: np.ndarray, source_labels: np.ndarray,
            raw_target: np.ndarray, target_labels: np.ndarray,
            raw_anchor: np.ndarray, seed: int, target_fpr: float) -> dict[str, Any]:
    if name in {"oracle_target", "oracle_target_reoriented"}:
        calibrator = _PlattCalibrator(seed + 2).fit(raw_target, target_labels)
        reference_raw, reference_labels = raw_target, target_labels
    else:
        calibrator = _PlattCalibrator(seed).fit(raw_source, source_labels)
        reference_raw, reference_labels = raw_source, source_labels
    if calibrator.constant is not None or (name != "oracle_target_reoriented" and calibrator.model.coef_[0, 0] <= 0):
        raise ValueError(f"Non-increasing {name} calibration; rank-invariance unavailable")
    reference_scores = calibrator.predict(reference_raw)
    if name == "target_real_anchor":
        selected = select_negative_anchor_threshold(calibrator.predict(raw_anchor), target_fpr)
    else:
        selected = select_fpr_threshold(reference_labels, reference_scores, target_fpr)
    quantifiers = {method: ScoreQuantifier(method).fit(reference_scores, reference_labels,
                                                     threshold=selected["threshold"])
                   for method in ["pacc", "pcc"]}
    return {"name": name, "calibrator": calibrator,
            "threshold": float(selected["threshold"]),
            "reference_fpr": float(selected["validation_fpr"]),
            "quantifiers": quantifiers,
            "uses_target_synthetic_labels": name in {"oracle_target", "oracle_target_reoriented"}}


def run_seed(config_path: Path, registry_path: Path | None, output_root: Path,
             seed: int, *, preflight: bool = False) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if seed not in cfg["seeds"] or cfg["detector"] != "c2st_lr" or cfg["downstream_model"] != "lr":
        raise ValueError("Seed/method not in frozen E2 design")
    if registry_path is None:
        registry_path = (config_path.parent / cfg["registry_path"]).resolve()
    source = RegistrySource(registry_path)
    table_id = str(cfg["table"])
    table = source.table(table_id)
    source_generator = str(cfg["source_generator"])
    target_generator = str(cfg["target_generator"])
    real_splits = _split(table.real, seed)
    target_splits = _split(table.synthetic[target_generator], seed + 31)
    source_real = real_splits["source"]
    anchor_n = int(cfg["target_real_anchor_size"])
    task_cal_n = int(cfg["task_calibration_size"])
    if len(source_real) < anchor_n + task_cal_n:
        raise ValueError("Source real split too small for disjoint anchor and task calibration")
    anchor = source_real.iloc[:anchor_n].copy()
    task_validation = source_real.iloc[anchor_n:anchor_n+task_cal_n].copy()
    detector_train, train_labels = _collect(source, (table_id,), (source_generator,), "detector_train", seed)
    detector_val, val_labels = _collect(source, (table_id,), (source_generator,), "detector_val", seed)
    target_val, target_labels = _collect(source, (table_id,), (target_generator,), "detector_val", seed)
    tune_idx, cal_idx = train_test_split(np.arange(len(detector_val)), test_size=.5,
                                          random_state=_seed(seed, "e2_source_cal"), stratify=val_labels)
    test = real_splits["final_test"]
    real_train = real_splits["downstream_train"]
    synthetic_train = target_splits["downstream_train"]
    partitions = [set(frame.record_id.astype(str)) for frame in
                  [anchor, task_validation, real_train, test]]
    if any(partitions[i] & partitions[j] for i in range(4) for j in range(i+1, 4)):
        raise AssertionError("Real data role overlap")
    features = _feature_columns(table.real, table.target_column)
    positive = sorted(table.real[table.target_column].astype(str).unique())[-1]
    if preflight:
        return {"status": "preflight_ok", "seed": seed,
                "detector_train_n": len(detector_train), "detector_tune_n": len(tune_idx),
                "detector_cal_n": len(cal_idx), "target_oracle_val_n": len(target_val),
                "real_anchor_n": len(anchor), "task_validation_n": len(task_validation),
                "real_bag_pool_n": len(real_train), "synthetic_bag_pool_n": len(synthetic_train),
                "test_n": len(test)}
    root = output_root / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    dataset_registry = pd.read_csv(registry_path)
    selected_rows = dataset_registry.loc[dataset_registry.table_id == table_id]
    source_paths = [registry_path.parent / str(selected_rows.real_path.iloc[0])]
    source_paths.extend(registry_path.parent / str(row.synthetic_path) for row in selected_rows.itertuples())
    context = {"seed": seed, "table": table_id, "source_generator": source_generator,
               "target_generator": target_generator,
               "config_sha256": sha256_file(config_path), "registry_sha256": sha256_file(registry_path),
               "implementation_sha256": sha256_file(Path(__file__)),
               "source_files_sha256": {str(path): sha256_file(path) for path in source_paths},
               "detector_train_ids": detector_train.record_id.astype(str).tolist(),
               "detector_tune_ids": detector_val.iloc[tune_idx].record_id.astype(str).tolist(),
               "detector_cal_ids": detector_val.iloc[cal_idx].record_id.astype(str).tolist(),
               "target_oracle_val_ids": target_val.record_id.astype(str).tolist(),
               "target_real_anchor_ids": anchor.record_id.astype(str).tolist(),
               "task_validation_ids": task_validation.record_id.astype(str).tolist(),
               "real_bag_pool_ids": real_train.record_id.astype(str).tolist(),
               "synthetic_bag_pool_ids": synthetic_train.record_id.astype(str).tolist(),
               "real_test_ids": test.record_id.astype(str).tolist()}
    existing_context = root / "context.json"
    if existing_context.exists():
        previous = json.loads(existing_context.read_text(encoding="utf-8"))
        if previous != context:
            raise ValueError("E2 seed output has different data, code or config")
    else:
        _write_json(existing_context, context)
    detector = C2STDetector("lr", seed=seed).fit(
        detector_train, train_labels, detector_val.iloc[tune_idx], val_labels[tune_idx])
    raw_source = detector.predict_score(detector_val.iloc[cal_idx])
    raw_target = detector.predict_score(target_val)
    raw_anchor = detector.predict_score(anchor)
    policies = {}
    policy_failures = {}
    for name in cfg["calibration_policies"]:
        try:
            policies[name] = _policy(name, raw_source, val_labels[cal_idx], raw_target, target_labels,
                                     raw_anchor, seed, float(cfg["target_fpr"]))
        except ValueError as exc:
            if name != "oracle_target":
                raise
            # A reversed target relation is a scientific failure of a
            # rank-preserving oracle calibration, not a reason to flip scores.
            policy_failures[name] = str(exc)
    _write_json(root / "calibration_diagnostics.json", {
        "source_raw_auroc": float(roc_auc_score(val_labels[cal_idx], raw_source)),
        "target_raw_auroc": float(roc_auc_score(target_labels, raw_target)),
        "policy_slopes": {name: float(policy["calibrator"].model.coef_[0, 0])
                          for name, policy in policies.items()},
        "policy_failures": policy_failures,
        "available_policies": list(policies),
    })
    detector.save(root / "detector.pkl")
    joblib.dump({name: {"calibrator": policy["calibrator"], "threshold": policy["threshold"],
                        "quantifiers": policy["quantifiers"]} for name, policy in policies.items()},
                root / "policies.joblib", compress=3)
    completed = 0
    tail_threshold = float(pd.to_numeric(real_train[cfg["tail_feature"]], errors="coerce").quantile(
        cfg["tail_quantile"]))
    for prevalence in cfg["prevalences"]:
        for bag_index in range(int(cfg["bags_per_rate"])):
            bag_id = f"{table_id}_{target_generator}_seed{seed}_p{float(prevalence):.2f}_b{bag_index}"
            final = root / bag_id
            if (final / "complete.json").exists():
                old = json.loads((final / "complete.json").read_text(encoding="utf-8"))
                if old["config_sha256"] != context["config_sha256"] or old["implementation_sha256"] != context["implementation_sha256"]:
                    raise ValueError(f"Completed bag belongs to another design: {bag_id}")
                completed += 1
                continue
            temp = root / f".{bag_id}.partial.{os.getpid()}"
            temp.mkdir(parents=True, exist_ok=False)
            began = time.monotonic()
            bag_seed = seed + int(round(float(prevalence)*1000))*17 + bag_index
            bag = exact_mixture(real_train, synthetic_train, int(cfg["bag_size"]),
                                float(prevalence), bag_seed, mode="replace")
            if bag.record_id.nunique() != len(bag):
                raise ValueError("Bag sampled duplicate record IDs; row position remains unique but evidence must be revised")
            raw_bag = detector.predict_score(bag)
            fixed_n = int(cfg["fixed_delete_count"])
            raw_top = top_k(raw_bag, fixed_n)
            source_labels = bag.source_label.to_numpy(dtype=int)
            rng = np.random.default_rng(_seed(seed, prevalence, bag_index, "e2_random"))
            random_order = rng.permutation(len(bag))
            oracle_order = oracle_top_k(source_labels, len(bag))
            all_positions = np.arange(len(bag))
            retained_cache: dict[tuple[int, ...], dict[str, Any]] = {}
            actions = []
            quant_rows = []
            score_rows = pd.DataFrame({"row_position": all_positions,
                                       "record_id": bag.record_id.astype(str),
                                       "source_label": source_labels,
                                       "task_label": bag[table.target_column].astype(str),
                                       "raw_score": raw_bag})

            def record_action(policy_name: str, action_name: str, removed: np.ndarray,
                              estimate: float | None, triggered: bool | None) -> None:
                removed = np.asarray(removed, dtype=int)
                if len(np.unique(removed)) != len(removed):
                    raise AssertionError("Duplicate removal index")
                keep = np.setdiff1d(all_positions, removed, assume_unique=True)
                cache_key = tuple(int(i) for i in keep)
                if cache_key not in retained_cache:
                    fingerprint = hashlib.sha256(np.asarray(keep, dtype=np.int64).tobytes()).hexdigest()[:16]
                    destination = temp / "task_models" / fingerprint
                    retained_cache[cache_key] = {"fingerprint": fingerprint,
                        **_fit_task(bag.iloc[keep].copy(), task_validation, test,
                                   table.target_column, positive, features,
                                   _seed(seed, prevalence, bag_index, "task_lr"),
                                   float(cfg["task_probability_clip"]),
                                   float(cfg["task_calibration_C"]), destination)}
                task = retained_cache[cache_key]
                removed_sources = source_labels[removed]
                kept = bag.iloc[keep]
                actions.append({"bag_id": bag_id, "seed": seed, "prevalence": prevalence,
                    "bag_index": bag_index, "policy": policy_name, "action": action_name,
                    "pacc_estimate": estimate, "decision_triggered": triggered,
                    "removed_n": len(removed), "removed_synthetic_n": int(removed_sources.sum()),
                    "removed_source_precision": float(removed_sources.mean()) if len(removed) else None,
                    "removed_positions": json.dumps(removed.tolist()),
                    "remaining_positive_rate": float(_labels(kept, table.target_column, positive).mean()),
                    "remaining_tail_rate": float((pd.to_numeric(kept[cfg["tail_feature"]], errors="coerce") >= tail_threshold).mean()),
                    "task_fingerprint": task["fingerprint"], "task_status": task["status"],
                    "task_calibration_status": task.get("task_calibration_status"),
                    **{f"{mode}_{metric}": (task["metrics"][mode] or {}).get(metric)
                       if task["metrics"] else None
                       for mode in ["raw", "calibrated"] for metric in
                       ["log_loss", "brier", "auroc", "auprc", "positive_log_loss"]}})

            record_action("shared", "keep_all", np.array([], dtype=int), None, None)
            record_action("shared", "random_fixed_k", random_order[:fixed_n], None, None)
            record_action("shared", "source_oracle_fixed_k", oracle_order[:fixed_n], None, None)
            record_action("shared", "raw_score_fixed_k", raw_top, None, None)
            for name, policy in policies.items():
                calibrated_scores = policy["calibrator"].predict(raw_bag)
                score_rows[f"score_{name}"] = calibrated_scores
                calibrated_top = top_k(calibrated_scores, fixed_n)
                if policy["calibrator"].model.coef_[0, 0] > 0 and not np.array_equal(calibrated_top, raw_top):
                    raise AssertionError(f"Fixed top-k rank invariance failed for {name}: {bag_id}")
                record_action(name, "calibrated_fixed_k", calibrated_top, None, None)
                estimates = {}
                for method, quantifier in policy["quantifiers"].items():
                    try:
                        estimate = float(quantifier.predict_prevalence(calibrated_scores)["clipped"])
                        status = "ok"
                    except ValueError as exc:
                        estimate = float("nan")
                        status = f"failed:{exc}"
                    estimates[method] = estimate
                    quant_rows.append({"bag_id": bag_id, "policy": name, "quantifier": method,
                                       "status": status, "estimated_prevalence": estimate,
                                       "true_prevalence": float(source_labels.mean()),
                                       "threshold": policy["threshold"],
                                       "reference_fpr": policy["reference_fpr"],
                                       "uses_target_synthetic_labels": policy["uses_target_synthetic_labels"]})
                main_estimate = estimates["pacc"]
                if not np.isfinite(main_estimate):
                    actions.append({"bag_id": bag_id, "seed": seed, "prevalence": prevalence,
                                    "bag_index": bag_index, "policy": name, "action": "decision_policy",
                                    "task_status": "undefined_pacc", "pacc_estimate": None})
                    continue
                triggered = main_estimate >= float(cfg["decision_prevalence_threshold"])
                removed = np.flatnonzero(calibrated_scores >= policy["threshold"]) if triggered else np.array([], dtype=int)
                record_action(name, "decision_policy", removed, main_estimate, triggered)
                if triggered:
                    count = len(removed)
                    record_action(name, "random_matched_n", random_order[:count], main_estimate, triggered)
                    record_action(name, "source_oracle_matched_n", oracle_order[:count], main_estimate, triggered)
            for name, reason in policy_failures.items():
                for method in ["pacc", "pcc"]:
                    quant_rows.append({"bag_id": bag_id, "policy": name, "quantifier": method,
                                       "status": f"unavailable:{reason}",
                                       "estimated_prevalence": float("nan"),
                                       "true_prevalence": float(source_labels.mean()),
                                       "uses_target_synthetic_labels": True})
                actions.append({"bag_id": bag_id, "seed": seed, "prevalence": prevalence,
                                "bag_index": bag_index, "policy": name,
                                "action": "decision_policy", "task_status": "unavailable_calibration",
                                "calibration_failure": reason})
            score_rows.to_csv(temp / "bag_scores.csv.gz", index=False, compression="gzip")
            pd.DataFrame(actions).to_csv(temp / "actions.csv", index=False)
            pd.DataFrame(quant_rows).to_csv(temp / "quantification.csv", index=False)
            result = {"status": "complete", "bag_id": bag_id, "seed": seed,
                      "prevalence": prevalence, "bag_index": bag_index,
                      "bag_size": len(bag), "true_prevalence": float(source_labels.mean()),
                      "config_sha256": context["config_sha256"],
                      "implementation_sha256": context["implementation_sha256"],
                      "action_rows": len(actions), "quantifier_rows": len(quant_rows),
                      "unique_task_fits": len(retained_cache),
                      "wall_seconds": round(time.monotonic()-began, 3)}
            _write_json(temp / "complete.json", result)
            if final.exists():
                raise FileExistsError(final)
            temp.replace(final)
            completed += 1
            print(json.dumps({"status": "bag_complete", **{k: result[k] for k in
                              ["bag_id", "action_rows", "unique_task_fits", "wall_seconds"]}}), flush=True)
    _write_json(root / "seed_complete.json", {"status": "complete", "seed": seed,
                                              "completed_bags": completed,
                                              "expected_bags": len(cfg["prevalences"])*int(cfg["bags_per_rate"])})
    return {"status": "complete", "seed": seed, "completed_bags": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_seed(args.config.resolve(),
                              args.registry.resolve() if args.registry else None,
                              args.output.resolve(), args.seed, preflight=args.preflight),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
