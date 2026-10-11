"""Frozen E3B target-domain detectability diagnosis.

The target evaluation labels are consumed only after every detector has been
fitted, its score direction fixed, and its operating threshold selected on the
training-domain calibration split.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from tabpollution.detectors.classical import C2STDetector
from tabpollution.governance.data import RegistrySource
from tabpollution.governance.metrics import select_fpr_threshold
from tabpollution.governance.pipeline import _split
from tabpollution.utils import sha256_file


LR_CELLS = {
    (train_generator, eval_generator)
    for train_generator in ("CTGAN", "TVAE")
    for eval_generator in ("CTGAN", "TVAE")
}
SCORE_COLUMNS = [
    "table", "seed", "detector", "train_generator", "eval_generator",
    "cell_id", "record_id", "split", "source_label", "model_direction", "raw_score",
]
METRIC_COLUMNS = [
    "table", "seed", "detector", "train_generator", "eval_generator", "cell_id",
    "auroc", "auprc", "score_mean_gap", "ks_distance", "tpr_at_calibrated_5pct_fpr",
    "observed_fpr", "calibration_threshold", "calibration_fpr", "calibration_tpr",
    "n", "n_real", "n_synthetic", "synthetic_fraction",
]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    partial.replace(path)


def _config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "study_id", "tables", "generators", "seeds", "primary_detector", "capacity_detector",
        "model_direction", "train_split", "calibration_split", "evaluation_split", "target_fpr",
        "bootstrap_replicates", "bootstrap_confidence", "transfer_min_auroc_gain",
        "near_random_auroc", "expected_replicate_id", "expected_replica_manifest_sha256",
        "expected_registry_sha256",
    }
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("E3B config fields do not match the frozen protocol")
    if cfg["tables"] != ["adult", "abalone"] or cfg["generators"] != ["CTGAN", "TVAE"]:
        raise ValueError("E3B tables/generators changed; version the protocol")
    if cfg["seeds"] != [2026, 2027, 2028]:
        raise ValueError("E3B split seeds changed; version the protocol")
    if cfg["primary_detector"] != "c2st_lr" or cfg["capacity_detector"] != "c2st_xgb":
        raise ValueError("E3B detectors changed; version the protocol")
    if cfg["model_direction"] != "high_score_is_synthetic":
        raise ValueError("E3B score direction must remain fixed")
    if [cfg["train_split"], cfg["calibration_split"], cfg["evaluation_split"]] != [
        "detector_train", "detector_val", "final_test"
    ]:
        raise ValueError("E3B split roles changed; version the protocol")
    if float(cfg["target_fpr"]) != .05 or float(cfg["transfer_min_auroc_gain"]) != .10:
        raise ValueError("E3B decision thresholds changed; version the protocol")
    if float(cfg["near_random_auroc"]) != .60:
        raise ValueError("E3B near-random rule changed; version the protocol")
    if not 20 <= int(cfg["bootstrap_replicates"]) <= 10000:
        raise ValueError("E3B bootstrap_replicates must be in [20, 10000]")
    if float(cfg["bootstrap_confidence"]) != .95:
        raise ValueError("E3B confidence level changed; version the protocol")
    return cfg


def preflight(config_path: Path, registry_path: Path, replica_manifest_path: Path) -> dict[str, Any]:
    cfg = _config(config_path.resolve())
    registry_path = registry_path.resolve()
    replica_manifest_path = replica_manifest_path.resolve()
    if sha256_file(replica_manifest_path) != cfg["expected_replica_manifest_sha256"]:
        raise ValueError("E3B replica manifest hash mismatch")
    if sha256_file(registry_path) != cfg["expected_registry_sha256"]:
        raise ValueError("E3B registry hash mismatch")
    manifest = json.loads(replica_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("E3B independent generator pool is incomplete")
    if manifest.get("plan", {}).get("replicate_id") != cfg["expected_replicate_id"]:
        raise ValueError("E3B independent generator pool identity mismatch")
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("E3B registry is not the one recorded by the replica manifest")
    registry = pd.read_csv(registry_path)
    required = {(table, generator) for table in cfg["tables"] for generator in cfg["generators"]}
    if set(zip(registry.table_id, registry.generator)) != required:
        raise ValueError("E3B registry matrix is incomplete or contains unexpected rows")
    runs = {(row["table_id"], row["generator"]): row for row in manifest["runs"]}
    if set(runs) != required or any(row.get("status") != "complete" for row in runs.values()):
        raise ValueError("E3B replica manifest does not contain four completed pools")
    data_hashes: dict[str, str] = {}
    for row in registry.itertuples(index=False):
        key = (str(row.table_id), str(row.generator))
        run_row = runs[key]
        real_path = (registry_path.parent / str(row.real_path)).resolve()
        synthetic_path = (registry_path.parent / str(row.synthetic_path)).resolve()
        if sha256_file(real_path) != run_row["real_sha256"]:
            raise ValueError(f"E3B real data hash mismatch: {key[0]}")
        if sha256_file(synthetic_path) != run_row["synthetic_sha256"]:
            raise ValueError(f"E3B synthetic data hash mismatch: {key[0]}/{key[1]}")
        data_hashes[f"{key[0]}/real"] = run_row["real_sha256"]
        data_hashes[f"{key[0]}/{key[1]}"] = run_row["synthetic_sha256"]
    return {
        "study_id": cfg["study_id"],
        "config_sha256": sha256_file(config_path),
        "replica_manifest_sha256": sha256_file(replica_manifest_path),
        "registry_sha256": sha256_file(registry_path),
        "data_hashes": data_hashes,
        "lr_cases": len(cfg["tables"]) * len(cfg["seeds"]),
        "protocol": {
            "tables": cfg["tables"], "generators": cfg["generators"], "seeds": cfg["seeds"],
            "target_fpr": float(cfg["target_fpr"]),
            "bootstrap_replicates": int(cfg["bootstrap_replicates"]),
            "bootstrap_confidence": float(cfg["bootstrap_confidence"]),
            "transfer_min_auroc_gain": float(cfg["transfer_min_auroc_gain"]),
            "near_random_auroc": float(cfg["near_random_auroc"]),
        },
    }


def _domain_splits(table: Any, generator: str, seed: int) -> dict[str, pd.DataFrame]:
    real = _split(table.real, seed)
    synthetic = _split(table.synthetic[generator], seed + 31)
    result: dict[str, pd.DataFrame] = {}
    for role, split_name in (
        ("detector_train", "detector_train"),
        ("calibration", "detector_val"),
        ("evaluation", "final_test"),
    ):
        combined = pd.concat([real[split_name], synthetic[split_name]], ignore_index=True, sort=False)
        result[role] = combined.sample(frac=1, random_state=seed + {"detector_train": 11, "calibration": 23, "evaluation": 37}[role]).reset_index(drop=True)
    return result


def _identity(frame: pd.DataFrame) -> pd.Series:
    return frame["generator"].astype(str) + "|" + frame["record_id"].astype(str)


def _assert_disjoint(splits: dict[str, pd.DataFrame], generator: str) -> None:
    identities = {name: set(_identity(frame)) for name, frame in splits.items()}
    if any(len(ids) != len(splits[name]) for name, ids in identities.items()):
        raise ValueError(f"Duplicate row identity inside E3B split: {generator}")
    roles = list(identities)
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            if identities[left] & identities[right]:
                raise ValueError(f"E3B split overlap: {generator} {left}/{right}")


def _labels(frame: pd.DataFrame) -> np.ndarray:
    labels = frame.source_label.to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("E3B split must contain real and synthetic rows")
    return labels


def _score_frame(detector: C2STDetector, frame: pd.DataFrame, *, table: str, seed: int,
                 detector_name: str, train_generator: str, eval_generator: str) -> pd.DataFrame:
    # Prediction happens before evaluation labels are copied into the output.
    raw_score = detector.predict_score(frame)
    cell_id = f"{train_generator}_to_{eval_generator}"
    return pd.DataFrame({
        "table": table,
        "seed": seed,
        "detector": detector_name,
        "train_generator": train_generator,
        "eval_generator": eval_generator,
        "cell_id": cell_id,
        "record_id": _identity(frame),
        "split": "evaluation",
        "source_label": frame.source_label.to_numpy(dtype=int),
        "model_direction": "high_score_is_synthetic",
        "raw_score": raw_score,
    }, columns=SCORE_COLUMNS)


def _metrics(scores: pd.DataFrame, threshold_info: dict[str, float]) -> dict[str, Any]:
    labels = scores.source_label.to_numpy(dtype=int)
    values = scores.raw_score.to_numpy(dtype=float)
    real, synthetic = values[labels == 0], values[labels == 1]
    predicted = values >= float(threshold_info["threshold"])
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
        "score_mean_gap": float(synthetic.mean() - real.mean()),
        "ks_distance": float(ks_2samp(real, synthetic).statistic),
        "tpr_at_calibrated_5pct_fpr": float(predicted[labels == 1].mean()),
        "observed_fpr": float(predicted[labels == 0].mean()),
        "calibration_threshold": float(threshold_info["threshold"]),
        "calibration_fpr": float(threshold_info["validation_fpr"]),
        "calibration_tpr": float(threshold_info["validation_tpr"]),
        "n": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_synthetic": int((labels == 1).sum()),
        "synthetic_fraction": float(labels.mean()),
    }


def _bootstrap(scores: pd.DataFrame, threshold: float, replicates: int,
               confidence: float, seed: int) -> dict[str, float]:
    labels = scores.source_label.to_numpy(dtype=int)
    values = scores.raw_score.to_numpy(dtype=float)
    real, synthetic = values[labels == 0], values[labels == 1]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("auroc", "auprc", "score_mean_gap", "ks_distance", "tpr")}
    for _ in range(replicates):
        sampled_real = real[rng.integers(0, len(real), len(real))]
        sampled_synthetic = synthetic[rng.integers(0, len(synthetic), len(synthetic))]
        y = np.r_[np.zeros(len(sampled_real), dtype=int), np.ones(len(sampled_synthetic), dtype=int)]
        s = np.r_[sampled_real, sampled_synthetic]
        draws["auroc"].append(roc_auc_score(y, s))
        draws["auprc"].append(average_precision_score(y, s))
        draws["score_mean_gap"].append(sampled_synthetic.mean() - sampled_real.mean())
        draws["ks_distance"].append(ks_2samp(sampled_real, sampled_synthetic).statistic)
        draws["tpr"].append((sampled_synthetic >= threshold).mean())
    alpha = (1 - confidence) / 2
    result: dict[str, float] = {}
    for name, values_ in draws.items():
        result[f"{name}_ci_low"] = float(np.quantile(values_, alpha))
        result[f"{name}_ci_high"] = float(np.quantile(values_, 1 - alpha))
    return result


def _paired_bootstrap(scores: pd.DataFrame, replicates: int, confidence: float,
                      seed: int) -> dict[str, float]:
    target = scores.loc[scores.eval_generator == "TVAE"]
    left = target.loc[target.train_generator == "CTGAN", ["record_id", "source_label", "raw_score"]]
    right = target.loc[target.train_generator == "TVAE", ["record_id", "source_label", "raw_score"]]
    merged = left.merge(right, on=["record_id", "source_label"], suffixes=("_ctgan", "_tvae"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("E3B paired target evaluation rows do not match")
    rng = np.random.default_rng(seed)
    index_by_label = {label: np.flatnonzero(merged.source_label.to_numpy() == label) for label in (0, 1)}
    auroc_delta, gap_delta = [], []
    for _ in range(replicates):
        indices = np.concatenate([
            idx[rng.integers(0, len(idx), len(idx))] for idx in index_by_label.values()
        ])
        y = merged.source_label.to_numpy()[indices]
        c = merged.raw_score_ctgan.to_numpy()[indices]
        t = merged.raw_score_tvae.to_numpy()[indices]
        auroc_delta.append(roc_auc_score(y, t) - roc_auc_score(y, c))
        gap_delta.append((t[y == 1].mean() - t[y == 0].mean()) - (c[y == 1].mean() - c[y == 0].mean()))
    alpha = (1 - confidence) / 2
    return {
        "auroc_gain": float(roc_auc_score(merged.source_label, merged.raw_score_tvae) - roc_auc_score(merged.source_label, merged.raw_score_ctgan)),
        "auroc_gain_ci_low": float(np.quantile(auroc_delta, alpha)),
        "auroc_gain_ci_high": float(np.quantile(auroc_delta, 1 - alpha)),
        "score_gap_gain": float(np.mean(gap_delta)),
        "score_gap_gain_ci_low": float(np.quantile(gap_delta, alpha)),
        "score_gap_gain_ci_high": float(np.quantile(gap_delta, 1 - alpha)),
    }


def _case_seed(seed: int, *parts: str) -> int:
    token = "|".join([str(seed), *parts]).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None, lineterminator="\n")


def _run_lr_case(table: Any, table_id: str, seed: int, cfg: dict[str, Any],
                 destination: Path, context: dict[str, Any]) -> None:
    splits = {generator: _domain_splits(table, generator, seed) for generator in cfg["generators"]}
    for generator, parts in splits.items():
        _assert_disjoint(parts, generator)
    assignments = []
    for generator, parts in splits.items():
        for role, frame in parts.items():
            assignments.append(pd.DataFrame({
                "table": table_id, "seed": seed, "generator": generator, "split": role,
                "record_id": _identity(frame), "source_label": frame.source_label.to_numpy(dtype=int),
            }))
    score_frames, metric_rows, interval_rows, thresholds = [], [], [], {}
    for train_generator in cfg["generators"]:
        train = splits[train_generator]["detector_train"]
        calibration = splits[train_generator]["calibration"]
        detector = C2STDetector("lr", seed=seed).fit(
            train, _labels(train), calibration, _labels(calibration)
        )
        calibration_scores = detector.predict_score(calibration)
        threshold = select_fpr_threshold(_labels(calibration), calibration_scores, float(cfg["target_fpr"]))
        thresholds[train_generator] = {**threshold, "detector_provenance": detector.get_provenance()}
        for eval_generator in cfg["generators"]:
            scored = _score_frame(
                detector, splits[eval_generator]["evaluation"], table=table_id, seed=seed,
                detector_name="c2st_lr", train_generator=train_generator, eval_generator=eval_generator,
            )
            score_frames.append(scored)
            identifiers = {
                "table": table_id, "seed": seed, "detector": "c2st_lr",
                "train_generator": train_generator, "eval_generator": eval_generator,
                "cell_id": f"{train_generator}_to_{eval_generator}",
            }
            metric_rows.append({**identifiers, **_metrics(scored, threshold)})
            interval_rows.append({
                **identifiers,
                **_bootstrap(scored, threshold["threshold"], int(cfg["bootstrap_replicates"]),
                             float(cfg["bootstrap_confidence"]),
                             _case_seed(seed, table_id, train_generator, eval_generator)),
            })
    scores = pd.concat(score_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    intervals = pd.DataFrame(interval_rows)
    paired = pd.DataFrame([{
        "table": table_id, "seed": seed, "eval_generator": "TVAE",
        **_paired_bootstrap(scores, int(cfg["bootstrap_replicates"]),
                            float(cfg["bootstrap_confidence"]), _case_seed(seed, table_id, "paired")),
    }])
    _write_frame(pd.concat(assignments, ignore_index=True), destination / "split_assignments.csv.gz")
    _write_frame(scores, destination / "scores.csv.gz")
    _write_frame(metrics, destination / "metrics.csv")
    _write_frame(intervals, destination / "bootstrap_intervals.csv")
    _write_frame(paired, destination / "paired_intervals.csv")
    _atomic_json(destination / "thresholds.json", thresholds)
    files = {
        name: sha256_file(destination / name) for name in (
            "split_assignments.csv.gz", "scores.csv.gz", "metrics.csv",
            "bootstrap_intervals.csv", "paired_intervals.csv", "thresholds.json",
        )
    }
    _atomic_json(destination / "complete.json", {
        "status": "complete", "case_type": "lr_matrix", "context": context,
        "table": table_id, "seed": seed, "matrix_cells": 4, "files_sha256": files,
    })


def _near_random_tables(output_root: Path, cfg: dict[str, Any]) -> list[str]:
    triggered = []
    for table in cfg["tables"]:
        rows = []
        for seed in cfg["seeds"]:
            case = output_root / "lr" / table / f"seed{seed}.complete"
            metrics = pd.read_csv(case / "metrics.csv")
            intervals = pd.read_csv(case / "bootstrap_intervals.csv")
            point = metrics.loc[(metrics.train_generator == "TVAE") & (metrics.eval_generator == "TVAE")].iloc[0]
            interval = intervals.loc[(intervals.train_generator == "TVAE") & (intervals.eval_generator == "TVAE")].iloc[0]
            rows.append((float(point.auroc), float(interval.auroc_ci_low), float(interval.auroc_ci_high)))
        mean_auroc = float(np.mean([row[0] for row in rows]))
        intervals_cover_half = sum(low <= .5 <= high for _, low, high in rows)
        if mean_auroc < float(cfg["near_random_auroc"]) or intervals_cover_half >= 2:
            triggered.append(table)
    return triggered


def _run_capacity_case(table: Any, table_id: str, seed: int, cfg: dict[str, Any],
                       destination: Path, context: dict[str, Any]) -> None:
    parts = _domain_splits(table, "TVAE", seed)
    _assert_disjoint(parts, "TVAE")
    train, calibration, evaluation = parts["detector_train"], parts["calibration"], parts["evaluation"]
    detector = C2STDetector("xgb", seed=seed).fit(train, _labels(train), calibration, _labels(calibration))
    calibration_scores = detector.predict_score(calibration)
    threshold = select_fpr_threshold(_labels(calibration), calibration_scores, float(cfg["target_fpr"]))
    scores = _score_frame(detector, evaluation, table=table_id, seed=seed,
                          detector_name="c2st_xgb", train_generator="TVAE", eval_generator="TVAE")
    identifiers = {"table": table_id, "seed": seed, "detector": "c2st_xgb",
                   "train_generator": "TVAE", "eval_generator": "TVAE", "cell_id": "TVAE_to_TVAE"}
    metrics = pd.DataFrame([{**identifiers, **_metrics(scores, threshold)}], columns=METRIC_COLUMNS)
    intervals = pd.DataFrame([{**identifiers, **_bootstrap(
        scores, threshold["threshold"], int(cfg["bootstrap_replicates"]),
        float(cfg["bootstrap_confidence"]), _case_seed(seed, table_id, "capacity"),
    )}])
    assignments = pd.concat([
        pd.DataFrame({"table": table_id, "seed": seed, "generator": "TVAE", "split": role,
                      "record_id": _identity(frame), "source_label": frame.source_label.to_numpy(dtype=int)})
        for role, frame in parts.items()
    ], ignore_index=True)
    _write_frame(assignments, destination / "split_assignments.csv.gz")
    _write_frame(scores, destination / "scores.csv.gz")
    _write_frame(metrics, destination / "metrics.csv")
    _write_frame(intervals, destination / "bootstrap_intervals.csv")
    _atomic_json(destination / "thresholds.json", {"TVAE": {**threshold, "detector_provenance": detector.get_provenance()}})
    files = {name: sha256_file(destination / name) for name in (
        "split_assignments.csv.gz", "scores.csv.gz", "metrics.csv", "bootstrap_intervals.csv", "thresholds.json",
    )}
    _atomic_json(destination / "complete.json", {
        "status": "complete", "case_type": "capacity_diagnostic", "context": context,
        "table": table_id, "seed": seed, "matrix_cells": 1, "files_sha256": files,
    })


def _atomic_case(case_root: Path, writer: Any) -> None:
    final = case_root.with_name(case_root.name + ".complete")
    partial = case_root.with_name(case_root.name + ".partial")
    if final.exists():
        complete = json.loads((final / "complete.json").read_text(encoding="utf-8"))
        if complete.get("status") != "complete":
            raise ValueError(f"E3B final case is not complete: {final}")
        for name, digest in complete["files_sha256"].items():
            if sha256_file(final / name) != digest:
                raise ValueError(f"E3B completed case changed: {final / name}")
        return
    if partial.exists():
        raise FileExistsError(f"E3B partial case requires inspection: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        writer(partial)
        if not (partial / "complete.json").is_file():
            raise ValueError("E3B case writer did not create complete.json")
        partial.replace(final)
    except Exception as exc:
        _atomic_json(partial / "failure.json", {"status": "failed", "error": repr(exc)})
        raise


def run(config_path: Path, registry_path: Path, replica_manifest_path: Path,
        output_root: Path) -> dict[str, Any]:
    cfg = _config(config_path.resolve())
    context = preflight(config_path, registry_path, replica_manifest_path)
    context["implementation_sha256"] = sha256_file(Path(__file__))
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    context_path = output_root / "context.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != context:
            raise ValueError("E3B output belongs to different code or inputs")
    else:
        if any(output_root.iterdir()):
            raise FileExistsError("Nonempty E3B output has no context")
        _atomic_json(context_path, context)
    source = RegistrySource(registry_path)
    for table_id in cfg["tables"]:
        table = source.table(table_id)
        for seed in cfg["seeds"]:
            root = output_root / "lr" / table_id / f"seed{seed}"
            _atomic_case(root, lambda destination, t=table, tid=table_id, s=seed:
                         _run_lr_case(t, tid, s, cfg, destination, context))
    capacity_tables = _near_random_tables(output_root, cfg)
    for table_id in capacity_tables:
        table = source.table(table_id)
        for seed in cfg["seeds"]:
            root = output_root / "gbdt" / table_id / f"seed{seed}"
            _atomic_case(root, lambda destination, t=table, tid=table_id, s=seed:
                         _run_capacity_case(t, tid, s, cfg, destination, context))
    summary = {
        "status": "complete", "context": context,
        "completed_lr_cases": len(cfg["tables"]) * len(cfg["seeds"]),
        "capacity_tables": capacity_tables,
        "completed_capacity_cases": len(capacity_tables) * len(cfg["seeds"]),
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary
