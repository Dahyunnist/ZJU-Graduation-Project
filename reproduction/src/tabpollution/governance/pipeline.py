"""Unified detection -> quantification -> governance decision experiment."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import importlib.util
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpollution.detectors.classical import C2STDetector, Char3GramDetector
from tabpollution.detectors.deep import DeepTextDetector
from tabpollution.governance.artifacts import artifact_only_auc
from tabpollution.governance.config import GovernanceConfig, load_governance_config
from tabpollution.governance.data import (
    META_COLUMNS,
    GovernanceDataSource,
    RegistrySource,
    SyntheticFixtureSource,
    exact_mixture,
    sample_rows,
)
from tabpollution.governance.metrics import (
    analytical_positive_rate,
    detection_metrics,
    finite_or_none,
    prevalence_error,
    select_fpr_threshold,
)
from tabpollution.mixing.protocols import validate_protocol
from tabpollution.quantification.methods import ScoreQuantifier
from tabpollution.utils import write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_resources(config: GovernanceConfig) -> None:
    value = str(config.max_cpu_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    if config.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _source(config: GovernanceConfig, seed: int) -> GovernanceDataSource:
    if config.data_mode == "synthetic_fixture":
        return SyntheticFixtureSource(config.fixture_rows_per_table, seed)
    assert config.registry_path is not None
    return RegistrySource(config.registry_path)


def _split(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    cuts = [int(n * .35), int(n * .55), int(n * .70), int(n * .90)]
    names = ("source", "detector_train", "detector_val", "downstream_train", "final_test")
    bounds = [0, *cuts, n]
    pieces = [shuffled.iloc[bounds[i]:bounds[i + 1]] for i in range(len(names))]
    return {name: piece.reset_index(drop=True) for name, piece in zip(names, pieces)}


def _collect(source: GovernanceDataSource, tables: tuple[str, ...], generators: tuple[str, ...],
             split_name: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for table_index, table_id in enumerate(tables):
        table = source.table(table_id)
        real_split = _split(table.real, seed + table_index * 113)[split_name]
        records.append(real_split)
        labels.append(np.zeros(len(real_split), dtype=int))
        for generator_index, generator in enumerate(generators):
            synth_split = _split(
                table.synthetic[generator], seed + table_index * 113 + generator_index * 997 + 31
            )[split_name]
            records.append(synth_split)
            labels.append(np.ones(len(synth_split), dtype=int))
    combined = pd.concat(records, ignore_index=True, sort=False)
    combined_labels = np.concatenate(labels)
    order = np.random.default_rng(seed + 7001).permutation(len(combined))
    return combined.iloc[order].reset_index(drop=True), combined_labels[order]


def _protocol_manifest(protocol: str, train: pd.DataFrame, test: pd.DataFrame,
                       train_tables: tuple[str, ...], test_tables: tuple[str, ...],
                       train_generators: tuple[str, ...], test_generators: tuple[str, ...],
                       source: GovernanceDataSource) -> dict[str, Any]:
    manifest = {
        "train_tables": list(train_tables), "test_tables": list(test_tables),
        "train_generators": list(train_generators), "test_generators": list(test_generators),
        "train_records": train["record_id"].astype(str).tolist(),
        "test_records": test["record_id"].astype(str).tolist(),
        "train_domains": sorted({source.table(t).domain for t in train_tables}),
        "test_domains": sorted({source.table(t).domain for t in test_tables}),
    }
    validate_protocol(protocol, manifest)
    return manifest


def _detector(name: str, seed: int, config: GovernanceConfig):
    if name == "char3gram":
        return Char3GramDetector(seed=seed, max_features=20000)
    if name == "c2st_lr":
        return C2STDetector("lr", seed=seed)
    if name == "c2st_xgb":
        return C2STDetector("xgb", seed=seed)
    modes = {
        "flat_transformer": "flat",
        "table_transformer": "table",
        "datum_transformer": "datum",
        "datum_ta": "datum_ta",
    }
    if name in modes:
        formal = config.run_type in {"pilot", "formal"}
        return DeepTextDetector(
            mode=modes[name], seed=seed,
            dim=64 if formal else 24,
            heads=4, layers=2 if formal else 1,
            max_len=512 if formal else 192,
            max_datum=64 if formal else 32,
            max_columns=64 if formal else 24,
            epochs=20 if formal else 2,
            batch_size=128 if formal else 32,
            device=config.device,
            table_classes=32,
        )
    raise ValueError(name)


class _PlattCalibrator:
    def __init__(self, seed: int):
        self.model = LogisticRegression(max_iter=300, random_state=seed)
        self.constant: float | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "_PlattCalibrator":
        scores = np.asarray(scores, float)
        if np.ptp(scores) < 1e-12:
            self.constant = float(np.mean(labels))
        else:
            self.model.fit(scores.reshape(-1, 1), labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, float)
        if self.constant is not None:
            return np.full(len(scores), self.constant)
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


def _target_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    features = frame.drop(columns=[target, *[c for c in META_COLUMNS if c in frame]], errors="ignore")
    labels = pd.Series(frame[target]).astype(str)
    classes = sorted(labels.unique())
    if len(classes) != 2:
        raise ValueError(f"Downstream target {target!r} must be binary, got {classes}")
    return features, (labels == classes[-1]).astype(int).to_numpy()


def _target_model(features: pd.DataFrame, seed: int, threads: int) -> Pipeline:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(max_iter=500, random_state=seed)),
    ])


def _utility(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int, threads: int) -> float:
    if len(train) < 10:
        return float("nan")
    x_train, y_train = _target_xy(train, target)
    x_test, y_test = _target_xy(test, target)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    model = _target_model(x_train, seed, threads)
    model.fit(x_train, y_train)
    if hasattr(model[-1], "predict_proba"):
        return float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))
    return float(accuracy_score(y_test, model.predict(x_test)))


def _safe_max(*values: float) -> float:
    finite = [float(x) for x in values if np.isfinite(x)]
    return max(finite) if finite else float("nan")


def _detector_cleanup(bag: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    return bag.loc[np.asarray(scores) < threshold].reset_index(drop=True)


def _source_cleanup(bag: pd.DataFrame) -> pd.DataFrame:
    return bag.loc[bag["source_label"].to_numpy() == 0].reset_index(drop=True)


def _quadrant(auroc: float, utility_delta: float, harm_tolerance: float) -> str:
    if not np.isfinite(utility_delta):
        return "harm_not_evaluated"
    detectable = auroc >= .75
    harmful = np.isfinite(utility_delta) and utility_delta <= -harm_tolerance
    return f"{'high' if detectable else 'low'}_detectability__{'high' if harmful else 'low'}_harm"


def _write_findings(rows: pd.DataFrame, artifact_rows: pd.DataFrame, output: Path,
                    primary_quantifier: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    detection = rows.groupby(["protocol", "detector"], as_index=False).agg(
        auroc=("detection_auroc", "mean"),
        auprc=("detection_auprc", "mean"),
        fpr=("detection_fpr", "mean"),
        ece=("detection_ece", "mean"),
    )
    detection.to_csv(output / "finding_1_transfer.csv", index=False)
    artifact_rows.to_csv(output / "finding_2_format_artifacts.csv", index=False)
    low = rows.loc[rows["true_prevalence"].isin([.05, .10])]
    low.groupby(["protocol", "detector", "quantifier", "true_prevalence"], as_index=False).agg(
        mae=("prevalence_absolute_error", "mean"),
        bias=("prevalence_error", "mean"),
        decision_error_rate=("decision_error", "mean"),
        false_positive_share=("false_positive_share", "mean"),
    ).to_csv(output / "finding_3_low_prevalence.csv", index=False)
    rows.groupby(["protocol", "quantifier"], as_index=False).agg(
        mae=("prevalence_absolute_error", "mean"),
        bias=("prevalence_error", "mean"),
        decision_regret=("decision_regret", "mean"),
    ).to_csv(output / "finding_4_quantifier_shift.csv", index=False)
    primary = rows.loc[rows["quantifier"] == primary_quantifier]
    primary.groupby(["protocol", "test_table", "test_generator", "true_prevalence"], as_index=False).agg(
        contaminated_delta=("contaminated_utility_delta", "mean"),
        detector_cleanup_delta=("detector_cleanup_delta", "mean"),
        oracle_cleanup_delta=("oracle_cleanup_delta", "mean"),
    ).to_csv(output / "finding_5_utility_curve.csv", index=False)
    primary.groupby(["protocol", "detectability_harm_quadrant"], as_index=False).agg(
        cases=("bag_id", "count"),
        mean_auroc=("detection_auroc", "mean"),
        mean_utility_delta=("contaminated_utility_delta", "mean"),
        mean_regret=("decision_regret", "mean"),
    ).to_csv(output / "finding_6_detectability_vs_harm.csv", index=False)


def validate_governance_setup(config_or_path: GovernanceConfig | str | Path) -> dict[str, Any]:
    config = config_or_path if isinstance(config_or_path, GovernanceConfig) else load_governance_config(config_or_path)
    if config.data_mode == "registry" and (config.registry_path is None or not config.registry_path.is_file()):
        return {
            "experiment_id": config.experiment_id,
            "passed": False,
            "data_mode": config.data_mode,
            "reason": "pool_registry_missing",
            "registry_path": str(config.registry_path),
            "dependency_checks": [],
            "protocol_checks": [],
            "output_dir": str(config.output_dir),
        }
    source = _source(config, config.seeds[0])
    dependency_checks: list[dict[str, Any]] = []
    if "c2st_xgb" in config.detectors:
        xgboost_available = importlib.util.find_spec("xgboost") is not None
        dependency_checks.append({
            "dependency": "xgboost", "passed": xgboost_available,
            "reason": None if xgboost_available else "xgboost_not_installed",
        })
    if config.device == "cuda":
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
        dependency_checks.append({
            "dependency": "torch_cuda", "passed": cuda_available,
            "reason": None if cuda_available else "cuda_unavailable",
        })
    available_tables = set(source.table_ids)
    available_generators = set(source.generators)
    checks: list[dict[str, Any]] = []
    for protocol, spec in config.protocols.items():
        missing_tables = sorted((set(spec.train_tables) | set(spec.test_tables)) - available_tables)
        missing_generators = sorted((set(spec.train_generators) | set(spec.test_generators)) - available_generators)
        checks.append({
            "protocol": protocol,
            "missing_tables": missing_tables,
            "missing_generators": missing_generators,
            "passed": not missing_tables and not missing_generators,
        })
    passed = all(row["passed"] for row in checks) and all(row["passed"] for row in dependency_checks)
    return {
        "experiment_id": config.experiment_id,
        "passed": passed,
        "data_mode": config.data_mode,
        "available_tables": sorted(available_tables),
        "available_generators": sorted(available_generators),
        "dependency_checks": dependency_checks,
        "protocol_checks": checks,
        "output_dir": str(config.output_dir),
    }


def run_governance_benchmark(config_or_path: GovernanceConfig | str | Path) -> dict[str, Any]:
    config = config_or_path if isinstance(config_or_path, GovernanceConfig) else load_governance_config(config_or_path)
    _configure_resources(config)
    preflight = validate_governance_setup(config)
    if not preflight["passed"]:
        raise ValueError(f"Governance preflight failed: {preflight['protocol_checks']}")
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(preflight, output / "preflight.json")
    resolved = json.loads(json.dumps(asdict(config), default=str))
    write_json(resolved, output / "resolved_config.json")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    protocol_manifests: dict[str, Any] = {}

    for seed in config.seeds:
        source = _source(config, seed)
        for protocol, spec in config.protocols.items():
            # One split seed is shared by every stage so record membership is
            # frozen; changing the model/evaluation stage must not reshuffle a
            # record into another partition.
            detector_train, train_labels = _collect(source, spec.train_tables, spec.train_generators, "detector_train", seed)
            detector_val, val_labels = _collect(source, spec.train_tables, spec.train_generators, "detector_val", seed)
            detector_test, test_labels = _collect(source, spec.test_tables, spec.test_generators, "final_test", seed)
            manifest = _protocol_manifest(
                protocol, detector_train, detector_test,
                spec.train_tables, spec.test_tables, spec.train_generators, spec.test_generators, source,
            )
            protocol_manifests[f"{seed}:{protocol}"] = manifest

            artifact = artifact_only_auc(detector_train, train_labels, detector_test, test_labels, seed)
            artifact_row = {
                "seed": seed, "protocol": protocol,
                "artifact_auroc": artifact["artifact_auroc"],
                "artifact_auroc_raw": artifact["artifact_auroc_raw"],
                "artifact_auc_gate": config.artifact_auc_gate,
                "artifact_gate_passed": artifact["artifact_auroc"] < config.artifact_auc_gate,
                "feature_coefficients": json.dumps(artifact["feature_coefficients"], sort_keys=True),
            }
            artifacts.append(artifact_row)

            for detector_name in config.detectors:
                if detector_name.startswith("c2st") and protocol in {"P3", "P4"}:
                    # Schema-bound C2ST is not a valid cross-table contestant.
                    continue
                detector = _detector(detector_name, seed, config)
                table_names = sorted(detector_train["table_id"].astype(str).unique())
                table_index = {name: index for index, name in enumerate(table_names)}
                table_labels = detector_train["table_id"].astype(str).map(table_index).to_numpy()
                detector.fit(
                    detector_train, train_labels, detector_val, val_labels,
                    table_labels=table_labels,
                )
                raw_val = detector.predict_score(detector_val)
                calibrator = _PlattCalibrator(seed).fit(raw_val, val_labels)
                val_scores = calibrator.predict(raw_val)
                threshold_info = select_fpr_threshold(val_labels, val_scores, config.detector_fpr_target)
                threshold = threshold_info["threshold"]
                test_scores = calibrator.predict(detector.predict_score(detector_test))
                det_metrics = detection_metrics(test_labels, test_scores, threshold)
                quantifiers: dict[str, ScoreQuantifier] = {}
                for method in config.quantifiers:
                    quantifiers[method] = ScoreQuantifier(method).fit(val_scores, val_labels, threshold=threshold)

                for table_index, test_table in enumerate(spec.test_tables):
                    table = source.table(test_table)
                    real_splits = _split(table.real, seed + table_index * 113)
                    pure_test = real_splits["final_test"]
                    downstream_real = real_splits["downstream_train"]
                    for generator_index, test_generator in enumerate(spec.test_generators):
                        synth_split = _split(
                            table.synthetic[test_generator], seed + table_index * 113 + generator_index * 997 + 31
                        )["downstream_train"]
                        for prevalence in config.prevalence_rates:
                            for bag_index in range(config.bags_per_rate):
                                bag_seed = seed + table_index * 100003 + generator_index * 1009 + int(prevalence * 1000) * 17 + bag_index
                                bag = exact_mixture(downstream_real, synth_split, config.bag_size, prevalence, bag_seed)
                                bag_scores = calibrator.predict(detector.predict_score(bag))
                                actual_prevalence = float(bag["source_label"].mean())
                                detector_clean = _detector_cleanup(bag, bag_scores, threshold)
                                oracle_clean = _source_cleanup(bag)
                                evaluate_utility = bag_index < config.utility_bags_per_rate
                                if evaluate_utility:
                                    clean_train = sample_rows(downstream_real, config.bag_size, bag_seed + 71)
                                    clean_utility = _utility(clean_train, pure_test, table.target_column, bag_seed, config.max_cpu_threads)
                                    contaminated_utility = _utility(bag, pure_test, table.target_column, bag_seed, config.max_cpu_threads)
                                    detector_cleanup_utility = _utility(
                                        detector_clean, pure_test, table.target_column, bag_seed, config.max_cpu_threads
                                    )
                                    oracle_cleanup_utility = _utility(
                                        oracle_clean, pure_test, table.target_column, bag_seed, config.max_cpu_threads
                                    )
                                else:
                                    clean_utility = contaminated_utility = float("nan")
                                    detector_cleanup_utility = oracle_cleanup_utility = float("nan")
                                predictions = bag_scores >= threshold
                                true_source = bag["source_label"].to_numpy(dtype=int)
                                cleanup_precision = float(true_source[predictions].mean()) if predictions.any() else 0.0
                                cleanup_recall = float(predictions[true_source == 1].mean()) if (true_source == 1).any() else 1.0
                                propagation = analytical_positive_rate(actual_prevalence, det_metrics["tpr"], det_metrics["fpr"])
                                for method, quantifier in quantifiers.items():
                                    try:
                                        estimate = quantifier.predict_prevalence(bag_scores)["clipped"]
                                        quantifier_status = "ok"
                                    except ValueError as exc:
                                        estimate = float("nan")
                                        quantifier_status = f"failed:{exc}"
                                    estimated_decision = bool(np.isfinite(estimate) and estimate >= config.governance_prevalence_threshold)
                                    detector_action_better = bool(
                                        np.isfinite(detector_cleanup_utility)
                                        and detector_cleanup_utility > contaminated_utility + config.harm_tolerance
                                    )
                                    policy_utility = detector_cleanup_utility if estimated_decision else contaminated_utility
                                    best_available = _safe_max(contaminated_utility, detector_cleanup_utility)
                                    decision_regret = (
                                        float(best_available - policy_utility)
                                        if np.isfinite(best_available) and np.isfinite(policy_utility) else float("nan")
                                    )
                                    error = prevalence_error(actual_prevalence, estimate) if np.isfinite(estimate) else {
                                        "prevalence_error": float("nan"),
                                        "prevalence_absolute_error": float("nan"),
                                        "prevalence_squared_error": float("nan"),
                                    }
                                    contaminated_delta = contaminated_utility - clean_utility
                                    row = {
                                        "experiment_id": config.experiment_id,
                                        "run_type": config.run_type,
                                        "seed": seed,
                                        "protocol": protocol,
                                        "detector": detector_name,
                                        "quantifier": method,
                                        "quantifier_status": quantifier_status,
                                        "test_table": test_table,
                                        "test_generator": test_generator,
                                        "bag_id": f"{seed}-{protocol}-{detector_name}-{test_table}-{test_generator}-{prevalence:.3f}-{bag_index}",
                                        "bag_index": bag_index,
                                        "bag_size": len(bag),
                                        "true_prevalence": actual_prevalence,
                                        "estimated_prevalence": estimate,
                                        **error,
                                        "detection_auroc": det_metrics["auroc"],
                                        "detection_auprc": det_metrics["auprc"],
                                        "detection_brier": det_metrics["brier"],
                                        "detection_ece": det_metrics["ece"],
                                        "detection_fpr": det_metrics["fpr"],
                                        "detection_tpr": det_metrics["tpr"],
                                        "detector_threshold": threshold,
                                        "artifact_auroc": artifact["artifact_auroc"],
                                        "artifact_gate_passed": artifact_row["artifact_gate_passed"],
                                        **propagation,
                                        "clean_utility": clean_utility,
                                        "contaminated_utility": contaminated_utility,
                                        "detector_cleanup_utility": detector_cleanup_utility,
                                        "oracle_cleanup_utility": oracle_cleanup_utility,
                                        "contaminated_utility_delta": contaminated_delta,
                                        "detector_cleanup_delta": detector_cleanup_utility - clean_utility if np.isfinite(detector_cleanup_utility) else float("nan"),
                                        "oracle_cleanup_delta": oracle_cleanup_utility - clean_utility if np.isfinite(oracle_cleanup_utility) else float("nan"),
                                        "cleanup_precision": cleanup_precision,
                                        "cleanup_recall": cleanup_recall,
                                        "governance_prevalence_threshold": config.governance_prevalence_threshold,
                                        "estimated_decision_remediate": estimated_decision,
                                        "best_available_decision_remediate": detector_action_better,
                                        "decision_error": int(estimated_decision != detector_action_better) if evaluate_utility else float("nan"),
                                        "policy_utility": policy_utility,
                                        "decision_regret": decision_regret,
                                        "oracle_cleanup_gap": oracle_cleanup_utility - detector_cleanup_utility if np.isfinite(oracle_cleanup_utility) and np.isfinite(detector_cleanup_utility) else float("nan"),
                                        "detectability_harm_quadrant": _quadrant(det_metrics["auroc"], contaminated_delta, config.harm_tolerance),
                                    }
                                    rows.append(row)

    evidence = pd.DataFrame(rows)
    artifact_frame = pd.DataFrame(artifacts)
    evidence.to_csv(output / "governance_evidence.csv", index=False)
    artifact_frame.to_csv(output / "format_artifact_audit.csv", index=False)
    _write_findings(evidence, artifact_frame, output, config.primary_quantifier)
    write_json(protocol_manifests, output / "protocol_manifests.json")
    summary = {
        "experiment_id": config.experiment_id,
        "run_type": config.run_type,
        "status": "complete",
        "formal_inclusion": config.run_type == "formal",
        "created_at": _utc_now(),
        "seconds": time.perf_counter() - started,
        "evidence_rows": len(evidence),
        "protocols": sorted(evidence["protocol"].unique().tolist()),
        "detectors": sorted(evidence["detector"].unique().tolist()),
        "quantifiers": sorted(evidence["quantifier"].unique().tolist()),
        "low_prevalence_rows": int(evidence["true_prevalence"].isin([.05, .10]).sum()),
        "artifact_gate_failures": int((~artifact_frame["artifact_gate_passed"]).sum()),
        "mean_prevalence_mae": finite_or_none(evidence["prevalence_absolute_error"].mean()),
        "mean_decision_regret": finite_or_none(evidence["decision_regret"].mean()),
        "outputs": {
            "evidence": str(output / "governance_evidence.csv"),
            "findings": [str(path) for path in sorted(output.glob("finding_*.csv"))],
        },
    }
    write_json(summary, output / "summary.json")
    return summary
