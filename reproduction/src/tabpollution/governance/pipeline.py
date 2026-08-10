"""Unified detection -> quantification -> governance decision experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import importlib.util
import os
from pathlib import Path
import time
from typing import Any
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import t as student_t, ttest_rel

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
    select_negative_anchor_threshold,
)
from tabpollution.mixing.protocols import validate_protocol
from tabpollution.quantification.methods import ScoreQuantifier
from tabpollution.utils import write_json
from tabpollution.valuation.methods import data_oob, knn_shapley


METHOD_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "formal_method_registry.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _method_registry(config: GovernanceConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        registry = yaml.safe_load(METHOD_REGISTRY_PATH.read_text(encoding="utf-8"))
        registered = registry["detectors"]
        missing = sorted(set(config.detectors) - set(registered))
        nonformal = sorted(
            name for name in config.detectors
            if config.run_type == "formal" and not bool(registered.get(name, {}).get("formal_ready"))
        )
        passed = not missing and not nonformal
        return registry, {
            "dependency": "formal_method_registry", "passed": passed,
            "registry_path": str(METHOD_REGISTRY_PATH),
            "missing_methods": missing, "nonformal_methods": nonformal,
            "reason": None if passed else "method_registry_contract_failed",
        }
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        return {}, {
            "dependency": "formal_method_registry", "passed": False,
            "registry_path": str(METHOD_REGISTRY_PATH),
            "reason": f"method_registry_unreadable:{type(exc).__name__}",
        }


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
        "column_positional_ablation": "table",
        "datum_transformer": "datum",
        "datum_ta": "datum_ta",
    }
    if name in modes:
        return DeepTextDetector(
            mode=modes[name], seed=seed,
            dim=config.deep_dim, heads=config.deep_heads, layers=config.deep_layers,
            max_len=config.deep_max_len, max_datum=config.deep_max_datum,
            max_columns=config.deep_max_columns, epochs=config.deep_epochs,
            batch_size=config.deep_batch_size,
            device=config.device,
            table_classes=32,
            learning_rate=config.deep_learning_rate,
            weight_decay=config.deep_weight_decay,
            gradient_clip_norm=config.deep_gradient_clip_norm,
            early_stopping_patience=config.deep_early_stopping_patience,
            min_epochs=config.deep_min_epochs,
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


@dataclass(frozen=True)
class _CalibrationPolicy:
    name: str
    calibrator: _PlattCalibrator
    threshold: float
    quantifier_scores: np.ndarray
    quantifier_labels: np.ndarray
    reference_fpr: float
    reference_tpr: float
    uses_target_real_records: bool
    uses_target_labels: bool
    deployment_status: str


def _target_real_anchor(source: GovernanceDataSource, tables: tuple[str, ...], seed: int,
                        limit: int) -> pd.DataFrame:
    frames = [
        _split(source.table(table_id).real, seed + table_index * 113)["source"]
        for table_index, table_id in enumerate(tables)
    ]
    anchor = pd.concat(frames, ignore_index=True, sort=False)
    if len(anchor) > limit:
        anchor = anchor.sample(n=limit, random_state=seed + 9109)
    return anchor.reset_index(drop=True)


def _calibration_policies(
    config: GovernanceConfig,
    detector: Any,
    source: GovernanceDataSource,
    spec: Any,
    seed: int,
    raw_source_val: np.ndarray,
    source_val_labels: np.ndarray,
) -> dict[str, _CalibrationPolicy]:
    policies: dict[str, _CalibrationPolicy] = {}
    for name in config.calibration_policies:
        if name == "source_only":
            fit_scores, fit_labels = raw_source_val, source_val_labels
            calibrator = _PlattCalibrator(seed).fit(fit_scores, fit_labels)
            quantifier_scores = calibrator.predict(fit_scores)
            threshold_info = select_fpr_threshold(fit_labels, quantifier_scores, config.detector_fpr_target)
            policies[name] = _CalibrationPolicy(
                name, calibrator, threshold_info["threshold"], quantifier_scores, fit_labels,
                threshold_info["validation_fpr"], threshold_info["validation_tpr"],
                False, False, "deployable_without_target_data",
            )
            continue

        if name == "target_real_anchor":
            anchor = _target_real_anchor(
                source, spec.test_tables, seed, config.target_real_anchor_size,
            )
            raw_anchor = detector.predict_score(anchor)
            positive = raw_source_val[source_val_labels == 1]
            n = min(len(raw_anchor), len(positive), config.target_real_anchor_size)
            if n < 1:
                raise RuntimeError("target_real_anchor requires clean target records and source positives")
            rng = np.random.default_rng(seed + 9119)
            positive = positive[rng.choice(len(positive), size=n, replace=False)]
            raw_anchor = raw_anchor[rng.choice(len(raw_anchor), size=n, replace=False)]
            fit_scores = np.concatenate([raw_anchor, positive])
            fit_labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
            calibrator = _PlattCalibrator(seed + 1).fit(fit_scores, fit_labels)
            quantifier_scores = calibrator.predict(fit_scores)
            calibrated_anchor = calibrator.predict(raw_anchor)
            threshold_info = select_negative_anchor_threshold(calibrated_anchor, config.detector_fpr_target)
            policies[name] = _CalibrationPolicy(
                name, calibrator, threshold_info["threshold"], quantifier_scores, fit_labels,
                threshold_info["validation_fpr"], float("nan"),
                True, False, "deployable_with_clean_target_anchor",
            )
            continue

        if name == "oracle_target":
            target_val, target_val_labels = _collect(
                source, spec.test_tables, spec.test_generators, "detector_val", seed,
            )
            raw_target_val = detector.predict_score(target_val)
            calibrator = _PlattCalibrator(seed + 2).fit(raw_target_val, target_val_labels)
            quantifier_scores = calibrator.predict(raw_target_val)
            threshold_info = select_fpr_threshold(
                target_val_labels, quantifier_scores, config.detector_fpr_target,
            )
            policies[name] = _CalibrationPolicy(
                name, calibrator, threshold_info["threshold"], quantifier_scores, target_val_labels,
                threshold_info["validation_fpr"], threshold_info["validation_tpr"],
                True, True, "diagnostic_oracle_not_deployable",
            )
            continue
        raise ValueError(f"Unsupported calibration policy: {name}")
    return policies


def _target_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    features = frame.drop(columns=[target, *[c for c in META_COLUMNS if c in frame]], errors="ignore")
    labels = pd.Series(frame[target]).astype(str)
    classes = sorted(labels.unique())
    if len(classes) != 2:
        raise ValueError(f"Downstream target {target!r} must be binary, got {classes}")
    return features, (labels == classes[-1]).astype(int).to_numpy()


def _target_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
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
    return ColumnTransformer(transformers)


def _target_model(features: pd.DataFrame, seed: int, threads: int) -> Pipeline:
    return Pipeline([
        ("preprocess", _target_preprocessor(features)),
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


def _valuation_rows(
    bag: pd.DataFrame,
    pure_test: pd.DataFrame,
    target: str,
    methods: tuple[str, ...],
    sample_limit: int,
    oob_estimators: int,
    seed: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    train = bag.sample(n=min(len(bag), sample_limit), random_state=seed, replace=False).reset_index(drop=True)
    test = pure_test.sample(
        n=min(len(pure_test), max(40, sample_limit // 2)), random_state=seed + 1, replace=False
    ).reset_index(drop=True)
    x_train_frame, y_train = _target_xy(train, target)
    x_test_frame, y_test = _target_xy(test, target)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return []
    preprocessor = _target_preprocessor(x_train_frame)
    x_train = preprocessor.fit_transform(x_train_frame)
    x_test = preprocessor.transform(x_test_frame)
    if hasattr(x_train, "toarray"):
        x_train = x_train.toarray()
        x_test = x_test.toarray()
    values_by_method: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    if "knn_shapley" in methods:
        values_by_method["knn_shapley"] = (
            knn_shapley(np.asarray(x_train), y_train, np.asarray(x_test), y_test, k=5), None
        )
    if "data_oob" in methods:
        values_by_method["data_oob"] = data_oob(
            np.asarray(x_train), y_train, n_estimators=oob_estimators, seed=seed
        )
    rows: list[dict[str, Any]] = []
    for method, (values, coverage) in values_by_method.items():
        for index, value in enumerate(values):
            rows.append({
                **metadata,
                "valuation_method": method,
                "valuation_row_index": index,
                "record_id": str(train.iloc[index].get("record_id", index)),
                "source_label": int(train.iloc[index]["source_label"]),
                "task_value": float(value) if np.isfinite(value) else float("nan"),
                "oob_coverage": int(coverage[index]) if coverage is not None else float("nan"),
            })
    return rows


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


def _write_findings(
    rows: pd.DataFrame,
    artifact_rows: pd.DataFrame,
    valuation_rows: pd.DataFrame,
    output: Path,
    primary_quantifier: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    detection = rows.groupby(["protocol", "detector", "calibration_policy"], as_index=False).agg(
        auroc=("detection_auroc", "mean"),
        auprc=("detection_auprc", "mean"),
        fpr=("detection_fpr", "mean"),
        ece=("detection_ece", "mean"),
    )
    detection.to_csv(output / "finding_1_transfer.csv", index=False)
    artifact_rows.to_csv(output / "finding_2_format_artifacts.csv", index=False)
    low = rows.loc[rows["true_prevalence"].isin([.05, .10])]
    low.groupby(["protocol", "detector", "calibration_policy", "quantifier", "contamination_mode", "true_prevalence"], as_index=False).agg(
        mae=("prevalence_absolute_error", "mean"),
        bias=("prevalence_error", "mean"),
        decision_error_rate=("decision_error", "mean"),
        false_positive_share=("false_positive_share", "mean"),
    ).to_csv(output / "finding_3_low_prevalence.csv", index=False)
    rows.groupby(["protocol", "calibration_policy", "quantifier", "contamination_mode"], as_index=False).agg(
        mae=("prevalence_absolute_error", "mean"),
        bias=("prevalence_error", "mean"),
        decision_regret=("decision_regret", "mean"),
    ).to_csv(output / "finding_4_quantifier_shift.csv", index=False)
    primary = rows.loc[rows["quantifier"] == primary_quantifier]
    primary.groupby(["protocol", "calibration_policy", "test_table", "test_generator", "contamination_mode", "true_prevalence"], as_index=False).agg(
        contaminated_delta=("contaminated_utility_delta", "mean"),
        detector_cleanup_delta=("detector_cleanup_delta", "mean"),
        oracle_cleanup_delta=("oracle_cleanup_delta", "mean"),
    ).to_csv(output / "finding_5_utility_curve.csv", index=False)
    primary.groupby(["protocol", "calibration_policy", "detectability_harm_quadrant"], as_index=False).agg(
        cases=("bag_id", "count"),
        mean_auroc=("detection_auroc", "mean"),
        mean_utility_delta=("contaminated_utility_delta", "mean"),
        mean_regret=("decision_regret", "mean"),
    ).to_csv(output / "finding_6_detectability_vs_harm.csv", index=False)
    primary.groupby([
        "protocol", "calibration_policy", "test_table", "test_generator", "contamination_mode",
    ], as_index=False).agg(
        generator_quality=("generator_quality", "mean"),
        detection_auroc=("detection_auroc", "mean"),
        mean_utility_delta=("contaminated_utility_delta", "mean"),
        worst_utility_delta=("contaminated_utility_delta", "min"),
    ).to_csv(output / "finding_6_quality_detectability_utility.csv", index=False)
    rows.groupby(["protocol", "detector", "calibration_policy"], as_index=False).agg(
        source_validation_auroc=("source_validation_auroc", "mean"),
        target_test_auroc=("detection_auroc", "mean"),
        ranking_shift=("ranking_shift", "mean"),
        source_validation_ece=("source_validation_ece", "mean"),
        target_test_ece=("detection_ece", "mean"),
        calibration_shift=("calibration_shift", "mean"),
        reference_fpr=("calibration_reference_fpr", "mean"),
        target_test_fpr=("detection_fpr", "mean"),
        threshold_fpr_shift=("threshold_fpr_shift", "mean"),
        prevalence_mae=("prevalence_absolute_error", "mean"),
        prevalence_decision_error_rate=("prevalence_decision_error", "mean"),
        governance_regret=("decision_regret", "mean"),
    ).to_csv(output / "finding_7_calibration_threshold_transfer.csv", index=False)
    rows.groupby([
        "protocol", "detector", "calibration_policy", "quantifier",
        "contamination_mode", "true_prevalence",
    ], as_index=False).agg(
        detection_error_contribution=("detection_error_contribution", "mean"),
        bag_sampling_error_contribution=("bag_sampling_error_contribution", "mean"),
        quantifier_adjustment_contribution=("quantifier_adjustment_contribution", "mean"),
        total_prevalence_error=("prevalence_error", "mean"),
        max_absolute_residual=("error_decomposition_residual", lambda x: float(np.abs(x).max())),
    ).to_csv(output / "finding_8_error_decomposition.csv", index=False)
    if not valuation_rows.empty:
        valuation_summary = valuation_rows.groupby([
            "protocol", "test_table", "test_generator", "contamination_mode",
            "true_prevalence", "valuation_method",
        ], as_index=False).agg(
            real_mean_value=("task_value", lambda x: float(x[valuation_rows.loc[x.index, "source_label"] == 0].mean())),
            synthetic_mean_value=("task_value", lambda x: float(x[valuation_rows.loc[x.index, "source_label"] == 1].mean())),
            synthetic_negative_value_rate=("task_value", lambda x: float((x[valuation_rows.loc[x.index, "source_label"] == 1] < 0).mean())),
            valued_records=("task_value", "count"),
        )
        valuation_summary["source_task_value_gap"] = (
            valuation_summary["real_mean_value"] - valuation_summary["synthetic_mean_value"]
        )
        valuation_summary.to_csv(output / "finding_6_source_task_value.csv", index=False)


def _write_statistical_inference(rows: pd.DataFrame, output: Path, primary_quantifier: str) -> None:
    group_columns = [
        "protocol", "detector", "calibration_policy", "quantifier", "contamination_mode", "true_prevalence",
    ]
    metrics = ["prevalence_absolute_error", "contaminated_utility_delta", "decision_regret"]
    summaries: list[dict[str, Any]] = []
    for keys, group in rows.groupby(group_columns, dropna=False):
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n > 1 else float("nan")
            half_width = float(student_t.ppf(.975, n - 1) * std / np.sqrt(n)) if n > 1 else float("nan")
            summaries.append({
                **base, "metric": metric, "n": n, "mean": mean, "std": std,
                "ci95_low": mean - half_width if np.isfinite(half_width) else float("nan"),
                "ci95_high": mean + half_width if np.isfinite(half_width) else float("nan"),
            })
    pd.DataFrame(summaries).to_csv(output / "statistical_summary_ci95.csv", index=False)

    primary = rows.loc[rows["quantifier"] == primary_quantifier]
    index = [
        "seed", "protocol", "test_table", "test_generator", "contamination_mode",
        "calibration_policy", "true_prevalence", "bag_index",
    ]
    paired_rows: list[dict[str, Any]] = []
    for (protocol, calibration_policy, mode, prevalence), group in primary.groupby([
        "protocol", "calibration_policy", "contamination_mode", "true_prevalence",
    ]):
        pivot = group.pivot_table(
            index=index, columns="detector", values="prevalence_absolute_error", aggfunc="first"
        )
        for left, right in combinations(sorted(pivot.columns), 2):
            paired = pivot[[left, right]].dropna()
            if len(paired) < 2:
                continue
            test = ttest_rel(paired[left], paired[right])
            paired_rows.append({
                "protocol": protocol, "contamination_mode": mode,
                "calibration_policy": calibration_policy,
                "true_prevalence": prevalence, "metric": "prevalence_absolute_error",
                "left_detector": left, "right_detector": right, "paired_n": len(paired),
                "mean_left": float(paired[left].mean()), "mean_right": float(paired[right].mean()),
                "mean_difference": float((paired[left] - paired[right]).mean()),
                "t_statistic": float(test.statistic), "p_value": float(test.pvalue),
            })
    paired_columns = [
        "protocol", "calibration_policy", "contamination_mode", "true_prevalence", "metric",
        "left_detector", "right_detector", "paired_n", "mean_left", "mean_right",
        "mean_difference", "t_statistic", "p_value",
    ]
    paired_frame = pd.DataFrame(paired_rows, columns=paired_columns)
    if not paired_frame.empty:
        order = np.argsort(paired_frame["p_value"].to_numpy())
        adjusted = np.empty(len(paired_frame), dtype=float)
        running = 0.0
        for rank, position in enumerate(order):
            candidate = min(1.0, (len(paired_frame) - rank) * paired_frame.iloc[position]["p_value"])
            running = max(running, candidate)
            adjusted[position] = running
        paired_frame["p_value_holm"] = adjusted
    paired_frame.to_csv(output / "paired_detector_tests.csv", index=False)


def validate_governance_setup(config_or_path: GovernanceConfig | str | Path) -> dict[str, Any]:
    config = config_or_path if isinstance(config_or_path, GovernanceConfig) else load_governance_config(config_or_path)
    _, registry_check = _method_registry(config)
    if not registry_check["passed"]:
        return {
            "experiment_id": config.experiment_id, "passed": False,
            "data_mode": config.data_mode, "reason": "method_registry_contract_failed",
            "dependency_checks": [registry_check], "protocol_checks": [],
            "output_dir": str(config.output_dir),
        }
    if config.data_mode == "registry" and (config.registry_path is None or not config.registry_path.is_file()):
        return {
            "experiment_id": config.experiment_id,
            "passed": False,
            "data_mode": config.data_mode,
            "reason": "pool_registry_missing",
            "registry_path": str(config.registry_path),
            "dependency_checks": [registry_check],
            "protocol_checks": [],
            "output_dir": str(config.output_dir),
        }
    source = _source(config, config.seeds[0])
    dependency_checks: list[dict[str, Any]] = [registry_check]
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
    registry, _ = _method_registry(config)
    write_json(registry, output / "method_registry_snapshot.json")
    resolved = json.loads(json.dumps(asdict(config), default=str))
    write_json(resolved, output / "resolved_config.json")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    valuation_records: list[dict[str, Any]] = []
    valuation_seen: set[tuple[Any, ...]] = set()
    protocol_manifests: dict[str, Any] = {}
    detector_diagnostics: dict[str, Any] = {}

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
                if detector_name == "datum_ta" and len(spec.train_tables) < 2:
                    # A table-adversarial head has no identifiable adaptation
                    # objective when every training record carries one table ID.
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
                raw_test = detector.predict_score(detector_test)
                diagnostic_key = f"{seed}:{protocol}:{detector_name}"
                detector_diagnostics[diagnostic_key] = {
                    "seed": seed, "protocol": protocol, "detector": detector_name,
                    "raw_validation_score_min": finite_or_none(np.min(raw_val)),
                    "raw_validation_score_max": finite_or_none(np.max(raw_val)),
                    "raw_validation_score_ptp": finite_or_none(np.ptp(raw_val)),
                    "raw_test_score_min": finite_or_none(np.min(raw_test)),
                    "raw_test_score_max": finite_or_none(np.max(raw_test)),
                    "raw_test_score_ptp": finite_or_none(np.ptp(raw_test)),
                    "provenance": detector.get_provenance(),
                }
                write_json(detector_diagnostics, output / "detector_diagnostics.json")
                if isinstance(detector, DeepTextDetector):
                    raw_validation_auroc = float(roc_auc_score(val_labels, raw_val))
                    detector_diagnostics[diagnostic_key]["raw_validation_auroc"] = raw_validation_auroc
                    write_json(detector_diagnostics, output / "detector_diagnostics.json")
                    if (
                        not np.isfinite(raw_val).all()
                        or np.ptp(raw_val) < 1e-8
                        or raw_validation_auroc < .52
                    ):
                        raise RuntimeError(
                            "degenerate_deep_detector_scores:"
                            f"{diagnostic_key}:ptp={np.ptp(raw_val):.3e}:"
                            f"auroc={raw_validation_auroc:.4f}"
                        )
                calibration_policies = _calibration_policies(
                    config, detector, source, spec, seed, raw_val, val_labels,
                )
                policy_metrics: dict[str, dict[str, float]] = {}
                policy_quantifiers: dict[str, dict[str, ScoreQuantifier]] = {}
                policy_diagnostics: dict[str, Any] = {}
                for policy_name, policy in calibration_policies.items():
                    val_scores = policy.calibrator.predict(raw_val)
                    test_scores = policy.calibrator.predict(raw_test)
                    policy_metrics[policy_name] = detection_metrics(test_labels, test_scores, policy.threshold)
                    policy_quantifiers[policy_name] = {
                        method: ScoreQuantifier(method).fit(
                            policy.quantifier_scores, policy.quantifier_labels, threshold=policy.threshold,
                        )
                        for method in config.quantifiers
                    }
                    source_metrics = detection_metrics(val_labels, val_scores, policy.threshold)
                    policy_diagnostics[policy_name] = {
                        "calibrated_source_validation_score_ptp": finite_or_none(np.ptp(val_scores)),
                        "calibrated_target_test_score_ptp": finite_or_none(np.ptp(test_scores)),
                        "selected_threshold": finite_or_none(policy.threshold),
                        "reference_fpr": finite_or_none(policy.reference_fpr),
                        "reference_tpr": finite_or_none(policy.reference_tpr),
                        "source_validation_auroc": source_metrics["auroc"],
                        "source_validation_ece": source_metrics["ece"],
                        "uses_target_real_records": policy.uses_target_real_records,
                        "uses_target_labels": policy.uses_target_labels,
                        "deployment_status": policy.deployment_status,
                    }
                detector_diagnostics[diagnostic_key]["calibration_policies"] = policy_diagnostics
                write_json(detector_diagnostics, output / "detector_diagnostics.json")

                for table_index, test_table in enumerate(spec.test_tables):
                    table = source.table(test_table)
                    real_splits = _split(table.real, seed + table_index * 113)
                    pure_test = real_splits["final_test"]
                    downstream_real = real_splits["downstream_train"]
                    for generator_index, test_generator in enumerate(spec.test_generators):
                        synth_split = _split(
                            table.synthetic[test_generator], seed + table_index * 113 + generator_index * 997 + 31
                        )["downstream_train"]
                        for contamination_mode, prevalence in [
                            (mode, rate)
                            for mode in config.contamination_modes
                            for rate in config.prevalence_rates
                            if not (mode == "append" and rate >= 1)
                        ]:
                            for bag_index in range(config.bags_per_rate):
                                bag_seed = seed + table_index * 100003 + generator_index * 1009 + int(prevalence * 1000) * 17 + bag_index
                                bag = exact_mixture(
                                    downstream_real, synth_split, config.bag_size, prevalence,
                                    bag_seed, mode=contamination_mode,
                                )
                                valuation_key = (
                                    seed, protocol, test_table, test_generator,
                                    contamination_mode, prevalence, bag_index,
                                )
                                if (
                                    config.valuation_enabled
                                    and bag_index < config.valuation_bags_per_rate
                                    and valuation_key not in valuation_seen
                                ):
                                    valuation_records.extend(_valuation_rows(
                                        bag, pure_test, table.target_column,
                                        config.valuation_methods, config.valuation_sample_limit,
                                        config.valuation_oob_estimators, bag_seed,
                                        {
                                            "seed": seed, "protocol": protocol,
                                            "test_table": test_table, "test_generator": test_generator,
                                            "contamination_mode": contamination_mode,
                                            "true_prevalence": float(bag["source_label"].mean()),
                                            "bag_index": bag_index,
                                        },
                                    ))
                                    valuation_seen.add(valuation_key)
                                raw_bag_scores = detector.predict_score(bag)
                                actual_prevalence = float(bag["source_label"].mean())
                                oracle_clean = _source_cleanup(bag)
                                evaluate_utility = bag_index < config.utility_bags_per_rate
                                if evaluate_utility:
                                    clean_train = sample_rows(downstream_real, config.bag_size, bag_seed + 71)
                                    clean_utility = _utility(clean_train, pure_test, table.target_column, bag_seed, config.max_cpu_threads)
                                    contaminated_utility = _utility(bag, pure_test, table.target_column, bag_seed, config.max_cpu_threads)
                                    oracle_cleanup_utility = _utility(
                                        oracle_clean, pure_test, table.target_column, bag_seed, config.max_cpu_threads
                                    )
                                else:
                                    clean_utility = contaminated_utility = float("nan")
                                    oracle_cleanup_utility = float("nan")
                                for calibration_policy, calibration in calibration_policies.items():
                                    det_metrics = policy_metrics[calibration_policy]
                                    threshold = calibration.threshold
                                    bag_scores = calibration.calibrator.predict(raw_bag_scores)
                                    detector_clean = _detector_cleanup(bag, bag_scores, threshold)
                                    if evaluate_utility:
                                        detector_cleanup_utility = _utility(
                                            detector_clean, pure_test, table.target_column, bag_seed,
                                            config.max_cpu_threads,
                                        )
                                    else:
                                        detector_cleanup_utility = float("nan")
                                    predictions = bag_scores >= threshold
                                    observed_positive_rate = float(predictions.mean())
                                    true_source = bag["source_label"].to_numpy(dtype=int)
                                    cleanup_precision = float(true_source[predictions].mean()) if predictions.any() else 0.0
                                    cleanup_recall = float(predictions[true_source == 1].mean()) if (true_source == 1).any() else 1.0
                                    propagation = analytical_positive_rate(
                                        actual_prevalence, det_metrics["tpr"], det_metrics["fpr"],
                                    )
                                    for method, quantifier in policy_quantifiers[calibration_policy].items():
                                        try:
                                            estimate = quantifier.predict_prevalence(bag_scores)["clipped"]
                                            quantifier_status = "ok"
                                        except ValueError as exc:
                                            estimate = float("nan")
                                            quantifier_status = f"failed:{exc}"
                                        estimated_decision = bool(np.isfinite(estimate) and estimate >= config.governance_prevalence_threshold)
                                        true_prevalence_decision = bool(actual_prevalence >= config.governance_prevalence_threshold)
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
                                        detection_contribution = propagation["expected_positive_rate"] - actual_prevalence
                                        sampling_contribution = observed_positive_rate - propagation["expected_positive_rate"]
                                        quantifier_contribution = estimate - observed_positive_rate if np.isfinite(estimate) else float("nan")
                                        decomposition_sum = detection_contribution + sampling_contribution + quantifier_contribution
                                        decomposition_residual = error["prevalence_error"] - decomposition_sum
                                        contaminated_delta = contaminated_utility - clean_utility
                                        source_diagnostics = policy_diagnostics[calibration_policy]
                                        row = {
                                        "experiment_id": config.experiment_id,
                                        "run_type": config.run_type,
                                        "seed": seed,
                                        "protocol": protocol,
                                        "detector": detector_name,
                                        "calibration_policy": calibration_policy,
                                        "calibration_deployment_status": calibration.deployment_status,
                                        "calibration_uses_target_real_records": calibration.uses_target_real_records,
                                        "calibration_uses_target_labels": calibration.uses_target_labels,
                                        "quantifier": method,
                                        "quantifier_status": quantifier_status,
                                        "test_table": test_table,
                                        "test_generator": test_generator,
                                        "generator_quality": table.synthetic_quality.get(test_generator, float("nan")),
                                        "bag_id": f"{seed}-{protocol}-{detector_name}-{calibration_policy}-{test_table}-{test_generator}-{contamination_mode}-{prevalence:.3f}-{bag_index}",
                                        "bag_index": bag_index,
                                        "bag_size": len(bag),
                                        "contamination_mode": contamination_mode,
                                        "nominal_prevalence": prevalence,
                                        "true_prevalence": actual_prevalence,
                                        "estimated_prevalence": estimate,
                                        **error,
                                        "source_validation_auroc": source_diagnostics["source_validation_auroc"],
                                        "source_validation_ece": source_diagnostics["source_validation_ece"],
                                        "detection_auroc": det_metrics["auroc"],
                                        "detection_auprc": det_metrics["auprc"],
                                        "detection_brier": det_metrics["brier"],
                                        "detection_ece": det_metrics["ece"],
                                        "detection_fpr": det_metrics["fpr"],
                                        "detection_tpr": det_metrics["tpr"],
                                        "detector_threshold": threshold,
                                        "calibration_reference_fpr": calibration.reference_fpr,
                                        "calibration_reference_tpr": calibration.reference_tpr,
                                        "ranking_shift": det_metrics["auroc"] - source_diagnostics["source_validation_auroc"],
                                        "calibration_shift": det_metrics["ece"] - source_diagnostics["source_validation_ece"],
                                        "threshold_fpr_shift": det_metrics["fpr"] - calibration.reference_fpr,
                                        "artifact_auroc": artifact["artifact_auroc"],
                                        "artifact_gate_passed": artifact_row["artifact_gate_passed"],
                                        **propagation,
                                        "observed_positive_rate": observed_positive_rate,
                                        "detection_error_contribution": detection_contribution,
                                        "bag_sampling_error_contribution": sampling_contribution,
                                        "quantifier_adjustment_contribution": quantifier_contribution,
                                        "error_decomposition_sum": decomposition_sum,
                                        "error_decomposition_residual": decomposition_residual,
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
                                        "true_prevalence_decision_remediate": true_prevalence_decision,
                                        "estimated_decision_remediate": estimated_decision,
                                        "prevalence_decision_error": int(estimated_decision != true_prevalence_decision),
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
    valuation_columns = [
        "seed", "protocol", "test_table", "test_generator", "contamination_mode",
        "true_prevalence", "bag_index", "valuation_method", "valuation_row_index",
        "record_id", "source_label", "task_value", "oob_coverage",
    ]
    valuation_frame = pd.DataFrame(valuation_records, columns=valuation_columns)
    evidence.to_csv(output / "governance_evidence.csv", index=False)
    artifact_frame.to_csv(output / "format_artifact_audit.csv", index=False)
    valuation_frame.to_csv(output / "record_valuation.csv", index=False)
    _write_findings(evidence, artifact_frame, valuation_frame, output, config.primary_quantifier)
    _write_statistical_inference(evidence, output, config.primary_quantifier)
    write_json(protocol_manifests, output / "protocol_manifests.json")
    quantifier_failures = evidence["quantifier_status"].astype(str).ne("ok")
    primary_failures = quantifier_failures & evidence["quantifier"].eq(config.primary_quantifier)
    analysis_ready = not bool(primary_failures.any())
    summary = {
        "experiment_id": config.experiment_id,
        "run_type": config.run_type,
        "status": "complete" if analysis_ready else "complete_with_method_failures",
        "analysis_ready": analysis_ready,
        "formal_inclusion": config.run_type == "formal" and analysis_ready,
        "created_at": _utc_now(),
        "seconds": time.perf_counter() - started,
        "evidence_rows": len(evidence),
        "protocols": sorted(evidence["protocol"].unique().tolist()),
        "detectors": sorted(evidence["detector"].unique().tolist()),
        "calibration_policies": sorted(evidence["calibration_policy"].unique().tolist()),
        "quantifiers": sorted(evidence["quantifier"].unique().tolist()),
        "max_absolute_error_decomposition_residual": finite_or_none(
            np.abs(evidence["error_decomposition_residual"].to_numpy(float)).max()
        ),
        "low_prevalence_rows": int(evidence["true_prevalence"].isin([.05, .10]).sum()),
        "artifact_gate_failures": int((~artifact_frame["artifact_gate_passed"]).sum()),
        "valuation_rows": len(valuation_frame),
        "quantifier_failure_rows": int(quantifier_failures.sum()),
        "primary_quantifier_failure_rows": int(primary_failures.sum()),
        "mean_prevalence_mae": finite_or_none(evidence["prevalence_absolute_error"].mean()),
        "mean_decision_regret": finite_or_none(evidence["decision_regret"].mean()),
        "outputs": {
            "evidence": str(output / "governance_evidence.csv"),
            "findings": [str(path) for path in sorted(output.glob("finding_*.csv"))],
            "valuation": str(output / "record_valuation.csv"),
            "statistical_summary": str(output / "statistical_summary_ci95.csv"),
            "paired_tests": str(output / "paired_detector_tests.csv"),
            "method_registry": str(output / "method_registry_snapshot.json"),
            "detector_diagnostics": str(output / "detector_diagnostics.json"),
        },
    }
    write_json(summary, output / "summary.json")
    return summary
