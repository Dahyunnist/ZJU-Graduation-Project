"""Paired replacement study of synthetic data's effect on real-test risk.

Exploratory only. The generator pools were trained once, so split seeds are not
independent generator replicates. No detector, quantifier, or test-set tuning is
used in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


META = {"record_id", "table_id", "generator", "source_label", "source_type", "schema_columns"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def _labels(frame: pd.DataFrame, target: str, positive: str) -> np.ndarray:
    values = frame[target].astype(str)
    unexpected = set(values.unique()) - {positive}
    if len(unexpected) > 1:
        raise ValueError(f"Unexpected target labels: {sorted(unexpected | {positive})}")
    return (values == positive).astype(int).to_numpy()


def _feature_columns(real: pd.DataFrame, target: str) -> list[str]:
    return [c for c in real.columns if c != target and c not in META]


def paired_split(real: pd.DataFrame, target: str, seed: int, test_fraction: float,
                 train_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    classes = sorted(real[target].astype(str).unique())
    if len(classes) != 2:
        raise ValueError(f"Binary task required, got {classes}")
    positive = classes[-1]
    indices = np.arange(len(real))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=seed,
        stratify=_labels(real, target, positive),
    )
    if len(train_idx) < train_size + train_size // 2:
        raise ValueError("Insufficient real training reserve for independent real-donor control")
    core_idx, reserve_idx = train_test_split(
        train_idx, train_size=train_size, random_state=_seed(seed, "core"),
        stratify=_labels(real.iloc[train_idx], target, positive),
    )
    assert not (set(core_idx) & set(test_idx) or set(reserve_idx) & set(test_idx)
                or set(core_idx) & set(reserve_idx))
    return core_idx, reserve_idx, test_idx, positive


def class_shuffle(frame: pd.DataFrame, target: str, feature_columns: list[str], seed: int) -> pd.DataFrame:
    """Preserve each feature's exact within-label multiset, scramble joint rows."""
    result = frame.copy()
    rng = np.random.default_rng(seed)
    labels = frame[target].astype(str).to_numpy()
    for klass in np.unique(labels):
        positions = np.flatnonzero(labels == klass)
        for column in feature_columns:
            result.iloc[positions, result.columns.get_loc(column)] = rng.permutation(frame.iloc[positions][column].to_numpy())
    return result


def label_permute(frame: pd.DataFrame, target: str, seed: int) -> pd.DataFrame:
    """Keep every feature row and the target multiset, break their pairing."""
    result = frame.copy()
    result[target] = np.random.default_rng(seed).permutation(frame[target].to_numpy())
    return result


def tail_undercoverage(pool: pd.DataFrame, template: pd.DataFrame, target: str,
                       feature: str, threshold: float, seed: int) -> pd.DataFrame:
    """Select non-tail generator records matching the natural draw's label counts."""
    rng = np.random.default_rng(seed)
    eligible = pool.loc[pd.to_numeric(pool[feature], errors="coerce") < threshold]
    selected: list[int] = []
    for label, count in template[target].astype(str).value_counts().items():
        candidates = eligible.index[eligible[target].astype(str) == label].to_numpy()
        if len(candidates) < count:
            raise ValueError(f"tail_undercoverage insufficient {label} donors: {len(candidates)} < {count}")
        selected.extend(rng.choice(candidates, size=int(count), replace=False).tolist())
    result = pool.loc[selected].sample(frac=1, random_state=seed)
    assert result[target].astype(str).value_counts().to_dict() == template[target].astype(str).value_counts().to_dict()
    return result


def _model(name: str, frame: pd.DataFrame, seed: int) -> Pipeline:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in frame if col not in numeric]
    branches = []
    if numeric:
        branches.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        branches.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(branches)
    if name == "lr":
        classifier = LogisticRegression(solver="liblinear", max_iter=500, random_state=seed)
    elif name == "rf":
        classifier = RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=3, random_state=seed, n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported learner {name}")
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def _risk(y: np.ndarray, probability: np.ndarray, test_tail: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "tail_log_loss": None,
        "non_tail_log_loss": None,
        "positive_log_loss": None,
    }
    for key, mask in (("tail_log_loss", test_tail), ("non_tail_log_loss", ~test_tail),
                      ("positive_log_loss", y == 1)):
        if mask.sum():
            result[key] = float(log_loss(y[mask], probability[mask], labels=[0, 1]))
    return result


def _record_fit(train: pd.DataFrame, test: pd.DataFrame, *, table: str, generator: str,
                seed: int, model_name: str, prevalence: float, intervention: str,
                target: str, positive: str, features: list[str], threshold: float,
                tail_feature: str, selected_ids: list[str], destination: Path) -> dict[str, Any]:
    train_y = _labels(train, target, positive)
    test_y = _labels(test, target, positive)
    if len(np.unique(train_y)) != 2:
        raise ValueError("Single-class training set")
    start = time.monotonic()
    model = _model(model_name, train[features], _seed(seed, table, model_name))
    model.fit(train[features], train_y)
    probability = model.predict_proba(test[features])[:, 1]
    elapsed = time.monotonic() - start
    test_tail = (pd.to_numeric(test[tail_feature], errors="coerce") >= threshold).fillna(False).to_numpy()
    name = f"{table}_{generator}_{model_name}_p{prevalence:g}_{intervention}"
    prediction = pd.DataFrame({
        "test_row_id": [f"{table}:real:{index}" for index in test.index],
        "y_true": test_y, "p_task": probability, "tail_group": test_tail.astype(int),
    })
    pred_path = destination / "predictions" / f"{name}.csv.gz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(pred_path, index=False, compression="gzip")
    result = {
        "table": table, "generator": generator, "seed": seed, "model": model_name,
        "prevalence": prevalence, "intervention": intervention,
        "train_n": len(train), "train_positive_rate": float(train_y.mean()),
        "train_tail_rate": float((pd.to_numeric(train[tail_feature], errors="coerce") >= threshold).mean()),
        "test_n": len(test), "test_positive_rate": float(test_y.mean()),
        "test_tail_n": int(test_tail.sum()), "wall_seconds": round(elapsed, 3),
        "train_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "prediction_file": str(pred_path.relative_to(destination)),
        **_risk(test_y, probability, test_tail),
    }
    return result


def _read_config(path: Path) -> tuple[dict[str, Any], Path]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must be a mapping")
    registry = (path.parent / config["registry_path"]).resolve()
    return config, registry


def run_shard(config_path: Path, registry_path: Path | None, output_root: Path, table: str, seed: int,
              *, preflight: bool = False) -> dict[str, Any]:
    config, configured_registry = _read_config(config_path)
    if registry_path is None:
        registry_path = configured_registry
    if table not in config["tables"] or seed not in config["seeds"]:
        raise ValueError("Requested table/seed not in frozen design")
    registry = pd.read_csv(registry_path)
    entries = registry.loc[registry.table_id == table]
    if set(entries.generator) != set(config["generators"]) or entries.target_column.nunique() != 1:
        raise ValueError(f"Incomplete registry entries for {table}")
    target = str(entries.target_column.iloc[0])
    real_path = registry_path.parent / str(entries.real_path.iloc[0])
    real = pd.read_csv(real_path)
    generators = {
        str(entry.generator): pd.read_csv(registry_path.parent / str(entry.synthetic_path))
        for entry in entries.itertuples()
    }
    features = _feature_columns(real, target)
    for generator, frame in generators.items():
        if set(features + [target]) - set(frame.columns):
            raise ValueError(f"Generator {generator} has incompatible columns")
    core_idx, reserve_idx, test_idx, positive = paired_split(
        real, target, seed, float(config["test_fraction"]), int(config["train_size"]),
    )
    core, reserve, test = real.iloc[core_idx].copy(), real.iloc[reserve_idx].copy(), real.iloc[test_idx].copy()
    feature = config["tail_features"][table]
    if feature not in features or not pd.api.types.is_numeric_dtype(real[feature]):
        raise ValueError(f"Invalid predeclared tail feature {feature}")
    threshold = float(pd.to_numeric(core[feature], errors="coerce").quantile(config["tail_quantile"]))
    maximum = max(round(len(core) * float(p)) for p in config["prevalences"])
    if len(reserve) < maximum or any(len(pool) < maximum for pool in generators.values()):
        raise ValueError("Insufficient generator or real-donor reserve")
    context = {
        "table": table, "seed": seed, "target": target, "positive_label": positive,
        "real_rows": len(real), "real_train_core": len(core), "real_train_reserve": len(reserve),
        "test_rows": len(test), "generator_rows": {k: len(v) for k, v in generators.items()},
        "tail_feature": feature, "tail_threshold": threshold,
        "core_ids_sha256": hashlib.sha256("\n".join(map(str, core.index)).encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256("\n".join(map(str, test.index)).encode()).hexdigest(),
        "config_sha256": sha256_file(config_path), "registry_sha256": sha256_file(registry_path),
        "real_sha256": sha256_file(real_path),
        "generator_sha256": {str(e.generator): sha256_file(registry_path.parent / str(e.synthetic_path)) for e in entries.itertuples()},
    }
    if preflight:
        return context

    final = output_root / f"{table}_seed{seed}"
    if (final / "complete.json").exists():
        existing = json.loads((final / "complete.json").read_text(encoding="utf-8"))
        if existing["context"]["config_sha256"] != context["config_sha256"] or existing["context"]["registry_sha256"] != context["registry_sha256"]:
            raise ValueError("Existing shard belongs to a different frozen config/registry")
        return {"status": "already_complete", "table": table, "seed": seed}
    partial = output_root / f".{table}_seed{seed}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for prevalence in config["prevalences"]:
        n_synthetic = round(len(core) * float(prevalence))
        slot_rng = np.random.default_rng(_seed(seed, table, prevalence, "slots"))
        replace_positions = slot_rng.choice(np.arange(len(core)), size=n_synthetic, replace=False)
        core_ids = [f"{table}:real:{i}" for i in core.index]
        reference = core.copy().reset_index(drop=True)
        if n_synthetic:
            donor_rng = np.random.default_rng(_seed(seed, table, prevalence, "real_donor"))
            donor_idx = donor_rng.choice(reserve.index.to_numpy(), size=n_synthetic, replace=False)
            real_control = reference.copy()
            real_control.iloc[replace_positions] = real.loc[donor_idx, reference.columns].to_numpy()
            control_ids = core_ids.copy()
            for pos, donor in zip(replace_positions, donor_idx):
                control_ids[int(pos)] = f"{table}:real_donor:{donor}"
        for model_name in config["models"]:
            baseline = _record_fit(reference, test, table=table, generator="none", seed=seed,
                model_name=model_name, prevalence=float(prevalence), intervention="real_reference",
                target=target, positive=positive, features=features, threshold=threshold,
                tail_feature=feature, selected_ids=core_ids, destination=partial)
            results.append(baseline)
            if n_synthetic:
                results.append(_record_fit(real_control, test, table=table, generator="none", seed=seed,
                    model_name=model_name, prevalence=float(prevalence), intervention="independent_real_donor",
                    target=target, positive=positive, features=features, threshold=threshold,
                    tail_feature=feature, selected_ids=control_ids, destination=partial))
        if n_synthetic == 0:
            continue
        for generator, pool in generators.items():
            chosen_rng = np.random.default_rng(_seed(seed, table, generator, prevalence, "natural"))
            chosen_indices = chosen_rng.choice(pool.index.to_numpy(), size=n_synthetic, replace=False)
            natural = pool.loc[chosen_indices].copy()
            conditions: dict[str, pd.DataFrame] = {"natural": natural}
            if "class_shuffle" in config["interventions"]:
                conditions["class_shuffle"] = class_shuffle(natural, target, features,
                    _seed(seed, table, generator, prevalence, "class_shuffle"))
            if "label_permute" in config["interventions"]:
                conditions["label_permute"] = label_permute(natural, target,
                    _seed(seed, table, generator, prevalence, "label_permute"))
            if "tail_undercoverage" in config["interventions"]:
                try:
                    conditions["tail_undercoverage"] = tail_undercoverage(pool, natural, target, feature,
                        threshold, _seed(seed, table, generator, prevalence, "tail_undercoverage"))
                except ValueError as exc:
                    failures.append({"table": table, "seed": seed, "generator": generator,
                        "prevalence": prevalence, "intervention": "tail_undercoverage", "reason": str(exc)})
            for intervention, synthetic in conditions.items():
                mixed = reference.copy()
                mixed.iloc[replace_positions] = synthetic[reference.columns].to_numpy()
                mixed_ids = core_ids.copy()
                ids = synthetic.index.to_numpy()
                for pos, pool_index in zip(replace_positions, ids):
                    mixed_ids[int(pos)] = f"{table}:{generator}:{int(pool_index)}:{intervention}"
                selections.append({
                    "table": table, "seed": seed, "generator": generator, "prevalence": prevalence,
                    "intervention": intervention, "replaced_positions": json.dumps(replace_positions.tolist()),
                    "synthetic_row_ids": json.dumps([str(x) for x in ids]),
                    "label_positive_rate": float(_labels(synthetic, target, positive).mean()),
                    "tail_rate": float((pd.to_numeric(synthetic[feature], errors="coerce") >= threshold).mean()),
                    "changed_label_count": int((synthetic[target].astype(str).to_numpy() != natural[target].astype(str).to_numpy()).sum()),
                })
                for model_name in config["models"]:
                    results.append(_record_fit(mixed, test, table=table, generator=generator,
                        seed=seed, model_name=model_name, prevalence=float(prevalence),
                        intervention=intervention, target=target, positive=positive,
                        features=features, threshold=threshold, tail_feature=feature,
                        selected_ids=mixed_ids, destination=partial))
    result_frame = pd.DataFrame(results)
    baseline = result_frame.loc[result_frame.intervention == "real_reference", ["model", "prevalence", "log_loss"]]
    if baseline.groupby("model").log_loss.nunique().max() != 1:
        raise AssertionError("Pi=0/reference invariance failed across rates")
    result_frame.to_csv(partial / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(partial / "selections.csv", index=False)
    pd.DataFrame(failures, columns=["table", "seed", "generator", "prevalence", "intervention", "reason"]).to_csv(partial / "infeasible.csv", index=False)
    completed = {"status": "complete", "context": context, "fit_count": len(results),
                 "infeasible_count": len(failures), "wall_seconds": round(time.monotonic() - started, 3)}
    _atomic_json(completed, partial / "complete.json")
    if final.exists():
        raise FileExistsError(final)
    partial.replace(final)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run_shard(args.config.resolve(), args.registry.resolve() if args.registry else None,
                       args.output.resolve(), args.table, args.seed, preflight=args.preflight)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
