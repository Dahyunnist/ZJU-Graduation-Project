"""Read-only audit and descriptive paired analysis of the frozen E1 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from tabpollution.studies.e1 import (
    _seed, class_shuffle, label_permute, paired_split, sha256_file, tail_undercoverage,
)


KEY = ["table", "seed", "model", "prevalence"]
METRICS = ["log_loss", "brier", "auroc", "auprc", "tail_log_loss", "non_tail_log_loss", "positive_log_loss"]
EXTRA = ["negative_log_loss", "clipped_1e3_log_loss", "wrong_endpoint_count", "top1pct_loss_share"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    registry = pd.read_csv(args.pools / "pool_registry.csv")
    all_metrics, all_structure, checks, inputs = [], [], [], {}

    def check(name, passed, shard, detail=""):
        checks.append({"shard": shard, "check": name, "passed": bool(passed), "detail": str(detail)})

    for table in config["tables"]:
        reg = registry.loc[registry.table_id == table]
        target = reg.target_column.iloc[0]
        real_path = args.pools / reg.real_path.iloc[0]
        real = pd.read_csv(real_path)
        pools = {r.generator: pd.read_csv(args.pools / r.synthetic_path) for r in reg.itertuples()}
        for seed in config["seeds"]:
            shard = f"{table}_seed{seed}"
            directory = args.input / shard
            complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            context = complete["context"]
            metrics = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
            selections = pd.read_csv(directory / "selections.csv")
            check("42_fits", len(metrics) == 42 == complete["fit_count"], shard)
            check("no_infeasible", complete["infeasible_count"] == 0 and pd.read_csv(directory / "infeasible.csv").empty, shard)
            check("unique_metric_keys", not metrics.duplicated(KEY + ["generator", "intervention"]).any(), shard)
            check("real_hash", sha256_file(real_path) == context["real_sha256"], shard)
            # Git normalizes line endings; compare normalized config bytes as well.
            normalized_config = args.config.read_bytes().replace(b"\r\n", b"\n")
            check("config_hash", hashlib.sha256(normalized_config).hexdigest() == context["config_sha256"], shard)
            for r in reg.itertuples():
                check(f"pool_hash_{r.generator}", sha256_file(args.pools / r.synthetic_path) == context["generator_sha256"][r.generator], shard)
            core_idx, reserve_idx, test_idx, positive = paired_split(real, target, seed, config["test_fraction"], config["train_size"])
            core = real.iloc[core_idx]
            test = real.iloc[test_idx]
            tail_feature = config["tail_features"][table]
            threshold = float(core[tail_feature].quantile(config["tail_quantile"]))
            check("tail_threshold", threshold == context["tail_threshold"], shard)
            features = [c for c in real if c != target]
            numeric = core[features].select_dtypes(include="number").columns.tolist()
            check("core_ids", hashlib.sha256("\n".join(map(str, core_idx)).encode()).hexdigest() == context["core_ids_sha256"], shard)
            check("test_ids", hashlib.sha256("\n".join(map(str, test_idx)).encode()).hexdigest() == context["test_ids_sha256"], shard)
            expected_y = (test[target].astype(str) == positive).astype(int).to_numpy()
            expected_ids = [f"{table}:real:{i}" for i in test_idx]
            expected_tail = (test[tail_feature] >= threshold).astype(int).to_numpy()
            baseline_predictions = {}
            for index, row in metrics.iterrows():
                path = directory / row.prediction_file
                pred = pd.read_csv(path, float_precision="round_trip")
                inputs[str(path.relative_to(args.input))] = sha256_file(path)
                p = pred.p_task.to_numpy()
                y = pred.y_true.to_numpy()
                check("prediction_alignment", pred.test_row_id.tolist() == expected_ids and np.array_equal(y, expected_y)
                      and np.array_equal(pred.tail_group.to_numpy(), expected_tail), shard, row.prediction_file)
                check("probability_valid", np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), shard, row.prediction_file)
                recomputed = {"log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
                              "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
                for metric, mask in [("tail_log_loss", expected_tail == 1), ("non_tail_log_loss", expected_tail == 0),
                                     ("positive_log_loss", y == 1)]:
                    recomputed[metric] = log_loss(y[mask], p[mask], labels=[0, 1])
                check("metric_recalculation", all(np.isclose(row[m], v, atol=1e-11, rtol=1e-10) for m, v in recomputed.items()),
                      shard, row.prediction_file + " " + json.dumps({m: float(v - row[m]) for m, v in recomputed.items()}))
                eps = np.finfo(float).eps
                clipped = np.clip(p, eps, 1 - eps)
                losses = -y * np.log(clipped) - (1-y) * np.log1p(-clipped)
                metrics.loc[index, "negative_log_loss"] = float(losses[y == 0].mean())
                metrics.loc[index, "clipped_1e3_log_loss"] = log_loss(y, np.clip(p, 1e-3, 1 - 1e-3), labels=[0, 1])
                metrics.loc[index, "wrong_endpoint_count"] = int(((y == 1) & (p == 0) | (y == 0) & (p == 1)).sum())
                metrics.loc[index, "top1pct_loss_share"] = float(np.sort(losses)[-max(1, int(np.ceil(len(y) * .01))):].sum() / losses.sum())
                if row.intervention == "real_reference":
                    if row.model in baseline_predictions:
                        check("reference_prediction_identity", np.array_equal(p, baseline_predictions[row.model]), shard, row.model)
                    else:
                        baseline_predictions[row.model] = p
            check("fixed_train_budget", (metrics.train_n == config["train_size"]).all(), shard)
            for prevalence in [.1, .5]:
                n = round(config["train_size"] * prevalence)
                positions = np.random.default_rng(_seed(seed, table, prevalence, "slots")).choice(np.arange(len(core)), n, replace=False)
                selected = selections.loc[selections.prevalence == prevalence]
                check("shared_replacement_positions", all(json.loads(value) == positions.tolist() for value in selected.replaced_positions), shard, prevalence)
                for generator, pool in pools.items():
                    chosen = np.random.default_rng(_seed(seed, table, generator, prevalence, "natural")).choice(pool.index.to_numpy(), n, replace=False)
                    natural = pool.loc[chosen].copy()
                    variants = {
                        "natural": natural,
                        "class_shuffle": class_shuffle(natural, target, features, _seed(seed, table, generator, prevalence, "class_shuffle")),
                        "label_permute": label_permute(natural, target, _seed(seed, table, generator, prevalence, "label_permute")),
                        "tail_undercoverage": tail_undercoverage(pool, natural, target, tail_feature, threshold,
                                                                  _seed(seed, table, generator, prevalence, "tail_undercoverage")),
                    }
                    for intervention, syn in variants.items():
                        selection = selected.loc[(selected.generator == generator) & (selected.intervention == intervention)].iloc[0]
                        check("synthetic_source_ids", json.loads(selection.synthetic_row_ids) == list(map(str, syn.index)), shard, f"{generator}/{prevalence}/{intervention}")
                        check("label_counts_preserved", syn[target].value_counts().to_dict() == natural[target].value_counts().to_dict(), shard, intervention)
                        if intervention == "class_shuffle":
                            preserved = all(sorted(syn.loc[syn[target] == label, feature].astype(str).tolist()) ==
                                            sorted(natural.loc[natural[target] == label, feature].astype(str).tolist())
                                            for label in natural[target].unique() for feature in features)
                            check("class_conditional_marginals_exact", preserved, shard, generator)
                        if intervention == "label_permute":
                            check("label_permute_features_exact", syn[features].equals(natural[features]), shard, generator)
                        if intervention == "tail_undercoverage":
                            check("tail_excluded", (syn[tail_feature] < threshold).all(), shard, generator)
                        mixed = core.copy().reset_index(drop=True)
                        mixed.iloc[positions] = syn[mixed.columns].to_numpy()
                        submetrics = metrics.loc[(metrics.generator == generator) & (metrics.prevalence == prevalence) & (metrics.intervention == intervention)]
                        check("mixture_label_rate", np.allclose(submetrics.train_positive_rate, (mixed[target].astype(str) == positive).mean()), shard, intervention)
                        check("mixture_tail_rate", np.allclose(submetrics.train_tail_rate, (mixed[tail_feature] >= threshold).mean()), shard, intervention)
                        correlation_errors = []
                        for label in core[target].unique():
                            rc = core.loc[core[target] == label, numeric].corr().to_numpy()
                            sc = syn.loc[syn[target] == label, numeric].corr().to_numpy()
                            upper = np.triu_indices(len(numeric), 1)
                            difference = np.abs(rc[upper] - sc[upper])
                            correlation_errors.extend(difference[np.isfinite(difference)].tolist())
                        all_structure.append({"table": table, "seed": seed, "generator": generator,
                            "prevalence": prevalence, "intervention": intervention,
                            "synthetic_n": len(syn), "label_rate": (syn[target].astype(str) == positive).mean(),
                            "tail_rate": (syn[tail_feature] >= threshold).mean(),
                            "class_numeric_corr_error": float(np.mean(correlation_errors)),
                            "label_change_fraction": float((syn[target].to_numpy() != natural[target].to_numpy()).mean()) if intervention != "tail_undercoverage" else None})
            all_metrics.append(metrics)
            for name in ["metrics.csv", "selections.csv", "complete.json"]:
                path = directory / name
                inputs[str(path.relative_to(args.input))] = sha256_file(path)

    all_data = pd.concat(all_metrics, ignore_index=True)
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(args.output / "audit_checks.csv", index=False)
    audit = {"checks": len(checks), "passed": int(checks_frame.passed.sum()),
             "failed": int((~checks_frame.passed).sum()), "shards": len(all_metrics), "metric_rows": len(all_data)}
    (args.output / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (args.output / "input_hashes.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    if audit["failed"]:
        raise AssertionError(checks_frame.loc[~checks_frame.passed].to_string())
    all_data.to_csv(args.output / "all_metrics.csv", index=False)
    pd.DataFrame(all_structure).to_csv(args.output / "structure_diagnostics.csv", index=False)
    contrasts = []
    for reference, comparison in [("real_reference", "vs_real"), ("independent_real_donor", "vs_real_donor"), ("natural", "vs_natural")]:
        join = KEY + (["generator"] if reference == "natural" else [])
        reference_frame = all_data.loc[all_data.intervention == reference, join + METRICS + EXTRA]
        reference_frame = reference_frame.rename(columns={m: f"reference_{m}" for m in METRICS + EXTRA})
        target = all_data.loc[~all_data.intervention.isin([reference, "real_reference"])].copy()
        if reference == "natural":
            target = target.loc[target.generator != "none"]
        paired = target.merge(reference_frame, on=join, how="inner", validate="many_to_one")
        paired["comparison"] = comparison
        for metric in METRICS + EXTRA:
            paired[f"delta_{metric}"] = paired[metric] - paired[f"reference_{metric}"]
        contrasts.append(paired)
    paired = pd.concat(contrasts, ignore_index=True)
    paired.to_csv(args.output / "paired_contrasts.csv", index=False)
    group = ["comparison", "table", "model", "generator", "prevalence", "intervention"]
    rows = []
    for keys, frame in paired.groupby(group, dropna=False):
        result = dict(zip(group, keys))
        result["seeds"] = len(frame)
        for metric in METRICS + EXTRA:
            values = frame[f"delta_{metric}"]
            result[f"{metric}_mean"] = values.mean()
            result[f"{metric}_min"] = values.min()
            result[f"{metric}_max"] = values.max()
            result[f"{metric}_positive_seeds"] = int((values > 0).sum())
        rows.append(result)
    pd.DataFrame(rows).to_csv(args.output / "descriptive_summary.csv", index=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
