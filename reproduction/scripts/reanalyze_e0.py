"""Read-only, bounded-memory E0 reanalysis of selected completed attempts.

No training imports. Historical inputs are never written. Outputs must be new.
Run with one BLAS thread and CUDA disabled on shared hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

BASE = ["seed", "protocol", "test_table", "test_generator", "contamination_mode", "nominal_prevalence", "bag_index"]
ACTION = BASE + ["detector", "calibration_policy"]
CELL = ["seed", "protocol", "detector", "calibration_policy", "quantifier", "test_table", "test_generator", "contamination_mode", "nominal_prevalence"]
BASE_VALUES = ["bag_size", "true_prevalence", "clean_utility", "contaminated_utility", "oracle_cleanup_utility", "contaminated_utility_delta"]
ACTION_VALUES = ["detector_cleanup_utility", "cleanup_precision", "cleanup_recall", "detector_threshold"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_checked(frame, keys, values):
    """Never choose an arbitrary first conflicting repeated measurement."""
    grouped = frame.groupby(keys, dropna=False, sort=False)
    sizes = grouped.size()
    for col in values:
        stats = grouped[col].agg(["min", "max", "count"])
        inconsistent = ((stats["max"] - stats["min"]).abs() > 1e-10) | (
            (stats["count"] > 0) & (stats["count"] != sizes)
        )
        if inconsistent.any():
            raise ValueError(f"Conflicting repeated field {col}: {stats.loc[inconsistent].head().to_dict()}")
    return frame[keys + values].drop_duplicates(keys).copy()


def check_mixture(frame, real_budget):
    p = frame.nominal_prevalence.to_numpy()
    append = frame.contamination_mode.eq("append").to_numpy()
    if np.any(append & (p >= 1)):
        raise ValueError("Undefined append=1 condition")
    ns = np.rint(real_budget * p).astype(int)
    ns[append] = np.rint(real_budget * p[append] / (1 - p[append])).astype(int)
    total = np.full(len(frame), real_budget)
    total[append] += ns[append]
    if not np.array_equal(total, frame.bag_size.to_numpy()):
        raise ValueError("Bag size does not match frozen mixture semantics")
    if not np.allclose(ns / total, frame.true_prevalence, atol=1e-12, rtol=0):
        raise ValueError("Actual prevalence mismatch")


def summarize(frame, groups, metrics):
    rows = []
    for key, block in frame.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(dict(zip(groups, key)) | {
                "metric": metric, "n_total": len(block), "n_valid": len(finite),
                "n_missing": len(block) - len(finite),
                "mean": float(finite.mean()) if len(finite) else np.nan,
            })
    return pd.DataFrame(rows)


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError("E0 output must be outside historical run directory")
    out.mkdir(parents=True, exist_ok=False)
    config_path = root / "resolved_config.json"
    manifest_path = root / "completed_shards_manifest.json"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    attempts = [Path(p) for p in manifest["attempts"]]
    if len(attempts) != len(set(attempts)):
        raise ValueError("Duplicate attempt paths")
    inputs = []
    for path in [config_path, manifest_path]:
        inputs.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
    bases, actions, quant, oldutility, detectors, value_summaries = [], [], [], [], [], []
    shard_keys = set()
    counts = {"evidence_rows": 0, "failed_quantifier_rows": 0, "valuation_rows": 0,
              "legacy_low_rows": 0, "corrected_low_rows": 0, "omitted_low_rows": 0,
              "zero_or_one_estimate_rows": 0}
    max_residual = 0.0
    schema = None
    for index, attempt in enumerate(attempts):
        path = attempt / "governance_evidence.csv"
        initial_hash = sha(path)
        frame = pd.read_csv(path)
        if schema is None:
            schema = list(frame.columns)
        elif list(frame.columns) != schema:
            raise ValueError("Inconsistent evidence schema")
        identity = frame[["seed", "protocol", "detector"]].drop_duplicates()
        if len(identity) != 1:
            raise ValueError("Attempt includes multiple shard identities")
        key = tuple(identity.iloc[0])
        if key in shard_keys:
            raise ValueError("Duplicate completed shard")
        shard_keys.add(key)
        if frame.duplicated(ACTION + ["quantifier"]).any():
            raise ValueError("Duplicate evidence primary keys")
        check_mixture(frame, config["bag_size"])
        expected = (len(config["contamination_modes"]) * len(config["prevalence_rates"]) - 1) * config["bags_per_rate"] * len(config["calibration_policies"]) * len(config["quantifiers"])
        if len(frame) != expected:
            raise ValueError(f"Unexpected row count {len(frame)} != {expected}")
        counts["evidence_rows"] += len(frame)
        ok = frame.quantifier_status.eq("ok")
        counts["failed_quantifier_rows"] += int((~ok).sum())
        if not np.isfinite(frame.loc[ok, "estimated_prevalence"]).all() or frame.loc[~ok, "estimated_prevalence"].notna().any():
            raise ValueError("Estimate/status mismatch")
        low_old = ok & frame.true_prevalence.isin([.05, .10])
        low_new = ok & frame.nominal_prevalence.isin([.05, .10])
        counts["legacy_low_rows"] += int(low_old.sum())
        counts["corrected_low_rows"] += int(low_new.sum())
        counts["omitted_low_rows"] += int((low_new & ~low_old).sum())
        counts["zero_or_one_estimate_rows"] += int((ok & frame.estimated_prevalence.isin([0, 1])).sum())
        recomputed = frame.estimated_prevalence - frame.true_prevalence
        if not np.allclose(recomputed[ok], frame.loc[ok, "prevalence_error"], atol=1e-12):
            raise ValueError("Stored error mismatch")
        residual = (recomputed - frame.detection_error_contribution - frame.bag_sampling_error_contribution - frame.quantifier_adjustment_contribution).abs().max()
        max_residual = max(max_residual, float(residual))
        bases.append(unique_checked(frame, BASE, BASE_VALUES))
        action = unique_checked(frame, ACTION, ACTION_VALUES)
        actions.append(action.loc[action.bag_index < config["utility_bags_per_rate"]])
        metrics = ["detection_auroc", "detection_fpr", "detection_tpr", "detection_brier", "detection_ece", "detector_threshold"]
        detectors.append(unique_checked(frame, ["seed", "protocol", "detector", "calibration_policy"], metrics))
        for group, block in frame.groupby(CELL, dropna=False):
            valid = block.quantifier_status.eq("ok")
            utility = block.bag_index < config["utility_bags_per_rate"]
            good_utility = utility & valid
            cleanup_valid = np.isfinite(block.detector_cleanup_utility)
            keep_valid = np.isfinite(block.contaminated_utility)
            estimates = block.loc[valid, "estimated_prevalence"]
            strict_policy = good_utility & np.isfinite(block.policy_utility)
            rows = dict(zip(CELL, group)) | {
                "n_total": len(block), "n_ok": int(valid.sum()), "n_failed": int((~valid).sum()),
                "failure_reasons": ";".join(sorted(block.loc[~valid, "quantifier_status"].unique())),
                "mae_success_only": block.loc[valid, "prevalence_absolute_error"].mean(),
                "bias_success_only": block.loc[valid, "prevalence_error"].mean(),
                "legacy_low_n": int((valid & block.true_prevalence.isin([.05,.10])).sum()),
                "corrected_low_n": int((valid & block.nominal_prevalence.isin([.05,.10])).sum()),
                "at_boundary_n": int(estimates.isin([0,1]).sum()),
                "n_utility_scheduled": int(utility.sum()),
                "n_utility_not_scheduled": int((~utility).sum()),
                "n_quantifier_failed_when_utility_scheduled": int((utility & ~valid).sum()),
                "n_selected_action_invalid": int((good_utility & ~np.isfinite(block.policy_utility)).sum()),
                "n_cleanup_invalid": int((utility & ~cleanup_valid).sum()),
                "n_strict_policy_valid": int(strict_policy.sum()),
                "policy_utility_success_only": block.loc[strict_policy, "policy_utility"].mean(),
                "policy_gain_vs_keep_success_only": (block.loc[strict_policy, "policy_utility"] - block.loc[strict_policy, "contaminated_utility"]).mean(),
                "regret_both_actions_valid_success_only": block.loc[good_utility & cleanup_valid & keep_valid, "decision_regret"].mean(),
                "n_both_actions_valid_success_only": int((good_utility & cleanup_valid & keep_valid).sum()),
                "zero_regret_with_invalid_cleanup_n": int((utility & ~cleanup_valid & block.decision_regret.eq(0)).sum()),
            }
            quant.append(rows)
        old = frame.loc[ok & frame.quantifier.eq(config["primary_quantifier"]) & (frame.bag_index < config["utility_bags_per_rate"])]
        oldutility.append(old[BASE + ["detector", "calibration_policy", "contaminated_utility_delta", "detector_cleanup_delta", "decision_regret"]])
        valpath = attempt / "record_valuation.csv"
        if valpath.exists():
            vf = pd.read_csv(valpath)
            counts["valuation_rows"] += len(vf)
            vg = ["seed", "protocol", "test_table", "test_generator", "contamination_mode", "true_prevalence", "valuation_method", "source_label"]
            for keys, block in vf.groupby(vg, dropna=False):
                finite = block.task_value[np.isfinite(block.task_value)]
                value_summaries.append(dict(zip(vg, keys)) | {"n_total": len(block), "n_valid": len(finite), "n_missing": len(block)-len(finite), "mean_value": finite.mean(), "negative_rate_finite_only": (finite < 0).mean() if keys[-2] == "knn_shapley" else np.nan})
            inputs.append({"path": str(valpath), "sha256": sha(valpath), "bytes": valpath.stat().st_size})
        if sha(path) != initial_hash:
            raise ValueError("Input changed while reading")
        inputs.append({"path": str(path), "sha256": initial_hash, "bytes": path.stat().st_size, "rows": len(frame)})
        print(f"E0 {index+1}/{len(attempts)} {key}", flush=True)

    base = unique_checked(pd.concat(bases, ignore_index=True), BASE, BASE_VALUES)
    base["utility_scheduled"] = base.bag_index < config["utility_bags_per_rate"]
    base["utility_status"] = np.where(~base.utility_scheduled, "not_scheduled", np.where(np.isfinite(base.clean_utility) & np.isfinite(base.contaminated_utility), "valid", "invalid_unknown_reason"))
    base["oracle_status"] = np.where(~base.utility_scheduled, "not_scheduled", np.where(np.isfinite(base.oracle_cleanup_utility), "valid", np.where(base.true_prevalence.eq(1), "empty_after_true_source_removal", "invalid_unknown_reason")))
    base.to_csv(out / "base_bags.csv", index=False)
    action = pd.concat(actions, ignore_index=True).merge(base[BASE + ["contaminated_utility", "clean_utility"]], on=BASE, validate="many_to_one")
    action["cleanup_status"] = np.where(np.isfinite(action.detector_cleanup_utility), "valid", "invalid_unknown_reason")
    action["cleanup_gain_vs_keep"] = action.detector_cleanup_utility - action.contaminated_utility
    action["cleanup_delta_vs_clean"] = action.detector_cleanup_utility - action.clean_utility
    action.to_csv(out / "actions.csv", index=False)
    q = pd.DataFrame(quant)
    q.to_csv(out / "quantifier_policy_cells.csv", index=False)
    q.loc[q.nominal_prevalence.isin([.05, .1])].to_csv(out / "low_prevalence_corrected.csv", index=False)
    pd.concat(detectors).to_csv(out / "detectors.csv", index=False)
    pd.DataFrame(value_summaries).to_csv(out / "valuation_coverage.csv", index=False)
    evaluated = base.loc[base.utility_scheduled]
    summarize(evaluated, ["seed", "protocol", "contamination_mode", "nominal_prevalence"], ["clean_utility", "contaminated_utility", "contaminated_utility_delta", "oracle_cleanup_utility"]).to_csv(out / "utility_seed_summary.csv", index=False)
    summarize(action, ["seed", "protocol", "detector", "calibration_policy", "contamination_mode", "nominal_prevalence"], ["cleanup_gain_vs_keep", "cleanup_delta_vs_clean"]).to_csv(out / "action_seed_summary.csv", index=False)
    old = pd.concat(oldutility)
    comparison = []
    for mode in config["contamination_modes"]:
        before = old.loc[old.contamination_mode.eq(mode), "contaminated_utility_delta"]
        after = evaluated.loc[evaluated.contamination_mode.eq(mode), "contaminated_utility_delta"]
        comparison.append({"mode": mode, "old_success_pacc_repeated_n": len(before), "old_row_weighted_mean": before.mean(), "new_unique_bag_n": len(after), "new_unique_bag_mean": after.mean(), "difference": after.mean()-before.mean()})
    pd.DataFrame(comparison).to_csv(out / "old_vs_new_utility.csv", index=False)
    # Explicit pairing diagnostic at zero contamination; clean baseline was resampled.
    zero = evaluated.loc[evaluated.nominal_prevalence.eq(0)].copy()
    zero.to_csv(out / "zero_prevalence_diagnostic.csv", index=False)
    inventory = []
    for scope in [root, args.checkpoints]:
        if scope is None or not scope.exists():
            continue
        for path in sorted(scope.rglob("*")):
            if path.is_file():
                inventory.append({"scope": str(scope), "path": str(path), "bytes": path.stat().st_size, "suffix": path.suffix})
    pd.DataFrame(inventory).to_csv(out / "file_inventory.csv", index=False)
    audit = counts | {"shards": len(attempts), "unique_base_bags": len(base), "utility_base_bags": len(evaluated), "utility_action_rows": len(action), "invalid_cleanup_action_rows": int(action.cleanup_status.ne("valid").sum()), "zero_prevalence_nonzero_deltas": int((zero.contaminated_utility_delta.abs() > 1e-12).sum()), "zero_prevalence_max_abs_delta": float(zero.contaminated_utility_delta.abs().max()), "max_decomposition_residual": max_residual, "checks_passed": True, "scientific_causal_validity_certified": False, "raw_estimate_available": "raw_estimated_prevalence" in schema, "evidence_columns": schema}
    dump_json(out / "audit.json", audit)
    dump_json(out / "provenance.json", {"analysis_id": out.name, "analysis_script_sha256": sha(__file__), "python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "inputs": inputs, "no_training": True, "input_selection": "completed_shards_manifest attempts, one attempt per shard", "inference": "descriptive; no new significance tests"})
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
