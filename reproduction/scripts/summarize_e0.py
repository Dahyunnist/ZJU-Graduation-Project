"""Finish E0 descriptive summaries and independent output checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reanalyze_e0 import sha, dump_json, BASE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    out = args.analysis
    if (out / "completion.json").exists():
        raise FileExistsError("E0 already finalized")
    cfg = json.loads((args.run / "resolved_config.json").read_text())
    plan = json.loads((args.run / "shard_plan.json").read_text())
    q = pd.read_csv(out / "quantifier_policy_cells.csv")
    base = pd.read_csv(out / "base_bags.csv")
    actions = pd.read_csv(out / "actions.csv")
    detectors = pd.read_csv(out / "detectors.csv")
    replay = pd.read_csv(out / "sampling_reconstruction.csv")
    planned = {(r["seed"], r["protocol"], r["detector"]) for r in plan["shards"]}
    observed = set(map(tuple, detectors[["seed", "protocol", "detector"]].drop_duplicates().to_numpy()))
    checks = {
        "exact_planned_shard_set": planned == observed,
        "each_quantifier_cell_has_100_bags": bool(q.n_total.eq(cfg["bags_per_rate"]).all()),
        "each_cell_has_5_scheduled_utilities": bool(q.n_utility_scheduled.eq(cfg["utility_bags_per_rate"]).all()),
        "all_nominal_rates_valid": set(q.nominal_prevalence) == set(cfg["prevalence_rates"]),
        "all_quantifiers_present": set(q.quantifier) == set(cfg["quantifiers"]),
        "all_policies_present": set(q.calibration_policy) == set(cfg["calibration_policies"]),
        "base_bag_key_unique": not base.duplicated(BASE).any(),
        "action_key_unique": not actions.duplicated(BASE+["detector", "calibration_policy"]).any(),
        "reconstruction_covers_scheduled_bags": len(replay) == int(base.utility_scheduled.sum()),
        "reconstructed_real_test_overlap_zero": bool(replay.real_test_id_overlap.eq(0).all()),
    }
    zero = base.loc[base.nominal_prevalence.eq(0)]
    zkeys = [k for k in BASE if k not in ("contamination_mode", "nominal_prevalence")]
    paired = zero.loc[zero.contamination_mode.eq("append")].merge(zero.loc[zero.contamination_mode.eq("replace")], on=zkeys, suffixes=("_a", "_r"), validate="one_to_one")
    checks["zero_append_replace_utilities_match"] = bool(np.allclose(paired.contaminated_utility_a, paired.contaminated_utility_r, equal_nan=True, atol=1e-12))
    checks["zero_append_replace_pair_count"] = len(paired) == len(zero)//2
    low = q.loc[q.nominal_prevalence.isin([.05,.10]) & q.quantifier.eq("pacc") & q.calibration_policy.eq("source_only")]
    low_summary = low.groupby(["protocol", "contamination_mode"], as_index=False).agg(mae=("mae_success_only", "mean"), n_total=("n_total", "sum"), n_ok=("n_ok", "sum"), legacy_included=("legacy_low_n", "sum"))
    low_summary.to_csv(out / "low_prevalence_headline.csv", index=False)
    evaluated = base.loc[base.utility_scheduled]
    utility = evaluated.groupby(["protocol", "contamination_mode"], as_index=False).agg(n_bags=("bag_index", "size"), mean_delta=("contaminated_utility_delta", "mean"), min_delta=("contaminated_utility_delta", "min"), max_delta=("contaminated_utility_delta", "max"))
    utility.to_csv(out / "utility_protocol_headline.csv", index=False)
    action_summary = actions.groupby(["protocol", "calibration_policy"], as_index=False).agg(n_total=("bag_index", "size"), n_valid=("cleanup_gain_vs_keep", "count"), mean_gain_success_only=("cleanup_gain_vs_keep", "mean"))
    action_summary["n_invalid"] = action_summary.n_total-action_summary.n_valid
    action_summary.to_csv(out / "cleanup_coverage_headline.csv", index=False)
    failed = q.loc[q.n_failed.gt(0)]
    failed.to_csv(out / "failed_method_cells.csv", index=False)
    policies = q.groupby(["protocol", "calibration_policy"], as_index=False)[["n_utility_scheduled", "n_quantifier_failed_when_utility_scheduled", "n_selected_action_invalid", "n_strict_policy_valid", "zero_regret_with_invalid_cleanup_n"]].sum()
    policies.to_csv(out / "policy_coverage_headline.csv", index=False)
    repeat = replay.groupby(["test_table", "contamination_mode", "nominal_prevalence"], as_index=False).agg(n_bags=("bag_index", "size"), mean_duplicate_fraction=("duplicate_fraction", "mean"), max_duplicate_fraction=("duplicate_fraction", "max"), min_real_pool=("n_real_pool", "min"), min_synthetic_pool=("n_synthetic_pool", "min"))
    repeat.to_csv(out / "sampling_coverage_headline.csv", index=False)
    inv = pd.read_csv(out / "file_inventory.csv")
    inv.loc[inv.suffix.isin([".pt", ".pth", ".ckpt", ".pkl", ".joblib", ".npz", ".npy", ".parquet", ".h5"])].to_csv(out / "cache_candidates.csv", index=False)
    result = {"checks": checks, "all_checks_passed": all(checks.values()), "failed_method_cells": len(failed), "selected_action_invalid_policy_rows": int(q.n_selected_action_invalid.sum()), "quantifier_failure_on_utility_rows": int(q.n_quantifier_failed_when_utility_scheduled.sum()), "zero_regret_with_invalid_cleanup_policy_rows": int(q.zero_regret_with_invalid_cleanup_n.sum()), "script_sha256": sha(__file__), "historical_shard_plan_sha256": sha(args.run / "shard_plan.json"), "scope": "descriptive E0 only; success-only means retain explicit coverage; no causal or significance claim"}
    dump_json(out / "completion.json", result)
    if not all(checks.values()):
        raise ValueError(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
