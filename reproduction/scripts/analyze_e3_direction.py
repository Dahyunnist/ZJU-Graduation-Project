"""Independently audit E3 gate-A scores, PACC estimates and decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from tabpollution.utils import sha256_file


def analyze(source: Path, destination: Path, expected_cases: int = 6) -> dict:
    complete_paths = sorted(source.glob("*/seed*/complete.json"))
    if len(complete_paths) != expected_cases:
        raise AssertionError(f"Expected {expected_cases} completed cases, found {len(complete_paths)}")
    context = json.loads((source / "context.json").read_text(encoding="utf-8"))
    cases, bags = [], []
    for complete_path in complete_paths:
        folder = complete_path.parent
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete["status"] != "complete" or complete["context"] != context:
            raise AssertionError(f"Stale/incomplete E3 case: {folder}")
        for name, digest in complete["files_sha256"].items():
            if sha256_file(folder / name) != digest:
                raise AssertionError(f"E3 file hash changed: {folder / name}")
        val = pd.read_csv(folder / "validation_scores.csv.gz")
        for partition, metric in [("source_cal", "source_raw_auroc"),
                                  ("target_eval", "target_raw_auroc")]:
            part = val.loc[val.partition == partition]
            observed = roc_auc_score(part.label, part.raw_score)
            if abs(observed - complete[metric]) > 1e-12:
                raise AssertionError(f"E3 validation AUROC mismatch: {folder} {partition}")
        scores = pd.read_csv(folder / "bag_scores.csv.gz")
        rows = pd.read_csv(folder / "bags.csv")
        policies = json.loads((folder / "policy_diagnostics.json").read_text(encoding="utf-8"))
        if len(rows) != complete["bag_rows"] or rows.duplicated(["bag_id", "policy"]).any():
            raise AssertionError(f"E3 bag result cardinality mismatch: {folder}")
        if set(rows.policy) != {"source_only", "target_real_anchor"}:
            raise AssertionError(f"E3 policy set changed: {folder}")
        for bag_id, group in scores.groupby("bag_id"):
            if group.record_id.duplicated().any() or not np.array_equal(
                    group.row_position.to_numpy(), np.arange(len(group))):
                raise AssertionError(f"E3 bag row identities invalid: {bag_id}")
            for row in rows.loc[rows.bag_id == bag_id].itertuples():
                actual_p = float(group.source_label.mean())
                if abs(actual_p - row.true_prevalence) > 1e-12:
                    raise AssertionError(f"E3 bag prevalence mismatch: {bag_id}")
                if row.policy not in policies:
                    if not row.status.startswith("unavailable:"):
                        raise AssertionError(f"E3 missing-policy status mismatch: {bag_id}")
                    continue
                p = policies[row.policy]
                calibrated = expit(p["platt_slope"] * group.raw_score.to_numpy() +
                                   p["platt_intercept"])
                denom = p["soft_tpr"] - p["soft_fpr"]
                if abs(denom) < 1e-8:
                    if not row.status.startswith("undefined:"):
                        raise AssertionError(f"E3 degenerate PACC status mismatch: {bag_id}")
                    continue
                estimate = float(np.clip((calibrated.mean() - p["soft_fpr"]) / denom, 0, 1))
                if row.status != "ok" or abs(estimate - row.pacc_estimate) > 1e-10:
                    raise AssertionError(f"E3 PACC mismatch: {bag_id} {row.policy}")
                triggered = estimate >= .10
                removed = np.flatnonzero(calibrated >= p["threshold"]) if triggered else np.array([], dtype=int)
                recorded_trigger = str(row.triggered).lower() == "true"
                if triggered != recorded_trigger or len(removed) != row.removed_n:
                    raise AssertionError(f"E3 governance action mismatch: {bag_id} {row.policy}")
                if len(removed):
                    precision = float(group.source_label.iloc[removed].mean())
                    if abs(precision - row.removed_source_precision) > 1e-12:
                        raise AssertionError(f"E3 removal precision mismatch: {bag_id} {row.policy}")
        cases.append({"table": complete["table"], "seed": complete["seed"],
                      "source_raw_auroc": complete["source_raw_auroc"],
                      "target_raw_auroc": complete["target_raw_auroc"],
                      "direction_reversed": complete["target_raw_auroc"] < .5,
                      "policy_errors": json.dumps(complete["policy_errors"], ensure_ascii=False)})
        bags.append(rows)
    destination.mkdir(parents=True, exist_ok=True)
    case_frame = pd.DataFrame(cases)
    bag_frame = pd.concat(bags, ignore_index=True)
    case_frame.to_csv(destination / "case_summary.csv", index=False)
    bag_frame.to_csv(destination / "bag_results.csv", index=False)
    summary = {"complete_cases": len(cases), "bag_policy_rows": len(bag_frame),
               "direction_reversed_cases": int(case_frame.direction_reversed.sum()),
               "audited": True}
    (destination / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.expected_cases)))
