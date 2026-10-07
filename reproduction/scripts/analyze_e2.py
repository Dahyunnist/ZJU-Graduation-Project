"""Audit E2 bag checkpoints and assemble decision-level paired evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(source: Path, destination: Path, expected_bags: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    bag_audits = []
    actions = []
    quantification = []
    for bag_dir in sorted(source.glob("seed*/*_b*")):
        if not bag_dir.is_dir():
            continue
        complete = json.loads((bag_dir / "complete.json").read_text(encoding="utf-8"))
        if complete["status"] != "complete":
            raise AssertionError(f"Incomplete bag {bag_dir}")
        action_frame = pd.read_csv(bag_dir / "actions.csv")
        quant_frame = pd.read_csv(bag_dir / "quantification.csv")
        scores = pd.read_csv(bag_dir / "bag_scores.csv.gz")
        diagnostics = json.loads((bag_dir.parent / "calibration_diagnostics.json").read_text(encoding="utf-8"))
        available_policies = diagnostics["available_policies"]
        monotone_policies = [name for name in available_policies
                             if diagnostics["policy_slopes"][name] > 0]
        if len(scores) != complete["bag_size"] or len(action_frame) != complete["action_rows"]:
            raise AssertionError(f"Bag size/action mismatch {bag_dir}")
        if len(quant_frame) != complete["quantifier_rows"]:
            raise AssertionError(f"Quantifier count mismatch {bag_dir}")
        if scores.record_id.duplicated().any() or not np.array_equal(
                scores.row_position.to_numpy(), np.arange(len(scores))):
            raise AssertionError(f"Bad row identities {bag_dir}")
        if abs(scores.source_label.mean() - complete["true_prevalence"]) > 1e-12:
            raise AssertionError(f"Source prevalence mismatch {bag_dir}")
        fixed = action_frame.loc[action_frame.action == "raw_score_fixed_k"]
        if len(fixed) != 1:
            raise AssertionError(f"Missing raw fixed budget {bag_dir}")
        reference_selection = json.loads(fixed.iloc[0].removed_positions)
        calibrated_fixed = action_frame.loc[action_frame.action == "calibrated_fixed_k"]
        if len(calibrated_fixed) != len(available_policies) or any(
                json.loads(row.removed_positions) != reference_selection
                for row in calibrated_fixed.itertuples() if row.policy in monotone_policies):
            raise AssertionError(f"Fixed-budget rank invariance failed {bag_dir}")
        reversed_rows = calibrated_fixed.loc[~calibrated_fixed.policy.isin(monotone_policies)]
        reversed_overlap = (len(set(json.loads(reversed_rows.iloc[0].removed_positions))
                                & set(reference_selection)) if len(reversed_rows) else None)
        keep_all = action_frame.loc[action_frame.action == "keep_all"]
        if len(keep_all) != 1 or int(keep_all.iloc[0].removed_n) != 0:
            raise AssertionError(f"Missing keep-all control {bag_dir}")
        for row in action_frame.itertuples():
            if row.task_status in {"undefined_pacc", "unavailable_calibration"}:
                continue
            removed = json.loads(row.removed_positions)
            if len(removed) != row.removed_n or len(set(removed)) != len(removed):
                raise AssertionError(f"Bad removal set {bag_dir}")
            if int(scores.iloc[removed].source_label.sum()) != row.removed_synthetic_n:
                raise AssertionError(f"Removal precision/source mismatch {bag_dir}")
            if row.task_status == "ok":
                pred_file = bag_dir / "task_models" / row.task_fingerprint / "predictions.csv.gz"
                prediction = pd.read_csv(pred_file)
                y = prediction.y_true.to_numpy()
                for mode, field in [("raw", "p_raw"), ("calibrated", "p_calibrated")]:
                    value = getattr(row, f"{mode}_log_loss")
                    if not np.isfinite(value):
                        continue
                    p = np.clip(prediction[field].to_numpy(), np.finfo(float).eps,
                                1-np.finfo(float).eps)
                    observed = float((-y*np.log(p) - (1-y)*np.log1p(-p)).mean())
                    if abs(observed-value) > 1e-9:
                        raise AssertionError(f"Task loss mismatch {bag_dir} {row.action}")
        policies = quant_frame.loc[quant_frame.quantifier == "pacc", ["policy", "estimated_prevalence"]]
        source_est = policies.loc[policies.policy == "source_only", "estimated_prevalence"].iloc[0]
        anchor_est = policies.loc[policies.policy == "target_real_anchor", "estimated_prevalence"].iloc[0]
        if np.isfinite(source_est) and np.isfinite(anchor_est) and abs(source_est-anchor_est) > 1e-12:
            raise AssertionError(f"Source/anchor PACC unexpectedly differ {bag_dir}")
        bag_audits.append({"bag_id": complete["bag_id"], "seed": complete["seed"],
                           "true_prevalence": complete["true_prevalence"],
                           "available_policies": ",".join(available_policies),
                           "target_raw_auroc": diagnostics["target_raw_auroc"],
                           "reoriented_fixed_k_overlap_with_raw": reversed_overlap,
                           "action_rows": len(action_frame),
                           "quantifier_rows": len(quant_frame),
                           "unique_task_fits": complete["unique_task_fits"],
                           "rank_invariant": True})
        actions.append(action_frame)
        quantification.append(quant_frame)
    if len(bag_audits) != expected_bags:
        raise AssertionError(f"Expected {expected_bags} complete bags, found {len(bag_audits)}")
    action_data = pd.concat(actions, ignore_index=True)
    quant_data = pd.concat(quantification, ignore_index=True)
    action_data.to_csv(destination / "all_actions.csv", index=False)
    quant_data.to_csv(destination / "all_quantification.csv", index=False)
    pd.DataFrame(bag_audits).to_csv(destination / "bag_audit.csv", index=False)
    contrasts = []
    for (bag_id, policy), group in action_data.groupby(["bag_id", "policy"]):
        if policy == "shared":
            continue
        policy_row = group.loc[group.action == "decision_policy"]
        if len(policy_row) != 1 or policy_row.iloc[0].task_status != "ok":
            continue
        decision = policy_row.iloc[0]
        baseline = action_data.loc[(action_data.bag_id == bag_id) &
                                   (action_data.action == "keep_all")].iloc[0]
        random = group.loc[group.action == "random_matched_n"]
        oracle = group.loc[group.action == "source_oracle_matched_n"]
        contrasts.append({"bag_id": bag_id, "policy": policy,
                          "seed": int(decision.seed), "prevalence": float(decision.prevalence),
                          "decision_triggered": str(decision.decision_triggered).lower() == "true",
                          "removed_n": int(decision.removed_n),
                          "removed_source_precision": decision.removed_source_precision,
                          "delta_auroc_vs_keep": decision.raw_auroc-baseline.raw_auroc,
                          "delta_log_loss_vs_keep": decision.raw_log_loss-baseline.raw_log_loss,
                          "delta_calibrated_log_loss_vs_keep":
                              decision.calibrated_log_loss-baseline.calibrated_log_loss,
                          "delta_auroc_vs_random_matched":
                              decision.raw_auroc-random.iloc[0].raw_auroc if len(random) else np.nan,
                          "delta_auroc_vs_source_oracle_matched":
                              decision.raw_auroc-oracle.iloc[0].raw_auroc if len(oracle) else np.nan})
    pd.DataFrame(contrasts).to_csv(destination / "decision_contrasts.csv", index=False)
    result = {"complete_bags": len(bag_audits), "action_rows": len(action_data),
              "quantifier_rows": len(quant_data), "decision_contrasts": len(contrasts),
              "fixed_budget_rank_invariance": True}
    (destination / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bags", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.expected_bags)))
