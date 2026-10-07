"""Audit E1D shards and compare each perturbation with its paired control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["log_loss", "brier", "auroc", "auprc", "positive_log_loss", "tail_log_loss"]
SYN = {"global_half": "real_global_half",
       "payment_status_full": "real_payment_status_full",
       "balances_full": "real_balances_full"}


def analyze(source: Path, e1c: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {(gen, seed) for gen in ["CTGAN", "TVAE"] for seed in [2026, 2027, 2028]}
    seen = set()
    all_rows = []
    contrast_rows = []
    structure_rows = []
    e1c_checks = []
    for shard in sorted(source.glob("credit_*_seed*")):
        if not shard.is_dir():
            continue
        complete = json.loads((shard / "complete.json").read_text(encoding="utf-8"))
        context = complete["context"]
        key = context["generator"], context["seed"]
        if key not in expected or key in seen:
            raise AssertionError(f"Unexpected or duplicate shard {key}")
        seen.add(key)
        groups = [set(context[field]) for field in
                  ["core_ids", "validation_ids", "donor_ids", "test_ids"]]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i+1, 4)):
            raise AssertionError(f"Split overlap {shard}")
        rows = pd.read_csv(shard / "metrics.csv")
        conditions = ["natural", *SYN, "independent_real_donor", *SYN.values()]
        if complete["fits"] != 8 or len(rows) != 16 or set(zip(rows.condition, rows["mode"])) != {
                (c, m) for c in conditions for m in ["raw", "calibrated"]}:
            raise AssertionError(f"Incomplete condition matrix {shard}")
        if rows[METRICS].isna().any().any() or (rows.calibration_slope <= 0).any():
            raise AssertionError(f"Invalid metric/calibration {shard}")
        for row in rows.itertuples():
            prediction = pd.read_csv(shard / row.predictions_path)
            if len(prediction) != row.n_test or prediction.row_id.duplicated().any():
                raise AssertionError(f"Bad prediction IDs {shard}")
            p = prediction.p_raw if row.mode == "raw" else prediction.p_calibrated
            if not np.isfinite(p).all() or not p.between(0, 1).all():
                raise AssertionError(f"Bad probability {shard}")
            bounded = np.clip(p.to_numpy(), np.finfo(float).eps, 1-np.finfo(float).eps)
            y = prediction.y_true.to_numpy()
            loss = float((-y*np.log(bounded) -(1-y)*np.log1p(-bounded)).mean())
            if abs(loss-row.log_loss) > 1e-9:
                raise AssertionError(f"Metric/prediction mismatch {shard}")
        anchor = pd.read_csv(e1c / shard.name / "metrics.csv")
        for condition in ["natural", "independent_real_donor"]:
            for mode in ["raw", "calibrated"]:
                current = rows.loc[(rows.condition == condition) & (rows["mode"] == mode)].iloc[0]
                previous = anchor.loc[(anchor.condition == condition) & (anchor["mode"] == mode)
                                      & (anchor.model == "rf")].iloc[0]
                delta = float(current.log_loss-previous.log_loss)
                e1c_checks.append({"generator": key[0], "seed": key[1],
                                   "condition": condition, "mode": mode,
                                   "e1d_minus_e1c_log_loss": delta})
                if abs(delta) > 1e-12:
                    raise AssertionError(f"Natural/real anchor mismatch {shard}")
        for condition, diagnostics in context["structure"].items():
            structure_rows.append({"generator": key[0], "seed": key[1],
                                   "condition": condition, **diagnostics})
        for mode, slice_ in rows.groupby("mode"):
            indexed = slice_.set_index("condition")
            for synthetic_condition, real_condition in SYN.items():
                record = {"generator": key[0], "seed": key[1], "mode": mode,
                          "intervention": synthetic_condition}
                for metric in METRICS:
                    syn_delta = indexed.loc[synthetic_condition, metric]-indexed.loc["natural", metric]
                    real_delta = indexed.loc[real_condition, metric]-indexed.loc["independent_real_donor", metric]
                    record[f"synthetic_delta_{metric}"] = float(syn_delta)
                    record[f"real_delta_{metric}"] = float(real_delta)
                    record[f"difference_in_differences_{metric}"] = float(syn_delta-real_delta)
                contrast_rows.append(record)
        all_rows.append(rows)
    if seen != expected:
        raise AssertionError(f"Missing shards {expected-seen}")
    pd.concat(all_rows, ignore_index=True).to_csv(destination / "all_metrics.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(destination / "paired_contrasts.csv", index=False)
    pd.DataFrame(structure_rows).to_csv(destination / "structure_diagnostics.csv", index=False)
    pd.DataFrame(e1c_checks).to_csv(destination / "e1c_anchor_checks.csv", index=False)
    result = {"shards": len(seen), "fits": 48, "metric_rows": 96,
              "paired_rows": len(contrast_rows),
              "max_abs_e1c_anchor_log_loss_diff": max(abs(x["e1d_minus_e1c_log_loss"])
                                                    for x in e1c_checks)}
    (destination / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--e1c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.e1c, args.output)))
