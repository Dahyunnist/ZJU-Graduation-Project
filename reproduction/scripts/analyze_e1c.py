"""Audit the frozen E1C Credit experiment and compute paired structural controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["log_loss", "brier", "auroc", "auprc", "positive_log_loss",
           "negative_log_loss", "tail_log_loss"]


def analyze(source: Path, destination: Path, e1_source: Path | None = None) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {(g, s) for g in ["CTGAN", "TVAE"] for s in [2026, 2027, 2028]}
    seen = set()
    metrics_list = []
    structure_rows = []
    e1_rows = []
    for shard in sorted(source.glob("credit_*_seed*")):
        if not shard.is_dir():
            continue
        complete = json.loads((shard / "complete.json").read_text(encoding="utf-8"))
        context = complete["context"]
        key = context["generator"], context["seed"]
        if key not in expected or key in seen:
            raise AssertionError(f"Unexpected/duplicate shard {key}")
        seen.add(key)
        metrics = pd.read_csv(shard / "metrics.csv")
        if complete["fits"] != 8 or len(metrics) != 16:
            raise AssertionError(f"Incomplete matrix {shard}")
        requested = {(m, c, mode) for m in ["lr", "rf"]
                     for c in ["natural", "class_shuffle", "independent_real_donor",
                               "real_donor_class_shuffle"] for mode in ["raw", "calibrated"]}
        if set(zip(metrics.model, metrics.condition, metrics["mode"])) != requested:
            raise AssertionError(f"Unexpected conditions {shard}")
        groups = [set(context[name]) for name in ["core_ids", "validation_ids", "donor_ids", "test_ids"]]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i+1, 4)):
            raise AssertionError(f"Split overlap {shard}")
        if (metrics.calibration_slope <= 0).any() or metrics[METRICS].isna().any().any():
            raise AssertionError(f"Invalid metric/calibration {shard}")
        for row in metrics.itertuples():
            predictions = pd.read_csv(shard / row.predictions_path)
            if len(predictions) != row.n_test or predictions.row_id.duplicated().any():
                raise AssertionError(f"Test prediction length/ID mismatch {shard}")
            values = predictions.p_raw if row.mode == "raw" else predictions.p_calibrated
            if not np.isfinite(values).all() or not values.between(0, 1).all():
                raise AssertionError(f"Invalid test probability {shard}")
            eps = np.finfo(float).eps
            p = np.clip(values.to_numpy(), eps, 1-eps)
            y = predictions.y_true.to_numpy()
            loss = float((-y*np.log(p) - (1-y)*np.log1p(-p)).mean())
            if abs(loss-row.log_loss) > 1e-9:
                raise AssertionError(f"Recomputed log loss mismatch {shard}")
        for condition, measurements in context["structure"].items():
            structure_rows.append({"generator": key[0], "seed": key[1],
                                   "condition": condition, **measurements})
        if e1_source is not None:
            old = pd.read_csv(e1_source / f"credit_seed{key[1]}" / "metrics.csv")
            for model in ["lr", "rf"]:
                for condition in ["natural", "class_shuffle"]:
                    old_row = old.loc[(old.generator == key[0]) & (old.model == model)
                                      & (old.prevalence == .5) & (old.intervention == condition)]
                    new_row = metrics.loc[(metrics.model == model) & (metrics.condition == condition)
                                          & (metrics["mode"] == "raw")]
                    if len(old_row) != 1 or len(new_row) != 1:
                        raise AssertionError("Missing E1 cross-check")
                    e1_rows.append({"generator": key[0], "seed": key[1], "model": model,
                                    "condition": condition,
                                    "e1c_minus_e1_log_loss": float(new_row.iloc[0].log_loss - old_row.iloc[0].log_loss)})
        metrics_list.append(metrics)
    if seen != expected:
        raise AssertionError(f"Missing shards {expected-seen}")
    all_metrics = pd.concat(metrics_list, ignore_index=True)
    paired = []
    for (generator, seed, model, mode), group in all_metrics.groupby(
            ["generator", "seed", "model", "mode"]):
        indexed = group.set_index("condition")
        row = {"generator": generator, "seed": seed, "model": model, "mode": mode}
        for metric in METRICS:
            synthetic = indexed.loc["class_shuffle", metric] - indexed.loc["natural", metric]
            real = indexed.loc["real_donor_class_shuffle", metric] - indexed.loc["independent_real_donor", metric]
            row[f"synthetic_delta_{metric}"] = float(synthetic)
            row[f"real_delta_{metric}"] = float(real)
            row[f"difference_in_differences_{metric}"] = float(synthetic-real)
        paired.append(row)
    paired_frame = pd.DataFrame(paired).sort_values(["generator", "model", "mode", "seed"])
    all_metrics.to_csv(destination / "all_metrics.csv", index=False)
    pd.DataFrame(structure_rows).to_csv(destination / "structure_diagnostics.csv", index=False)
    paired_frame.to_csv(destination / "paired_contrasts.csv", index=False)
    if e1_rows:
        pd.DataFrame(e1_rows).to_csv(destination / "e1_comparison.csv", index=False)
    result = {"shards": len(seen), "fits": int(sum(len(frame)//2 for frame in metrics_list)),
              "metric_rows": len(all_metrics), "paired_rows": len(paired_frame),
              "max_abs_e1_log_loss_diff": max((abs(row["e1c_minus_e1_log_loss"])
                                                 for row in e1_rows), default=None)}
    (destination / "audit_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e1", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.e1)))
