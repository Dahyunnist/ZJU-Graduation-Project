"""Audit and summarize the frozen E1B probability experiment without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["log_loss", "brier", "auroc", "auprc", "positive_log_loss",
           "negative_log_loss", "tail_log_loss", "clipped_1e3_log_loss",
           "wrong_endpoint_count", "top1pct_loss_share"]


def analyze(source: Path, destination: Path, e1_source: Path | None = None) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    frames = []
    audits = []
    expected_cases = {(t, g, s) for t, g in [("abalone", "TVAE"),
                                               ("credit", "TVAE"), ("adult", "CTGAN")]
                      for s in [2026, 2027, 2028]}
    seen_cases = set()
    for shard in sorted(source.glob("*_seed*")):
        if not shard.is_dir():
            continue
        summary = json.loads((shard / "complete.json").read_text(encoding="utf-8"))
        context = summary["context"]
        key = (context["table"], context["generator"], context["seed"])
        if key not in expected_cases or key in seen_cases:
            raise AssertionError(f"Unexpected or duplicate E1B shard: {key}")
        seen_cases.add(key)
        metrics = pd.read_csv(shard / "metrics.csv")
        if len(metrics) != 32 or summary["fits"] != 16 or summary["metric_rows"] != 32:
            raise AssertionError(f"Incomplete fit matrix: {shard}")
        if (metrics.calibration_slope <= 0).any() or metrics[METRICS].isna().any().any():
            raise AssertionError(f"Invalid metric or calibration slope: {shard}")
        groups = [set(context[field]) for field in
                  ["core_ids", "validation_ids", "donor_ids", "test_ids"]]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
            raise AssertionError(f"Split overlap: {shard}")
        expected = {(model, condition, mode)
                    for model in ["lr", "rf_leaf3", "rf_leaf10", "rf_leaf30"]
                    for condition in ["real_reference", "independent_real_donor",
                                      "natural", "label_permute"]
                    for mode in ["raw", "calibrated"]}
        actual = set(zip(metrics.model, metrics.condition, metrics["mode"]))
        if actual != expected:
            raise AssertionError(f"Missing or duplicate condition: {shard}")
        for (model, condition), pair in metrics.groupby(["model", "condition"]):
            prediction = pd.read_csv(shard / pair.iloc[0].predictions_path)
            validation = pd.read_csv(shard / pair.iloc[0].validation_path)
            if len(prediction) != int(pair.iloc[0].n_test) or len(validation) != 500:
                raise AssertionError(f"Prediction count mismatch: {shard} {model} {condition}")
            if prediction.row_id.duplicated().any() or prediction.row_id.isna().any():
                raise AssertionError(f"Duplicate/missing test row id: {shard}")
            for mode, column in [("raw", "p_raw"), ("calibrated", "p_calibrated")]:
                p = prediction[column].to_numpy()
                y = prediction.y_true.to_numpy()
                if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                    raise AssertionError(f"Bad predictions: {shard} {model} {condition} {mode}")
                eps = np.finfo(float).eps
                clipped = np.clip(p, eps, 1 - eps)
                loss = float((-y * np.log(clipped) - (1-y) * np.log1p(-clipped)).mean())
                recorded = float(pair.loc[pair["mode"] == mode, "log_loss"].iloc[0])
                if abs(loss - recorded) > 1e-9:
                    raise AssertionError(f"Log loss mismatch: {shard} {model} {condition} {mode}")
        frames.append(metrics)
        audits.append({"table": key[0], "generator": key[1], "seed": key[2],
                       "fits": summary["fits"], "metric_rows": len(metrics),
                       "test_n": len(context["test_ids"]), "split_overlap": False,
                       "source_hash": context["real_sha256"],
                       "synthetic_hash": context["synthetic_sha256"]})
    if seen_cases != expected_cases:
        raise AssertionError(f"Missing E1B shards: {expected_cases - seen_cases}")
    combined = pd.concat(frames, ignore_index=True)
    paired = []
    for (table, generator, seed, model, mode), group in combined.groupby(
            ["table", "generator", "seed", "model", "mode"]):
        natural = group.set_index("condition").loc["natural"]
        permuted = group.set_index("condition").loc["label_permute"]
        row = {"table": table, "generator": generator, "seed": seed,
               "model": model, "mode": mode}
        row.update({f"delta_{metric}": float(permuted[metric] - natural[metric])
                    for metric in METRICS})
        paired.append(row)
    contrast = pd.DataFrame(paired).sort_values(["table", "generator", "model", "mode", "seed"])
    summary = contrast.groupby(["table", "generator", "model", "mode"], as_index=False)[
        [f"delta_{metric}" for metric in METRICS]].agg(["mean", "min", "max"])
    summary.columns = ["_".join(str(part) for part in col if part) if isinstance(col, tuple)
                       else col for col in summary.columns]
    comparison = []
    if e1_source is not None:
        for case in audits:
            table, generator, seed = case["table"], case["generator"], case["seed"]
            e1 = pd.read_csv(e1_source / f"{table}_seed{seed}" / "metrics.csv")
            for model, current_model in [("lr", "lr"), ("rf", "rf_leaf3")]:
                for condition, old in [("natural", "natural"), ("label_permute", "label_permute")]:
                    previous = e1.loc[(e1.generator == generator) & (e1.model == model)
                                      & (e1.prevalence == .5) & (e1.intervention == old)]
                    current = combined.loc[(combined.table == table) & (combined.generator == generator)
                                           & (combined.seed == seed) & (combined.model == current_model)
                                           & (combined.condition == condition) & (combined["mode"] == "raw")]
                    if len(previous) != 1 or len(current) != 1:
                        raise AssertionError(f"Missing E1 comparison: {table} {seed} {model} {old}")
                    diff = float(current.iloc[0].log_loss - previous.iloc[0].log_loss)
                    comparison.append({"table": table, "generator": generator, "seed": seed,
                                       "model": model, "condition": condition,
                                       "e1b_minus_e1_log_loss": diff})
    pd.DataFrame(audits).to_csv(destination / "audit.csv", index=False)
    combined.to_csv(destination / "all_metrics.csv", index=False)
    contrast.to_csv(destination / "paired_contrasts.csv", index=False)
    summary.to_csv(destination / "contrast_summary.csv", index=False)
    if comparison:
        pd.DataFrame(comparison).to_csv(destination / "e1_comparison.csv", index=False)
    result = {"shards": len(audits), "fits": int(sum(row["fits"] for row in audits)),
              "metric_rows": len(combined), "paired_rows": len(contrast),
              "max_abs_e1_log_loss_diff": max((abs(row["e1b_minus_e1_log_loss"])
                                                for row in comparison), default=None)}
    (destination / "audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e1", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, args.e1), ensure_ascii=False))
