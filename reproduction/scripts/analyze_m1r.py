"""Independently audit all M1R row scores and summarize generator-level variation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from analyze_e3b_detectability import analyze as analyze_e3b
from tabpollution.studies.m1r_repeats import decide_cross_instance, load_config


def analyze(config_path: Path, results_root: Path, destination: Path) -> dict:
    cfg = load_config(config_path)
    final = destination.resolve()
    partial = final.with_name(final.name + ".partial")
    if final.exists() or partial.exists():
        raise FileExistsError("M1R audit destination already exists")
    partial.mkdir(parents=True)
    instances = [cfg["historical_instance"], *cfg["new_instances"]]
    decisions, metrics = [], []
    try:
        for instance in instances:
            instance_id = instance["instance_id"]
            source = (
                results_root / instance["historical_e3b_output"]
                if "historical_e3b_output" in instance
                else results_root / cfg["study_id"] / "detection" / instance_id
            )
            audit_dir = partial / "instance_audits" / instance_id
            audit = analyze_e3b(source, audit_dir, expected_lr_cases=6)
            if not audit.get("audited"):
                raise AssertionError(f"M1R instance audit failed: {instance_id}")
            d = pd.read_csv(audit_dir / "table_decisions.csv")
            d.insert(0, "instance_id", instance_id)
            decisions.append(d)
            m = pd.read_csv(audit_dir / "matrix_metrics.csv")
            m.insert(0, "instance_id", instance_id)
            metrics.append(m)
        decision_frame = pd.concat(decisions, ignore_index=True)
        metric_frame = pd.concat(metrics, ignore_index=True)
        instance_points = (
            metric_frame.groupby(
                ["instance_id", "table", "train_generator", "eval_generator", "cell_id"],
                as_index=False,
            )[["auroc", "auprc", "score_mean_gap", "ks_distance", "tpr_at_calibrated_5pct_fpr"]]
            .mean()
        )
        between = (
            instance_points.groupby(["table", "train_generator", "eval_generator", "cell_id"])
            .agg(
                instances=("instance_id", "nunique"),
                auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                auroc_min=("auroc", "min"), auroc_max=("auroc", "max"),
                score_gap_mean=("score_mean_gap", "mean"), score_gap_sd=("score_mean_gap", "std"),
                score_gap_min=("score_mean_gap", "min"), score_gap_max=("score_mean_gap", "max"),
            ).reset_index()
        )
        result = {
            "audited": True,
            "complete_instances": int(decision_frame.instance_id.nunique()),
            "instance_ids": sorted(decision_frame.instance_id.unique().tolist()),
            **decide_cross_instance(decision_frame, int(cfg["minimum_complete_instances"])),
            "independent_unit": "generator_training_instance",
            "split_seeds_are_independent_generator_repeats": False,
        }
        decision_frame.to_csv(partial / "instance_decisions.csv", index=False)
        metric_frame.to_csv(partial / "split_level_metrics.csv", index=False)
        instance_points.to_csv(partial / "instance_level_metrics.csv", index=False)
        between.to_csv(partial / "between_instance_summary.csv", index=False)
        (partial / "audit_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        partial.replace(final)
        return result
    except Exception:
        (partial / "failure.json").write_text(
            json.dumps({"status": "failed"}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config, args.results_root, args.output), ensure_ascii=False))

