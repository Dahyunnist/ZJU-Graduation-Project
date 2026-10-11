"""Independently audit E3B row scores, metrics, intervals and branch decision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from tabpollution.utils import sha256_file


EXPECTED_CELLS = {
    (train_generator, eval_generator)
    for train_generator in ("CTGAN", "TVAE")
    for eval_generator in ("CTGAN", "TVAE")
}


def _metric(row: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = row.source_label.to_numpy(dtype=int)
    score = row.raw_score.to_numpy(dtype=float)
    real, synthetic = score[y == 0], score[y == 1]
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "score_mean_gap": float(synthetic.mean() - real.mean()),
        "ks_distance": float(ks_2samp(real, synthetic).statistic),
        "tpr_at_calibrated_5pct_fpr": float((synthetic >= threshold).mean()),
        "observed_fpr": float((real >= threshold).mean()),
    }


def _seed(seed: int, *parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join([str(seed), *parts]).encode()).digest()[:4], "big")


def _bootstrap(row: pd.DataFrame, threshold: float, replicates: int, confidence: float,
               seed: int) -> dict[str, float]:
    y = row.source_label.to_numpy(dtype=int)
    score = row.raw_score.to_numpy(dtype=float)
    real, synthetic = score[y == 0], score[y == 1]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("auroc", "auprc", "score_mean_gap", "ks_distance", "tpr")}
    for _ in range(replicates):
        r = real[rng.integers(0, len(real), len(real))]
        s = synthetic[rng.integers(0, len(synthetic), len(synthetic))]
        labels, values = np.r_[np.zeros(len(r), int), np.ones(len(s), int)], np.r_[r, s]
        draws["auroc"].append(roc_auc_score(labels, values))
        draws["auprc"].append(average_precision_score(labels, values))
        draws["score_mean_gap"].append(s.mean() - r.mean())
        draws["ks_distance"].append(ks_2samp(r, s).statistic)
        draws["tpr"].append((s >= threshold).mean())
    alpha = (1 - confidence) / 2
    return {f"{name}_ci_{side}": float(np.quantile(values, quantile))
            for name, values in draws.items()
            for side, quantile in (("low", alpha), ("high", 1 - alpha))}


def _paired(scores: pd.DataFrame, replicates: int, confidence: float, seed: int) -> dict[str, float]:
    target = scores.loc[scores.eval_generator == "TVAE"]
    c = target.loc[target.train_generator == "CTGAN", ["record_id", "source_label", "raw_score"]]
    t = target.loc[target.train_generator == "TVAE", ["record_id", "source_label", "raw_score"]]
    merged = c.merge(t, on=["record_id", "source_label"], suffixes=("_ctgan", "_tvae"), validate="one_to_one")
    if len(merged) != len(c) or len(merged) != len(t):
        raise AssertionError("E3B target evaluation identity mismatch")
    labels = merged.source_label.to_numpy(int)
    cscore, tscore = merged.raw_score_ctgan.to_numpy(float), merged.raw_score_tvae.to_numpy(float)
    groups = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    rng = np.random.default_rng(seed)
    auc, gap = [], []
    for _ in range(replicates):
        idx = np.concatenate([group[rng.integers(0, len(group), len(group))] for group in groups.values()])
        y, cs, ts = labels[idx], cscore[idx], tscore[idx]
        auc.append(roc_auc_score(y, ts) - roc_auc_score(y, cs))
        gap.append((ts[y == 1].mean() - ts[y == 0].mean()) - (cs[y == 1].mean() - cs[y == 0].mean()))
    alpha = (1 - confidence) / 2
    return {
        "auroc_gain": float(roc_auc_score(labels, tscore) - roc_auc_score(labels, cscore)),
        "auroc_gain_ci_low": float(np.quantile(auc, alpha)),
        "auroc_gain_ci_high": float(np.quantile(auc, 1 - alpha)),
        "score_gap_gain": float(np.mean(gap)),
        "score_gap_gain_ci_low": float(np.quantile(gap, alpha)),
        "score_gap_gain_ci_high": float(np.quantile(gap, 1 - alpha)),
    }


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def analyze(source: Path, destination: Path, *, expected_lr_cases: int = 6) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if list(source.glob("**/*.partial")):
        raise AssertionError("E3B contains partial cases")
    context = json.loads((source / "context.json").read_text(encoding="utf-8"))
    cfg = context["protocol"]
    lr_cases = sorted(source.glob("lr/*/seed*.complete/complete.json"))
    if len(lr_cases) != expected_lr_cases:
        raise AssertionError(f"Expected {expected_lr_cases} completed E3B LR cases, found {len(lr_cases)}")
    all_metrics, all_intervals, all_paired = [], [], []
    for complete_path in lr_cases:
        folder = complete_path.parent
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "complete" or complete.get("context") != context:
            raise AssertionError(f"Stale/incomplete E3B case: {folder}")
        for name, digest in complete["files_sha256"].items():
            if sha256_file(folder / name) != digest:
                raise AssertionError(f"E3B file hash changed: {folder / name}")
        assignments = pd.read_csv(folder / "split_assignments.csv.gz")
        for generator, group in assignments.groupby("generator"):
            identities = {role: set(rows.record_id) for role, rows in group.groupby("split")}
            if set(identities) != {"detector_train", "calibration", "evaluation"}:
                raise AssertionError(f"E3B split set incomplete: {folder}/{generator}")
            if any(identities[a] & identities[b] for a, b in (("detector_train", "calibration"), ("detector_train", "evaluation"), ("calibration", "evaluation"))):
                raise AssertionError(f"E3B split leakage: {folder}/{generator}")
        scores = pd.read_csv(folder / "scores.csv.gz")
        if set(zip(scores.train_generator, scores.eval_generator)) != EXPECTED_CELLS:
            raise AssertionError(f"E3B 2x2 matrix incomplete: {folder}")
        if set(scores.model_direction) != {"high_score_is_synthetic"}:
            raise AssertionError(f"E3B target-dependent score flip detected: {folder}")
        metrics = pd.read_csv(folder / "metrics.csv")
        intervals = pd.read_csv(folder / "bootstrap_intervals.csv")
        paired = pd.read_csv(folder / "paired_intervals.csv")
        for recorded in metrics.itertuples(index=False):
            row = scores.loc[(scores.train_generator == recorded.train_generator) &
                             (scores.eval_generator == recorded.eval_generator)]
            observed = _metric(row, recorded.calibration_threshold)
            for name, value in observed.items():
                if not _close(value, getattr(recorded, name)):
                    raise AssertionError(f"E3B metric mismatch: {folder} {recorded.cell_id} {name}")
            expected_n = len(row)
            if recorded.n != expected_n or recorded.n_real != int((row.source_label == 0).sum()) or recorded.n_synthetic != int((row.source_label == 1).sum()):
                raise AssertionError(f"E3B sample count mismatch: {folder} {recorded.cell_id}")
            interval = intervals.loc[(intervals.train_generator == recorded.train_generator) &
                                     (intervals.eval_generator == recorded.eval_generator)].iloc[0]
            observed_interval = _bootstrap(
                row, recorded.calibration_threshold, int(cfg["bootstrap_replicates"]),
                float(cfg["bootstrap_confidence"]),
                _seed(int(recorded.seed), str(recorded.table), str(recorded.train_generator), str(recorded.eval_generator)),
            )
            for name, value in observed_interval.items():
                if not _close(value, interval[name]):
                    raise AssertionError(f"E3B interval mismatch: {folder} {recorded.cell_id} {name}")
        observed_paired = _paired(scores, int(cfg["bootstrap_replicates"]), float(cfg["bootstrap_confidence"]),
                                  _seed(int(paired.seed.iloc[0]), str(paired.table.iloc[0]), "paired"))
        for name, value in observed_paired.items():
            if not _close(value, paired[name].iloc[0]):
                raise AssertionError(f"E3B paired interval mismatch: {folder} {name}")
        all_metrics.append(metrics)
        all_intervals.append(intervals)
        all_paired.append(paired)
    metrics = pd.concat(all_metrics, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    paired = pd.concat(all_paired, ignore_index=True)
    summary_rows, capacity_tables = [], []
    for table in cfg["tables"]:
        same = metrics.loc[(metrics.table == table) & (metrics.train_generator == "TVAE") & (metrics.eval_generator == "TVAE")].sort_values("seed")
        same_ci = intervals.loc[(intervals.table == table) & (intervals.train_generator == "TVAE") & (intervals.eval_generator == "TVAE")].sort_values("seed")
        paired_table = paired.loc[paired.table == table].sort_values("seed")
        transfer = bool(
            (same.auroc > .5).all() and (same.score_mean_gap > 0).all()
            and (paired_table.auroc_gain >= float(cfg["transfer_min_auroc_gain"])).all()
            and (same_ci.auroc_ci_low > .5).all() and (same_ci.score_mean_gap_ci_low > 0).all()
            and (paired_table.auroc_gain_ci_low > 0).all()
        )
        covers = ((same_ci.auroc_ci_low <= .5) & (same_ci.auroc_ci_high >= .5)).sum()
        near_random = bool(same.auroc.mean() < float(cfg["near_random_auroc"]) or covers >= 2)
        if near_random:
            capacity_tables.append(table)
        summary_rows.append({
            "table": table, "lr_transfer_rule_passed": transfer, "lr_near_random": near_random,
            "tvae_same_domain_auroc_mean": float(same.auroc.mean()),
            "tvae_same_domain_gap_mean": float(same.score_mean_gap.mean()),
            "paired_auroc_gain_mean": float(paired_table.auroc_gain.mean()),
            "same_domain_positive_direction_seeds": int((same.score_mean_gap > 0).sum()),
        })
    capacity_paths = sorted(source.glob("gbdt/*/seed*.complete/complete.json"))
    expected_capacity = len(capacity_tables) * len(cfg["seeds"])
    if len(capacity_paths) != expected_capacity:
        raise AssertionError(f"Expected {expected_capacity} capacity cases, found {len(capacity_paths)}")
    capacity_metrics, capacity_intervals = [], []
    for complete_path in capacity_paths:
        folder = complete_path.parent
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "complete" or complete.get("context") != context:
            raise AssertionError(f"Stale/incomplete E3B capacity case: {folder}")
        for name, digest in complete["files_sha256"].items():
            if sha256_file(folder / name) != digest:
                raise AssertionError(f"E3B capacity file hash changed: {folder / name}")
        assignments = pd.read_csv(folder / "split_assignments.csv.gz")
        identities = {role: set(rows.record_id) for role, rows in assignments.groupby("split")}
        if set(identities) != {"detector_train", "calibration", "evaluation"}:
            raise AssertionError(f"E3B capacity split set incomplete: {folder}")
        if any(identities[a] & identities[b] for a, b in (("detector_train", "calibration"), ("detector_train", "evaluation"), ("calibration", "evaluation"))):
            raise AssertionError(f"E3B capacity split leakage: {folder}")
        scores = pd.read_csv(folder / "scores.csv.gz")
        if set(zip(scores.train_generator, scores.eval_generator)) != {("TVAE", "TVAE")}:
            raise AssertionError(f"E3B capacity cell changed: {folder}")
        if set(scores.model_direction) != {"high_score_is_synthetic"}:
            raise AssertionError(f"E3B capacity score flip detected: {folder}")
        metric_frame = pd.read_csv(folder / "metrics.csv")
        interval_frame = pd.read_csv(folder / "bootstrap_intervals.csv")
        recorded = metric_frame.iloc[0]
        observed = _metric(scores, float(recorded.calibration_threshold))
        for name, value in observed.items():
            if not _close(value, recorded[name]):
                raise AssertionError(f"E3B capacity metric mismatch: {folder} {name}")
        observed_interval = _bootstrap(
            scores, float(recorded.calibration_threshold), int(cfg["bootstrap_replicates"]),
            float(cfg["bootstrap_confidence"]), _seed(int(recorded.seed), str(recorded.table), "capacity"),
        )
        for name, value in observed_interval.items():
            if not _close(value, interval_frame[name].iloc[0]):
                raise AssertionError(f"E3B capacity interval mismatch: {folder} {name}")
        capacity_metrics.append(metric_frame)
        capacity_intervals.append(interval_frame)
    capacity = pd.concat(capacity_metrics, ignore_index=True) if capacity_metrics else pd.DataFrame()
    capacity_ci = pd.concat(capacity_intervals, ignore_index=True) if capacity_intervals else pd.DataFrame()
    decisions = []
    for row in summary_rows:
        table = row["table"]
        if row["lr_transfer_rule_passed"]:
            branch = "E3C-Transfer"
        elif row["lr_near_random"]:
            g = capacity.loc[capacity.table == table] if not capacity.empty else pd.DataFrame()
            ci = capacity_ci.loc[capacity_ci.table == table] if not capacity_ci.empty else pd.DataFrame()
            stable = (len(g) == 3 and len(ci) == 3 and (g.auroc > .5).all()
                      and (g.score_mean_gap > 0).all() and (ci.auroc_ci_low > .5).all()
                      and (ci.score_mean_gap_ci_low > 0).all())
            branch = "E3C-Capacity" if stable else "E3C-Overlap"
        else:
            branch = "Uncertain"
        decisions.append({**row, "branch": branch})
    branch_set = {row["branch"] for row in decisions}
    overall = next(iter(branch_set)) if len(branch_set) == 1 else "Uncertain"
    destination.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(destination / "matrix_metrics.csv", index=False)
    intervals.to_csv(destination / "bootstrap_intervals.csv", index=False)
    paired.to_csv(destination / "paired_auroc_gain.csv", index=False)
    pd.DataFrame(decisions).to_csv(destination / "table_decisions.csv", index=False)
    if not capacity.empty:
        capacity.to_csv(destination / "capacity_metrics.csv", index=False)
        capacity_ci.to_csv(destination / "capacity_bootstrap_intervals.csv", index=False)
    result = {
        "audited": True, "complete_lr_cases": len(lr_cases),
        "matrix_cells": len(metrics), "capacity_cases": len(capacity_paths),
        "table_branches": {row["table"]: row["branch"] for row in decisions},
        "overall_branch": overall,
    }
    (destination / "audit_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-lr-cases", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output, expected_lr_cases=args.expected_lr_cases), ensure_ascii=False))
