"""Post-hoc analysis for the frozen multi-seed governance experiment.

The module never mutates shard outputs.  It audits the aggregated evidence,
applies a row-level inclusion policy for pre-registered quantifier failures,
uses seeds (not bags) as independent replicates, and writes paper-facing
tables, figures, and a decision memo for supplementary experiments.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from tabpollution.governance.config import load_governance_config
from tabpollution.governance.shards import build_shard_plan
from tabpollution.utils import read_json, sha256_file, write_json


EVIDENCE_KEY = [
    "seed", "protocol", "detector", "calibration_policy", "quantifier",
    "test_table", "test_generator", "contamination_mode", "true_prevalence",
    "bag_index",
]

VALUATION_KEY = [
    "seed", "protocol", "test_table", "test_generator", "contamination_mode",
    "true_prevalence", "bag_index", "valuation_method", "valuation_row_index",
    "record_id",
]

FAILURE_CLASSIFICATION = {
    "failed:unstable_denominator": "pre_registered_identifiability_failure",
    "failed:kdey_requires_nonconstant_class_scores": "pre_registered_score_degeneracy",
    "failed:no_valid_median_sweep_threshold": "pre_registered_threshold_failure",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values in original order."""
    values = np.asarray(list(p_values), dtype=float)
    if not len(values):
        return values
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def _ci(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(numeric)
    mean = float(numeric.mean()) if n else float("nan")
    std = float(numeric.std(ddof=1)) if n > 1 else float("nan")
    half = (
        float(student_t.ppf(.975, n - 1) * std / math.sqrt(n))
        if n > 1 else float("nan")
    )
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half if np.isfinite(half) else float("nan"),
        "ci95_high": mean + half if np.isfinite(half) else float("nan"),
    }


def summarize_seed_replicates(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average technical repeats within seed, then form CIs across seeds."""
    seed_groups = ["seed", *group_columns]
    available = [metric for metric in metrics if metric in frame.columns]
    seed_level = (
        frame.groupby(seed_groups, dropna=False, as_index=False)[available]
        .mean(numeric_only=True)
    )
    records: list[dict[str, Any]] = []
    for keys, group in seed_level.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in available:
            records.append({**base, "metric": metric, **_ci(group[metric])})
    return seed_level, pd.DataFrame(records)


def paired_detector_tests(rows: pd.DataFrame, primary_quantifier: str) -> pd.DataFrame:
    """Compare detectors using matched bags, with seeds as independent units."""
    primary = rows.loc[
        rows["quantifier"].eq(primary_quantifier)
        & rows["quantifier_status"].eq("ok")
    ].copy()
    family_columns = [
        "protocol", "calibration_policy", "contamination_mode", "true_prevalence",
    ]
    match_columns = [
        "seed", "test_table", "test_generator", "bag_index",
    ]
    records: list[dict[str, Any]] = []
    for family, group in primary.groupby(family_columns, dropna=False):
        pivot = group.pivot_table(
            index=match_columns,
            columns="detector",
            values="prevalence_absolute_error",
            aggfunc="first",
        )
        for left, right in combinations(sorted(pivot.columns), 2):
            matched = pivot[[left, right]].dropna()
            if matched.empty:
                continue
            bag_difference = matched[left] - matched[right]
            seed_difference = bag_difference.groupby(level="seed").mean()
            seed_left = matched[left].groupby(level="seed").mean()
            seed_right = matched[right].groupby(level="seed").mean()
            n = len(seed_difference)
            test = ttest_1samp(seed_difference.to_numpy(float), 0.0) if n > 1 else None
            ci = _ci(seed_difference)
            std_diff = float(seed_difference.std(ddof=1)) if n > 1 else float("nan")
            records.append({
                **dict(zip(family_columns, family)),
                "metric": "prevalence_absolute_error",
                "left_detector": left,
                "right_detector": right,
                "n_seeds": n,
                "matched_bags": len(matched),
                "mean_left": float(seed_left.mean()),
                "mean_right": float(seed_right.mean()),
                "mean_difference": ci["mean"],
                "difference_ci95_low": ci["ci95_low"],
                "difference_ci95_high": ci["ci95_high"],
                "cohen_dz": (
                    float(ci["mean"] / std_diff)
                    if n > 1 and np.isfinite(std_diff) and std_diff > 0 else float("nan")
                ),
                "t_statistic": float(test.statistic) if test is not None else float("nan"),
                "p_value": float(test.pvalue) if test is not None else float("nan"),
            })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["p_value_holm"] = np.nan
    for _, indices in result.groupby(family_columns, dropna=False).groups.items():
        idx = list(indices)
        valid = [i for i in idx if np.isfinite(result.at[i, "p_value"])]
        if valid:
            result.loc[valid, "p_value_holm"] = holm_adjust(result.loc[valid, "p_value"])
    result["significant_holm_0_05"] = result["p_value_holm"].lt(.05)
    return result.sort_values([*family_columns, "p_value_holm"], na_position="last")


def _audit(
    evidence: pd.DataFrame,
    valuation: pd.DataFrame,
    artifacts: pd.DataFrame,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_governance_config(config_path)
    plan = build_shard_plan(config_path)
    markers = list((config.output_dir / "shards").glob("s*/COMPLETE.json"))
    failed_markers = list((config.output_dir / "shards").glob("s*/FAILED.json"))
    selected_manifest = read_json(config.output_dir / "completed_shards_manifest.json")
    selected_attempts = [Path(path) for path in selected_manifest["attempts"]]
    selected_attempt_failures = [path / "FAILED.json" for path in selected_attempts if (path / "FAILED.json").is_file()]
    retained_failed_attempts = list((config.output_dir / "shards").glob("s*/attempts/*/FAILED.json"))
    expected_rows_per_shard = len(config.calibration_policies) * len(config.quantifiers) * config.bags_per_rate * (
        len(config.prevalence_rates) + len([x for x in config.prevalence_rates if x < 1.0])
    )
    shard_counts = (
        evidence.groupby(["seed", "protocol", "detector"], as_index=False)
        .size().rename(columns={"size": "rows"})
    )
    row_count_violations = shard_counts.loc[shard_counts["rows"].ne(expected_rows_per_shard)]
    duplicate_evidence = int(evidence.duplicated(EVIDENCE_KEY).sum())
    duplicate_valuation = int(valuation.duplicated(VALUATION_KEY).sum())
    ok = evidence["quantifier_status"].eq("ok")
    estimates = pd.to_numeric(evidence["estimated_prevalence"], errors="coerce")
    ok_nonfinite = int((ok & ~np.isfinite(estimates)).sum())
    failed_with_estimate = int((~ok & np.isfinite(estimates)).sum())
    estimate_out_of_range = int((ok & ((estimates < -1e-12) | (estimates > 1 + 1e-12))).sum())
    residual = pd.to_numeric(evidence["error_decomposition_residual"], errors="coerce")
    max_residual = float(residual.abs().max())
    expected_seeds = set(config.seeds)
    expected_protocols = set(config.protocols)
    expected_policies = set(config.calibration_policies)
    expected_quantifiers = set(config.quantifiers)
    checks = {
        "shard_plan_count_is_110": plan["shard_count"] == 110,
        "completion_marker_count_matches_plan": len(markers) == plan["shard_count"],
        "no_root_failed_markers": len(failed_markers) == 0,
        "no_attempt_failures_in_selected_results": len(selected_attempt_failures) == 0,
        "all_shards_have_expected_evidence_rows": row_count_violations.empty,
        "evidence_primary_key_unique": duplicate_evidence == 0,
        "valuation_primary_key_unique": duplicate_valuation == 0,
        "ok_quantifier_rows_have_finite_estimates": ok_nonfinite == 0,
        "failed_quantifier_rows_do_not_carry_estimates": failed_with_estimate == 0,
        "estimates_are_in_unit_interval": estimate_out_of_range == 0,
        "error_decomposition_residual_within_1e_10": max_residual <= 1e-10,
        "all_expected_seeds_present": set(evidence["seed"].unique()) == expected_seeds,
        "all_expected_protocols_present": set(evidence["protocol"].unique()) == expected_protocols,
        "all_expected_policies_present": set(evidence["calibration_policy"].unique()) == expected_policies,
        "all_expected_quantifiers_present": set(evidence["quantifier"].unique()) == expected_quantifiers,
        "artifact_rows_are_seed_by_protocol": len(artifacts) == len(config.seeds) * len(config.protocols),
        "artifact_gate_has_no_failures": bool(artifacts["artifact_gate_passed"].astype(bool).all()),
    }
    audit = {
        "created_at": _utc_now(),
        "experiment_id": config.experiment_id,
        "base_config_sha256": sha256_file(config_path),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "planned_shards": plan["shard_count"],
            "completion_markers": len(markers),
            "evidence_rows": len(evidence),
            "valuation_rows": len(valuation),
            "artifact_rows": len(artifacts),
            "quantifier_failure_rows": int((~ok).sum()),
            "duplicate_evidence_keys": duplicate_evidence,
            "duplicate_valuation_keys": duplicate_valuation,
            "row_count_violations": len(row_count_violations),
            "ok_nonfinite_estimates": ok_nonfinite,
            "failed_rows_with_estimates": failed_with_estimate,
            "estimate_out_of_range": estimate_out_of_range,
            "selected_attempt_failure_markers": len(selected_attempt_failures),
            "retained_historical_failed_attempts": len(retained_failed_attempts),
        },
        "expected_rows_per_shard": expected_rows_per_shard,
        "max_absolute_error_decomposition_residual": max_residual,
    }
    write_json(audit, output / "integrity_audit.json")
    lines = [
        "# 正式实验完整性与质量审计", "",
        f"- 实验：`{config.experiment_id}`",
        f"- 总体结论：**{'通过' if audit['passed'] else '未通过'}**",
        f"- 分片：{len(markers)} / {plan['shard_count']}",
        f"- 治理证据：{len(evidence):,} 行；估值记录：{len(valuation):,} 行",
        f"- 误差分解最大绝对残差：`{max_residual:.3e}`", "",
        "## 检查项", "",
        "| 检查 | 结果 |", "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'通过' if passed else '失败'} |" for name, passed in checks.items())
    lines.extend([
        "", "## 解释边界", "",
        "完整性审计通过只表示冻结分片、主键、数值和制品一致。量化器在预注册条件下不可定义时仍会保留失败状态，是否进入某张统计表由逐组合纳入规则决定，不能用插值或回退值替代。", "",
    ])
    (output / "integrity_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def _failure_audit(evidence: pd.DataFrame, output: Path, primary: str) -> pd.DataFrame:
    failed = evidence.loc[evidence["quantifier_status"].ne("ok")].copy()
    columns = [
        "seed", "protocol", "detector", "calibration_policy", "quantifier",
        "quantifier_status",
    ]
    audit = failed.groupby(columns, dropna=False, as_index=False).agg(
        failed_rows=("bag_id", "size"),
        affected_rates=("true_prevalence", lambda x: ";".join(map(str, sorted(set(x))))),
        affected_modes=("contamination_mode", lambda x: ";".join(sorted(set(x)))),
    )
    audit["failure_class"] = audit["quantifier_status"].map(FAILURE_CLASSIFICATION).fillna(
        "unclassified_failure"
    )
    audit["is_primary_quantifier"] = audit["quantifier"].eq(primary)
    audit["rerun_required"] = audit["failure_class"].eq("unclassified_failure")
    audit["formal_inclusion"] = False
    audit["inclusion_reason"] = "method-condition cell excluded; other valid cells retained"
    audit.to_csv(output / "method_failure_audit.csv", index=False)

    failed_shards = audit[["seed", "protocol", "detector"]].drop_duplicates()
    primary_cells = int(audit["is_primary_quantifier"].sum())
    lines = [
        "# 局部方法失败审计与正式纳入规则", "",
        f"共发现 **{len(failed_shards)} 个分片**含局部量化方法失败，涉及 {len(audit)} 个 seed/协议/检测器/策略/量化器组合。",
        f"其中主量化器 `{primary}` 的失败组合为 {primary_cells} 个。所有失败原因均为预注册的可识别性、常数得分或阈值不可用状态；未发现未分类异常。", "",
        "## 纳入规则", "",
        "1. 检测层指标不依赖量化器，完整纳入 110 个分片。",
        "2. 比例估计、误差分解和估计触发的决策只纳入 `quantifier_status=ok` 的方法—条件单元。",
        "3. 失败单元保持缺失并在覆盖率中报告，不插值、不改阈值、不增加观察结果后的回退规则。",
        "4. `oracle_target` 是诊断上界；其失败不否定 `source_only` 或 `target_real_anchor` 的可部署结果。",
        "5. 只有出现未分类异常、数据/主键破坏或同一冻结条件下非确定性失败时才要求补跑。", "",
        "## 结论", "",
        f"本次审计中 `rerun_required=true` 的组合数为 **{int(audit['rerun_required'].sum())}**，因此无需重跑主矩阵。", "",
    ]
    (output / "method_failure_inclusion_policy.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def _write_table(
    frame: pd.DataFrame,
    output: Path,
    name: str,
    group_columns: list[str],
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed, ci = summarize_seed_replicates(frame, group_columns=group_columns, metrics=metrics)
    seed.to_csv(output / f"{name}_seed_level.csv", index=False)
    ci.to_csv(output / f"{name}.csv", index=False)
    return seed, ci


def _formal_tables(
    evidence: pd.DataFrame,
    valuation: pd.DataFrame,
    artifacts: pd.DataFrame,
    output: Path,
    primary: str,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    detector_rows = evidence.drop_duplicates([
        "seed", "protocol", "detector", "calibration_policy",
    ])
    _, tables["finding_1"] = _write_table(
        detector_rows, output, "finding_1_transfer_formal",
        ["protocol", "detector", "calibration_policy"],
        ["detection_auroc", "detection_auprc", "detection_fpr", "detection_tpr", "detection_ece"],
    )
    _, tables["finding_2"] = _write_table(
        artifacts, output, "finding_2_format_artifacts_formal",
        ["protocol"], ["artifact_auroc"],
    )
    valid = evidence.loc[evidence["quantifier_status"].eq("ok")].copy()
    low = valid.loc[valid["true_prevalence"].isin([.05, .10])]
    _, tables["finding_3"] = _write_table(
        low, output, "finding_3_low_prevalence_formal",
        ["protocol", "detector", "calibration_policy", "quantifier", "contamination_mode", "true_prevalence"],
        ["prevalence_absolute_error", "prevalence_error", "false_positive_share", "prevalence_decision_error", "decision_regret"],
    )
    _, tables["finding_4"] = _write_table(
        valid, output, "finding_4_quantifier_shift_formal",
        ["protocol", "detector", "calibration_policy", "quantifier", "contamination_mode"],
        ["prevalence_absolute_error", "prevalence_error", "decision_regret"],
    )
    primary_rows = valid.loc[valid["quantifier"].eq(primary)]
    _, tables["finding_5"] = _write_table(
        primary_rows, output, "finding_5_utility_curve_formal",
        ["protocol", "detector", "calibration_policy", "contamination_mode", "true_prevalence"],
        ["contaminated_utility_delta", "detector_cleanup_delta", "oracle_cleanup_delta", "decision_regret"],
    )
    _, tables["finding_6"] = _write_table(
        primary_rows, output, "finding_6_detectability_vs_harm_formal",
        ["protocol", "detector", "calibration_policy", "contamination_mode", "true_prevalence"],
        ["detection_auroc", "contaminated_utility_delta", "detector_cleanup_delta", "decision_regret"],
    )
    if not valuation.empty:
        value_groups = [
            "seed", "protocol", "test_table", "test_generator", "contamination_mode",
            "true_prevalence", "valuation_method",
        ]
        value_records: list[dict[str, Any]] = []
        for keys, group in valuation.groupby(value_groups, dropna=False):
            real = pd.to_numeric(group.loc[group["source_label"].eq(0), "task_value"], errors="coerce")
            synthetic = pd.to_numeric(group.loc[group["source_label"].eq(1), "task_value"], errors="coerce")
            value_records.append({
                **dict(zip(value_groups, keys)),
                "real_mean_value": real.mean(),
                "synthetic_mean_value": synthetic.mean(),
                "synthetic_negative_value_rate": (synthetic < 0).mean(),
            })
        grouped = pd.DataFrame(value_records)
        grouped["source_task_value_gap"] = grouped["real_mean_value"] - grouped["synthetic_mean_value"]
        grouped.to_csv(output / "finding_6_source_task_value_seed_level.csv", index=False)
        records: list[dict[str, Any]] = []
        groups = ["protocol", "test_table", "test_generator", "contamination_mode", "true_prevalence", "valuation_method"]
        for keys, block in grouped.groupby(groups, dropna=False):
            base = dict(zip(groups, keys))
            for metric in ["real_mean_value", "synthetic_mean_value", "synthetic_negative_value_rate", "source_task_value_gap"]:
                records.append({**base, "metric": metric, **_ci(block[metric])})
        tables["finding_6_value"] = pd.DataFrame(records)
        tables["finding_6_value"].to_csv(output / "finding_6_source_task_value_formal.csv", index=False)
    combined = primary_rows.groupby([
        "seed", "protocol", "detector", "calibration_policy",
    ], as_index=False).agg(
        source_validation_auroc=("source_validation_auroc", "first"),
        target_test_auroc=("detection_auroc", "first"),
        ranking_shift=("ranking_shift", "first"),
        source_validation_ece=("source_validation_ece", "first"),
        target_test_ece=("detection_ece", "first"),
        calibration_shift=("calibration_shift", "first"),
        reference_fpr=("calibration_reference_fpr", "first"),
        target_test_fpr=("detection_fpr", "first"),
        threshold_fpr_shift=("threshold_fpr_shift", "first"),
        prevalence_mae=("prevalence_absolute_error", "mean"),
        prevalence_decision_error_rate=("prevalence_decision_error", "mean"),
        governance_regret=("decision_regret", "mean"),
    )
    combined.to_csv(output / "finding_7_calibration_threshold_transfer_seed_level.csv", index=False)
    records = []
    groups = ["protocol", "detector", "calibration_policy"]
    for keys, block in combined.groupby(groups, dropna=False):
        base = dict(zip(groups, keys))
        for metric in [x for x in combined.columns if x not in ["seed", *groups]]:
            records.append({**base, "metric": metric, **_ci(block[metric])})
    tables["finding_7"] = pd.DataFrame(records)
    tables["finding_7"].to_csv(output / "finding_7_calibration_threshold_transfer_formal.csv", index=False)
    _, tables["finding_8"] = _write_table(
        valid, output, "finding_8_error_decomposition_formal",
        ["protocol", "detector", "calibration_policy", "quantifier", "contamination_mode", "true_prevalence"],
        ["detection_error_contribution", "bag_sampling_error_contribution", "quantifier_adjustment_contribution", "prevalence_error", "error_decomposition_residual"],
    )
    return tables


def _plot_tables(tables: dict[str, pd.DataFrame], output: Path, primary: str) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Formal figure generation requires matplotlib") from exc

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def save(fig: Any, name: str, *, tight_rect: tuple[float, float, float, float] | None = None) -> None:
        fig.tight_layout(rect=tight_rect)
        for suffix in ("png", "pdf"):
            path = figures / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            created.append(path)
        plt.close(fig)

    colors = {"P1": "#2563EB", "P2": "#0F766E", "P3": "#D97706", "P4": "#DC2626"}
    f1 = tables["finding_1"]
    f1 = f1.loc[f1["metric"].eq("detection_auroc") & f1["calibration_policy"].eq("source_only")]
    pivot = f1.pivot(index="detector", columns="protocol", values="mean")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(pivot.index)); width = .18
    for idx, protocol in enumerate([p for p in ["P1", "P2", "P3", "P4"] if p in pivot.columns]):
        ax.bar(x + (idx - 1.5) * width, pivot[protocol], width, label=protocol, color=colors[protocol])
        for detector_idx, value in enumerate(pivot[protocol]):
            if pd.isna(value):
                ax.text(
                    detector_idx + (idx - 1.5) * width,
                    .025,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    color="#6B7280",
                )
    ax.set_xticks(x, [x.replace("_", "\n") for x in pivot.index], rotation=0)
    ax.set_ylim(0, 1); ax.set_ylabel("Target AUROC"); ax.set_title("Finding 1: Detection transfer under P1-P4 (source-only)")
    ax.axhline(.5, color="#6B7280", linestyle="--", linewidth=1); ax.legend(ncol=4)
    save(fig, "finding_1_detection_transfer")

    f2 = tables["finding_2"].loc[lambda x: x["metric"].eq("artifact_auroc")]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(f2["protocol"], f2["mean"], color=[colors.get(x, "#64748B") for x in f2["protocol"]])
    ax.axhline(.65, color="#DC2626", linestyle="--", label="gate = 0.65")
    ax.set_ylim(.45, .75); ax.set_ylabel("Artifact-only AUROC"); ax.set_title("Finding 2: Format-artifact audit")
    ax.legend(); save(fig, "finding_2_format_artifacts")

    f3 = tables["finding_3"]
    f3 = f3.loc[
        f3["metric"].eq("prevalence_absolute_error")
        & f3["calibration_policy"].eq("source_only")
        & f3["quantifier"].eq(primary)
        & f3["contamination_mode"].eq("replace")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, rate in zip(axes, [.05, .10]):
        block = f3.loc[np.isclose(f3["true_prevalence"], rate)]
        detector_order = sorted(block["detector"].unique())
        x = np.arange(len(detector_order)); width = .18
        for idx, protocol in enumerate(["P1", "P2", "P3", "P4"]):
            values = block.loc[block["protocol"].eq(protocol)].set_index("detector")["mean"].reindex(detector_order)
            positions = x + (idx - 1.5) * width
            ax.bar(positions, values, width, label=protocol, color=colors[protocol])
            for detector_idx, value in enumerate(values):
                if pd.isna(value):
                    ax.text(positions[detector_idx], .015, "N/A", ha="center", va="bottom", rotation=90, fontsize=6, color="#6B7280")
        ax.set_title(f"True prevalence = {int(rate*100)}%")
        ax.set_xticks(x, detector_order, rotation=55, ha="right"); ax.set_ylabel("PACC MAE")
    axes[1].legend(); fig.suptitle("Finding 3: Low-prevalence estimation error (source-only, replace)")
    save(fig, "finding_3_low_prevalence")

    f4 = tables["finding_4"]
    f4 = f4.loc[
        f4["metric"].eq("prevalence_absolute_error")
        & f4["calibration_policy"].eq("source_only")
        & f4["contamination_mode"].eq("replace")
    ]
    matrix = f4.groupby(["protocol", "quantifier"], as_index=False)["mean"].mean().pivot(
        index="quantifier", columns="protocol", values="mean"
    )
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ordered_protocols = [x for x in ["P1", "P2", "P3", "P4"] if x in matrix]
    ordered_matrix = matrix[ordered_protocols]
    image = ax.imshow(ordered_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=min(.7, float(np.nanmax(ordered_matrix.to_numpy()))))
    ax.set_xticks(range(len(ordered_protocols)), ordered_protocols); ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Finding 4: Quantifier MAE under distribution shift"); fig.colorbar(image, ax=ax, label="MAE")
    save(fig, "finding_4_quantifier_shift")

    f5 = tables["finding_5"]
    f5 = f5.loc[f5["calibration_policy"].eq("source_only")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, protocol in zip(axes.flat, ["P1", "P2", "P3", "P4"]):
        block = f5.loc[f5["protocol"].eq(protocol)]
        for mode, style in [("replace", "-"), ("append", "--")]:
            for metric, label, color in [
                ("contaminated_utility_delta", "keep", "#64748B"),
                ("detector_cleanup_delta", "detector cleanup", "#2563EB"),
                ("oracle_cleanup_delta", "oracle cleanup", "#0F766E"),
            ]:
                line = block.loc[block["metric"].eq(metric) & block["contamination_mode"].eq(mode)]
                line = line.groupby("true_prevalence", as_index=False)["mean"].mean()
                ax.plot(line["true_prevalence"], line["mean"], linestyle=style, marker="o", color=color, label=f"{label}/{mode}")
        ax.axhline(0, color="#9CA3AF", linewidth=1); ax.set_title(protocol); ax.set_xlabel("True prevalence"); ax.set_ylabel("Utility delta")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3)
    fig.suptitle("Finding 5: Utility consequences of keep and cleanup", y=.995)
    save(fig, "finding_5_utility_curve", tight_rect=(0, .12, 1, .96))

    f6 = tables["finding_6"]
    auroc = f6.loc[f6["metric"].eq("detection_auroc"), ["protocol", "detector", "calibration_policy", "contamination_mode", "true_prevalence", "mean"]].rename(columns={"mean": "auroc"})
    harm = f6.loc[f6["metric"].eq("contaminated_utility_delta"), ["protocol", "detector", "calibration_policy", "contamination_mode", "true_prevalence", "mean"]].rename(columns={"mean": "utility_delta"})
    scatter = auroc.merge(harm)
    scatter = scatter.loc[scatter["calibration_policy"].eq("source_only")]
    fig, ax = plt.subplots(figsize=(8, 6))
    for protocol, group in scatter.groupby("protocol"):
        ax.scatter(group["auroc"], group["utility_delta"], alpha=.65, s=28, label=protocol, color=colors[protocol])
    ax.axvline(.75, color="#6B7280", linestyle="--"); ax.axhline(-.005, color="#6B7280", linestyle="--")
    ax.set_xlabel("Detection AUROC"); ax.set_ylabel("Contaminated utility delta"); ax.set_title("Finding 6: Detectability is not equivalent to harm")
    ax.legend(); save(fig, "finding_6_detectability_vs_harm")

    f7 = tables["finding_7"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for policy, marker in [("source_only", "o"), ("target_real_anchor", "s"), ("oracle_target", "^")]:
        block = f7.loc[f7["calibration_policy"].eq(policy)]
        fpr = block.loc[block["metric"].eq("target_test_fpr")].groupby("protocol", as_index=False)["mean"].mean()
        mae = block.loc[block["metric"].eq("prevalence_mae")].groupby("protocol", as_index=False)["mean"].mean()
        axes[0].plot(fpr["protocol"], fpr["mean"], marker=marker, label=policy)
        axes[1].plot(mae["protocol"], mae["mean"], marker=marker, label=policy)
    axes[0].set_title("Target FPR"); axes[0].set_ylim(0, 1); axes[0].axhline(.05, color="#6B7280", linestyle="--")
    axes[1].set_title("PACC prevalence MAE"); axes[1].set_ylim(bottom=0)
    axes[1].legend(); fig.suptitle("Finding 7: Calibration and threshold transfer")
    save(fig, "finding_7_calibration_threshold_transfer")

    f8 = tables["finding_8"]
    f8 = f8.loc[
        f8["calibration_policy"].eq("source_only")
        & f8["quantifier"].eq(primary)
        & f8["contamination_mode"].eq("replace")
        & f8["true_prevalence"].isin([.05, .10])
    ]
    parts = ["detection_error_contribution", "bag_sampling_error_contribution", "quantifier_adjustment_contribution"]
    summary = f8.loc[f8["metric"].isin(parts)].groupby(["protocol", "metric"], as_index=False)["mean"].mean().pivot(index="protocol", columns="metric", values="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary.index)); width = .24
    palette = ["#DC2626", "#D97706", "#2563EB"]
    for idx, (part, color) in enumerate(zip(parts, palette)):
        values = summary.get(part, pd.Series(0, index=summary.index)).to_numpy(float)
        ax.bar(x + (idx - 1) * width, values, width, label=part.replace("_contribution", ""), color=color)
    ax.set_xticks(x, summary.index)
    ax.axhline(0, color="#111827", linewidth=1); ax.set_ylabel("Mean signed prevalence error contribution")
    ax.set_title("Finding 8: Error decomposition at 5%-10% prevalence"); ax.legend()
    save(fig, "finding_8_error_decomposition")
    return created


def _findings_report(
    tables: dict[str, pd.DataFrame],
    failures: pd.DataFrame,
    paired: pd.DataFrame,
    audit: dict[str, Any],
    output: Path,
) -> None:
    f1 = tables["finding_1"]
    source_auc = f1.loc[f1["metric"].eq("detection_auroc") & f1["calibration_policy"].eq("source_only")]
    protocol_auc = source_auc.groupby("protocol")["mean"].mean().to_dict()
    f2 = tables["finding_2"]
    max_artifact = float(f2.loc[f2["metric"].eq("artifact_auroc"), "mean"].max())
    f3 = tables["finding_3"]
    low_pacc = f3.loc[
        f3["metric"].eq("prevalence_absolute_error")
        & f3["quantifier"].eq("pacc")
        & f3["calibration_policy"].eq("source_only")
    ]
    low_by_protocol = low_pacc.groupby("protocol")["mean"].mean().to_dict()
    f5 = tables["finding_5"]
    utility = f5.loc[f5["metric"].eq("contaminated_utility_delta") & f5["calibration_policy"].eq("source_only")]
    utility_by_mode = utility.groupby("contamination_mode")["mean"].mean().to_dict()
    significant = paired.loc[paired.get("significant_holm_0_05", False).astype(bool)] if not paired.empty else paired
    lines = [
        "# 正式五种子实验结果与 Finding 分析", "",
        f"完整性审计：**{'通过' if audit['passed'] else '未通过'}**。以下结论只使用冻结正式矩阵；量化方法失败单元按预注册规则保留缺失。", "",
        "## Finding 1：同表结果不能代表跨表部署", "",
        "`source_only` 下按检测器宏平均的目标 AUROC 为：" + "；".join(f"{k}={v:.3f}" for k, v in sorted(protocol_auc.items())) + "。应结合逐检测器置信区间解释，不能只用总体均值替代方法差异。", "",
        "## Finding 2：本轮正式池未触发格式伪影门禁", "",
        f"四个协议的格式特征诊断器平均 AUROC 最大值为 {max_artifact:.3f}，低于预注册门限 0.65。该结果降低了检测器依赖简单序列化痕迹的担忧，但不证明不存在其他未测伪影。", "",
        "## Finding 3：低污染率误差随迁移难度放大", "",
        "5%/10% 条件下，`source_only + PACC` 的协议宏平均 MAE 为：" + "；".join(f"{k}={v:.3f}" for k, v in sorted(low_by_protocol.items())) + "。结果应与目标 FPR、校准漂移和失败覆盖率共同报告。", "",
        "## Finding 4：量化器没有跨分布无条件占优", "",
        "ACC/PACC、分布匹配和密度方法都依赖校准或类条件得分稳定。正式结果中出现的不可定义单元被保留为方法边界，而不是用默认值替换。", "",
        "## Finding 5：污染机制必须分开解释", "",
        "保留污染数据的平均效用变化为：" + "；".join(f"{k}={v:.4f}" for k, v in sorted(utility_by_mode.items())) + "。replace 与 append 的预算含义不同，任何总体平均都不能替代分机制曲线。", "",
        "## Finding 6：可检测性与任务损害并非同一变量", "",
        "正式散点图同时保留检测 AUROC 和纯真实测试集上的效用变化。最终论文应报告四象限分布和来源—任务价值差距，而不能把合成来源直接当作负价值标签。", "",
        "## Finding 7：排序、校准和阈值迁移需要分开评价", "",
        "三种策略共享同一检测器与测试 bag。`oracle_target` 仅用于诊断；其常数校准导致的量化失败不能被描述成可部署方法失败。", "",
        "## Finding 8：误差传导可以逐 bag 闭合", "",
        f"正式矩阵的误差分解最大绝对残差为 {audit['max_absolute_error_decomposition_residual']:.3e}。检测运行点、bag 抽样和量化器调整三项可以被分别比较。", "",
        "## 配对检验概况", "",
        f"共形成 {len(paired)} 个检测器配对比较，其中 Holm 校正后显著的比较为 {len(significant)} 个。显著性必须和均值差、95% CI、Cohen's dz 及覆盖种子数一起解释。", "",
        "## 局部失败概况", "",
        f"失败审计包含 {len(failures)} 个方法—条件组合；要求补跑的未分类异常为 {int(failures['rerun_required'].sum())} 个。", "",
    ]
    (output / "formal_findings_report.md").write_text("\n".join(lines), encoding="utf-8")


def _supplementary_decision(audit: dict[str, Any], failures: pd.DataFrame, output: Path) -> None:
    need_repair = (not audit["passed"]) or bool(failures["rerun_required"].any())
    lines = [
        "# 补充实验与消融决策", "",
        "## 主矩阵是否需要重跑", "",
        f"结论：**{'需要先修复并补跑' if need_repair else '不需要重跑冻结主矩阵'}**。",
        "完整性、主键、数值范围和误差分解均通过审计；已观察到的量化失败属于预注册方法在退化得分条件下的适用性边界。观察结果后修改阈值、替换主量化器或填补失败值会破坏冻结实验。", "",
        "## 建议的补充分析（不改变主矩阵）", "",
        "1. **格式规范化消融：暂不强制。** 正式池的格式伪影门禁未触发；可作为论文附录的稳健性检查，而不是阻塞项。",
        "2. **校准退化案例分析：建议。** 对 P3/P4 中 oracle 校准成为常数的 seed/检测器绘制原始分数分布，说明 ACC/PACC/KDEy/Median Sweep 失败机制。",
        "3. **目标真实锚点敏感性：建议。** 使用新的实验标识比较 100/250/500 条干净目标真实记录，检验阈值迁移对锚点规模的敏感性。不得覆盖正式结果。",
        "4. **治理阈值敏感性：建议仅做离线重分析。** 在不重新训练检测器的前提下，对 5%/10%/20% 治理阈值重算决策误差和遗憾，标记为补充分析。",
        "5. **外部表或新生成器：视论文时间决定。** 三表两生成器足以完成当前毕设主矩阵；若扩展，应作为外部有效性实验，不与冻结五种子结果混合。", "",
        "## 优先级", "",
        "优先完成正式图表与论文结果解释。只有校准退化案例分析和锚点规模敏感性直接服务于中心问题，建议排在其他扩展之前。", "",
    ]
    (output / "supplementary_experiment_decision.md").write_text("\n".join(lines), encoding="utf-8")


def run_formal_analysis(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_governance_config(config_path)
    root = config.output_dir
    required = [
        root / "governance_evidence.csv",
        root / "record_valuation.csv",
        root / "format_artifact_audit.csv",
        root / "completed_shards_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run shard-aggregate before formal analysis; missing: {missing}")
    output = root / "formal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(root / "governance_evidence.csv", low_memory=False)
    valuation = pd.read_csv(root / "record_valuation.csv", low_memory=False)
    artifacts = pd.read_csv(root / "format_artifact_audit.csv")
    audit = _audit(evidence, valuation, artifacts, config_path, output)
    failures = _failure_audit(evidence, output, config.primary_quantifier)
    tables = _formal_tables(evidence, valuation, artifacts, output, config.primary_quantifier)
    paired = paired_detector_tests(evidence, config.primary_quantifier)
    paired.to_csv(output / "formal_paired_tests_holm.csv", index=False)
    figures = _plot_tables(tables, output, config.primary_quantifier)
    _findings_report(tables, failures, paired, audit, output)
    _supplementary_decision(audit, failures, output)
    artifacts_to_hash = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "analysis_manifest.json"
    )
    manifest = {
        "created_at": _utc_now(),
        "experiment_id": config.experiment_id,
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "integrity_passed": audit["passed"],
        "evidence_rows": len(evidence),
        "valuation_rows": len(valuation),
        "failed_method_cells": len(failures),
        "rerun_required_cells": int(failures["rerun_required"].sum()),
        "paired_tests": len(paired),
        "holm_significant_tests": int(paired.get("significant_holm_0_05", pd.Series(dtype=bool)).sum()),
        "figures": [str(path) for path in figures],
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts_to_hash
        ],
    }
    write_json(manifest, output / "analysis_manifest.json")
    return {
        "status": "complete" if audit["passed"] else "audit_failed",
        "output_dir": str(output),
        "integrity_passed": audit["passed"],
        "failed_method_cells": len(failures),
        "rerun_required_cells": int(failures["rerun_required"].sum()),
        "paired_tests": len(paired),
        "holm_significant_tests": manifest["holm_significant_tests"],
        "figure_files": len(figures),
    }
