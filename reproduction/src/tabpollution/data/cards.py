"""Human- and machine-readable dataset cards."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from tabpollution.data.cleaning import CleaningReport
from tabpollution.data.registry import DatasetSpec
from tabpollution.data.splits import SplitSummary
from tabpollution.utils import write_json


def build_card_payload(
    spec: DatasetSpec,
    frame: pd.DataFrame,
    cleaning: CleaningReport,
    raw_sha256: str,
    processed_sha256: str,
    retrieval_date: str,
    splits: list[SplitSummary],
) -> dict[str, Any]:
    target_counts = frame[spec.target_column].astype(str).value_counts(dropna=False)
    target_distribution = {
        str(key): {"count": int(value), "proportion": float(value / len(frame))}
        for key, value in target_counts.items()
    }
    missing = {
        column: {"count": int(frame[column].isna().sum()), "rate": float(frame[column].isna().mean())}
        for column in spec.feature_columns
    }
    return {
        "dataset_id": spec.dataset_id,
        "canonical_name": spec.canonical_name,
        "source_page": spec.source_page,
        "download_url": spec.download_url,
        "uci_id": spec.uci_id,
        "doi": spec.doi,
        "license": spec.license,
        "retrieval_date": retrieval_date,
        "raw_sha256": raw_sha256,
        "processed_sha256": processed_sha256,
        "task_type": spec.task_type,
        "target_column": spec.target_column,
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "cleaning": asdict(cleaning),
        "feature_count": len(spec.feature_columns),
        "target_distribution": target_distribution,
        "missing": missing,
        "split_summaries": [asdict(summary) for summary in splits],
        "known_risks": [
            "Preprocessors must be fit only on the permitted training partition.",
            "row_id, split and provenance fields are metadata and must never become model features.",
            "Categorical codes in Credit carry semantics and are not continuous measurements.",
        ],
    }


def _markdown_card(payload: dict[str, Any]) -> str:
    cleaning = payload["cleaning"]
    lines = [
        f"# 数据卡：{payload['canonical_name']}",
        "",
        "## 来源与许可",
        "",
        f"- 数据集ID：`{payload['dataset_id']}`",
        f"- UCI ID：{payload['uci_id']}；DOI：{payload['doi']}",
        f"- 来源页面：{payload['source_page']}",
        f"- 许可：{payload['license']}",
        f"- 获取日期：{payload['retrieval_date']}",
        f"- 原始文件 SHA-256：`{payload['raw_sha256']}`",
        f"- 处理文件 SHA-256：`{payload['processed_sha256']}`",
        "",
        "## 数据概况",
        "",
        f"- 任务：{payload['task_type']}；目标列：`{payload['target_column']}`",
        f"- 原始行数：{cleaning['raw_rows']}；去除完全重复后：{cleaning['cleaned_rows']}",
        f"- 完全重复行移除数：{cleaning['exact_duplicate_rows_removed']}",
        f"- 特征数：{payload['feature_count']}（数值 {len(payload['numeric_columns'])}，类别 {len(payload['categorical_columns'])}）",
        f"- 数值列：{', '.join(payload['numeric_columns'])}",
        f"- 类别列：{', '.join(payload['categorical_columns'])}",
        "",
        "### 目标分布",
        "",
        "|取值|数量|比例|",
        "|---|---:|---:|",
    ]
    for key, value in payload["target_distribution"].items():
        lines.append(f"|{key}|{value['count']}|{value['proportion']:.6f}|")
    lines.extend(["", "### 缺失情况", "", "|列|缺失数|缺失率|", "|---|---:|---:|"])
    for column, value in payload["missing"].items():
        lines.append(f"|{column}|{value['count']}|{value['rate']:.6f}|")
    lines.extend(
        [
            "",
            "## 冻结分区",
            "",
            "采用顺序分层划分：先留出15%最终测试，再从其余85%留出10%验证，最后将75%拆为60%来源训练和15%检测训练。",
            "",
            "|seed|R_source_train|R_detector_train|R_detector_val|R_final_test|assignment SHA-256|",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in payload["split_summaries"]:
        counts = summary["counts"]
        lines.append(
            f"|{summary['seed']}|{counts['R_source_train']}|{counts['R_detector_train']}|"
            f"{counts['R_detector_val']}|{counts['R_final_test']}|`{summary['assignment_sha256']}`|"
        )
    lines.extend(["", "### 各分区目标分布", ""])
    for summary in payload["split_summaries"]:
        lines.append(f"- seed={summary['seed']}")
        for split_name, distribution in summary["target_distribution"].items():
            rendered = ", ".join(f"{label}={rate:.6f}" for label, rate in distribution.items())
            lines.append(f"  - `{split_name}`：{rendered}")
    lines.extend(["", "## 已知风险", ""])
    lines.extend(f"- {item}" for item in payload["known_risks"])
    return "\n".join(lines) + "\n"


def write_dataset_card(payload: dict[str, Any], report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dataset_card_{payload['dataset_id']}"
    write_json(payload, report_dir / f"{stem}.json")
    (report_dir / f"{stem}.md").write_text(_markdown_card(payload), encoding="utf-8", newline="\n")
