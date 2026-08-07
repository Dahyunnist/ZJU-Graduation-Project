"""Reproduce the phase-1 Adult baseline.

The baseline combines:
1. SDMetrics statistical quality metrics for real vs synthetic tables;
2. a scikit-learn Logistic Regression detector that reports AUROC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

from run_minimal_loop import RANDOM_STATE, load_adult, run_detector, train_ctgan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adult SDMetrics + sklearn baseline")
    parser.add_argument("--train-rows", type=int, default=3000, help="Rows used to fit CTGAN")
    parser.add_argument("--sample-rows", type=int, default=1000, help="Synthetic rows to evaluate")
    parser.add_argument("--epochs", type=int, default=20, help="CTGAN training epochs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adult_baseline"))
    parser.add_argument("--cuda", action="store_true", help="Use GPU for CTGAN if available")
    parser.add_argument("--real-csv", type=Path, help="Reuse an existing real Adult CSV")
    parser.add_argument("--synthetic-csv", type=Path, help="Reuse an existing synthetic Adult CSV")
    return parser.parse_args()


def build_metadata(real_data: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    return metadata


def load_or_generate_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bool(args.real_csv) != bool(args.synthetic_csv):
        raise ValueError("--real-csv and --synthetic-csv must be provided together.")

    if args.real_csv and args.synthetic_csv:
        real_data = pd.read_csv(args.real_csv)
        synthetic_data = pd.read_csv(args.synthetic_csv)
        return real_data, synthetic_data

    real_data = load_adult(args.train_rows)
    synthesizer = train_ctgan(real_data, epochs=args.epochs, cuda=args.cuda)
    synthetic_data = synthesizer.sample(args.sample_rows)
    return real_data, synthetic_data


def run_sdmetrics(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    metadata = build_metadata(real_data)
    metadata_dict = metadata.to_dict()

    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata_dict, verbose=True)
    report.save(output_dir / "sdmetrics_quality_report.pkl")

    properties = report.get_properties()
    properties.to_csv(output_dir / "sdmetrics_properties.csv", index=False)

    detail_files: dict[str, str] = {}
    for property_name in properties["Property"]:
        details = report.get_details(property_name)
        safe_name = property_name.lower().replace(" ", "_")
        detail_path = output_dir / f"sdmetrics_details_{safe_name}.csv"
        details.to_csv(detail_path, index=False)
        detail_files[property_name] = str(detail_path)

    return {
        "quality_score": float(report.get_score()),
        "properties": properties.to_dict(orient="records"),
        "detail_files": detail_files,
        "metadata": metadata_dict,
    }


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sdmetrics_result: dict[str, Any],
    detector_result: dict[str, float | str],
) -> None:
    summary = {
        "dataset": "UCI Adult",
        "random_state": RANDOM_STATE,
        "generator": "CTGAN",
        "real_rows": len(real_data),
        "synthetic_rows": len(synthetic_data),
        "epochs": args.epochs,
        "classifier": "LogisticRegression",
        "sdmetrics_quality_score": sdmetrics_result["quality_score"],
        "sdmetrics_properties": sdmetrics_result["properties"],
        "classifier_accuracy": detector_result["accuracy"],
        "classifier_auroc": detector_result["roc_auc"],
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    properties_md = "\n".join(
        f"- {item['Property']}: {item['Score']:.4f}" for item in sdmetrics_result["properties"]
    )
    report_md = (
        "# Adult Baseline Result\n\n"
        "## Setup\n\n"
        f"- Dataset: UCI Adult\n"
        f"- Generator: CTGAN, epochs={args.epochs}\n"
        f"- Real rows evaluated: {len(real_data)}\n"
        f"- Synthetic rows evaluated: {len(synthetic_data)}\n"
        f"- Classifier: scikit-learn LogisticRegression\n\n"
        "## SDMetrics\n\n"
        f"- Overall quality score: {sdmetrics_result['quality_score']:.4f}\n"
        f"{properties_md}\n\n"
        "## Classifier Detector\n\n"
        f"- Accuracy: {detector_result['accuracy']:.4f}\n"
        f"- AUROC: {detector_result['roc_auc']:.4f}\n\n"
        "## Classification Report\n\n"
        "```text\n"
        f"{detector_result['report']}"
        "```\n"
    )
    (output_dir / "baseline_summary.md").write_text(report_md, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading or generating Adult real/synthetic data...")
    real_data, synthetic_data = load_or_generate_data(args)
    real_path = args.output_dir / "adult_real.csv"
    synthetic_path = args.output_dir / "adult_synthetic_ctgan.csv"
    real_data.to_csv(real_path, index=False)
    synthetic_data.to_csv(synthetic_path, index=False)
    print(f"Saved real data: {real_path}")
    print(f"Saved synthetic data: {synthetic_path}")

    print("[2/4] Running SDMetrics statistical quality report...")
    sdmetrics_result = run_sdmetrics(real_data, synthetic_data, args.output_dir)

    print("[3/4] Training scikit-learn real-vs-synthetic detector...")
    detector_result = run_detector(real_data, synthetic_data)
    (args.output_dir / "classifier_report.txt").write_text(
        f"Accuracy: {detector_result['accuracy']:.4f}\n"
        f"AUROC: {detector_result['roc_auc']:.4f}\n\n"
        f"{detector_result['report']}",
        encoding="utf-8",
    )

    print("[4/4] Writing baseline summary...")
    write_summary(args.output_dir, args, real_data, synthetic_data, sdmetrics_result, detector_result)

    print(f"SDMetrics quality score: {sdmetrics_result['quality_score']:.4f}")
    print(f"Classifier AUROC: {detector_result['roc_auc']:.4f}")
    print(f"Saved baseline outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
