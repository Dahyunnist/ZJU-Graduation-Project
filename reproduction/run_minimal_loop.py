"""Run the week-0 minimal loop for synthetic tabular data detection.

This script follows the project note:
1. load the UCI Adult dataset;
2. train an SDV CTGAN synthesizer;
3. sample 1000 synthetic rows;
4. plot feature distributions;
5. train a Logistic Regression detector for real vs synthetic rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week-0 CTGAN minimal loop")
    parser.add_argument("--train-rows", type=int, default=3000, help="Rows used to train CTGAN")
    parser.add_argument("--sample-rows", type=int, default=1000, help="Synthetic rows to generate")
    parser.add_argument("--epochs", type=int, default=20, help="CTGAN training epochs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/week0"))
    parser.add_argument("--cuda", action="store_true", help="Use GPU for CTGAN if available")
    return parser.parse_args()


def load_adult(max_rows: int) -> pd.DataFrame:
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    real_data = adult.data.copy()
    real_data["income"] = adult.target

    # Adult uses '?' for missing categories. Normalize them before metadata detection.
    real_data = real_data.replace("?", pd.NA)
    category_columns = real_data.select_dtypes(include=["category"]).columns
    real_data[category_columns] = real_data[category_columns].astype("object")
    real_data = real_data.sample(n=min(max_rows, len(real_data)), random_state=RANDOM_STATE)
    return real_data.reset_index(drop=True)


def train_ctgan(real_data: pd.DataFrame, epochs: int, cuda: bool) -> CTGANSynthesizer:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        cuda=cuda,
        verbose=True,
    )
    synthesizer.fit(real_data)
    return synthesizer


def plot_distributions(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("age", "numeric"),
        ("hours-per-week", "numeric"),
        ("education", "categorical"),
        ("income", "categorical"),
    ]

    for column, kind in plot_specs:
        if column not in real_data.columns or column not in synthetic_data.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        if kind == "numeric":
            sns.histplot(real_data[column], label="real", stat="density", bins=30, alpha=0.45, ax=ax)
            sns.histplot(synthetic_data[column], label="synthetic", stat="density", bins=30, alpha=0.45, ax=ax)
        else:
            compare = pd.concat(
                [
                    real_data[[column]].assign(source="real"),
                    synthetic_data[[column]].assign(source="synthetic"),
                ],
                ignore_index=True,
            )
            order = real_data[column].value_counts(dropna=False).head(12).index
            sns.countplot(data=compare, y=column, hue="source", order=order, ax=ax)

        ax.set_title(f"Real vs Synthetic Distribution: {column}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"distribution_{column}.png", dpi=160)
        plt.close(fig)


def build_detector(data: pd.DataFrame) -> Pipeline:
    categorical_columns = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in data.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE),
            ),
        ]
    )


def run_detector(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float | str]:
    n = min(len(real_data), len(synthetic_data))
    detector_data = pd.concat(
        [
            real_data.sample(n=n, random_state=RANDOM_STATE).assign(is_synthetic=0),
            synthetic_data.sample(n=n, random_state=RANDOM_STATE).assign(is_synthetic=1),
        ],
        ignore_index=True,
    )

    y = detector_data.pop("is_synthetic")
    x_train, x_test, y_train, y_test = train_test_split(
        detector_data,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    detector = build_detector(detector_data)
    detector.fit(x_train, y_train)

    probabilities = detector.predict_proba(x_test)[:, 1]
    predictions = detector.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "report": classification_report(y_test, predictions, digits=4),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading UCI Adult dataset...")
    real_data = load_adult(args.train_rows)
    real_path = args.output_dir / "adult_real_sample.csv"
    real_data.to_csv(real_path, index=False)
    print(f"Saved real sample: {real_path}")

    print("[2/5] Training SDV CTGAN synthesizer...")
    synthesizer = train_ctgan(real_data, epochs=args.epochs, cuda=args.cuda)

    print(f"[3/5] Sampling {args.sample_rows} synthetic rows...")
    synthetic_data = synthesizer.sample(args.sample_rows)
    synthetic_path = args.output_dir / "adult_synthetic_ctgan.csv"
    synthetic_data.to_csv(synthetic_path, index=False)
    print(f"Saved synthetic data: {synthetic_path}")

    print("[4/5] Plotting feature distributions...")
    plot_distributions(real_data, synthetic_data, args.output_dir)
    print(f"Saved plots under: {args.output_dir}")

    print("[5/5] Training real-vs-synthetic Logistic Regression detector...")
    metrics = run_detector(real_data, synthetic_data)
    report_path = args.output_dir / "detector_report.txt"
    report_path.write_text(
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"ROC-AUC: {metrics['roc_auc']:.4f}\n\n"
        f"{metrics['report']}",
        encoding="utf-8",
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved detector report: {report_path}")


if __name__ == "__main__":
    main()
