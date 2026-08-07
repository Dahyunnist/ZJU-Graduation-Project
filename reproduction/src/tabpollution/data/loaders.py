"""Official-source download and parsing for Track A datasets."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from tabpollution.data.registry import DatasetSpec, validate_dataframe_schema


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def ensure_raw_file(spec: DatasetSpec, raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / spec.raw_filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    request = urllib.request.Request(
        spec.download_url,
        headers={"User-Agent": "tabpollution-benchmark/0.1 (academic dataset preparation)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    return destination


def _read_adult(raw_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(raw_path) as archive:
        for member in ("adult.data", "adult.test"):
            with archive.open(member) as stream:
                frame = pd.read_csv(
                    stream,
                    names=ADULT_COLUMNS,
                    header=None,
                    skipinitialspace=True,
                    comment="|",
                    na_values=["?"],
                )
                frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["income"] = data["income"].astype("string").str.strip().str.rstrip(".")
    return data


def _read_credit(raw_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".xls")]
        if len(members) != 1:
            raise ValueError(f"Expected one XLS member in {raw_path}, found {members}")
        payload = archive.read(members[0])
    frame = pd.read_excel(io.BytesIO(payload), header=1, engine="xlrd")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(
        columns={
            "default payment next month": "default_payment_next_month",
            "default.payment.next.month": "default_payment_next_month",
        }
    )
    if "ID" in frame.columns:
        frame = frame.drop(columns=["ID"])
    return frame


def load_raw_dataset(spec: DatasetSpec, raw_path: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_path)
    if spec.dataset_id == "adult":
        frame = _read_adult(raw_path)
    elif spec.dataset_id == "credit":
        frame = _read_credit(raw_path)
    else:
        raise ValueError(f"No loader is registered for {spec.dataset_id}")
    validate_dataframe_schema(frame.columns.tolist(), spec)
    return frame.loc[:, [*spec.feature_columns, spec.target_column]].copy()


def load_processed_dataset(spec: DatasetSpec, path: str | Path) -> pd.DataFrame:
    """Load a canonical CSV without allowing pandas to reinterpret category codes as numbers."""
    dtype = {"row_id": "string", spec.target_column: "string"}
    dtype.update({column: "string" for column in spec.categorical_columns})
    frame = pd.read_csv(path, dtype=dtype)
    validate_dataframe_schema(frame.columns.tolist(), spec)
    for column in spec.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.loc[:, ["row_id", *spec.feature_columns, spec.target_column]].copy()
