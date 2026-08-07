"""Generator contracts, deterministic seeds and access auditing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


PROVENANCE_COLUMNS = {
    "row_id",
    "split",
    "synth_row_id",
    "dataset_id",
    "generator_name",
    "generator_seed",
    "sample_seed",
    "pool_name",
    "mix_row_id",
    "source_type",
    "source_row_id",
    "condition",
    "p",
    "mix_seed",
}


class GeneratorError(RuntimeError):
    pass


class TabularGenerator(Protocol):
    def fit(self, real_records: pd.DataFrame, metadata: Any, seed: int, **config: Any) -> None: ...
    def sample(self, n: int, sample_seed: int, pool_name: str) -> pd.DataFrame: ...
    def save(self, path: str) -> None: ...
    def get_provenance(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AccessAudit:
    allowed_partition: str
    actual_partitions: tuple[str, ...]
    row_count: int
    row_ids_sha256: str
    passed: bool


def stable_ids_hash(row_ids: list[str]) -> str:
    payload = "\n".join(sorted(row_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_generator_access(records: pd.DataFrame, allowed_ids: set[str]) -> AccessAudit:
    if "row_id" not in records or "split" not in records:
        raise GeneratorError("Generator input must include row_id and split for access auditing")
    actual_ids = records["row_id"].astype(str).tolist()
    forbidden = sorted(set(actual_ids) - allowed_ids)
    partitions = tuple(sorted(set(records["split"].astype(str))))
    passed = not forbidden and partitions == ("R_source_train",)
    if not passed:
        raise GeneratorError(
            f"Generator access violation: partitions={partitions}, forbidden_row_ids={forbidden[:5]}"
        )
    return AccessAudit(
        allowed_partition="R_source_train",
        actual_partitions=partitions,
        row_count=len(records),
        row_ids_sha256=stable_ids_hash(actual_ids),
        passed=True,
    )


def model_frame(records: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in records.columns if column not in PROVENANCE_COLUMNS]
    result = records.loc[:, columns].copy()
    leaked = set(result.columns) & PROVENANCE_COLUMNS
    if leaked:
        raise GeneratorError(f"Provenance fields leaked into model frame: {sorted(leaked)}")
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def derive_pool_seeds(generator_seed: int, offsets: dict[str, int]) -> dict[str, int]:
    values = {name: int(generator_seed + offset) for name, offset in offsets.items()}
    if len(values) != len(set(values.values())):
        raise GeneratorError("Derived pool sample seeds are not unique")
    return values


def stable_synth_ids(
    frame: pd.DataFrame,
    dataset_id: str,
    generator_name: str,
    generator_seed: int,
    sample_seed: int,
    pool_name: str,
) -> list[str]:
    ids = []
    for ordinal, values in enumerate(frame.itertuples(index=False, name=None)):
        body = json.dumps([str(value) for value in values], ensure_ascii=False, separators=(",", ":"))
        payload = f"{dataset_id}|{generator_name}|{generator_seed}|{sample_seed}|{pool_name}|{ordinal}|{body}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        ids.append(f"syn:{dataset_id}:{generator_name}:{pool_name}:{digest}")
    if len(ids) != len(set(ids)):
        raise GeneratorError("Synthetic row ID collision")
    return ids

