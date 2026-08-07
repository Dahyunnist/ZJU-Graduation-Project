"""Data sources for smoke fixtures and registry-backed formal experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


META_COLUMNS = {"record_id", "table_id", "generator", "source_label", "source_type"}


@dataclass(frozen=True)
class TableData:
    table_id: str
    domain: str
    target_column: str
    real: pd.DataFrame
    synthetic: dict[str, pd.DataFrame]


class GovernanceDataSource(Protocol):
    @property
    def table_ids(self) -> tuple[str, ...]: ...
    @property
    def generators(self) -> tuple[str, ...]: ...
    def table(self, table_id: str) -> TableData: ...


def _labelled(frame: pd.DataFrame, table_id: str, source: str, generator: str = "real") -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    native_schema = "\x1f".join(str(column) for column in result.columns)
    result["record_id"] = [f"{table_id}:{source}:{i:06d}" for i in range(len(result))]
    result["table_id"] = table_id
    result["generator"] = generator
    result["source_label"] = int(source == "synthetic")
    result["source_type"] = source
    result["schema_columns"] = native_schema
    return result


def _make_real_table(table_id: str, n: int, seed: int) -> tuple[pd.DataFrame, str, str]:
    rng = np.random.default_rng(seed)
    latent_a = rng.normal(size=n)
    latent_b = rng.normal(size=n)
    category = rng.choice(["alpha", "beta", "gamma"], size=n, p=[.45, .35, .20])
    category_signal = np.select([category == "alpha", category == "beta"], [.45, -.25], default=.1)
    logits = 1.15 * latent_a - .75 * latent_b + category_signal + rng.normal(0, .55, n)
    target = (logits > np.median(logits)).astype(int)
    if table_id == "table_a":
        frame = pd.DataFrame({
            "age": np.round(42 + 11 * latent_a, 3),
            "income": np.round(np.exp(10.2 + .35 * latent_b), 2),
            "segment": category,
            "target": target,
        })
        return frame, "target", "social"
    if table_id == "table_b":
        frame = pd.DataFrame({
            "duration_months": np.maximum(1, np.round(36 + 9 * latent_a)).astype(int),
            "balance_ratio": np.round(1 / (1 + np.exp(-latent_b)), 6),
            "channel": np.where(category == "alpha", "web", np.where(category == "beta", "branch", "partner")),
            "risk_band": pd.cut(latent_a + latent_b, [-np.inf, -.5, .7, np.inf], labels=["low", "mid", "high"]).astype(str),
            "defaulted": target,
        })
        return frame, "defaulted", "finance"
    frame = pd.DataFrame({
        "temperature": np.round(18 + 4.5 * latent_a, 4),
        "pressure": np.round(100 + 7 * latent_b, 4),
        "site_code": np.where(category == "alpha", "N", np.where(category == "beta", "S", "W")),
        "alarm": target,
    })
    return frame, "alarm", "industrial"


def _synthesize(real: pd.DataFrame, generator: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(real)
    if generator == "marginal":
        # Preserve univariate marginals while breaking cross-column dependence.
        return pd.DataFrame({column: rng.permutation(real[column].to_numpy()) for column in real.columns})
    sampled = real.iloc[rng.integers(0, n, size=n)].reset_index(drop=True).copy()
    numeric = sampled.select_dtypes(include=["number", "bool"]).columns
    if generator == "noisy":
        for column in numeric:
            # Preserve discrete labels/codes; perturb only genuinely continuous
            # attributes so the downstream task remains well-defined.
            if real[column].nunique(dropna=True) <= 10:
                continue
            scale = max(float(pd.to_numeric(real[column]).std()), 1e-6)
            sampled[column] = pd.to_numeric(sampled[column]) + rng.normal(0, .18 * scale, n)
        return sampled
    if generator == "rounded":
        for column in numeric:
            if sampled[column].nunique() > 2:
                sampled[column] = pd.to_numeric(sampled[column]).round(1)
        # Deliberately retain a mild format trace for the artifact gate to detect.
        for column in sampled.select_dtypes(exclude=["number", "bool"]).columns:
            sampled[column] = sampled[column].astype(str).str.strip()
        return sampled
    raise ValueError(f"Unknown fixture generator: {generator}")


class SyntheticFixtureSource:
    """Deterministic multi-table fixture that exercises P1-P4 end to end."""

    def __init__(self, rows_per_table: int, seed: int):
        self._tables: dict[str, TableData] = {}
        generators = ("marginal", "noisy", "rounded")
        for offset, table_id in enumerate(("table_a", "table_b", "table_c")):
            raw, target, domain = _make_real_table(table_id, rows_per_table, seed + offset * 101)
            real = _labelled(raw, table_id, "real")
            synthetic = {
                generator: _labelled(
                    _synthesize(raw, generator, seed + offset * 101 + (index + 1) * 1009),
                    table_id, "synthetic", generator,
                )
                for index, generator in enumerate(generators)
            }
            self._tables[table_id] = TableData(table_id, domain, target, real, synthetic)

    @property
    def table_ids(self) -> tuple[str, ...]:
        return tuple(self._tables)

    @property
    def generators(self) -> tuple[str, ...]:
        return ("marginal", "noisy", "rounded")

    def table(self, table_id: str) -> TableData:
        return self._tables[table_id]


class RegistrySource:
    """Load formal real/synthetic pools from a compact CSV registry.

    Required columns: table_id, domain, target_column, generator, real_path,
    synthetic_path. Relative paths are resolved against the registry file.
    """

    REQUIRED = {"table_id", "domain", "target_column", "generator", "real_path", "synthetic_path"}

    def __init__(self, registry_path: str | Path):
        path = Path(registry_path).resolve()
        registry = pd.read_csv(path)
        missing = sorted(self.REQUIRED - set(registry.columns))
        if missing:
            raise ValueError(f"Pool registry missing columns: {missing}")
        if registry.duplicated(["table_id", "generator"]).any():
            raise ValueError("Pool registry contains duplicate table_id/generator rows")
        self._tables: dict[str, TableData] = {}
        for table_id, group in registry.groupby("table_id", sort=True):
            targets = group["target_column"].astype(str).unique()
            domains = group["domain"].astype(str).unique()
            real_paths = group["real_path"].astype(str).unique()
            if len(targets) != 1 or len(domains) != 1 or len(real_paths) != 1:
                raise ValueError(f"Inconsistent registry metadata for table {table_id}")
            real_path = (path.parent / real_paths[0]).resolve()
            if not real_path.is_file():
                raise FileNotFoundError(real_path)
            real_raw = pd.read_csv(real_path)
            target = targets[0]
            if target not in real_raw:
                raise ValueError(f"Target {target!r} absent from {real_path}")
            synthetic: dict[str, pd.DataFrame] = {}
            for row in group.itertuples(index=False):
                synth_path = (path.parent / str(row.synthetic_path)).resolve()
                if not synth_path.is_file():
                    raise FileNotFoundError(synth_path)
                synth_raw = pd.read_csv(synth_path)
                if set(synth_raw.columns) != set(real_raw.columns):
                    raise ValueError(f"Schema mismatch: {synth_path}")
                synthetic[str(row.generator)] = _labelled(synth_raw[real_raw.columns], str(table_id), "synthetic", str(row.generator))
            self._tables[str(table_id)] = TableData(
                str(table_id), domains[0], target,
                _labelled(real_raw, str(table_id), "real"), synthetic,
            )

    @property
    def table_ids(self) -> tuple[str, ...]:
        return tuple(self._tables)

    @property
    def generators(self) -> tuple[str, ...]:
        return tuple(sorted({g for table in self._tables.values() for g in table.synthetic}))

    def table(self, table_id: str) -> TableData:
        return self._tables[table_id]


def sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    replace = n > len(frame)
    indices = rng.choice(len(frame), size=n, replace=replace)
    return frame.iloc[indices].reset_index(drop=True).copy()


def exact_mixture(real: pd.DataFrame, synthetic: pd.DataFrame, size: int, prevalence: float, seed: int) -> pd.DataFrame:
    synthetic_count = int(round(size * prevalence))
    real_count = size - synthetic_count
    result = pd.concat([
        sample_rows(real, real_count, seed + 17),
        sample_rows(synthetic, synthetic_count, seed + 29),
    ], ignore_index=True)
    return result.sample(frac=1, random_state=seed + 41).reset_index(drop=True)
