"""Deterministic contamination conditions with explicit provenance."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pandas as pd


MIX_METADATA_COLUMNS = {
    "mix_row_id",
    "source_type",
    "source_row_id",
    "source_synth_row_id",
    "generator",
    "pool",
    "p",
    "mix_seed",
    "condition",
}
CONDITIONS = ("real_only", "real_append", "synthetic_append", "synthetic_replace")


def half_up_count(n: int, proportion: float) -> int:
    if n < 0 or proportion < 0 or proportion > 1:
        raise ValueError("n must be non-negative and proportion must be in [0,1]")
    return int(math.floor(proportion * n + 0.5))


def _with_mix_metadata(
    frame: pd.DataFrame,
    condition: str,
    proportion: float,
    mix_seed: int,
    source_type: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["source_type"] = source_type
    result["source_row_id"] = result.get("row_id", pd.Series([pd.NA] * len(result), index=result.index))
    result["source_synth_row_id"] = result.get(
        "synth_row_id", pd.Series([pd.NA] * len(result), index=result.index)
    )
    result["generator"] = result.get(
        "generator_name", pd.Series([pd.NA] * len(result), index=result.index)
    )
    result["pool"] = result.get("pool_name", pd.Series([pd.NA] * len(result), index=result.index))
    result["p"] = proportion
    result["mix_seed"] = mix_seed
    result["condition"] = condition
    ids = []
    for ordinal, row in enumerate(result.itertuples(index=False)):
        source_id = getattr(row, "row_id", None) or getattr(row, "synth_row_id", None) or ordinal
        digest = hashlib.sha256(
            f"{condition}|{proportion:.8f}|{mix_seed}|{source_type}|{source_id}|{ordinal}".encode()
        ).hexdigest()[:24]
        ids.append(f"mix:{digest}")
    result.insert(0, "mix_row_id", ids)
    return result


def build_contamination(
    real_base: pd.DataFrame,
    synthetic_pool: pd.DataFrame,
    condition: str,
    proportion: float,
    mix_seed: int,
    real_extra_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown contamination condition: {condition}")
    n = len(real_base)
    count = half_up_count(n, proportion)
    real = real_base.sample(frac=1, random_state=mix_seed).reset_index(drop=True)
    synthetic = synthetic_pool.sample(n=count, replace=count > len(synthetic_pool), random_state=mix_seed + 1)
    synthetic = synthetic.reset_index(drop=True)

    if condition == "real_only":
        result = _with_mix_metadata(real, condition, proportion, mix_seed, "real")
    elif condition == "real_append":
        if real_extra_pool is None:
            raise ValueError("real_append requires an explicit real_extra_pool")
        extra = real_extra_pool.sample(n=count, replace=count > len(real_extra_pool), random_state=mix_seed + 2)
        real_part = _with_mix_metadata(
            real, "real_append_bootstrap_control", proportion, mix_seed, "real"
        )
        if count == 0:
            result = real_part
        else:
            result = pd.concat(
                [
                    real_part,
                    _with_mix_metadata(
                        extra.reset_index(drop=True),
                        "real_append_bootstrap_control",
                        proportion,
                        mix_seed,
                        "real_bootstrap",
                    ),
                ],
                ignore_index=True,
            )
    elif condition == "synthetic_append":
        real_part = _with_mix_metadata(real, condition, proportion, mix_seed, "real")
        result = (
            real_part
            if count == 0
            else pd.concat(
                [real_part, _with_mix_metadata(synthetic, condition, proportion, mix_seed, "synthetic")],
                ignore_index=True,
            )
        )
    else:
        kept = real.iloc[count:].reset_index(drop=True)
        kept_part = _with_mix_metadata(kept, condition, proportion, mix_seed, "real")
        result = (
            kept_part
            if count == 0
            else pd.concat(
                [kept_part, _with_mix_metadata(synthetic, condition, proportion, mix_seed, "synthetic")],
                ignore_index=True,
            )
        )
    return result.reset_index(drop=True)


def contamination_summary(frame: pd.DataFrame, base_n: int) -> dict[str, Any]:
    synthetic_count = int((frame["source_type"] == "synthetic").sum())
    return {
        "rows": len(frame),
        "base_n": base_n,
        "real_count": int(frame["source_type"].isin(["real", "real_bootstrap"]).sum()),
        "synthetic_count": synthetic_count,
        "actual_synthetic_proportion": synthetic_count / len(frame) if len(frame) else 0.0,
        "condition": str(frame["condition"].iloc[0]) if len(frame) else None,
        "p_requested": float(frame["p"].iloc[0]) if len(frame) else None,
        "rounding": "half_up=floor(p*N+0.5)",
    }


def mixture_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not set(feature_columns).issubset(frame.columns):
        raise ValueError("Mixture is missing required feature columns")
    return frame.loc[:, feature_columns].copy()
