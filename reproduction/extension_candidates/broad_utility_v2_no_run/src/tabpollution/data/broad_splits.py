"""New-cohort split preparation; never migrates an existing frozen split."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROLES = ("train", "tune", "evaluation", "final_test")
FRACTIONS = (.50, .15, .15, .20)

# Metadata-derived adapter routes for the twelve non-pilot tables. These routes are
# declarations, not source verification; blocked routes must remain blocked until
# exact member/schema evidence is checked against the approved archive.
SOURCE_SPLIT_ADAPTERS = {
    "adult": "official_member_holdout",
    "credit": "iid_stratified",
    "bank_marketing": "chronological",
    "online_shoppers": "chronological_session_policy_review",
    "magic_gamma": "iid_group_semantics_review",
    "pendigits": "official_member_holdout",
    "letter_recognition": "official_ordered_holdout",
    "abalone": "iid_historical_anchor_refresh",
    "concrete_strength": "iid_stratified",
    "energy_efficiency_y1": "iid_design_provenance_review",
    "bike_hourly": "chronological",
    "airfoil_self_noise": "grouped_measurement_review",
}


def assign_official_holdout_roles(ids, source_partitions, *, heldout_partition,
                                  development_partition, seed, labels=None):
    """Keep a documented official holdout intact; split only its development member."""
    ids = pd.Series(ids).reset_index(drop=True).astype("string")
    parts = pd.Series(source_partitions).reset_index(drop=True).astype("string")
    if len(ids) != len(parts) or ids.isna().any() or ids.duplicated().any() or parts.isna().any():
        raise ValueError("official split needs aligned unique IDs and nonmissing member labels")
    if {heldout_partition, development_partition} - set(parts):
        raise ValueError("declared official members are missing")
    if set(parts) != {heldout_partition, development_partition}:
        raise ValueError("unexpected source member; refuse implicit merging")
    dev = np.flatnonzero(parts.to_numpy() == development_partition)
    heldout = np.flatnonzero(parts.to_numpy() == heldout_partition)
    if len(dev) < 20 or not len(heldout):
        raise ValueError("official development/heldout member is too small")
    y = None if labels is None else pd.Series(labels).reset_index(drop=True)
    if y is not None and (len(y) != len(ids) or y.isna().any()):
        raise ValueError("invalid labels for official split adapter")
    rng_labels = None if y is None else y.iloc[dev].to_numpy()
    # Allocate development as 62.5/18.75/18.75, yielding overall 50/15/15
    # when the official holdout is 20%; preserve the published holdout fraction otherwise.
    train_idx, rest_idx = train_test_split(dev, train_size=.625, random_state=seed,
                                         stratify=rng_labels)
    rest_y = None if y is None else y.iloc[rest_idx].to_numpy()
    tune_idx, evaluation_idx = train_test_split(rest_idx, train_size=.5,
        random_state=seed + 1, stratify=rest_y)
    roles = np.empty(len(ids), dtype=object)
    roles[train_idx], roles[tune_idx], roles[evaluation_idx] = "train", "tune", "evaluation"
    roles[heldout] = "final_test"
    if y is not None:
        expected = set(y.iloc[dev].astype(str))
        for role in ("train", "tune", "evaluation"):
            if set(y[roles == role].astype(str)) != expected:
                raise ValueError(f"class absent from {role}; preserve official holdout and fail without reroll")
    return pd.DataFrame({"record_id": ids, "role": roles,
                         "source_partition": parts})


def assign_ordered_holdout_roles(ids, *, development_count, seed, labels=None):
    """Preserve a documented prefix/suffix boundary, splitting only the prefix."""
    ids = pd.Series(ids).reset_index(drop=True).astype("string")
    n = len(ids)
    if ids.isna().any() or ids.duplicated().any() or not 20 <= development_count < n:
        raise ValueError("invalid ordered source boundary or IDs")
    y = None if labels is None else pd.Series(labels).reset_index(drop=True)
    if y is not None and (len(y) != n or y.isna().any()):
        raise ValueError("invalid labels for ordered split adapter")
    development = np.arange(development_count)
    dev_labels = None if y is None else y.iloc[development].to_numpy()
    train_idx, rest_idx = train_test_split(development, train_size=.625, random_state=seed,
                                           stratify=dev_labels)
    rest_labels = None if y is None else y.iloc[rest_idx].to_numpy()
    tune_idx, evaluation_idx = train_test_split(rest_idx, train_size=.5,
        random_state=seed + 1, stratify=rest_labels)
    roles = np.empty(n, dtype=object)
    roles[train_idx], roles[tune_idx], roles[evaluation_idx] = "train", "tune", "evaluation"
    roles[development_count:] = "final_test"
    if y is not None:
        expected = set(y.iloc[:development_count].astype(str))
        for role in ("train", "tune", "evaluation"):
            if set(y[roles == role].astype(str)) != expected:
                raise ValueError(f"class absent from {role}; preserve ordered holdout and fail without reroll")
    return pd.DataFrame({"record_id": ids, "role": roles,
                         "source_order": np.arange(n, dtype=int)})


def prepare_assignment(ids, *, seed, labels=None, groups=None, times=None, max_fraction_error=.10):
    ids = pd.Series(ids).reset_index(drop=True).astype("string")
    if ids.isna().any() or ids.duplicated().any() or len(ids) < 20:
        raise ValueError("unique nonmissing IDs and at least20 rows required")
    n = len(ids)
    for values in (labels, groups, times):
        if values is not None and len(values) != n:
            raise ValueError("split metadata length mismatch")
    if groups is not None and times is not None:
        raise ValueError("group+time needs dataset-specific purged split; no silent random fallback")
    assigned = np.empty(n, dtype=object)
    if times is not None:
        timestamps = pd.to_datetime(pd.Series(times), errors="raise", utc=True)
        if timestamps.isna().any():
            raise ValueError("missing chronology")
        unique = sorted(timestamps.unique())
        if len(unique) < 8:
            raise ValueError("insufficient independent time blocks")
        # Allocate whole timestamp blocks; chronological roles have no tied boundary.
        order = {t: i for i, t in enumerate(unique)}
        ranks = timestamps.map(order).to_numpy()
        bounds = np.floor(np.cumsum(FRACTIONS) * len(unique)).astype(int)
        bins = np.searchsorted(bounds[:-1], ranks, side="right")
        assigned[:] = np.array(ROLES, dtype=object)[bins]
    elif groups is not None:
        group_values = pd.Series(groups).astype("string")
        if group_values.isna().any():
            raise ValueError("missing independent-unit group")
        for group in sorted(group_values.unique()):
            digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
            u = int.from_bytes(digest[:8], "big") / 2**64
            index = min(int(np.searchsorted(np.cumsum(FRACTIONS), u)), 3)
            assigned[group_values.eq(group).to_numpy(dtype=bool)] = ROLES[index]
    else:
        remaining = np.arange(n)
        y = None if labels is None else np.asarray(labels)
        remaining_mass = 1.
        for role, fraction in zip(ROLES[:-1], FRACTIONS[:-1]):
            selected, remaining = train_test_split(remaining, train_size=fraction / remaining_mass,
                random_state=seed, stratify=None if y is None else y[remaining])
            assigned[selected] = role
            remaining_mass -= fraction
        assigned[remaining] = ROLES[-1]
    if set(assigned) != set(ROLES):
        raise ValueError("empty role; do not reroll seeds")
    for role, fraction in zip(ROLES, FRACTIONS):
        if abs(float(np.mean(assigned == role))-fraction) > max_fraction_error:
            raise ValueError("role fraction outside declared tolerance; do not reroll")
    if labels is not None:
        y = pd.Series(labels).reset_index(drop=True)
        if y.isna().any():
            raise ValueError("missing classification label")
        required = set(y.astype(str))
        for role in ROLES:
            if set(y[assigned == role].astype(str)) != required:
                raise ValueError(f"class absent from {role}; fail, do not silently rebalance")
    return pd.DataFrame({"record_id": ids, "role": assigned})


def duplicate_feature_groups(frame, features):
    """Exact feature-identical records stay together; no target-dependent clustering."""
    if not features or not set(features) <= set(frame.columns):
        raise ValueError("invalid feature grouping schema")
    return pd.util.hash_pandas_object(frame[list(features)], index=False).astype(str)
