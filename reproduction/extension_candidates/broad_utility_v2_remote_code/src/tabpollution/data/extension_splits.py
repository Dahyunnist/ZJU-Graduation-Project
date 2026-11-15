"""Split adapters for the Bank/Online/Pendigits expansion candidate."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

ROLES = ("train", "tune", "evaluation", "final_test")
FRACTIONS = (.50, .15, .15, .20)


def assign_purged_ordered_roles(ids, *, groups, labels, max_fraction_error=.10):
    """Use contiguous source-order blocks; purge, never move, groups crossing cuts."""
    ids = pd.Series(ids).reset_index(drop=True).astype("string")
    groups = pd.Series(groups).reset_index(drop=True).astype("string")
    labels = pd.Series(labels).reset_index(drop=True).astype("string")
    n = len(ids)
    if n < 40 or len(groups) != n or len(labels) != n or ids.isna().any() or ids.duplicated().any():
        raise ValueError("ordered split needs aligned IDs, groups, labels and at least 40 rows")
    if groups.isna().any() or labels.isna().any():
        raise ValueError("ordered split has missing group or class")
    cuts = np.floor(np.cumsum(FRACTIONS[:-1]) * n).astype(int)
    role_index = np.searchsorted(cuts, np.arange(n), side="right")
    assigned = np.asarray(ROLES, dtype=object)[role_index]
    role_by_group = pd.DataFrame({"group": groups, "role": assigned}).groupby("group").role.nunique()
    crossing = set(role_by_group[role_by_group > 1].index)
    assigned[groups.isin(crossing).to_numpy()] = "purged"
    for role, fraction in zip(ROLES, FRACTIONS):
        observed = float(np.mean(assigned == role))
        if int(np.sum(assigned == role)) < 20 or abs(observed - fraction) > max_fraction_error:
            raise ValueError(f"ordered role fraction outside tolerance for {role}; fail without reroll")
        if set(labels[assigned == role]) != set(labels):
            raise ValueError(f"class absent from ordered role {role}; fail without reroll")
    result = pd.DataFrame({"record_id": ids, "role": assigned,
                           "source_order": np.arange(n, dtype=int)})
    return result


def assign_group_stratified_iid_roles(ids, *, groups, labels, strata, seed,
                                      max_fraction_error=.10):
    """Greedily balance label×stratum counts while keeping each group intact."""
    ids = pd.Series(ids).reset_index(drop=True).astype("string")
    groups = pd.Series(groups).reset_index(drop=True).astype("string")
    labels = pd.Series(labels).reset_index(drop=True).astype("string")
    strata = pd.Series(strata).reset_index(drop=True).astype("string")
    n = len(ids)
    if n < 20 or any(len(x) != n for x in (groups, labels, strata)):
        raise ValueError("grouped IID split needs aligned data and at least 20 rows")
    if ids.isna().any() or ids.duplicated().any() or groups.isna().any() or labels.isna().any() or strata.isna().any():
        raise ValueError("grouped IID split has duplicate IDs or missing split metadata")
    cells = pd.Series([json.dumps([str(a), str(b)], ensure_ascii=False)
                       for a, b in zip(labels, strata)])
    cell_order = sorted(cells.unique())
    cell_index = {value: i for i, value in enumerate(cell_order)}
    target = np.zeros(len(cell_order), dtype=float)
    grouped_rows = {}
    for group, indices in groups.groupby(groups, sort=False).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        vector = np.zeros(len(cell_order), dtype=float)
        for cell, count in cells.iloc[rows].value_counts().items():
            vector[cell_index[cell]] = count
        grouped_rows[str(group)] = (rows, vector)
        target += vector
    desired = np.asarray(FRACTIONS)[:, None] * target[None, :]
    counts = np.zeros_like(desired)
    assignment = {}
    tie_key = lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
    ordered_groups = sorted(grouped_rows,
        key=lambda group: (-len(grouped_rows[group][0]), tie_key(group), group))
    scale = np.maximum(desired, 1.)
    for group in ordered_groups:
        rows, vector = grouped_rows[group]
        choices = []
        for index, role in enumerate(ROLES):
            candidate = counts.copy()
            candidate[index] += vector
            loss = float(np.sum(np.square(candidate - desired) / scale))
            tie = hashlib.sha256(f"{seed}:{group}:{role}".encode()).hexdigest()
            choices.append((loss, tie, index))
        chosen = min(choices)[2]
        assignment[group] = ROLES[chosen]
        counts[chosen] += vector
    assigned = groups.map(assignment).to_numpy(dtype=object)
    if set(assigned) != set(ROLES):
        raise ValueError("grouped IID split produced an empty role; fail without reroll")
    for role, fraction in zip(ROLES, FRACTIONS):
        if int(np.sum(assigned == role)) < 20 or abs(float(np.mean(assigned == role)) - fraction) > max_fraction_error:
            raise ValueError(f"grouped IID role fraction outside tolerance for {role}; fail without reroll")
        if set(labels[assigned == role]) != set(labels):
            raise ValueError(f"class absent from grouped IID role {role}; fail without reroll")
    result = pd.DataFrame({"record_id": ids, "role": assigned})
    if result.assign(group=groups).groupby("group").role.nunique().max() != 1:
        raise AssertionError("internal error: duplicate group crossed roles")
    return result


def validate_model_row_permissions(role_rows, operation_rows):
    """Reject any model input outside train rows or score input outside evaluation."""
    roles = pd.Series(role_rows.role.to_numpy(), index=role_rows.record_id.astype(str))
    if roles.index.duplicated().any():
        raise ValueError("role IDs are not unique")
    allowed = {"generator_fit": {"train"}, "learner_fit": {"train"},
               "evaluation_score": {"evaluation"}}
    for operation, ids in operation_rows.items():
        if operation not in allowed:
            raise PermissionError(f"operation is not authorized: {operation}")
        ids = pd.Index([str(value) for value in ids])
        if ids.has_duplicates or not ids.isin(roles.index).all():
            raise PermissionError("model input contains unknown or duplicate role IDs")
        if not set(roles.loc[ids].astype(str)) <= allowed[operation]:
            raise PermissionError(f"row role is not allowed for {operation}")
    return True


def plan_extension_budget(preflight_statuses):
    """Return a plan-only budget; any failed dataset gate blocks all first fits."""
    if len(preflight_statuses) != 3 or any(status != "pass" for status in preflight_statuses):
        return {"status": "BLOCKED_BEFORE_FIRST_FIT", "generator_fits": 0,
                "downstream_fits": 0, "max_attempts": 0}
    return {"status": "PLAN_ONLY", "generator_fits": 27,
            "downstream_fits": 756, "max_attempts": 1}
