"""Fail-closed inputs and execution contract for plan-only broad utility candidates."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path



def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


LEGACY_TOTAL_WALL_SECONDS = 8 * 60 * 60
DEFAULT_UNIT_WALL_SECONDS = 2700


def resolve_config_path(package, requested=None):
    package = Path(package).resolve()
    if requested is not None:
        path = Path(requested)
        if not path.is_absolute():
            path = package / path
        if path.is_symlink():
            raise PermissionError("config path must not be a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(package):
            raise PermissionError("config must be inside the verified package")
        return resolved
    candidates = (
        package / "broad_utility_formal_candidate_v1.json",
        package / "configs" / "broad_utility_formal_candidate_v1.json",
        package / "broad_utility_future_v2_candidate_NO_RUN.json",
    )
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            return path.resolve(strict=True)
    raise FileNotFoundError("no v1 or v2 candidate config found in package")


def calculate_runtime_limits(config, plan):
    budget = config.get("budget", {})
    units = int(plan["unit_count"])
    per_unit = int(budget.get("per_unit_wall_seconds", DEFAULT_UNIT_WALL_SECONDS))
    total = int(budget.get("total_wall_seconds", LEGACY_TOTAL_WALL_SECONDS))
    if units < 1 or per_unit < 1 or total < 1:
        raise ValueError("runtime limits must be positive")
    if total < units * per_unit:
        raise ValueError("batch wall cap is below unit caps summed serially")
    return {"per_unit_wall_seconds": per_unit, "total_wall_seconds": total}


def validate_runtime_limits(config, plan, *, total_wall_seconds, per_unit_wall_seconds):
    expected = calculate_runtime_limits(config, plan)
    actual = {"total_wall_seconds": int(total_wall_seconds),
              "per_unit_wall_seconds": int(per_unit_wall_seconds)}
    if actual != expected:
        raise PermissionError("worker runtime limits differ from the verified candidate budget")
    return expected


def require_execution_enabled(config):
    if config.get("protocol_id") == "broad-utility-v1-preparation":
        return
    execution = config.get("execution", {})
    if (str(config.get("status", "")).startswith("NO-RUN")
            or execution.get("enabled") is not True):
        raise PermissionError("candidate execution is disabled")
    if execution.get("authorization_required") is True:
        receipt = execution.get("authorization_receipt_sha256", "")
        if (not isinstance(receipt, str) or len(receipt) != 64
                or any(char not in "0123456789abcdef" for char in receipt.lower())):
            raise PermissionError("candidate requires an authorization receipt SHA-256")


def calculate_fit_plan(dataset_count, seeds, generator_count, learner_count, replacement_doses, augmentation_doses):
    """Compute planned outcomes and actual fits after shared-control cache reuse."""
    if min(dataset_count, len(seeds), generator_count, learner_count) < 1:
        raise ValueError("plan dimensions must be positive")
    arm_count = 3 + 2 * len(replacement_doses) + 2 * len(augmentation_doses)
    cached_control_arms = 2 + len(replacement_doses) + len(augmentation_doses)
    units = dataset_count * len(seeds)
    generator_fits = units * generator_count
    fits_per_unit = learner_count * (arm_count + max(0, generator_count - 1) * (arm_count - cached_control_arms))
    return {"unit_count": units, "generator_fit_calls": generator_fits,
            "planned_outcomes": units * generator_count * learner_count * arm_count,
            "downstream_fit_calls": units * fits_per_unit}


def build_plan(config):
    """Reuse the existing plan scheduler; this stage is exploratory breadth, not confirmation."""
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plan_broad_utility import schedule

    protocol = {
        "breadth_screen": {"generator_seeds": config["seeds"]},
        "learners": list(config["learner_names"]),
        "generators_main": list(config["generators"]),
        "replacement_doses": config["utility"]["replacement_doses"],
        "augmentation_doses": config["utility"]["augmentation_doses"],
        "primary_metric": config["utility"]["primary_metric"],
        "generator_configs": config["generators"],
        "learner_configs": {"per_learner": config["learners"], "preprocessing": config["preprocessing"]},
        "bootstrap_replicates": config["utility"]["paired_bootstrap_replicates"],
    }
    datasets = [row["dataset_id"] for row in config["datasets"]]
    plan = schedule(protocol, datasets, stage="breadth_screen")
    plan.update({"batch_id": config["batch_id"], "phase": config["phase"],
                 "formal_inclusion": False,
                 "source_archive_hashes": {x["dataset_id"]: x.get("archive_sha256") for x in config["datasets"]},
                 "role_assignment_hashes": {x["dataset_id"]: x.get("role_assignment_sha256") for x in config["datasets"]},
                 "role_mode": config.get("role_mode", "recovery02_verified"),
                 "role_algorithm_sha256": config.get("role_algorithm_sha256"),
                 "selection_definition": config["source_cache"]["selection_definition"],
                 "selection_definition_sha256": sha_bytes(canonical(config["source_cache"]["selection_definition"]).encode()),
                 "final_test_evaluated": False})
    if config.get("role_candidate_manifest_file"):
        plan["candidate_role_manifest_file"] = config["role_candidate_manifest_file"]
        plan["candidate_role_manifest_sha256"] = config.get("role_candidate_manifest_sha256")
        plan["candidate_role_assignment_hashes"] = {
            x["dataset_id"]: x.get("candidate_role_assignment_sha256")
            for x in config["datasets"]}
    budget = calculate_fit_plan(len(datasets), config["seeds"], len(config["generators"]),
        len(config["learner_names"]), config["utility"]["replacement_doses"],
        config["utility"]["augmentation_doses"])
    declared_budget = config.get("budget", {})
    for key, expected in (
            ("unit_count", budget["unit_count"]),
            ("learned_generator_fits", budget["generator_fit_calls"]),
            ("bootstrap_generator_fits", 0),
            ("downstream_fits", budget["downstream_fit_calls"]),
            ("planned_outcomes", budget["planned_outcomes"])):
        if key in declared_budget and declared_budget[key] != expected:
            raise ValueError(f"declared candidate budget mismatch: {key}")
    if config.get("protocol_id") == "broad-utility-v1-preparation":
        if (plan["downstream_fit_calls"], plan["independent_generator_fit_calls"]) != (756, 27):
            raise ValueError("frozen three-table breadth plan must be 756 downstream / 27 generator fits")
    else:
        if (plan["downstream_fit_calls"], plan["independent_generator_fit_calls"]) != (
                budget["downstream_fit_calls"], budget["generator_fit_calls"]):
            raise ValueError("candidate plan differs from calculated dynamic budget")
    plan.update(budget)
    return plan


def rebuild_recovery02_roles(frame, *, spec, features, target, task):
    """Reproduce r5 roles byte-for-byte; reject any mismatch before any fit."""
    from tabpollution.data.broad_splits import duplicate_feature_groups, prepare_assignment

    groups = duplicate_feature_groups(frame, features)
    labels = None if task == "regression" else frame[target]
    roles = prepare_assignment(frame.record_id, seed=20261029, labels=labels, groups=groups)
    csv_bytes = roles.to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest = sha_bytes(csv_bytes)
    if digest != spec["role_assignment_sha256"]:
        raise PermissionError(f"reconstructed recovery02 role hash mismatch for {spec['dataset_id']}")
    counts = roles.role.value_counts().to_dict()
    if counts != spec["role_counts"] or set(counts) != {"train", "tune", "evaluation", "final_test"}:
        raise ValueError(f"reconstructed recovery02 role counts mismatch for {spec['dataset_id']}")
    return roles, digest


def cap_fixed_role(frame, *, cap, task, target, seed=20261029):
    """Apply r5's deterministic 2000/1000 cap without changing outer roles."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    if len(frame) <= cap:
        return frame.reset_index(drop=True)
    stratify = None if task == "regression" else frame[target]
    selected, _ = train_test_split(np.arange(len(frame)), train_size=cap,
        random_state=seed, stratify=stratify)
    return frame.iloc[selected].reset_index(drop=True)


def require_source_role_identities(config):
    """Reject execution until source and candidate-role identities pass pre-fit gates."""
    if config.get("role_mode") == "extension_first_assignment":
        package = Path(__file__).resolve().parents[3]
        if config.get("external_role_dependencies_file"):
            from tabpollution.data.candidate_roles import require_external_role_dependencies
            require_external_role_dependencies(package, config)
        else:
            from tabpollution.data.candidate_roles import require_candidate_role_manifest_ready
            require_candidate_role_manifest_ready(package, config)
    for spec in config.get("datasets", []):
        if not spec.get("archive_sha256") or not spec.get("role_assignment_sha256"):
            raise PermissionError(f"source and role identities required before execution: {spec.get('dataset_id')}")


def load_and_validate_all(config, catalog, input_root):
    """Read/cache-check all sources and recreate every accepted role before any fit."""
    require_source_role_identities(config)
    if config.get("role_mode") == "extension_first_assignment":
        if not config.get("role_algorithm_sha256"):
            raise PermissionError("extension split algorithm commitment is required")
        import tabpollution.data.extension_splits as extension_module
        if sha_bytes(Path(extension_module.__file__).read_bytes()) != config["role_algorithm_sha256"]:
            raise PermissionError("extension split algorithm hash mismatch")
        return load_and_validate_extension_all(config, catalog, input_root)
    if any(not item.get("role_assignment_sha256") for item in config["datasets"]):
        raise PermissionError("historical roles require accepted assignment hashes")
    import pandas as pd
    from tabpollution.intake import read_allowlisted_table
    import tabpollution.intake as intake_module
    import tabpollution.data.broad_splits as split_module

    input_root = Path(input_root)
    if input_root.is_symlink():
        raise PermissionError("source cache root must not be a symlink")
    input_root = input_root.resolve(strict=True)
    lineage = config["source_cache"].get("role_reproduction_source")
    if lineage:
        for name, module, expected in (
            ("intake", intake_module, lineage["intake_module_sha256"]),
            ("split", split_module, lineage["split_module_sha256"]),
        ):
            if sha_bytes(Path(module.__file__).read_bytes()) != expected:
                raise PermissionError(f"r5 {name} implementation hash mismatch; stop before fitting")
    cards = {row["dataset_id"]: row for row in catalog["records"]}
    prepared = {}
    for spec in config["datasets"]:
        dataset = spec["dataset_id"]
        card = cards.get(dataset)
        if card is None or card.get("status") != "prepared":
            raise ValueError(f"source catalog gate failed for {dataset}")
        archive = input_root / spec["archive_file"]
        frame, intake = read_allowlisted_table(catalog, dataset, archive,
            input_root=input_root, approved_sha256=spec["archive_sha256"], allow_source_data_read=True)
        if int(len(frame)) != int(spec["source_rows"]) or int(len(frame)) != int(card["official_instance_count_page"]):
            raise ValueError(f"source row count mismatch for {dataset}")
        features, target = card["features"], card["target"]
        for feature in features:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        task = spec["task"]
        frame[target] = frame[target].astype(str) if task != "regression" else pd.to_numeric(frame[target], errors="raise")
        frame.insert(0, "record_id", [f"public:{dataset}:{i}" for i in range(len(frame))])
        roles, role_digest = rebuild_recovery02_roles(frame, spec=spec, features=features, target=target, task=task)
        assignment = roles.set_index("record_id").role
        frame["split_role"] = frame.record_id.map(assignment)
        if frame.split_role.isna().any():
            raise ValueError(f"role join failed for {dataset}")
        train = frame.loc[frame.split_role.eq("train"), ["record_id", *features, target]].reset_index(drop=True)
        evaluation = frame.loc[frame.split_role.eq("evaluation"), ["record_id", *features, target]].reset_index(drop=True)
        train = cap_fixed_role(train, cap=config["source_cache"]["train_cap"], task=task, target=target)
        evaluation = cap_fixed_role(evaluation, cap=config["source_cache"]["evaluation_cap"], task=task, target=target)
        prepared[dataset] = {"task": task, "features": features, "target": target,
            "train": train, "evaluation": evaluation, "roles": roles,
            "source_sha256": intake["archive_sha256"],
            "roles_sha256": role_digest,
            "intake": {"dataset_id": dataset, "archive_sha256": intake["archive_sha256"],
                "archive_member": spec["archive_member"], "source_rows": len(frame),
                "role_counts": spec["role_counts"]},
            "group_features": spec["group_features"]}
    if set(prepared) != {x["dataset_id"] for x in config["datasets"]}:
        raise ValueError("all source and role gates must pass before fitting")
    return prepared


def load_and_validate_extension_all(config, catalog, input_root):
    require_source_role_identities(config)
    """Prepare three extension datasets and first-time role hashes; never read .tes."""
    import pandas as pd
    import tabpollution.intake as intake_module
    from tabpollution.data.broad_splits import assign_official_holdout_roles
    from tabpollution.data.extension_splits import (
        assign_group_stratified_iid_roles, assign_purged_ordered_roles,
    )

    root = Path(input_root)
    if root.is_symlink():
        raise PermissionError("source cache root must not be a symlink")
    root = root.resolve(strict=True)
    prepared = {}
    for spec in config["datasets"]:
        dataset, task = spec["dataset_id"], spec["task"]
        archive = root / spec["archive_file"]
        catalog_view = {**catalog, "records": [
            {**row, "member": spec.get("archive_member", spec.get("archive_members", {}).get("development"))}
            for row in catalog["records"] if row["dataset_id"] == dataset]}
        if not catalog_view["records"]:
            raise ValueError(f"extension source catalog gate failed for {dataset}")
        frame, intake = intake_module.read_allowlisted_table(catalog_view, dataset, archive,
            input_root=root, approved_sha256=spec["archive_sha256"], allow_source_data_read=True,
            approved_nested_sha256=spec.get("nested_archive_sha256"))
        if len(frame) != int(spec["source_rows"]):
            raise ValueError(f"extension row count mismatch for {dataset}")
        features, target = list(spec["features"]), spec["target"]
        if target not in frame or not set(features) <= set(frame.columns):
            raise ValueError(f"extension schema mismatch for {dataset}")
        frame = frame[[*features, target]].copy()
        frame.insert(0, "record_id", [f"public:{dataset}:{i}" for i in range(len(frame))])
        frame[target] = frame[target].astype(str) if task != "regression" else pd.to_numeric(frame[target], errors="raise")
        numeric, categorical = list(spec["numeric_features"]), list(spec["categorical_features"])
        if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
            raise ValueError(f"extension feature type partition mismatch for {dataset}")
        for column in features:
            if column in categorical:
                frame[column] = frame[column].fillna("__MISSING__").astype(str)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
        if spec.get("candidate_role_assignment_file"):
            from tabpollution.data.candidate_roles import load_candidate_role_assignment
            roles = load_candidate_role_assignment(
                Path(__file__).resolve().parents[3], spec, len(frame))
        elif dataset == "bank_marketing":
            groups = pd.util.hash_pandas_object(frame[features], index=False).astype(str)
            roles = assign_purged_ordered_roles(frame.record_id, groups=groups, labels=frame[target])
        elif dataset == "online_shoppers":
            groups = pd.util.hash_pandas_object(frame[features], index=False).astype(str)
            roles = assign_group_stratified_iid_roles(frame.record_id, groups=groups,
                labels=frame[target], strata=frame["Month"], seed=config["source_cache"]["role_seed"])
        elif dataset == "pendigits":
            parts = pd.Series(["pendigits.tra"] * len(frame))
            heldout_n = int(spec["member_rows"]["pendigits.tes"])
            full_ids = pd.concat([frame.record_id, pd.Series(
                [f"public:{dataset}:tes:{i}" for i in range(heldout_n)])], ignore_index=True)
            full_parts = pd.concat([parts, pd.Series(["pendigits.tes"] * heldout_n)], ignore_index=True)
            labels = pd.concat([frame[target], pd.Series(["__SEALED__"] * heldout_n)], ignore_index=True)
            roles = assign_official_holdout_roles(full_ids, full_parts,
                heldout_partition="pendigits.tes", development_partition="pendigits.tra",
                seed=config["source_cache"]["role_seed"], labels=labels)
            # Retain sealed-member metadata in the identity ledger without opening
            # that member. Only development rows exist in frame and reach models.
        else:
            raise ValueError(f"unsupported extension dataset {dataset}")
        role_hash = sha_bytes(roles.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        candidate_hash = spec.get("candidate_role_assignment_sha256")
        if candidate_hash and role_hash != candidate_hash:
            raise PermissionError(f"candidate role assignment digest mismatch for {dataset}")
        if spec.get("role_assignment_sha256") and role_hash != spec["role_assignment_sha256"]:
            raise PermissionError(f"frozen extension role hash mismatch for {dataset}")
        frame["split_role"] = frame.record_id.map(roles.set_index("record_id").role)
        if frame.split_role.isna().any():
            raise ValueError(f"extension role join failed for {dataset}")
        train = cap_fixed_role(frame.loc[frame.split_role.eq("train"), ["record_id", *features, target]],
            cap=config["source_cache"]["train_cap"], task=task, target=target,
            seed=config["source_cache"]["cap_seed"])
        evaluation = cap_fixed_role(frame.loc[frame.split_role.eq("evaluation"), ["record_id", *features, target]],
            cap=config["source_cache"]["evaluation_cap"], task=task, target=target,
            seed=config["source_cache"]["cap_seed"])
        prepared[dataset] = {"task": task, "features": features, "target": target,
            "train": train, "evaluation": evaluation, "roles": roles,
            "source_sha256": intake["archive_sha256"], "roles_sha256": role_hash,
            "intake": {"dataset_id": dataset, "archive_sha256": intake["archive_sha256"],
                "source_rows": len(frame), "role_counts": roles.role.value_counts().to_dict()},
            "numeric": numeric, "categorical": categorical,
            "group_features": list(spec.get("group_features", numeric[:1] + categorical[:1]))}
    if set(prepared) != {row["dataset_id"] for row in config["datasets"]}:
        raise ValueError("all extension sources and first-time roles must pass before fitting")
    return prepared


def build_identity_for_unit(config, plan, dataset, seed, data, package_sha256, runtime_sha256):
    from tabpollution.governance.broad_batch_registry import build_unit_identity

    selection = config["source_cache"]["selection_definition"]
    return build_unit_identity(dataset_id=dataset, task=data["task"], seed=seed,
        source_sha256=data["source_sha256"], role_assignments=data["roles"],
        roles_assignment_sha256=data["roles_sha256"],
        train_ids=data["train"].record_id.astype(str).tolist(),
        evaluation_ids=data["evaluation"].record_id.astype(str).tolist(),
        generator_config=plan["generator_config"], learner_config=plan["learner_config"],
        selection_definition=selection, package_sha256=package_sha256, runtime_sha256=runtime_sha256)


def account_matrix(matrix, config):
    """Account every planned outcome without concealing invalid synthetic arms."""
    arms = {"TRTR", "TSTR", "bootstrap_TSTR"}
    for kind, doses in (("replace", config["utility"]["replacement_doses"]),
                        ("augment", config["utility"]["augmentation_doses"])):
        for dose in doses:
            arms.update({f"{kind}_{dose:g}", f"bootstrap_{kind}_{dose:g}"})
    roster = {(learner, arm) for learner in config["learner_names"] for arm in arms}
    cases = matrix.get("cases", {})
    if set(cases) != set(config["generators"]) or matrix.get("reference_control_cache_entries") != 24:
        raise ValueError("generator/control roster mismatch")
    actual = invalid = 0
    for result in cases.values():
        rows = result.get("results", [])
        if len(rows) != len(roster) or {(r["learner"], r["arm"]) for r in rows} != roster:
            raise ValueError("planned arm roster incomplete or duplicated")
        case_calls = 0
        for row in rows:
            if row.get("status") == "complete":
                case_calls += int(not row.get("fit_reused", False))
            elif row.get("status") == "invalid_training_class_support":
                if row["arm"] == "TRTR" or row["arm"].startswith("bootstrap_") or row.get("metrics") is not None:
                    raise ValueError("invalid shared control or fabricated invalid-arm metric")
                counts = row.get("class_support", {})
                if not (counts.get("missing_class_count", 0) or counts.get("unexpected_class_count", 0)):
                    raise ValueError("invalid arm lacks class-support evidence")
                invalid += 1
            else:
                raise ValueError("unaccounted downstream outcome")
        if case_calls != result.get("model_calls"):
            raise ValueError("case fit accounting differs")
        actual += case_calls
    if actual != matrix.get("actual_downstream_model_calls") or actual + invalid != 84:
        raise ValueError("actual fits plus unavailable synthetic outcomes must equal 84")
    return actual, invalid


def execute_unit(data, *, seed, plan, config, generator_factory, progress_callback=None):
    from tabpollution.studies.broad_utility import run_generator_matrix

    train, evaluation = data["train"], data["evaluation"]
    target, task, features = data["target"], data["task"], data["features"]
    classes = () if task == "regression" else tuple(sorted(train[target].astype(str).unique()))
    if task != "regression" and set(evaluation[target].astype(str)) != set(classes):
        raise ValueError("fixed evaluation role has incomplete class support; no reroll")
    matrix = run_generator_matrix(train, evaluation, target=target,
        numeric=data.get("numeric", features), categorical=data.get("categorical", ()), task=task, classes=classes,
        group_features=data["group_features"], generator_names=list(config["generators"]),
        generator_factory=generator_factory, seed=seed,
        learners=list(config["learner_names"]),
        bootstrap_reps=config["utility"]["paired_bootstrap_replicates"],
        replacement=tuple(config["utility"]["replacement_doses"]),
        augmentation=tuple(config["utility"]["augmentation_doses"]),
        dose_rounding=config["utility"].get("dose_rounding", "half_up"),
        progress_callback=progress_callback)
    actual, invalid = account_matrix(matrix, config)
    if matrix["unit"]["generator_fit_calls"] != 3:
        raise ValueError("unit generator fit-call budget mismatch")
    return {"status": "complete", "formal_inclusion": False, "phase": config["phase"],
        "dataset_id": data["intake"]["dataset_id"], "task": task, "seed": seed,
        "source_sha256": data["source_sha256"], "roles_assignment_sha256": data["roles_sha256"],
        "generator_config": plan["generator_config"], "learner_config": plan["learner_config"],
        "generator_fit_calls": 3, "learned_generator_fit_calls": 3,
        "bootstrap_generator_fit_calls": 0, "downstream_fit_calls": actual,
        "downstream_fit_budget": 84, "invalid_training_outcomes": invalid,
        "utility_availability": "complete" if invalid == 0 else "contains_unavailable_synthetic_arms",
        "final_test_evaluated": False, "row_outputs": False, "matrix": matrix}
