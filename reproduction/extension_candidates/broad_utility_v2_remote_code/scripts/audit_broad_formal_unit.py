"""Independent aggregate-only auditor for one formal-candidate dataset/seed unit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from tabpollution.governance.broad_batch_registry import identity_sha256

FORBIDDEN_KEYS = {"predictions", "row_ids", "y_true", "training_rows", "record_id", "raw_data", "feature_values"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_row_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"row-level field in aggregate: {key}")
            reject_row_values(item)
    elif isinstance(value, list):
        for item in value:
            reject_row_values(item)


def audit(aggregate_path, identity_path, output_path, *, package_sha, runtime_sha, config_path, plan_path):
    aggregate_path, identity_path, output_path = map(Path, (aggregate_path, identity_path, output_path))
    if output_path.exists():
        raise FileExistsError("auditor refuses to overwrite receipt")
    report = json.loads(aggregate_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    from tabpollution.studies.broad_formal_batch import build_plan
    trusted_plan = build_plan(config)
    required_plan_fields = ("jobs_sha256", "generator_config_sha256", "learner_config_sha256",
        "downstream_fit_calls", "independent_generator_fit_calls", "unit_count", "generator_fit_calls")
    if any(field not in plan or field not in trusted_plan or plan[field] != trusted_plan[field]
           for field in required_plan_fields):
        raise ValueError("resolved plan differs from trusted config or lacks bound budget fields")
    from tabpollution.governance.broad_batch_registry import validate_identity
    validate_identity(identity)
    identity_digest = identity_sha256(identity)
    matrix = report.get("matrix", {})
    generators = list(config["generators"])
    provenance = matrix.get("unit", {}).get("generator_provenance", {})
    from tabpollution.studies.broad_formal_batch import calculate_fit_plan
    unit_budget = calculate_fit_plan(1, [identity["seed"]], len(generators),
        len(config["learner_names"]), config["utility"]["replacement_doses"],
        config["utility"]["augmentation_doses"])
    expected_cache_entries = (2 + len(config["utility"]["replacement_doses"])
        + len(config["utility"]["augmentation_doses"])) * len(config["learner_names"])
    if (report.get("status") != "complete" or report.get("row_outputs") is not False
            or report.get("formal_inclusion") is not False
            or report.get("identity_sha256") != identity_digest
            or report.get("identity") != identity
            or report.get("package_sha256") != package_sha
            or report.get("runtime_sha256") != runtime_sha
            or identity["generator_config_sha256"] != plan["generator_config_sha256"]
            or identity["learner_config_sha256"] != plan["learner_config_sha256"]
            or report.get("generator_fit_calls") != len(generators)
            or report.get("downstream_fit_budget") != unit_budget["downstream_fit_calls"]
            or report.get("generator_names") != generators
            or matrix.get("reference_control_cache_entries") != expected_cache_entries
            or set(provenance) != set(generators)
            or matrix.get("row_outputs") is not False
            or report.get("final_test_evaluated") is not False):
        raise ValueError("aggregate does not match the frozen unit plan and identity")
    for name, settings in config["generators"].items():
        if provenance[name].get("config") != settings:
            raise ValueError(f"resolved generator config differs for {name}")
    cases = matrix.get("cases", {})
    if set(cases) != set(generators):
        raise ValueError("generator case roster mismatch")
    expected_arms = {"TRTR", "TSTR", "bootstrap_TSTR"}
    for dose in config["utility"]["replacement_doses"]:
        expected_arms |= {f"replace_{dose:g}", f"bootstrap_replace_{dose:g}"}
    for dose in config["utility"]["augmentation_doses"]:
        expected_arms |= {f"augment_{dose:g}", f"bootstrap_augment_{dose:g}"}
    required = {(learner, arm) for learner in config["learner_names"] for arm in expected_arms}
    control_counts = {}
    actual_calls = unavailable = 0
    for generator, result in cases.items():
        if result.get("row_outputs") is not False or result.get("model_calls", 0) < 0:
            raise ValueError("case contains row outputs or invalid call count")
        rows = result.get("results", [])
        if len(rows) != len(required) or {(r.get("learner"), r.get("arm")) for r in rows} != required:
            raise ValueError("missing, duplicate or unknown planned outcome")
        case_calls = 0
        for row in rows:
            control = row["arm"] == "TRTR" or row["arm"].startswith("bootstrap_")
            if row.get("status") == "complete":
                if not isinstance(row.get("metrics"), dict) or not isinstance(row.get("fit_reused"), bool):
                    raise ValueError("complete outcome lacks metrics/fit evidence")
                if row["fit_reused"] and not control:
                    raise ValueError("synthetic model fits must not be reused")
                case_calls += int(not row["fit_reused"])
                if control:
                    key = (row["learner"], row["arm"])
                    counts = control_counts.setdefault(key, [0, 0])
                    counts[int(row["fit_reused"])] += 1
            elif row.get("status") == "invalid_training_class_support":
                counts = row.get("class_support", {})
                keys = {"expected_class_count", "observed_class_count", "missing_class_count", "unexpected_class_count"}
                if (control or identity["task"] == "regression" or row.get("metrics") is not None
                        or row.get("groups") or row.get("fit_reused", False)
                        or set(counts) != keys or any(type(v) is not int or v < 0 for v in counts.values())
                        or counts["expected_class_count"] < 2 or counts["observed_class_count"] < 1
                        or counts["missing_class_count"] + counts["unexpected_class_count"] == 0
                        or counts["expected_class_count"] - counts["observed_class_count"]
                            != counts["missing_class_count"] - counts["unexpected_class_count"]):
                    raise ValueError("invalid class-support outcome lacks honest unavailable evidence")
                unavailable += 1
            else:
                raise ValueError("unknown/unhandled downstream failure")
        if case_calls != result["model_calls"]:
            raise ValueError("case actual fit count mismatch")
        actual_calls += case_calls
    if (len(control_counts) != 24 or any(counts != [1, 2] for counts in control_counts.values())
            or actual_calls + unavailable != 84
            or matrix.get("actual_downstream_model_calls") != actual_calls
            or report.get("downstream_fit_calls") != actual_calls
            or report.get("invalid_training_outcomes") != unavailable):
        raise ValueError("independent unique-fit/control/unavailable accounting failed")
    reject_row_values(report)
    receipt = {"status": "pass", "independent": True,
        "auditor_id": "broad-formal-unit-auditor:" + sha(__file__),
        "aggregate_sha256": sha(aggregate_path), "identity_sha256": identity_digest,
        "package_sha256": package_sha, "runtime_sha256": runtime_sha,
        "generator_fit_calls": len(generators), "downstream_fit_calls": actual_calls,
        "downstream_fit_budget": unit_budget["downstream_fit_calls"], "invalid_training_outcomes": unavailable,
        "row_outputs": False, "final_test_evaluated": False}
    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--package-sha256", required=True)
    p.add_argument("--runtime-sha256", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args()
    receipt = audit(args.aggregate, args.identity, args.output,
        package_sha=args.package_sha256, runtime_sha=args.runtime_sha256,
        config_path=args.config, plan_path=args.plan)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
