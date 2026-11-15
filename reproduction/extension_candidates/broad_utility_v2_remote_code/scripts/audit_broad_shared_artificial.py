"""Independent audit of fixed-real-adapter artificial-only generator/learner aggregates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(aggregate_path, output_path, package_sha, runtime_sha):
    report = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    tasks = {"binary", "multiclass", "regression"}
    generators = {"GaussianCopula", "CTGAN", "TVAE", "Bootstrap"}
    if (report.get("status") != "complete" or report.get("adapter_mode") != "fixed_real_sdv_artificial_smoke"
            or report.get("formal_inclusion") is not False or report.get("artificial_only") is not True
            or report.get("real_data_reads") != 0 or report.get("row_outputs") is not False
            or report.get("final_test_evaluated") is not False or report.get("package_sha256") != package_sha
            or report.get("runtime_sha256") != runtime_sha or set(report.get("generator_names", [])) != generators):
        raise ValueError("fixed artificial adapter scope mismatch")
    configs = report.get("generator_configs")
    if configs != {"GaussianCopula": {}, "CTGAN": {"epochs": 2, "batch_size": 40, "pac": 10},
                   "TVAE": {"epochs": 2, "batch_size": 40}}:
        raise ValueError("frozen smoke configurations differ")
    if configs["CTGAN"]["batch_size"] % configs["CTGAN"]["pac"] != 0:
        raise ValueError("CTGAN batch_size must be divisible by pac")
    runtime_versions = report.get("runtime", {}).get("dependency_closure", {})
    for name, expected in {"sdv": "1.37.3", "sdmetrics": "0.28.0"}.items():
        if runtime_versions.get(name, {}).get("version") != expected:
            raise ValueError(f"runtime {name} version mismatch")
    if {case.get("task") for case in report.get("cases", [])} != tasks or len(report["cases"]) != 3:
        raise ValueError("task roster mismatch")
    learned = bootstrap = downstream = 0
    for case in report["cases"]:
        if (case.get("generator_fit_calls") != 4 or case.get("learned_generator_fit_calls") != 3
                or case.get("bootstrap_generator_fit_calls") != 1 or case.get("downstream_fit_calls") != 104
                or case.get("final_test_evaluated") is not False):
            raise ValueError("per-task generator/downstream budget differs")
        inner = case.get("case", {})
        if (inner.get("row_outputs") is not False or inner.get("actual_downstream_model_calls") != 104
                or inner.get("reference_control_cache_entries") != 24
                or set(inner.get("cases", {})) != generators):
            raise ValueError("shared-control cache or aggregate schema mismatch")
        calls = sum(row.get("model_calls", -1) for row in inner["cases"].values())
        if calls != 104 or any(row.get("row_outputs") is not False for row in inner["cases"].values()):
            raise ValueError("nested model call sum/output boundary mismatch")
        learned += case["learned_generator_fit_calls"]
        bootstrap += case["bootstrap_generator_fit_calls"]
        downstream += case["downstream_fit_calls"]
    if (learned, bootstrap, downstream) != (9, 3, 312):
        raise ValueError("overall frozen call budget mismatch")
    identity = {key: report[key] for key in ("protocol", "adapter_mode", "package_sha256",
        "runtime_sha256", "generator_names", "generator_configs", "learner_config", "analysis_seed")}
    identity["cases"] = [{key: case[key] for key in ("task", "source_sha256", "roles_assignment_sha256",
        "train_sample_ids_sha256", "evaluation_sample_ids_sha256", "generator_config_sha256",
        "learner_config_sha256", "seed")} for case in report["cases"]]
    identity_sha = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if report.get("identity_sha256") != identity_sha:
        raise ValueError("source/roles/sample/config/package/runtime identity mismatch")
    receipt = {"status": "pass", "independent": True,
        "auditor_id": f"fixed_sdv_artificial_audit:{sha(__file__)}",
        "aggregate_sha256": sha(aggregate_path), "identity_sha256": identity_sha,
        "package_sha256": package_sha, "runtime_sha256": runtime_sha,
        "learned_generator_fits_checked": learned, "bootstrap_generator_fits_checked": bootstrap,
        "downstream_fits_checked": downstream, "task_types_checked": sorted(tasks),
        "real_data_reads": 0, "final_test_evaluated": False, "row_outputs": False}
    Path(output_path).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()
    audit(args.aggregate, args.output, args.package_sha256, args.runtime_sha256)


if __name__ == "__main__":
    main()
