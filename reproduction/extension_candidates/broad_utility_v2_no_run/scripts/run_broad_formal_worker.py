"""One supervised worker for the bounded, identity-bound breadth-screen batch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "scripts"))
from broad_shared_runtime import runtime_fingerprint, sha_file, verify_package
from plan_broad_utility import ledger_from_registry
from tabpollution.governance.broad_batch_registry import identity_sha256, read_registry, register_unit
from tabpollution.generators.sdv_adapter import create_generator
from tabpollution.studies.broad_formal_batch import (
    build_identity_for_unit, build_plan, calculate_runtime_limits, execute_unit,
    load_and_validate_all, load_json, require_execution_enabled, resolve_config_path,
    validate_runtime_limits,
)

TOTAL_WALL_SECONDS = 28800
UNIT_WALL_SECONDS = 2700
RSS_BYTES = 8 * 1024**3
OUTPUT_BYTES = 64 * 1024**2


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_or_verify(path, value, *, stable_keys=None):
    path = Path(path)
    expected = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if not path.exists():
        write_once(path, value)
        return
    actual = json.loads(path.read_text(encoding="utf-8"))
    keys = stable_keys or tuple(expected)
    if any(actual.get(key) != expected.get(key) for key in keys):
        raise PermissionError(f"existing {path.name} does not match the current plan")


def _timeout(_signum, _frame):
    raise TimeoutError("unit exceeded 2700 second wall limit")


def _authorize_worker(authority_path, package_sha, runtime_sha):
    authority = json.loads(Path(authority_path).read_text(encoding="utf-8"))
    token = os.environ.get("BROAD_FORMAL_GATE_TOKEN", "")
    if (not token or hashlib.sha256(token.encode()).hexdigest() != authority.get("token_sha256")
            or os.getppid() != authority.get("parent_pid")
            or authority.get("package_sha256") != package_sha
            or authority.get("runtime_sha256") != runtime_sha
            or authority.get("run_id") != "broad_utility_formal_candidate_v1"):
        raise PermissionError("formal worker authority mismatch")
    parent_cmd = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().decode(errors="strict").split("\0")
    if (str(PACKAGE / "scripts/run_broad_formal_batch.py") not in parent_cmd
            or "--execute" not in parent_cmd):
        raise PermissionError("formal worker must be spawned by the execute parent")


def run(args):
    if not sys.platform.startswith("linux"):
        raise PermissionError("real-source batch executes only on Linux")
    package = Path(args.package_root).resolve(strict=True)
    if package != PACKAGE.resolve():
        raise PermissionError("worker package root mismatch")
    verify_package(package, args.package_sha256)
    runtime, runtime_sha = runtime_fingerprint(package)
    if runtime_sha != args.runtime_sha256:
        raise RuntimeError("runtime changed between parent and worker")
    _authorize_worker(args.authority, args.package_sha256, args.runtime_sha256)
    config_path = resolve_config_path(package, args.config)
    config = load_json(config_path)
    require_execution_enabled(config)
    plan = build_plan(config)
    limits = calculate_runtime_limits(config, plan)
    validate_runtime_limits(config, plan, total_wall_seconds=args.total_wall_seconds,
                            per_unit_wall_seconds=args.per_unit_wall_seconds)
    TOTAL_WALL_SECONDS = limits["total_wall_seconds"]
    UNIT_WALL_SECONDS = limits["per_unit_wall_seconds"]
    resource.setrlimit(resource.RLIMIT_AS, (RSS_BYTES, RSS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (TOTAL_WALL_SECONDS, TOTAL_WALL_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
    output = Path(args.output).resolve(strict=True)

    catalog = load_json(package / "intake_catalog_v2.json")
    plan_path = output / "broad_formal_plan_resolved.json"
    if not plan_path.is_file():
        raise FileNotFoundError("parent did not stage the resolved plan")
    staged_plan = load_json(plan_path)
    if (staged_plan.get("jobs_sha256") != plan["jobs_sha256"]
            or staged_plan.get("package_sha256") != args.package_sha256
            or staged_plan.get("runtime_sha256") != args.runtime_sha256):
        raise PermissionError("staged plan differs from current frozen plan")

    registry_path = output / "registry.jsonl"
    prepared = load_and_validate_all(config, catalog, args.input_root)
    expected = {}
    for dataset, data in prepared.items():
        for seed in config["seeds"]:
            expected[f"{dataset}|seed={seed}"] = build_identity_for_unit(
                config, plan, dataset, seed, data, args.package_sha256, args.runtime_sha256)
    existing = read_registry(registry_path)
    resume = ledger_from_registry(plan, registry_path, expected)
    if any(value.get("status") == "failed_consumed" for value in resume.values()):
        raise PermissionError("a failed identity is already consumed; no retry")
    accepted_unit_keys = {row["unit_key"] for row in existing if row["status"] == "accepted"}

    preflight = {
        "status": "pass", "dataset_count": len(prepared), "source_rows": {k: v["intake"]["source_rows"] for k, v in prepared.items()},
        "source_sha256": {k: v["source_sha256"] for k, v in prepared.items()},
        "roles_assignment_sha256": {k: v["roles_sha256"] for k, v in prepared.items()},
        "role_counts": {k: v["intake"]["role_counts"] for k, v in prepared.items()},
        "all_inputs_validated_before_any_fit": True, "final_test_evaluated": False}
    write_or_verify(output / "source_role_preflight.json", preflight)

    started = time.monotonic()
    worker_sha = sha_file(__file__)
    for dataset in [x["dataset_id"] for x in config["datasets"]]:
        data = prepared[dataset]
        for seed in config["seeds"]:
            identity = expected[f"{dataset}|seed={seed}"]
            if identity_sha256(identity) in accepted_unit_keys:
                continue
            unit_dir = output / "units" / dataset / f"seed_{seed}"
            if unit_dir.exists():
                raise FileExistsError("unregistered unit output exists; consumed status is ambiguous, no retry")
            unit_dir.mkdir(parents=True, exist_ok=False)
            identity_path = unit_dir / "identity.json"
            aggregate_path = unit_dir / "aggregate.json"
            audit_path = unit_dir / "audit_receipt.json"
            write_once(identity_path, identity)
            write_once(unit_dir / "unit_started.json", {
                "status": "started_consumed", "identity_sha256": identity_sha256(identity),
                "started_unix_time": time.time(), "retry_allowed": False,
                "role_assignment_sha256": identity["roles_assignment_sha256"],
                "final_test_evaluated": False})
            stage = "generator_fit_or_sample"
            fit_calls = 0
            progress_path = output / "progress.jsonl"
            if not progress_path.exists():
                progress_path.write_text("", encoding="utf-8")

            def progress(event):
                nonlocal stage, fit_calls
                stage = "downstream_fit_or_metric"
                fit_calls += 1
                safe = {"dataset": dataset, "seed": seed, "event": event.get("event"),
                    "learner": event.get("learner"), "arm": event.get("arm"),
                    "fit_calls_in_case": event.get("local_fit_calls")}
                with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(safe, sort_keys=True) + "\n")

            old_handler = signal.signal(signal.SIGALRM, _timeout)
            signal.setitimer(signal.ITIMER_REAL, UNIT_WALL_SECONDS)
            try:
                matrix_result = execute_unit(data, seed=seed, plan=plan, config=config,
                    generator_factory=lambda name: create_generator(name, config=config["generators"][name]),
                    progress_callback=progress)
                aggregate = {**matrix_result, "producer_id": worker_sha,
                    "identity": identity, "identity_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
                    "package_sha256": args.package_sha256, "runtime_sha256": args.runtime_sha256,
                    "generator_names": list(config["generators"])}
                write_once(aggregate_path, aggregate)
                stage = "aggregate_audit"
                audit_log = unit_dir / "auditor.log"
                command = [sys.executable, "-B", str(package / "scripts/audit_broad_formal_unit.py"),
                    "--aggregate", str(aggregate_path), "--identity", str(identity_path),
                    "--output", str(audit_path), "--package-sha256", args.package_sha256,
                    "--runtime-sha256", args.runtime_sha256, "--config", str(config_path),
                    "--plan", str(plan_path)]
                with audit_log.open("xb") as log:
                    auditor = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                             "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
                        preexec_fn=lambda: (resource.setrlimit(resource.RLIMIT_CPU, (300, 300)),
                            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3)),
                            resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))))
                    try:
                        audit_code = auditor.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        auditor.kill()
                        auditor.wait()
                        raise TimeoutError("independent auditor exceeded 300 seconds")
                if audit_code != 0:
                    raise ValueError("independent aggregate auditor failed")
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit.get("status") != "pass" or audit.get("independent") is not True:
                    raise ValueError("independent audit receipt did not pass")
                register_unit(registry_path, identity=identity, status="accepted",
                    aggregate_path=aggregate_path, audit_receipt_path=audit_path)
            except BaseException as exc:
                if not any(row.get("unit_key") == hashlib.sha256(canonical(identity).encode()).hexdigest()
                           for row in read_registry(registry_path)):
                    category = stage if stage in {"source_integrity_or_parse", "role_or_schema_gate", "generator_fit_or_sample",
                        "downstream_fit_or_metric", "aggregate_audit", "runtime_or_resource"} else "runtime_or_resource"
                    if category == "downstream_fit_or_metric":
                        category = "downstream_fit_or_metric"
                    try:
                        register_unit(registry_path, identity=identity, status="failed_consumed", failure_category=category)
                    except Exception:
                        pass
                write_once(unit_dir / "failure.json", {"status": "failed_consumed", "stage": stage,
                    "failure_type": type(exc).__name__, "downstream_fit_calls_started": fit_calls,
                    "retry_allowed": False})
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

    final_rows = read_registry(registry_path)
    accepted_records = [row for row in final_rows if row["status"] == "accepted"]
    if len(accepted_records) != plan["unit_count"]:
        raise ValueError("not all planned dataset-seed units have accepted audited results")
    accepted_reports = [load_json(row["aggregate_path"]) for row in accepted_records]
    actual_calls = sum(row["downstream_fit_calls"] for row in accepted_reports)
    invalid_outcomes = sum(row.get("invalid_training_outcomes", 0) for row in accepted_reports)
    budget = plan["downstream_fit_calls"]
    if config.get("protocol_id") == "broad-utility-v1-preparation" and (budget, plan["generator_fit_calls"], plan["unit_count"]) != (756, 27, 9):
        raise ValueError("recovery02 plan budget changed")
    if actual_calls + invalid_outcomes != budget:
        raise ValueError("batch outcome accounting differs from resolved plan")
    summary = {"status": "complete", "formal_inclusion": False, "plan_jobs_sha256": plan["jobs_sha256"],
        "unit_count": plan["unit_count"], "generator_fit_calls": plan["generator_fit_calls"], "downstream_fit_calls": actual_calls,
        "downstream_fit_budget": budget, "invalid_training_outcomes": invalid_outcomes,
        "accepted_units": len(accepted_records), "package_sha256": args.package_sha256,
        "runtime_sha256": args.runtime_sha256, "elapsed_seconds": time.monotonic()-started,
        "real_data_reads": len(prepared), "final_test_evaluated": False, "row_outputs": False,
        "auditors_reaped": True, "children_reaped": True}
    write_or_verify(output / "batch_summary.json", summary, stable_keys=(
        "status", "formal_inclusion", "plan_jobs_sha256", "unit_count", "generator_fit_calls",
        "downstream_fit_calls", "downstream_fit_budget", "invalid_training_outcomes", "accepted_units", "package_sha256", "runtime_sha256",
        "real_data_reads", "final_test_evaluated", "row_outputs"))
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--total-wall-seconds", type=int, required=True)
    p.add_argument("--per-unit-wall-seconds", type=int, required=True)
    p.add_argument("--package-sha256", required=True)
    p.add_argument("--runtime-sha256", required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--authority", type=Path, required=True)
    args = p.parse_args()
    if not args.worker:
        raise PermissionError("worker entry requires the supervised public batch parent")
    run(args)


if __name__ == "__main__":
    main()
