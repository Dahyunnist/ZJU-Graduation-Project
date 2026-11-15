"""Public offline runner for hash-pinned broad-utility candidate batches.

Default behavior is plan-only. Execution reads only already-cached, hash-pinned
archives and immutable role CSVs; it never downloads sources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "scripts"))
from broad_shared_runtime import canonical, runtime_fingerprint, sha_file, verify_package
from tabpollution.studies.broad_formal_batch import (
    build_plan, calculate_runtime_limits, load_json, require_execution_enabled, resolve_config_path,
)

TOTAL_WALL_SECONDS = 8 * 60 * 60
RSS_BYTES = 8 * 1024**3
OUTPUT_BYTES = 64 * 1024**2


def _load_shared_supervisor(total_wall_seconds):
    path = PACKAGE / "scripts" / "run_broad_shared_artificial_gate.py"
    spec = importlib.util.spec_from_file_location("shared_gate_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MAX_SECONDS = total_wall_seconds
    module.MAX_RSS_BYTES = RSS_BYTES
    module.MAX_OUTPUT_BYTES = OUTPUT_BYTES
    return module


def _sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _bytes_under(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def verify_external_commitments(package, expected_package_sha, expected_runtime_sha):
    if not expected_package_sha or not expected_runtime_sha:
        raise PermissionError("execute requires externally approved package and runtime SHA-256 values")
    actual_package_sha = sha_file(Path(package) / "package_manifest.json")
    if actual_package_sha != expected_package_sha:
        raise PermissionError("frozen package differs from the external package commitment")
    verify_package(package, expected_package_sha)
    runtime, actual_runtime_sha = runtime_fingerprint(package)
    if actual_runtime_sha != expected_runtime_sha:
        raise PermissionError("Linux runtime differs from the external runtime commitment")
    return actual_package_sha, runtime, actual_runtime_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the fixed source-backed exploratory batch")
    parser.add_argument("--resume", action="store_true", help="resume only exact accepted identities in an existing output")
    parser.add_argument("--config", type=Path, help="candidate config inside the verified package")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-package-sha256")
    parser.add_argument("--expected-runtime-sha256")
    args = parser.parse_args()
    config_path = resolve_config_path(PACKAGE, args.config)
    config = load_json(config_path)
    plan = build_plan(config)
    limits = calculate_runtime_limits(config, plan)
    total_wall_seconds = limits["total_wall_seconds"]
    per_unit_wall_seconds = limits["per_unit_wall_seconds"]
    summary = {"status": "NO-RUN", "protocol_id": config["protocol_id"], "config_path": str(config_path),
        "phase": plan["phase"], "dataset_count": len(config["datasets"]),
        "datasets": [x["dataset_id"] for x in config["datasets"]], "seeds": config["seeds"],
        "unit_count": plan["unit_count"], "jobs_sha256": plan["jobs_sha256"],
        "generator_fit_calls": plan["generator_fit_calls"],
        "planned_outcomes": plan["planned_outcomes"], "downstream_fit_calls": plan["downstream_fit_calls"],
        "resource_limits": {**limits, "cpu_cores": config.get("budget", {}).get("cpu_cores", 1),
            "address_space_bytes": config.get("budget", {}).get("address_space_bytes", RSS_BYTES),
            "aggregate_output_bytes": config.get("budget", {}).get("aggregate_output_bytes", OUTPUT_BYTES)},
        "execution_enabled": config.get("execution", {}).get("enabled", False),
        "source_rows_read": 0, "final_test_evaluated": False,
        "input_root_expected": config["source_cache"]["input_root"],
        "role_rebuild": config["source_cache"]["role_algorithm"],
        "source_role_gate": "all source hashes, approved roles and role/class checks must pass before any fit"}
    if not args.execute:
        if args.resume:
            parser.error("--resume requires --execute")
        print(json.dumps(summary))
        return
    require_execution_enabled(config)
    import resource
    if args.input_root is None or args.output is None:
        parser.error("--execute requires --input-root and --output")
    if not sys.platform.startswith("linux"):
        raise PermissionError("source-backed batch execution requires Linux")
    if args.input_root.is_symlink() or not args.input_root.is_dir():
        raise PermissionError("input root must be an existing non-symlink directory")
    package_sha, runtime, runtime_sha = verify_external_commitments(
        PACKAGE, args.expected_package_sha256, args.expected_runtime_sha256)
    if args.output.is_symlink():
        raise PermissionError("output must not be a symlink")
    output = args.output.resolve()
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError("--resume requires an existing output directory")
        staged = json.loads((output / "broad_formal_plan_resolved.json").read_text(encoding="utf-8"))
        if (staged.get("jobs_sha256") != plan["jobs_sha256"]
                or staged.get("package_sha256") != package_sha
                or staged.get("runtime_sha256") != runtime_sha):
            raise PermissionError("resume plan/package identity differs; refuse to skip or retry")
    else:
        if output.exists():
            raise FileExistsError("new run output already exists; refusing overwrite")
        output.mkdir(parents=True, exist_ok=False)
        plan["package_sha256"] = package_sha
        plan["runtime_sha256"] = runtime_sha
        plan_text = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (output / "broad_formal_plan_resolved.json").write_text(plan_text, encoding="utf-8")
    if args.resume:
        plan = staged

    supervisor = _load_shared_supervisor(total_wall_seconds)
    start = time.monotonic()
    cpu_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    log_path = output.with_name(output.name + f".formal_worker_{time.time_ns()}.log")
    prior_logs = tuple(output.parent.glob(output.name + ".formal_worker_*.log"))
    token = secrets.token_hex(32)
    authority = output / "worker_authority.json"
    authority_digest = hashlib.sha256(token.encode()).hexdigest()
    if authority.exists():
        raise FileExistsError("worker authority file exists; refusing to resume ambiguous process state")
    authority.write_text(json.dumps({"parent_pid": os.getpid(), "package_sha256": package_sha,
        "runtime_sha256": runtime_sha, "token_sha256": authority_digest,
        "run_id": "broad_utility_formal_candidate_v1"}, sort_keys=True) + "\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PACKAGE / "src"), str(PACKAGE / "scripts")]),
        "BROAD_FORMAL_GATE_TOKEN": token, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    try:
        os.nice(19)
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        code, child_peak, parent_peak = supervisor._run_child(
            [sys.executable, "-B", str(PACKAGE / "scripts/run_broad_formal_worker.py"),
             "--worker", "--package-root", str(PACKAGE), "--config", str(config_path),
             "--total-wall-seconds", str(total_wall_seconds),
             "--per-unit-wall-seconds", str(per_unit_wall_seconds), "--package-sha256", package_sha,
             "--runtime-sha256", runtime_sha, "--input-root", str(args.input_root.resolve(strict=True)),
             "--output", str(output),
             "--authority", str(authority)],
            output=output, log_path=log_path, env=env, deadline=start + total_wall_seconds,
            sidecar_paths=prior_logs)
        if code != 0:
            failure = {"status": "failed_consumed_or_preflight_blocked", "worker_exit_code": code,
                "elapsed_seconds": time.monotonic() - start,
                "log_path": str(log_path), "package_sha256": package_sha, "runtime_sha256": runtime_sha}
            (output / "gate_failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"formal batch worker failed; no automatic retry; see {log_path}")
        batch = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
        cpu_end = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        receipt = {**batch, "status": "accepted_exploratory_breadth_screen",
            "formal_inclusion": False, "elapsed_seconds": time.monotonic() - start,
            "peak_combined_rss_bytes": child_peak + parent_peak,
            "total_output_bytes": _bytes_under(output) + sum(p.stat().st_size for p in (*prior_logs, log_path) if p.exists()),
            "worker_reaped": True, "auditors_reaped": batch.get("auditors_reaped") is True,
            "child_cpu_seconds": cpu_end - cpu_start,
            "limits": {"wall_seconds": total_wall_seconds, "per_unit_wall_seconds": per_unit_wall_seconds,
                "rss_bytes": RSS_BYTES, "output_bytes": OUTPUT_BYTES, "cpu_threads": 1, "gpu": False}}
        if receipt["total_output_bytes"] > OUTPUT_BYTES:
            raise MemoryError("aggregate output and logs exceeded 64 MiB")
        receipt_path = output / "batch_receipt.json"
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            keys = ("status", "formal_inclusion", "plan_jobs_sha256", "accepted_units",
                    "downstream_fit_calls", "downstream_fit_budget", "invalid_training_outcomes",
                    "generator_fit_calls", "package_sha256", "runtime_sha256")
            if any(prior.get(key) != receipt.get(key) for key in keys):
                raise PermissionError("existing completion receipt differs from the verified batch")
            print(json.dumps(prior))
            return
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt))
    finally:
        authority.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
