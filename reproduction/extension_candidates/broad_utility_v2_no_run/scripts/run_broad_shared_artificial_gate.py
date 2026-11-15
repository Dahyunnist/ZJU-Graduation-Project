"""Linux parent supervisor for the fixed real-SDV, artificial-only shared gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Package hash/budget can still be reviewed outside Linux.
    resource = None
import secrets
import signal
import subprocess
import sys
import time

from broad_shared_runtime import runtime_fingerprint, sha_file, verify_package

PACKAGE = Path(__file__).resolve().parents[1]
MAX_SECONDS = 900
MAX_RSS_BYTES = 8 * 1024**3
MAX_OUTPUT_BYTES = 64 * 1024**2
GENERATOR_FITS = 12  # 9 learned SDV fits + 3 Bootstrap fits.
DOWNSTREAM_FITS = 312


def _rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _output_bytes(output, log_path):
    total = log_path.stat().st_size if log_path.exists() else 0
    if output.exists():
        total += sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    return total


def _run_child(command, *, output, log_path, env, deadline, sidecar_paths=()):
    log = log_path.open("xb")
    def set_child_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_SECONDS, MAX_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (MAX_RSS_BYTES, MAX_RSS_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, preexec_fn=set_child_limits)
    peak = parent_peak = 0
    try:
        while proc.poll() is None:
            peak = max(peak, _rss_bytes(proc.pid))
            parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            if time.monotonic() > deadline:
                raise TimeoutError("shared artificial gate exceeded 900 second wall limit")
            if peak + parent_peak > MAX_RSS_BYTES:
                raise MemoryError("combined parent/child RSS exceeded 8 GiB")
            observed_output = _output_bytes(output, log_path) + sum(
                path.stat().st_size for path in sidecar_paths if path.exists())
            if observed_output > MAX_OUTPUT_BYTES:
                raise MemoryError("combined output and log exceeded 64 MiB")
            time.sleep(.2)
        log.flush()
    except BaseException:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()
        raise
    finally:
        log.close()
    peak = max(peak, _rss_bytes(proc.pid))
    parent_peak = max(parent_peak, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    return proc.returncode, peak, parent_peak


def execute(package_sha, output):
    started = time.monotonic()
    if not sys.platform.startswith("linux"):
        raise PermissionError("fixed real-adapter artificial gate executes only on Linux")
    if resource is None:
        raise RuntimeError("Linux resource controls are unavailable")
    verify_package(PACKAGE, package_sha)
    runtime, runtime_sha = runtime_fingerprint(PACKAGE)  # fails closed; never installs/upgrades.
    output = Path(output)
    if output.is_symlink() or output.exists() or output.resolve().parent != PACKAGE.parent.resolve():
        raise FileExistsError("use a new non-symlink sibling output directory")
    output.mkdir(parents=False)
    log_path = output.with_suffix(".log")
    token = secrets.token_hex(32)
    authority_path = output / "worker_authority.json"
    authority_path.write_text(json.dumps({"package_sha256": package_sha,
        "runtime_sha256": runtime_sha, "parent_pid": os.getpid(),
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "run_id": "broad_shared_artificial_fixed_sdv_v1"}, sort_keys=True) + "\n", encoding="utf-8")
    os.nice(19)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    env = {**os.environ, "PYTHONPATH": str(PACKAGE / "src"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
        "BROAD_SHARED_GATE_TOKEN": token}
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    try:
        code, worker_peak, parent_peak = _run_child([sys.executable, "-B",
            str(PACKAGE / "scripts" / "run_broad_shared_artificial_worker.py"),
            "--worker", "--package-sha256", package_sha, "--runtime-sha256", runtime_sha,
            "--output", str(output), "--authority", str(authority_path)],
            output=output, log_path=log_path, env=env, deadline=started + MAX_SECONDS)
        authority_path.unlink(missing_ok=True)
        if code:
            raise RuntimeError("artificial worker failed; no retry: " + log_path.read_text(errors="replace")[-2000:])
        aggregate = output / "aggregate.json"
        if not aggregate.is_file() or _output_bytes(output, log_path) > MAX_OUTPUT_BYTES:
            raise ValueError("worker aggregate missing or total output exceeds limit")
        audit_path = output / "audit_receipt.json"
        audit_log = output.with_suffix(".audit.log")
        auditor_env = {**env, "BROAD_SHARED_GATE_TOKEN": ""}
        audit_code, audit_peak, audit_parent_peak = _run_child([sys.executable, "-B",
            str(PACKAGE / "scripts" / "audit_broad_shared_artificial.py"),
            "--aggregate", str(aggregate), "--output", str(audit_path),
            "--package-sha256", package_sha, "--runtime-sha256", runtime_sha],
            output=output, log_path=audit_log, env=auditor_env,
            deadline=min(started + MAX_SECONDS, time.monotonic() + 60), sidecar_paths=(log_path,))
        if audit_code:
            raise RuntimeError("independent auditor failed; no retry: " + audit_log.read_text(errors="replace")[-2000:])
        peak = max(worker_peak + parent_peak, audit_peak + audit_parent_peak)
        total = _output_bytes(output, log_path) + (audit_log.stat().st_size if audit_log.exists() else 0)
        if total > MAX_OUTPUT_BYTES:
            raise MemoryError("aggregate, audit and logs exceed 64 MiB")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (audit.get("status") != "pass" or audit.get("independent") is not True
                or audit.get("aggregate_sha256") != sha_file(aggregate)
                or audit.get("package_sha256") != package_sha
                or audit.get("runtime_sha256") != runtime_sha):
            raise ValueError("independent artificial audit did not pass")
        receipt = {"status": "artificial_real_adapter_smoke_accepted", "formal_inclusion": False,
            "package_sha256": package_sha, "runtime_sha256": runtime_sha,
            "aggregate_sha256": sha_file(aggregate), "audit_receipt_sha256": sha_file(audit_path),
            "generator_fits": GENERATOR_FITS, "learned_generator_fits": 9,
            "bootstrap_generator_fits": 3, "downstream_fits": DOWNSTREAM_FITS,
            "elapsed_seconds": time.monotonic() - started, "peak_combined_rss_bytes": peak,
            "child_cpu_seconds": (resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - cpu_before),
            "max_output_bytes_observed": total, "child_reaped": True, "auditor_reaped": True,
            "real_data_reads": 0, "final_test_evaluated": False,
            "runtime_versions": {name: row["version"] for name, row in runtime["dependency_closure"].items()},
            "limits": {"wall_seconds": MAX_SECONDS, "rss_bytes": MAX_RSS_BYTES,
                       "output_bytes": MAX_OUTPUT_BYTES, "cpu_threads": 1, "gpu": False}}
        receipt_path = output / "gate_receipt.json"
        for _ in range(3):
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            measured_total = _output_bytes(output, log_path) + (audit_log.stat().st_size if audit_log.exists() else 0)
            if measured_total == receipt["max_output_bytes_observed"]:
                break
            receipt["max_output_bytes_observed"] = measured_total
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if _output_bytes(output, log_path) + (audit_log.stat().st_size if audit_log.exists() else 0) > MAX_OUTPUT_BYTES:
            raise MemoryError("final aggregate, audit, receipt and logs exceed 64 MiB")
        print(json.dumps(receipt))
    finally:
        authority_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        verify_package(PACKAGE, args.package_sha256)
        print(json.dumps({"status": "NO-RUN", "adapter_mode": "fixed_real_sdv_artificial_smoke",
            "learned_generator_fits": 9, "bootstrap_generator_fits": 3,
            "downstream_fits": DOWNSTREAM_FITS,
            "generators": ["GaussianCopula", "CTGAN", "TVAE", "Bootstrap"],
            "tasks": ["binary", "multiclass", "regression"], "real_data_reads": 0}))
        return
    if args.output is None:
        parser.error("--output is required with --execute")
    execute(args.package_sha256, args.output)


if __name__ == "__main__":
    main()
