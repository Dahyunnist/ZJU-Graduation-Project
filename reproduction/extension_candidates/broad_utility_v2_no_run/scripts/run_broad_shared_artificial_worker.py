"""Authenticated worker for the fixed, real-SDV/artificial-record acceptance gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Unit tests may inspect the frozen route config on Windows.
    resource = None
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from broad_shared_runtime import canonical, runtime_fingerprint, sha_file, verify_package
from tabpollution.generators.sdv_adapter import create_generator

MAX_SECONDS = 900
MAX_RSS_BYTES = 8 * 1024**3
MAX_FILE_BYTES = 64 * 1024**2
TASKS = ("binary", "multiclass", "regression")
GENERATORS = ("GaussianCopula", "CTGAN", "TVAE", "Bootstrap")
LEARNERS = ("linear", "rf", "histgb", "mlp")
ANALYSIS_SEED = 203101
GENERATOR_CONFIGS = {
    "GaussianCopula": {},
    # Tiny artificial interface smoke only; these are NOT formal-run settings.
    # batch_size is even and divisible by pac, as required by the CTGAN adapter.
    "CTGAN": {"epochs": 2, "batch_size": 40, "pac": 10},
    "TVAE": {"epochs": 2, "batch_size": 40},
}
LEARNER_CONFIG = {"learners": list(LEARNERS), "replacement": [.25, .5],
    "augmentation": [.5, 1.0], "bootstrap_replicates": 100}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def fixed_artificial_factory(name):
    """Use the production factory with frozen, intentionally tiny artificial-only settings."""
    if name == "Bootstrap":
        return create_generator(name)
    return create_generator(name, config=GENERATOR_CONFIGS[name])


def _authorize_worker(authority_path, package_sha, runtime_sha):
    authority = json.loads(Path(authority_path).read_text(encoding="utf-8"))
    token = os.environ.get("BROAD_SHARED_GATE_TOKEN", "")
    if (not token or _sha(token.encode()) != authority.get("token_sha256")
            or os.getppid() != authority.get("parent_pid")
            or authority.get("package_sha256") != package_sha
            or authority.get("runtime_sha256") != runtime_sha
            or authority.get("run_id") != "broad_shared_artificial_fixed_sdv_v1"):
        raise PermissionError("internal worker authority mismatch")
    parent_cmd = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().decode(errors="strict").split("\0")
    if (str(PACKAGE / "scripts" / "run_broad_shared_artificial_gate.py") not in parent_cmd
            or "--execute" not in parent_cmd):
        raise PermissionError("worker must be launched by the authenticated gate parent")
    expected = GENERATOR_CONFIGS["CTGAN"]
    if expected["batch_size"] % expected["pac"] != 0 or expected["batch_size"] % 2:
        raise ValueError("fixed CTGAN artificial config violates batch/pac constraints")


def run(output, package_sha, expected_runtime_sha, authority_path):
    if resource is None or not sys.platform.startswith("linux"):
        raise PermissionError("authenticated artificial worker runs only on Linux")
    verify_package(PACKAGE, package_sha)
    runtime, runtime_sha = runtime_fingerprint(PACKAGE)
    if runtime_sha != expected_runtime_sha:
        raise RuntimeError("runtime changed between parent check and worker start")
    _authorize_worker(authority_path, package_sha, runtime_sha)
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_SECONDS, MAX_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RSS_BYTES, MAX_RSS_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})

    import pandas as pd
    from tabpollution.studies.broad_utility import artificial_records, run_generator_matrix

    started = time.monotonic()
    cases = []
    learned_fits = bootstrap_fits = downstream_fits = 0
    for task in TASKS:
        train, evaluation = artificial_records(task, seed=811, n=240)
        columns = ["x0", "x1", "x2", "x3", "category", "target"]
        source = pd.concat([train, evaluation], ignore_index=True)
        source_sha = _sha(source[columns].to_csv(index=False, lineterminator="\n").encode())
        role_records = ([{"record_id": str(row.record_id), "role": "train"} for row in train.itertuples()]
            + [{"record_id": str(row.record_id), "role": "evaluation"} for row in evaluation.itertuples()])
        role_sha = _sha(canonical(sorted(role_records, key=lambda item: item["record_id"])).encode())
        train_ids = sorted(train.record_id.astype(str).tolist())
        evaluation_ids = sorted(evaluation.record_id.astype(str).tolist())
        class_order = () if task == "regression" else tuple(sorted(train.target.astype(str).unique()))

        result = run_generator_matrix(
            train, evaluation, target="target", numeric=("x0", "x1", "x2", "x3"),
            categorical=("category",), task=task, classes=class_order,
            group_features=("x0", "category"), generator_names=GENERATORS,
            generator_factory=fixed_artificial_factory, seed=ANALYSIS_SEED, learners=LEARNERS,
            bootstrap_reps=100)
        if result["actual_downstream_model_calls"] != 104 or result["reference_control_cache_entries"] != 24:
            raise ValueError("per-task shared utility call budget or control reuse mismatch")
        provenance = result["unit"]["generator_provenance"]
        if any(provenance[name]["config"] != GENERATOR_CONFIGS[name]
               for name in ("GaussianCopula", "CTGAN", "TVAE")):
            raise ValueError("actual SDV adapter config differs from the frozen artificial smoke config")
        case_identity = {"task": task, "source_sha256": source_sha, "roles_assignment_sha256": role_sha,
            "train_sample_ids_sha256": _sha(canonical(train_ids).encode()),
            "evaluation_sample_ids_sha256": _sha(canonical(evaluation_ids).encode()),
            "generator_config_sha256": _sha(canonical({"routes": list(GENERATORS), "configs": GENERATOR_CONFIGS}).encode()),
            "learner_config_sha256": _sha(canonical(LEARNER_CONFIG).encode()),
            "package_sha256": package_sha, "runtime_sha256": runtime_sha, "seed": ANALYSIS_SEED}
        cases.append({**case_identity, "generator_fit_calls": 4,
            "learned_generator_fit_calls": 3, "bootstrap_generator_fit_calls": 1,
            "downstream_fit_calls": result["actual_downstream_model_calls"],
            "reference_control_cache_entries": result["reference_control_cache_entries"],
            "case": result, "final_test_evaluated": False})
        learned_fits += 3
        bootstrap_fits += 1
        downstream_fits += result["actual_downstream_model_calls"]

    if (learned_fits, bootstrap_fits, downstream_fits) != (9, 3, 312):
        raise ValueError("frozen artificial call budget mismatch")
    aggregate = {"status": "complete", "producer_id": sha_file(__file__),
        "protocol": "broad-utility-v1-preparation", "adapter_mode": "fixed_real_sdv_artificial_smoke",
        "formal_inclusion": False, "artificial_only": True, "real_data_reads": 0,
        "row_outputs": False, "final_test_evaluated": False,
        "generator_names": list(GENERATORS), "generator_configs": GENERATOR_CONFIGS,
        "learner_config": LEARNER_CONFIG, "analysis_seed": ANALYSIS_SEED,
        "generator_fits": learned_fits + bootstrap_fits, "learned_generator_fits": learned_fits,
        "bootstrap_generator_fits": bootstrap_fits, "downstream_fits": downstream_fits,
        "runtime": runtime, "runtime_sha256": runtime_sha, "package_sha256": package_sha,
        "cases": cases, "elapsed_seconds": time.monotonic() - started,
        "resource_limits": {"cpu_time_seconds": MAX_SECONDS, "address_space_bytes": MAX_RSS_BYTES,
            "max_file_bytes": MAX_FILE_BYTES, "cpu_threads": 1, "gpu": False}}
    identity = {key: aggregate[key] for key in ("protocol", "adapter_mode", "package_sha256",
        "runtime_sha256", "generator_names", "generator_configs", "learner_config", "analysis_seed")}
    identity["cases"] = [{key: case[key] for key in ("task", "source_sha256", "roles_assignment_sha256",
        "train_sample_ids_sha256", "evaluation_sample_ids_sha256", "generator_config_sha256",
        "learner_config_sha256", "seed")} for case in cases]
    aggregate["identity_sha256"] = _sha(canonical(identity).encode())
    path = Path(output) / "aggregate.json"
    path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "worker_complete", "aggregate_sha256": sha_file(path),
        "learned_generator_fits": learned_fits, "bootstrap_generator_fits": bootstrap_fits,
        "downstream_fits": downstream_fits}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker:
        raise PermissionError("worker entry requires authenticated gate parent")
    run(args.output, args.package_sha256, args.runtime_sha256, args.authority)


if __name__ == "__main__":
    main()
