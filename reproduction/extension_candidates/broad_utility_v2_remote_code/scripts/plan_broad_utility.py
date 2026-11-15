"""Build a deduplicated aggregate experiment schedule; never opens dataset rows."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


FAILURE_CATEGORIES = {
    "source": "source_integrity_or_parse",
    "split": "role_or_schema_gate",
    "generator": "generator_fit_or_sample",
    "utility": "downstream_fit_or_metric",
    "audit": "aggregate_audit",
    "runtime": "runtime_or_resource",
}


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def schedule(protocol, datasets, *, stage="pilot"):
    cfg = protocol[stage]
    seeds = cfg["generator_seeds"]
    generators = cfg["generators"] if "generators" in cfg else protocol["generators_main"]
    learners = protocol["learners"]
    replacements = protocol["replacement_doses"]
    augmentations = protocol["augmentation_doses"]
    jobs = []
    controls = ["TRTR", "bootstrap_TSTR"] + [f"bootstrap_replace_{x:g}" for x in replacements] + [f"bootstrap_augment_{x:g}" for x in augmentations]
    synthetic = ["TSTR"] + [f"replace_{x:g}" for x in replacements] + [f"augment_{x:g}" for x in augmentations]
    for dataset in datasets:
        for seed in seeds:
            for learner in learners:
                for arm in controls:
                    jobs.append({"dataset": dataset, "seed": seed, "learner": learner,
                                 "generator": "shared_real_or_resampling", "arm": arm})
                for generator in generators:
                    for arm in synthetic:
                        jobs.append({"dataset": dataset, "seed": seed, "learner": learner,
                                     "generator": generator, "arm": arm})
    keys = [json.dumps(x, sort_keys=True) for x in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate fit key")
    generator_config = {"generators": list(generators),
        "per_generator": protocol.get("generator_configs", {}),
        "sample_size": "training_role_row_count"}
    learner_config = {"learners": list(learners),
        "per_learner": protocol.get("learner_configs", {}),
        "replacement_doses": list(replacements), "augmentation_doses": list(augmentations),
        "bootstrap_replicates": cfg.get("bootstrap_replicates", 2000),
        "primary_metric": protocol.get("primary_metric", {})}
    return {"stage": stage, "status": "PLAN_ONLY", "dataset_count": len(datasets),
            "downstream_fit_calls": len(jobs), "independent_generator_fit_calls": len(datasets)*len(seeds)*len(generators),
            "max_downstream_fit_calls": len(jobs), "max_generator_fit_calls": len(datasets)*len(seeds)*len(generators),
            "max_attempts_per_unit": 1,
            "bootstrap_fit_calls": 0, "bootstrap_note": "empirical row draws; not independent learned generator fits",
            "reference_controls_reused_across_generators": True,
            "generator_config": generator_config, "generator_config_sha256": config_hash(generator_config),
            "learner_config": learner_config, "learner_config_sha256": config_hash(learner_config),
            "source_role_policy": "freeze and reuse one assignment per dataset across all analysis seeds; seeds vary generator/sample/learner randomness only",
            "evaluation_exposure_policy": "recovery02 evaluation is development-exposed; never relabel it as a new final_test; final_test remains sealed and unevaluated",
            "jobs_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(), "jobs": jobs}


def apply_ledger(plan, ledger):
    """Resume only unconsumed units; accepted units skip and consumed failures stay closed.

    Ledger maps each job's canonical key to {status, failure_category?, result_sha256?}.
    A missing entry is pending. No automatic retry or failure replacement is allowed.
    """
    known = {json.dumps(job, sort_keys=True): job for job in plan["jobs"]}
    if set(ledger) - set(known):
        raise ValueError("ledger contains jobs outside this immutable plan")
    runnable, accepted, failed = [], [], []
    for key, job in known.items():
        entry = ledger.get(key)
        if entry is None or entry.get("status") == "pending":
            runnable.append(job)
        elif entry.get("status") == "accepted":
            if not entry.get("result_sha256"):
                raise ValueError("accepted job requires result_sha256")
            accepted.append(job)
        elif entry.get("status") == "failed_consumed":
            if entry.get("failure_category") not in FAILURE_CATEGORIES.values():
                raise ValueError("consumed failure requires a declared failure category")
            failed.append(job)
        else:
            raise ValueError("unknown ledger status; consumed work is never inferred as pending")
    return {"status": "RESUME_PLAN_ONLY", "runnable_jobs": runnable,
            "accepted_skipped": len(accepted), "consumed_failures_skipped": len(failed),
            "runnable_fit_calls": len(runnable), "max_attempts_per_unit": 1,
            "accepted_keys": [json.dumps(j, sort_keys=True) for j in accepted],
            "consumed_failure_keys": [json.dumps(j, sort_keys=True) for j in failed]}


def ledger_from_registry(plan, registry_path, expected_identities):
    """Use the registry verifier; never trust a bare accepted status or digest string."""
    import sys
    source = Path(__file__).resolve().parents[1] / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from tabpollution.governance.broad_batch_registry import ledger_from_registry as verified_ledger
    return verified_ledger(plan, registry_path, expected_identities)


def readiness_rows(catalog, generators=("Bootstrap", "GaussianCopula", "CTGAN", "TVAE"),
                   learners=("linear", "rf", "histgb", "mlp")):
    """Development evidence matrix; recovery02 proves a narrow one-seed three-table slice."""
    validated = {"rice_cammeo", "dry_bean", "wine_white"}
    split_adapters = {
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
    rows = []
    for item in catalog["records"]:
        source_status = item.get("status", "blocked")
        in_validated_slice = item["dataset_id"] in validated
        blocker = ("single-seed GC development check only; expand to three seeds and CTGAN/TVAE"
                   if in_validated_slice else item.get("blocked_reason") or
                   "metadata only; source archive, exact split and fresh role hashes unverified")
        row = {"dataset_id": item["dataset_id"], "task_type": item["task_type"],
               "source_metadata_status": source_status,
               "source_parser_adapter": f"{item.get('source_format', 'unknown')}" +
                   ("_multi_member_preserve_partition" if "+" in item.get("member", "") else "_single_member"),
               "split_adapter": "recovery02_frozen_roles" if in_validated_slice else split_adapters.get(item["dataset_id"], "unmapped_blocked"),
               "source_and_split_evidence": "recovery02_one_seed_audit_pass" if in_validated_slice else "blocked_unverified",
               "GC_common_utility": "recovery02_one_seed_44_fits_complete" if in_validated_slice else "blocked_source_split",
               "CTGAN_common_utility": "pending_shared_runner_and_runtime_gate",
               "TVAE_common_utility": "pending_shared_runner_and_runtime_gate",
               "overall_status": "development_slice_validated_expand_scope" if in_validated_slice else "blocked",
               "blocker": blocker}
        for gen in generators:
            row[f"generator:{gen}"] = ("one-seed development evidence" if in_validated_slice and gen == "GaussianCopula"
                                         else "needs shared-runner/runtime validation")
        for learner in learners:
            row[f"learner:{learner}"] = ("one-seed development evidence" if in_validated_slice
                                           else "implemented; source acceptance pending")
        rows.append(row)
    return rows


def write_readiness_csv(catalog_path, output_path):
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    rows = readiness_rows(catalog)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refuse to overwrite readiness matrix")
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--stage", choices=("pilot", "breadth_screen", "formal"), default="pilot")
    p.add_argument("--dataset-ids", nargs="+")
    p.add_argument("--accepted-registry", type=Path,
                   help="verified broad batch registry; accepted outcomes are rehashed and independently audited")
    p.add_argument("--expected-identities", type=Path,
                   help="JSON mapping dataset|seed to full source/roles/sample/config/package/runtime identity")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding="utf-8"))
    datasets = a.dataset_ids or protocol["pilot_datasets"]
    if a.output.exists():
        raise FileExistsError("refuse to overwrite schedule")
    result = schedule(protocol, datasets, stage=a.stage)
    if bool(a.accepted_registry) != bool(a.expected_identities):
        raise ValueError("registry resume requires both --accepted-registry and --expected-identities")
    if a.accepted_registry:
        expected = json.loads(a.expected_identities.read_text(encoding="utf-8"))
        result["resume"] = apply_ledger(result, ledger_from_registry(result, a.accepted_registry, expected))
    result["failure_categories"] = FAILURE_CATEGORIES
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "jobs"}))
