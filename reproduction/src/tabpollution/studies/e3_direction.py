"""Frozen E3 gate A: score direction -> PACC -> governance trigger.

Target synthetic labels are read only to evaluate AUROC. Deployable policies
are fitted with source labels and, optionally, target *real-only* anchors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tabpollution.detectors.classical import C2STDetector
from tabpollution.governance.data import RegistrySource, exact_mixture
from tabpollution.governance.pipeline import _collect, _split
from tabpollution.studies.e1 import _seed, sha256_file
from tabpollution.studies.e2 import _policy


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    part.replace(path)


def _config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"study_id", "tables", "source_generator", "target_generator", "seeds",
                "detector", "target_real_anchor_size", "target_fpr", "bag_size",
                "prevalences", "bags_per_rate", "primary_quantifier",
                "decision_prevalence_threshold"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError("E3 direction config fields do not match frozen protocol")
    if cfg["detector"] != "c2st_lr" or cfg["primary_quantifier"] != "pacc":
        raise ValueError("E3 direction methods are fixed to C2ST-LR and PACC")
    if cfg["source_generator"] != "CTGAN" or cfg["target_generator"] != "TVAE":
        raise ValueError("E3 direction source/target generators are fixed")
    if cfg["tables"] != ["adult", "abalone"] or cfg["seeds"] != [2026, 2027, 2028]:
        raise ValueError("E3 direction tables/seeds changed; version the protocol")
    if int(cfg["bag_size"]) <= 0 or int(cfg["bags_per_rate"]) <= 0:
        raise ValueError("E3 bag construction must be positive")
    if list(map(float, cfg["prevalences"])) != [.05, .10, .25] or float(cfg["target_fpr"]) != .05:
        raise ValueError("E3 prevalence/FPR values changed; version the protocol")
    if float(cfg["decision_prevalence_threshold"]) != .10:
        raise ValueError("E3 decision threshold changed; version the protocol")
    return cfg


def preflight(config_path: Path, registry_path: Path, replica_manifest_path: Path) -> dict[str, Any]:
    config_path, registry_path = config_path.resolve(), registry_path.resolve()
    replica_manifest_path = replica_manifest_path.resolve()
    cfg = _config(config_path)
    manifest = json.loads(replica_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("Independent pool is incomplete or its registry hash changed")
    if manifest.get("plan", {}).get("replicate_id") != "e3-independent-pools-r3031":
        raise ValueError("Unexpected independent pool identity")
    complete = {(r["table_id"], r["generator"]) for r in manifest["runs"] if r["status"] == "complete"}
    required = {(table, gen) for table in cfg["tables"] for gen in
                (cfg["source_generator"], cfg["target_generator"])}
    if complete != required:
        raise ValueError(f"Expected exactly {sorted(required)} generated pools, got {sorted(complete)}")
    registry = pd.read_csv(registry_path)
    if set(zip(registry.table_id, registry.generator)) != required:
        raise ValueError("Independent registry has unexpected table/generator entries")
    for row in manifest["runs"]:
        matching = registry.loc[(registry.table_id == row["table_id"]) &
                                (registry.generator == row["generator"])]
        if len(matching) != 1:
            raise ValueError("Duplicate/missing independent registry row")
        synthetic = (registry_path.parent / str(matching.synthetic_path.iloc[0])).resolve()
        if sha256_file(synthetic) != row["synthetic_sha256"]:
            raise ValueError("Independent synthetic file hash changed")
        real = (registry_path.parent / str(matching.real_path.iloc[0])).resolve()
        if sha256_file(real) != row["real_sha256"]:
            raise ValueError("Fixed real evaluation pool hash changed")
    return {"study_id": cfg["study_id"], "config_sha256": sha256_file(config_path),
            "replica_manifest_sha256": sha256_file(replica_manifest_path),
            "registry_sha256": sha256_file(registry_path),
            "cases": len(cfg["tables"]) * len(cfg["seeds"]),
            "expected_bags": len(cfg["tables"]) * len(cfg["seeds"]) *
            len(cfg["prevalences"]) * int(cfg["bags_per_rate"])}


def run(config_path: Path, registry_path: Path, replica_manifest_path: Path,
        output_root: Path) -> dict[str, Any]:
    context = preflight(config_path, registry_path, replica_manifest_path)
    context["implementation_sha256"] = sha256_file(Path(__file__))
    cfg = _config(config_path)
    source = RegistrySource(registry_path)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    root_context = output_root / "context.json"
    if root_context.exists():
        if json.loads(root_context.read_text(encoding="utf-8")) != context:
            raise ValueError("E3 direction output belongs to different code or inputs")
    else:
        if any(output_root.iterdir()):
            raise FileExistsError("Nonempty E3 direction output has no context")
        _atomic_json(root_context, context)
    completed = 0
    for table_id in cfg["tables"]:
        table = source.table(table_id)
        for seed in cfg["seeds"]:
            case_dir = output_root / table_id / f"seed{seed}"
            final = case_dir / "complete.json"
            if final.exists():
                old = json.loads(final.read_text(encoding="utf-8"))
                if old["context"] != context or old["status"] != "complete":
                    raise ValueError(f"Stale E3 direction case {case_dir}")
                for name, digest in old["files_sha256"].items():
                    if sha256_file(case_dir / name) != digest:
                        raise ValueError(f"E3 direction result changed after completion: {case_dir / name}")
                completed += 1
                continue
            if case_dir.exists() and any(case_dir.iterdir()):
                raise FileExistsError(f"Partial E3 case needs inspection: {case_dir}")
            case_dir.mkdir(parents=True, exist_ok=True)
            real_splits = _split(table.real, seed)
            synth_splits = _split(table.synthetic[cfg["target_generator"]], seed + 31)
            detector_train, train_labels = _collect(source, (table_id,),
                (cfg["source_generator"],), "detector_train", seed)
            detector_val, val_labels = _collect(source, (table_id,),
                (cfg["source_generator"],), "detector_val", seed)
            target_val, target_labels = _collect(source, (table_id,),
                (cfg["target_generator"],), "detector_val", seed)
            tune, cal = train_test_split(np.arange(len(detector_val)), test_size=.5,
                random_state=_seed(seed, "e2_source_cal"), stratify=val_labels)
            anchor = real_splits["source"].iloc[:int(cfg["target_real_anchor_size"])]
            if len(anchor) != cfg["target_real_anchor_size"]:
                raise ValueError(f"Insufficient target real-only anchor for {table_id}")
            detector = C2STDetector("lr", seed=seed).fit(detector_train, train_labels,
                detector_val.iloc[tune], val_labels[tune])
            source_raw = detector.predict_score(detector_val.iloc[cal])
            target_raw = detector.predict_score(target_val)
            anchor_raw = detector.predict_score(anchor)
            policies: dict[str, dict[str, Any]] = {}
            policy_errors = {}
            for policy_name in ("source_only", "target_real_anchor"):
                try:
                    policies[policy_name] = _policy(policy_name, source_raw, val_labels[cal],
                        target_raw, target_labels, anchor_raw, seed, float(cfg["target_fpr"]))
                except ValueError as exc:
                    policy_errors[policy_name] = str(exc)
            source_auc = float(roc_auc_score(val_labels[cal], source_raw))
            target_auc = float(roc_auc_score(target_labels, target_raw))
            diagnostics = {name: {
                "platt_slope": float(policy["calibrator"].model.coef_[0, 0]),
                "platt_intercept": float(policy["calibrator"].model.intercept_[0]),
                "threshold": float(policy["threshold"]),
                "reference_fpr": float(policy["reference_fpr"]),
                "soft_tpr": float(policy["quantifiers"]["pacc"].state["soft_tpr"]),
                "soft_fpr": float(policy["quantifiers"]["pacc"].state["soft_fpr"]),
            } for name, policy in policies.items()}
            _atomic_json(case_dir / "policy_diagnostics.json", diagnostics)
            pd.DataFrame({"partition": ["source_cal"] * len(source_raw) +
                          ["target_eval"] * len(target_raw),
                          "label": np.r_[val_labels[cal], target_labels],
                          "raw_score": np.r_[source_raw, target_raw]}).to_csv(
                              case_dir / "validation_scores.csv.gz", index=False, compression="gzip")
            bag_rows = []
            score_rows = []
            for prevalence in cfg["prevalences"]:
                for bag_index in range(int(cfg["bags_per_rate"])):
                    bag_seed = seed + int(round(float(prevalence) * 1000)) * 17 + bag_index
                    bag = exact_mixture(real_splits["downstream_train"],
                        synth_splits["downstream_train"], int(cfg["bag_size"]),
                        float(prevalence), bag_seed, mode="replace")
                    if bag.record_id.duplicated().any():
                        raise ValueError(f"Duplicate E3 bag IDs for {table_id}/{seed}")
                    raw_bag = detector.predict_score(bag)
                    bag_id = f"{table_id}_seed{seed}_p{prevalence}_b{bag_index}"
                    score_rows.extend({"bag_id": bag_id, "row_position": i,
                                       "record_id": str(bag.record_id.iloc[i]),
                                       "source_label": int(bag.source_label.iloc[i]),
                                       "raw_score": float(raw_bag[i])}
                                      for i in range(len(bag)))
                    for policy_name in ("source_only", "target_real_anchor"):
                        if policy_name not in policies:
                            bag_rows.append({"bag_id": bag_id, "table": table_id, "seed": seed,
                                "true_prevalence": float(bag.source_label.mean()),
                                "policy": policy_name, "status": f"unavailable:{policy_errors[policy_name]}",
                                "pacc_estimate": None, "triggered": None, "removed_n": None})
                            continue
                        policy = policies[policy_name]
                        score = policy["calibrator"].predict(raw_bag)
                        try:
                            estimate = float(policy["quantifiers"]["pacc"].predict_prevalence(score)["clipped"])
                            triggered = estimate >= float(cfg["decision_prevalence_threshold"])
                            removed = np.flatnonzero(score >= policy["threshold"]) if triggered else np.array([], dtype=int)
                            status = "ok"
                        except ValueError as exc:
                            estimate, triggered, removed, status = None, None, np.array([], dtype=int), f"undefined:{exc}"
                        bag_rows.append({"bag_id": bag_id, "table": table_id, "seed": seed,
                            "true_prevalence": float(bag.source_label.mean()),
                            "policy": policy_name, "status": status,
                            "pacc_estimate": estimate, "triggered": triggered,
                            "removed_n": int(len(removed)) if triggered is not None else None,
                            "removed_source_precision": float(bag.source_label.iloc[removed].mean()) if len(removed) else None,
                            "reference_fpr": policy["reference_fpr"]})
            pd.DataFrame(score_rows).to_csv(case_dir / "bag_scores.csv.gz", index=False, compression="gzip")
            pd.DataFrame(bag_rows).to_csv(case_dir / "bags.csv", index=False)
            files = {name: sha256_file(case_dir / name) for name in
                     ("validation_scores.csv.gz", "bag_scores.csv.gz", "bags.csv", "policy_diagnostics.json")}
            _atomic_json(final, {"status": "complete", "context": context,
                "table": table_id, "seed": seed, "source_raw_auroc": source_auc,
                "target_raw_auroc": target_auc, "detector_provenance": detector.get_provenance(),
                "policy_errors": policy_errors, "bag_rows": len(bag_rows), "files_sha256": files})
            completed += 1
    summary = {"status": "complete", "context": context, "completed_cases": completed}
    _atomic_json(output_root / "summary.json", summary)
    return summary
