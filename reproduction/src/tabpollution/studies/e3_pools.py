"""Independent generator repeats on the *unchanged* governance real split.

This deliberately does not use the original pool builder's ``seed`` field:
that field changes both the source/evaluation split and the generator seed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml

from tabpollution.generators.sdv_adapter import create_generator
from tabpollution.utils import sha256_file


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {"replicate_id", "seed", "tables", "generators"}:
        raise ValueError("E3 config requires exactly replicate_id, seed, tables, generators")
    if not cfg["replicate_id"] or int(cfg["seed"]) <= 0:
        raise ValueError("Invalid replicate identity or seed")
    if not isinstance(cfg["tables"], list) or len(cfg["tables"]) != len(set(cfg["tables"])):
        raise ValueError("Tables must be a nonempty unique list")
    if not cfg["tables"] or not set(cfg["tables"]) <= {"adult", "credit", "abalone"}:
        raise ValueError("Unexpected table selection")
    if not isinstance(cfg["generators"], dict) or not cfg["generators"]:
        raise ValueError("Generators must be a mapping")
    if not set(cfg["generators"]) <= {"CTGAN", "TVAE"} or not all(
        isinstance(spec, dict) and spec.get("cuda") is True for spec in cfg["generators"].values()
    ):
        raise ValueError("E3 generator configurations must be CTGAN/TVAE GPU variants")
    return cfg


def preflight(config_path: Path, base_root: Path, output_root: Path,
              checkpoint_root: Path) -> dict[str, Any]:
    config_path, base_root = config_path.resolve(), base_root.resolve()
    output_root, checkpoint_root = output_root.resolve(), checkpoint_root.resolve()
    cfg = _config(config_path)
    if output_root == base_root or base_root in output_root.parents:
        raise ValueError("E3 output must not modify the fixed base pool")
    if checkpoint_root == base_root or base_root in checkpoint_root.parents:
        raise ValueError("E3 checkpoints must not modify the fixed base pool")
    if output_root == checkpoint_root or output_root in checkpoint_root.parents or checkpoint_root in output_root.parents:
        raise ValueError("E3 output and checkpoints must be separate roots")
    registry_path = base_root / "pool_registry.csv"
    manifest_path = base_root / "pool_build_manifest.json"
    registry = pd.read_csv(registry_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Base pool is not marked complete")
    expected = {(r["table_id"], r["generator"]): r for r in manifest["runs"]}
    runs = []
    for table_index, table in enumerate(cfg["tables"]):
        table_rows = registry.loc[registry.table_id == table]
        if table_rows.empty:
            raise ValueError(f"Base registry missing {table}")
        if table_rows.real_path.nunique() != 1 or table_rows.target_column.nunique() != 1:
            raise ValueError(f"Inconsistent base metadata for {table}")
        source_path = base_root / "source_train" / f"{table}.csv"
        real_path = (base_root / str(table_rows.real_path.iloc[0])).resolve()
        source_sha, real_sha = sha256_file(source_path), sha256_file(real_path)
        source = pd.read_csv(source_path)
        real = pd.read_csv(real_path)
        if list(source.columns) != list(real.columns):
            raise ValueError(f"Source/evaluation schema mismatch for {table}")
        for generator_index, (generator, settings) in enumerate(cfg["generators"].items()):
            original = expected.get((table, generator))
            if original is None or original["source_train_sha256"] != source_sha or original["real_sha256"] != real_sha:
                raise ValueError(f"Base provenance mismatch for {table}/{generator}")
            generator_seed = int(cfg["seed"]) + table_index * 1009 + generator_index * 101
            if generator_seed == int(original["generator_seed"]):
                raise ValueError("A purported independent generator uses the base training seed")
            runs.append({"table_id": table, "generator": generator,
                         "generator_seed": generator_seed,
                         "sample_seed": generator_seed + 1_000_003,
                         "source_train_rows": len(source), "evaluation_real_rows": len(real),
                         "source_train_sha256": source_sha, "real_sha256": real_sha,
                         "generator_config_sha256": hashlib.sha256(json.dumps(
                             settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         "target_column": str(table_rows.target_column.iloc[0]),
                         "domain": str(table_rows.domain.iloc[0]),
                         "source_path": str(source_path), "real_path": str(real_path)})
    return {"replicate_id": cfg["replicate_id"], "config_sha256": sha256_file(config_path),
            "base_registry_sha256": sha256_file(registry_path),
            "base_manifest_sha256": sha256_file(manifest_path),
            "base_root": str(base_root), "output_root": str(output_root),
            "checkpoint_root": str(checkpoint_root), "runs": runs}


def build(config_path: Path, base_root: Path, output_root: Path,
          checkpoint_root: Path, *, resume: bool = False) -> dict[str, Any]:
    plan = preflight(config_path, base_root, output_root, checkpoint_root)
    cfg = _config(config_path)
    output_root, checkpoint_root = output_root.resolve(), checkpoint_root.resolve()
    manifest_path = output_root / "replica_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("E3 output exists; use --resume after checking its manifest")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["plan"] != plan:
            raise ValueError("E3 plan or base data changed; use a new output root")
        completed = {(r["table_id"], r["generator"]): r for r in previous["runs"]}
    else:
        launch_files = {"worker.lock", "preflight.json", "worker.log",
                        "job-status.txt", "job-status.txt.partial"}
        if output_root.exists() and any(path.name not in launch_files for path in output_root.iterdir()):
            raise FileExistsError("Nonempty E3 output has no manifest; refusing to overwrite")
        completed = {}
        _write_json(manifest_path, {"status": "running", "plan": plan, "runs": []})
    registry_rows = []
    runs = []
    for case in plan["runs"]:
        table, generator_name = case["table_id"], case["generator"]
        synthetic_path = output_root / "synthetic" / table / f"{generator_name}.csv"
        model_path = checkpoint_root / table / f"{generator_name}.pkl"
        case_checkpoint = output_root / "case_checkpoints" / table / f"{generator_name}.json"
        old = completed.get((table, generator_name))
        if old is None and resume and case_checkpoint.is_file():
            old = json.loads(case_checkpoint.read_text(encoding="utf-8"))
        if old is not None:
            if old["status"] != "complete" or not synthetic_path.is_file() or not model_path.is_file():
                raise ValueError(f"Incomplete E3 resume checkpoint for {table}/{generator_name}")
            if old["synthetic_sha256"] != sha256_file(synthetic_path) or old["model_sha256"] != sha256_file(model_path):
                raise ValueError(f"E3 checkpoint hash mismatch for {table}/{generator_name}")
            if any(old[key] != case[key] for key in
                   ("generator_seed", "sample_seed", "source_train_sha256",
                    "real_sha256", "generator_config_sha256")):
                raise ValueError(f"E3 case checkpoint belongs to another plan: {table}/{generator_name}")
            result = old
        else:
            if synthetic_path.exists() or model_path.exists():
                raise FileExistsError(f"Unmanifested E3 output for {table}/{generator_name}")
            source = pd.read_csv(case["source_path"])
            real = pd.read_csv(case["real_path"])
            if sha256_file(Path(case["source_path"])) != case["source_train_sha256"]:
                raise ValueError("Source training file changed after preflight")
            if sha256_file(Path(case["real_path"])) != case["real_sha256"]:
                raise ValueError("Fixed real evaluation pool changed after preflight")
            generator = create_generator(generator_name, cfg["generators"][generator_name])
            metadata = generator.build_metadata(source)
            started = time.perf_counter()
            generator.fit(source, metadata, case["generator_seed"])
            fit_seconds = time.perf_counter() - started
            synthetic = generator.sample(len(real), case["sample_seed"], "e3_independent")
            if set(synthetic.columns) != set(real.columns) or len(synthetic) != len(real):
                raise ValueError(f"E3 generated schema/size mismatch for {table}/{generator_name}")
            synthetic = synthetic[real.columns]
            _write_csv(synthetic, synthetic_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_partial = model_path.with_suffix(model_path.suffix + ".partial")
            generator.save(model_partial)
            model_partial.replace(model_path)
            result = {"table_id": table, "generator": generator_name,
                      "status": "complete", "fit_seconds": fit_seconds,
                      "synthetic_rows": len(synthetic),
                      "synthetic_sha256": sha256_file(synthetic_path),
                      "model_sha256": sha256_file(model_path),
                      "generator_seed": case["generator_seed"],
                      "sample_seed": case["sample_seed"],
                      "source_train_sha256": case["source_train_sha256"],
                      "real_sha256": case["real_sha256"],
                      "generator_config_sha256": case["generator_config_sha256"],
                      "provenance": generator.get_provenance()}
            _write_json(case_checkpoint, result)
            del generator
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        runs.append(result)
        registry_rows.append({"table_id": table, "domain": case["domain"],
                              "target_column": case["target_column"], "generator": generator_name,
                              "real_path": Path(os.path.relpath(case["real_path"], output_root)).as_posix(),
                              "synthetic_path": synthetic_path.relative_to(output_root).as_posix()})
        _write_json(manifest_path, {"status": "running", "plan": plan, "runs": runs})
    _write_csv(pd.DataFrame(registry_rows), output_root / "pool_registry.csv")
    summary = {"status": "complete", "plan": plan, "runs": runs,
               "registry_sha256": sha256_file(output_root / "pool_registry.csv")}
    _write_json(manifest_path, summary)
    return summary
