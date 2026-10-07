import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tabpollution.studies.e3_direction import preflight, run
from tabpollution.utils import sha256_file


def test_gate_a_full_toy_case_and_independent_audit(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_e3_direction.py"
    spec = spec_from_file_location("analyze_e3_direction", script_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(12)
    pool = tmp_path / "replica"
    (pool / "real").mkdir(parents=True)
    runs, rows = [], []
    for table in ("adult", "abalone"):
        x = rng.normal(size=500)
        real = pd.DataFrame({"x": x, "target": (x > 0).astype(int)})
        real_path = pool / "real" / f"{table}.csv"
        real.to_csv(real_path, index=False)
        for generator, offset in (("CTGAN", .8), ("TVAE", -.8)):
            syn = pd.DataFrame({"x": x + offset, "target": (x > 0).astype(int)})
            syn_path = pool / "synthetic" / table / f"{generator}.csv"
            syn_path.parent.mkdir(parents=True, exist_ok=True)
            syn.to_csv(syn_path, index=False)
            rows.append({"table_id": table, "domain": "toy", "target_column": "target",
                         "generator": generator, "real_path": f"real/{table}.csv",
                         "synthetic_path": f"synthetic/{table}/{generator}.csv"})
            runs.append({"table_id": table, "generator": generator, "status": "complete",
                         "synthetic_sha256": sha256_file(syn_path),
                         "real_sha256": sha256_file(real_path)})
    registry = pool / "pool_registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    manifest = pool / "replica_manifest.json"
    manifest.write_text(json.dumps({"status": "complete",
        "registry_sha256": sha256_file(registry),
        "plan": {"replicate_id": "e3-independent-pools-r3031"}, "runs": runs}), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"study_id": "toy-e3", "tables": ["adult", "abalone"],
        "source_generator": "CTGAN", "target_generator": "TVAE",
        "seeds": [2026, 2027, 2028], "detector": "c2st_lr",
        "target_real_anchor_size": 20, "target_fpr": .05, "bag_size": 40,
        "prevalences": [.05, .10, .25], "bags_per_rate": 2,
        "primary_quantifier": "pacc", "decision_prevalence_threshold": .10}), encoding="utf-8")
    assert preflight(cfg, registry, manifest)["cases"] == 6
    output = tmp_path / "results"
    assert run(cfg, registry, manifest, output)["completed_cases"] == 6
    assert run(cfg, registry, manifest, output)["completed_cases"] == 6
    audited = module.analyze(output, tmp_path / "audit")
    assert audited["complete_cases"] == 6 and audited["bag_policy_rows"] == 72
    assert 0 <= audited["direction_reversed_cases"] <= 6 and audited["audited"]
    score_path = output / "adult" / "seed2026" / "bag_scores.csv.gz"
    with score_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(AssertionError, match="file hash changed"):
        module.analyze(output, tmp_path / "audit-tampered")
