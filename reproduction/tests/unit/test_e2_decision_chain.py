from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tabpollution.studies.e2 import oracle_top_k, run_seed, top_k


def test_fixed_budget_order_and_source_oracle():
    scores = np.array([.2, .9, .9, .1])
    assert top_k(scores, 2).tolist() == [1, 2]
    assert top_k(scores * .5 + .1, 2).tolist() == [1, 2]
    assert oracle_top_k(np.array([0, 1, 0, 1]), 2).tolist() == [1, 3]


def test_e2_toy_preflight(tmp_path: Path):
    rng = np.random.default_rng(5)
    real = pd.DataFrame({"age": rng.normal(40, 10, 500), "income": rng.normal(size=500)})
    real["target"] = (real.age/10 + real.income > 4).astype(int)
    ctgan = real.sample(350, random_state=3).reset_index(drop=True)
    tvae = real.sample(350, random_state=4).reset_index(drop=True)
    pools = tmp_path / "pool"
    (pools / "real").mkdir(parents=True)
    (pools / "synthetic").mkdir()
    real.to_csv(pools / "real/one.csv", index=False)
    ctgan.to_csv(pools / "synthetic/ctgan.csv", index=False)
    tvae.to_csv(pools / "synthetic/tvae.csv", index=False)
    pd.DataFrame([
        {"table_id": "one", "domain": "toy", "target_column": "target", "generator": "CTGAN",
         "real_path": "real/one.csv", "synthetic_path": "synthetic/ctgan.csv"},
        {"table_id": "one", "domain": "toy", "target_column": "target", "generator": "TVAE",
         "real_path": "real/one.csv", "synthetic_path": "synthetic/tvae.csv"},
    ]).to_csv(pools / "pool_registry.csv", index=False)
    config = {"registry_path": "pool/pool_registry.csv", "table": "one",
              "source_generator": "CTGAN", "target_generator": "TVAE", "seeds": [11],
              "bag_size": 40, "prevalences": [.1], "bags_per_rate": 1,
              "detector": "c2st_lr", "downstream_model": "lr",
              "target_real_anchor_size": 20, "task_calibration_size": 20,
              "target_fpr": .05, "fixed_delete_count": 3,
              "decision_prevalence_threshold": .1,
              "task_probability_clip": 1e-6, "task_calibration_C": 1000,
              "tail_feature": "age", "tail_quantile": .9}
    path = tmp_path / "design.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_seed(path, None, tmp_path / "output", 11, preflight=True)
    assert result["status"] == "preflight_ok"
    assert result["real_anchor_n"] == 20 and result["task_validation_n"] == 20
