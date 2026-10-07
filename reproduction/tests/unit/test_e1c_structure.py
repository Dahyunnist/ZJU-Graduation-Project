from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tabpollution.studies.e1c import run_case


def test_e1c_preserves_marginals_and_information_splits(tmp_path: Path):
    rng = np.random.default_rng(42)
    real = pd.DataFrame({"a": rng.normal(size=180), "b": rng.normal(size=180)})
    real["target"] = (real.a + .3 * real.b > 0).astype(int)
    synthetic = real.sample(100, random_state=11).reset_index(drop=True)
    pools = tmp_path / "pools"
    (pools / "real").mkdir(parents=True)
    (pools / "synthetic").mkdir()
    real.to_csv(pools / "real/toy.csv", index=False)
    synthetic.to_csv(pools / "synthetic/toy.csv", index=False)
    pd.DataFrame([{"table_id": "toy", "generator": "GEN", "target_column": "target",
                   "real_path": "real/toy.csv", "synthetic_path": "synthetic/toy.csv"}]).to_csv(
                       pools / "pool_registry.csv", index=False)
    config = {"registry_path": "pools/pool_registry.csv", "table": "toy",
              "generators": ["GEN"], "seeds": [11], "train_size": 50,
              "test_fraction": .25, "prevalence": .5, "calibration_size": 20,
              "real_donor_size": 25,
              "conditions": ["natural", "class_shuffle", "independent_real_donor",
                             "real_donor_class_shuffle"],
              "learners": ["lr", "rf"],
              "calibration": {"method": "sigmoid_on_clipped_logit", "input_clip": 1e-6,
                              "regularization_C": 1000},
              "tail_feature": "a", "tail_quantile": .9}
    path = tmp_path / "design.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    assert run_case(path, None, output, "GEN", 11, preflight=True)["status"] == "preflight_ok"
    result = run_case(path, None, output, "GEN", 11)
    assert result["fits"] == 8 and result["metric_rows"] == 16
    metrics = pd.read_csv(output / "toy_GEN_seed11/metrics.csv")
    assert set(metrics.condition) == set(config["conditions"])
    assert metrics.calibration_slope.gt(0).all()
    context = __import__("json").loads((output / "toy_GEN_seed11/context.json").read_text())
    groups = [set(context[field]) for field in ["core_ids", "validation_ids", "donor_ids", "test_ids"]]
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i+1, 4))
    assert run_case(path, None, output, "GEN", 11)["status"] == "already_complete"
