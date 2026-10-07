from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tabpollution.studies.e1d import partial_class_shuffle, run_case


def test_partial_class_shuffle_preserves_within_class_marginals():
    frame = pd.DataFrame({"x": range(20), "z": range(20, 40), "y": [0]*10+[1]*10})
    changed = partial_class_shuffle(frame, "y", ["x", "z"], .5, 5)
    assert changed.y.equals(frame.y)
    for label in [0, 1]:
        for feature in ["x", "z"]:
            assert sorted(changed.loc[changed.y == label, feature]) == sorted(
                frame.loc[frame.y == label, feature])


def test_e1d_toy_shard(tmp_path: Path):
    rng = np.random.default_rng(3)
    real = pd.DataFrame({"a": rng.normal(size=180), "b": rng.normal(size=180),
                         "c": rng.normal(size=180), "d": rng.normal(size=180)})
    real["target"] = (real.a + real.b/3 > 0).astype(int)
    synthetic = real.sample(100, random_state=4).reset_index(drop=True)
    pool = tmp_path / "pool"
    (pool / "real").mkdir(parents=True)
    (pool / "synthetic").mkdir()
    real.to_csv(pool / "real/toy.csv", index=False)
    synthetic.to_csv(pool / "synthetic/toy.csv", index=False)
    pd.DataFrame([{"table_id": "toy", "generator": "GEN", "target_column": "target",
                   "real_path": "real/toy.csv", "synthetic_path": "synthetic/toy.csv"}]).to_csv(
                       pool / "pool_registry.csv", index=False)
    config = {"registry_path": "pool/pool_registry.csv", "table": "toy",
              "generators": ["GEN"], "seeds": [11], "train_size": 50,
              "test_fraction": .25, "prevalence": .5, "calibration_size": 20,
              "real_donor_size": 25, "learner": "rf",
              "conditions": ["natural", "global_half", "payment_status_full", "balances_full",
                             "independent_real_donor", "real_global_half",
                             "real_payment_status_full", "real_balances_full"],
              "feature_blocks": {"payment_status": ["a", "b"], "balances": ["c", "d"]},
              "calibration": {"method": "sigmoid_on_clipped_logit",
                              "input_clip": 1e-6, "regularization_C": 1000},
              "tail_feature": "a", "tail_quantile": .9}
    path = tmp_path / "design.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "result"
    assert run_case(path, None, output, "GEN", 11, preflight=True)["status"] == "preflight_ok"
    assert run_case(path, None, output, "GEN", 11)["fits"] == 8
    metrics = pd.read_csv(output / "toy_GEN_seed11/metrics.csv")
    assert len(metrics) == 16 and metrics.calibration_slope.gt(0).all()
    assert run_case(path, None, output, "GEN", 11)["status"] == "already_complete"
