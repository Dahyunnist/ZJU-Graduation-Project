from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tabpollution.studies.e1b import _apply_calibrator, _fit_calibrator, run_case


def test_e1b_disjoint_calibration_and_monotone_probabilities(tmp_path: Path):
    generator = np.random.default_rng(3)
    real = pd.DataFrame({"x": generator.normal(size=180), "z": generator.normal(size=180)})
    real["target"] = (real.x + real.z / 4 > 0).astype(int)
    synthetic = real.sample(100, random_state=4).reset_index(drop=True)
    pools = tmp_path / "pools"
    (pools / "real").mkdir(parents=True)
    (pools / "synthetic").mkdir()
    real.to_csv(pools / "real/toy.csv", index=False)
    synthetic.to_csv(pools / "synthetic/toy.csv", index=False)
    pd.DataFrame([{"table_id": "toy", "generator": "GEN", "target_column": "target",
                   "real_path": "real/toy.csv", "synthetic_path": "synthetic/toy.csv"}]).to_csv(
                       pools / "pool_registry.csv", index=False)
    config = {
        "registry_path": "pools/pool_registry.csv", "cases": [{"table": "toy", "generator": "GEN"}],
        "seeds": [11], "train_size": 50, "test_fraction": .25, "prevalence": .5,
        "calibration_size": 20, "real_donor_size": 25,
        "conditions": ["real_reference", "independent_real_donor", "natural", "label_permute"],
        "learners": {"lr": [None], "rf": [3]},
        "calibration": {"method": "sigmoid_on_clipped_logit", "input_clip": 1e-6,
                        "regularization_C": 1000},
        "tail_features": {"toy": "x"}, "tail_quantile": .9,
    }
    config_file = tmp_path / "design.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    preflight = run_case(config_file, None, output, "toy", "GEN", 11, preflight=True)
    assert preflight["positive_validation_n"] > 0
    result = run_case(config_file, None, output, "toy", "GEN", 11)
    assert result["fits"] == 8 and result["metric_rows"] == 16
    context = __import__("json").loads((output / "toy_GEN_seed11/context.json").read_text())
    groups = [set(context[key]) for key in ["core_ids", "validation_ids", "donor_ids", "test_ids"]]
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))
    metrics = pd.read_csv(output / "toy_GEN_seed11/metrics.csv")
    assert set(metrics["mode"]) == {"raw", "calibrated"}
    assert metrics.calibration_slope.gt(0).all()
    assert run_case(config_file, None, output, "toy", "GEN", 11)["status"] == "already_complete"


def test_calibration_uses_only_supplied_validation_labels():
    raw = np.linspace(.1, .9, 50)
    labels = (raw > .5).astype(int)
    calibrator = _fit_calibrator(raw, labels, 1e-6, 1000)
    transformed = _apply_calibrator(calibrator, np.array([.1, .2, .8, .9]), 1e-6)
    assert np.diff(transformed).min() > 0
    assert np.isfinite(transformed).all()
