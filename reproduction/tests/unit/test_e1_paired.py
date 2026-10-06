from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tabpollution.studies.e1 import class_shuffle, label_permute, run_shard, tail_undercoverage


def test_controlled_interventions_preserve_declared_quantities():
    frame = pd.DataFrame({
        "x": np.arange(100), "z": np.arange(100) * 2,
        "category": np.where(np.arange(100) % 3 == 0, "A", "B"),
        "target": np.arange(100) % 2,
    })
    shuffled = class_shuffle(frame, "target", ["x", "z", "category"], 5)
    assert shuffled.target.tolist() == frame.target.tolist()
    for label in [0, 1]:
        for column in ["x", "z", "category"]:
            assert sorted(shuffled.loc[shuffled.target == label, column].tolist()) == sorted(
                frame.loc[frame.target == label, column].tolist())
    permuted = label_permute(frame, "target", 6)
    assert permuted.drop(columns="target").equals(frame.drop(columns="target"))
    assert sorted(permuted.target.tolist()) == sorted(frame.target.tolist())
    covered = tail_undercoverage(frame, frame.iloc[:20], "target", "x", 80, 7)
    assert len(covered) == 20
    assert covered.target.value_counts().to_dict() == frame.iloc[:20].target.value_counts().to_dict()
    assert covered.x.max() < 80


def test_small_e1_shard_is_paired_and_restartable(tmp_path: Path):
    rng = np.random.default_rng(1)
    real = pd.DataFrame({"x": rng.normal(size=240), "z": rng.normal(size=240)})
    real["target"] = (real.x + 0.2 * real.z > 0).astype(int)
    synthetic = real.sample(n=150, random_state=2).reset_index(drop=True)
    data = tmp_path / "data"
    (data / "real").mkdir(parents=True)
    (data / "synthetic").mkdir()
    real.to_csv(data / "real" / "toy.csv", index=False)
    synthetic.to_csv(data / "synthetic" / "toy.csv", index=False)
    pd.DataFrame([{
        "table_id": "toy", "target_column": "target", "generator": "GEN",
        "real_path": "real/toy.csv", "synthetic_path": "synthetic/toy.csv",
    }]).to_csv(data / "pool_registry.csv", index=False)
    config = tmp_path / "design.yaml"
    config.write_text(yaml.safe_dump({
        "registry_path": "data/pool_registry.csv", "tables": ["toy"], "generators": ["GEN"],
        "seeds": [11], "train_size": 100, "test_fraction": 0.25,
        "prevalences": [0.0, 0.1], "models": ["lr"],
        "interventions": ["natural", "class_shuffle", "label_permute", "tail_undercoverage"],
        "tail_features": {"toy": "x"}, "tail_quantile": 0.9,
    }), encoding="utf-8")
    output = tmp_path / "outputs"
    preview = run_shard(config, None, output, "toy", 11, preflight=True)
    assert preview["real_train_core"] == 100
    result = run_shard(config, None, output, "toy", 11)
    assert result["status"] == "complete"
    assert run_shard(config, None, output, "toy", 11)["status"] == "already_complete"
    metrics = pd.read_csv(output / "toy_seed11" / "metrics.csv")
    assert metrics.loc[metrics.intervention == "real_reference", "log_loss"].nunique() == 1
    assert (metrics.train_n == 100).all()
    assert metrics.loc[metrics.intervention == "natural", "prevalence"].tolist() == [0.1]
