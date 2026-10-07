import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tabpollution.studies import e3_pools
from tabpollution.utils import sha256_file


class FakeGenerator:
    fits = 0

    def __init__(self, name, settings):
        self.name = name
        self.seed = None

    def build_metadata(self, source):
        return {"columns": source.columns.tolist()}

    def fit(self, source, metadata, seed):
        self.seed = seed
        FakeGenerator.fits += 1

    def sample(self, n, sample_seed, pool_name):
        return pd.DataFrame({"x": range(n), "target": [0, 1] * (n // 2)})

    def save(self, path):
        Path(path).write_text(str(self.seed), encoding="utf-8")

    def get_provenance(self):
        return {"generator_seed": self.seed}


def _fixture(tmp_path):
    base = tmp_path / "base"
    (base / "source_train").mkdir(parents=True)
    (base / "real").mkdir()
    source = base / "source_train" / "adult.csv"
    real = base / "real" / "adult.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_csv(source, index=False)
    pd.DataFrame({"x": [5, 6, 7, 8], "target": [0, 1, 0, 1]}).to_csv(real, index=False)
    pd.DataFrame([{"table_id": "adult", "domain": "social", "target_column": "target",
                   "generator": "CTGAN", "real_path": "real/adult.csv",
                   "synthetic_path": "synthetic/adult/CTGAN.csv"}]).to_csv(base / "pool_registry.csv", index=False)
    (base / "pool_build_manifest.json").write_text(json.dumps({
        "status": "complete", "runs": [{"table_id": "adult", "generator": "CTGAN",
        "source_train_sha256": sha256_file(source), "real_sha256": sha256_file(real),
        "generator_seed": 2026}]}), encoding="utf-8")
    config = tmp_path / "e3.yaml"
    config.write_text(yaml.safe_dump({"replicate_id": "test-r3031", "seed": 3031,
        "tables": ["adult"], "generators": {"CTGAN": {"cuda": True, "epochs": 1}}}), encoding="utf-8")
    return config, base, tmp_path / "replica", tmp_path / "models", source, real


def test_independent_generator_repeat_keeps_real_split_and_resumes(tmp_path, monkeypatch):
    config, base, output, models, source, real = _fixture(tmp_path)
    monkeypatch.setattr(e3_pools, "create_generator", FakeGenerator)
    FakeGenerator.fits = 0
    plan = e3_pools.preflight(config, base, output, models)
    assert plan["runs"][0]["generator_seed"] == 3031
    assert plan["runs"][0]["source_train_sha256"] == sha256_file(source)
    result = e3_pools.build(config, base, output, models)
    assert result["status"] == "complete" and FakeGenerator.fits == 1
    registry = pd.read_csv(output / "pool_registry.csv")
    assert (output / registry.real_path.iloc[0]).resolve() == real.resolve()
    assert e3_pools.build(config, base, output, models, resume=True)["status"] == "complete"
    assert FakeGenerator.fits == 1
    source.write_text("x,target\n0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Base provenance mismatch"):
        e3_pools.preflight(config, base, output, models)


def test_refuses_base_pool_output(tmp_path):
    config, base, _, models, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must not modify"):
        e3_pools.preflight(config, base, base / "replica", models)
