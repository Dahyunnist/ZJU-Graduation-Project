from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

from tabpollution.governance.data import RegistrySource
from tabpollution.governance.pools import (
    PoolBuildConfig,
    PoolDatasetSpec,
    _parse_abalone_archive,
    build_governance_pools,
    load_pool_build_config,
)


class FakeGenerator:
    def __init__(self, name: str, config: dict):
        self.name, self.config = name, config
        self.real: pd.DataFrame | None = None

    def build_metadata(self, real: pd.DataFrame):
        return {"columns": real.columns.tolist()}

    def fit(self, real: pd.DataFrame, metadata, seed: int):
        self.real = real.copy()

    def sample(self, n: int, sample_seed: int, pool_name: str) -> pd.DataFrame:
        assert self.real is not None
        return self.real.sample(n=n, replace=n > len(self.real), random_state=sample_seed).reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-model", encoding="utf-8")

    def get_provenance(self) -> dict:
        return {"implementation": "test_fake"}


def test_shipped_pool_config_is_structurally_valid() -> None:
    config = load_pool_build_config("configs/governance_pool_build.yaml")
    assert {spec.table_id for spec in config.datasets} == {"adult", "abalone", "credit"}
    assert set(config.generators) == {"CTGAN", "TVAE"}
    abalone = next(spec for spec in config.datasets if spec.table_id == "abalone")
    assert abalone.target_transform == "threshold_binary"
    assert abalone.target_threshold == 9


def test_pool_pilot_is_small_and_isolated_from_formal_artifacts() -> None:
    pilot = load_pool_build_config("configs/governance_pool_build_pilot.yaml")
    formal = load_pool_build_config("configs/governance_pool_build.yaml")
    assert {spec.table_id for spec in pilot.datasets} == {"adult"}
    assert set(pilot.generators) == {"CTGAN", "TVAE"}
    assert pilot.sample_size == 2000
    assert all(config["epochs"] <= 10 for config in pilot.generators.values())
    assert pilot.output_dir != formal.output_dir
    assert pilot.checkpoint_dir != formal.checkpoint_dir


def test_pool_builder_writes_atomic_registry_and_binary_target(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({
        "row_id": [f"r{i}" for i in range(40)],
        "feature": list(range(40)),
        "rings": [5 + i % 12 for i in range(40)],
    }).to_csv(source, index=False)
    config = PoolBuildConfig(
        build_id="test-pools", seed=2026, sample_size="match_real", source_train_fraction=0.35,
        datasets=(PoolDatasetSpec(
            table_id="abalone", domain="scientific", source_path=source,
            source_target="rings", target_column="rings_high", target_transform="median_binary",
        ),),
        generators={"CTGAN": {"epochs": 1}},
        output_dir=tmp_path / "data", checkpoint_dir=tmp_path / "models",
    )
    monkeypatch.setattr("tabpollution.governance.pools.importlib_metadata.version", lambda _: "test")
    monkeypatch.setattr(
        "tabpollution.governance.pools.create_generator",
        lambda name, params: FakeGenerator(name, params),
    )
    monkeypatch.setattr(
        "tabpollution.governance.pools.sdmetrics_quality",
        lambda real, synthetic: {"overall": 0.75, "properties": {}},
    )
    result = build_governance_pools(config)
    assert result["status"] == "complete"
    registry_path = tmp_path / "data" / "pool_registry.csv"
    assert registry_path.is_file()
    assert pd.read_csv(registry_path)["quality_score"].iloc[0] == 0.75
    loaded = RegistrySource(registry_path).table("abalone")
    assert loaded.target_column == "rings_high"
    assert set(loaded.real["rings_high"].unique()) == {0, 1}
    assert "rings" not in loaded.real.columns
    source_train = pd.read_csv(tmp_path / "data" / "source_train" / "abalone.csv")
    assert set(source_train["feature"]).isdisjoint(set(loaded.real["feature"]))
    resumed = build_governance_pools(config, resume=True)
    assert resumed["runs"][0]["status"] == "reused_complete"


def test_abalone_archive_parser_freezes_stable_row_ids(tmp_path: Path) -> None:
    archive_path = tmp_path / "abalone.zip"
    rows = [f"M,0.1,0.2,0.3,0.4,0.5,0.6,0.7,{1 + index % 20}" for index in range(4177)]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("abalone.data", "\n".join(rows))
    first = _parse_abalone_archive(archive_path)
    second = _parse_abalone_archive(archive_path)
    assert list(first.columns) == [
        "row_id", "sex", "length", "diameter", "height", "whole_weight",
        "shucked_weight", "viscera_weight", "shell_weight", "rings",
    ]
    assert first["row_id"].tolist() == second["row_id"].tolist()
    assert first["row_id"].is_unique
