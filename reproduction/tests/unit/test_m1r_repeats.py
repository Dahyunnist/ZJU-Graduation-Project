from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tabpollution.studies.m1r_repeats import decide_cross_instance, load_config, make_e3b_binding
from tabpollution.studies.e3b_detectability import _config as load_e3b_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "m1r_independent_repeats_v1.yaml"


def test_frozen_m1r_config_loads() -> None:
    cfg = load_config(CONFIG)
    assert cfg["minimum_complete_instances"] == 3
    assert len(cfg["new_instances"]) == 2


def test_m1r_rejects_split_seed_change(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["split_seeds"].append(2029)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="split seeds changed"):
        load_config(path)


def test_cross_instance_requires_unanimous_branches() -> None:
    stable = pd.DataFrame([
        {"instance_id": instance, "table": table, "branch": branch}
        for instance in ("a", "b", "c")
        for table, branch in (("adult", "E3C-Transfer"), ("abalone", "E3C-Capacity"))
    ])
    assert decide_cross_instance(stable)["overall_result"] == "Stable-Split-Branches"
    mixed = stable.copy()
    mixed.loc[(mixed.instance_id == "c") & (mixed.table == "adult"), "branch"] = "E3C-Capacity"
    result = decide_cross_instance(mixed)
    assert result["table_results"]["adult"] == "Conditional-Heterogeneity"
    assert result["overall_result"] == "Conditional-Heterogeneity"


def test_cross_instance_does_not_replace_missing_instance_with_splits() -> None:
    incomplete = pd.DataFrame([
        {"instance_id": instance, "table": table, "branch": branch}
        for instance in ("a", "b")
        for table, branch in (("adult", "E3C-Transfer"), ("abalone", "E3C-Capacity"))
    ])
    result = decide_cross_instance(incomplete)
    assert result["overall_result"] == "Inconclusive"


def test_instance_binding_preserves_frozen_e3b_protocol(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "replica_manifest.json").write_text("{}", encoding="utf-8")
    (pool / "pool_registry.csv").write_text("table_id,generator\n", encoding="utf-8")
    path = make_e3b_binding(cfg, "unit-instance", pool, tmp_path / "binding.yaml")
    binding = load_e3b_config(path)
    assert binding["seeds"] == [2026, 2027, 2028]
    assert binding["model_direction"] == "high_score_is_synthetic"
    assert binding["transfer_min_auroc_gain"] == 0.10

