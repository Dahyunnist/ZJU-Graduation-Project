"""Strict configuration contract for the governance benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROTOCOLS = ("P1", "P2", "P3", "P4")
DETECTORS = (
    "char3gram", "c2st_lr", "c2st_xgb", "flat_transformer",
    "table_transformer", "datum_transformer", "datum_ta",
)
QUANTIFIERS = ("cc", "pcc", "acc", "pacc", "emq", "hdy", "dys", "median_sweep", "kdey")
DATA_MODES = ("synthetic_fixture", "registry")


class GovernanceConfigError(ValueError):
    """Raised when a governance experiment is underspecified or unsafe."""


@dataclass(frozen=True)
class ProtocolSpec:
    train_tables: tuple[str, ...]
    test_tables: tuple[str, ...]
    train_generators: tuple[str, ...]
    test_generators: tuple[str, ...]


@dataclass(frozen=True)
class GovernanceConfig:
    experiment_id: str
    run_type: str
    seeds: tuple[int, ...]
    protocols: dict[str, ProtocolSpec]
    prevalence_rates: tuple[float, ...]
    bags_per_rate: int
    utility_bags_per_rate: int
    bag_size: int
    detectors: tuple[str, ...]
    quantifiers: tuple[str, ...]
    primary_quantifier: str
    data_mode: str
    registry_path: Path | None
    fixture_rows_per_table: int
    detector_fpr_target: float
    artifact_auc_gate: float
    governance_prevalence_threshold: float
    harm_tolerance: float
    output_dir: Path
    max_cpu_threads: int
    device: str


def _unique_tuple(value: Any, field: str, item_type: type) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, item_type) for x in value):
        type_name = "/".join(t.__name__ for t in item_type) if isinstance(item_type, tuple) else item_type.__name__
        raise GovernanceConfigError(f"{field} must be a non-empty list of {type_name}")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise GovernanceConfigError(f"{field} contains duplicates")
    return result


def _bounded_number(value: Any, field: str, low: float, high: float, *, inclusive: bool = True) -> float:
    if not isinstance(value, (int, float)):
        raise GovernanceConfigError(f"{field} must be numeric")
    number = float(value)
    valid = low <= number <= high if inclusive else low < number < high
    if not valid:
        brackets = "[" if inclusive else "("
        closing = "]" if inclusive else ")"
        raise GovernanceConfigError(f"{field} must be in {brackets}{low}, {high}{closing}")
    return number


def _protocol_specs(raw: Any) -> dict[str, ProtocolSpec]:
    if not isinstance(raw, dict) or not raw:
        raise GovernanceConfigError("protocols must be a non-empty mapping")
    unknown = sorted(set(raw) - set(PROTOCOLS))
    if unknown:
        raise GovernanceConfigError(f"Unknown protocols: {unknown}")
    specs: dict[str, ProtocolSpec] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise GovernanceConfigError(f"protocols.{name} must be a mapping")
        required = {"train_tables", "test_tables", "train_generators", "test_generators"}
        if set(value) != required:
            raise GovernanceConfigError(f"protocols.{name} must contain exactly {sorted(required)}")
        spec = ProtocolSpec(
            train_tables=_unique_tuple(value["train_tables"], f"{name}.train_tables", str),
            test_tables=_unique_tuple(value["test_tables"], f"{name}.test_tables", str),
            train_generators=_unique_tuple(value["train_generators"], f"{name}.train_generators", str),
            test_generators=_unique_tuple(value["test_generators"], f"{name}.test_generators", str),
        )
        if name == "P1" and (set(spec.train_tables) != set(spec.test_tables) or set(spec.train_generators) != set(spec.test_generators)):
            raise GovernanceConfigError("P1 requires identical train/test tables and generators")
        if name == "P2" and (set(spec.train_tables) != set(spec.test_tables) or set(spec.train_generators) & set(spec.test_generators)):
            raise GovernanceConfigError("P2 requires identical tables and disjoint generators")
        if name == "P3" and set(spec.train_tables) & set(spec.test_tables):
            raise GovernanceConfigError("P3 requires disjoint train/test tables")
        if name == "P4" and (set(spec.train_tables) & set(spec.test_tables) or set(spec.train_generators) & set(spec.test_generators)):
            raise GovernanceConfigError("P4 requires disjoint train/test tables and generators")
        specs[name] = spec
    return specs


def validate_governance_config(raw: dict[str, Any], base_dir: Path = Path(".")) -> GovernanceConfig:
    required = {
        "experiment_id", "run_type", "seeds", "protocols", "prevalence_rates",
        "bags_per_rate", "utility_bags_per_rate", "bag_size", "detectors", "quantifiers", "primary_quantifier",
        "data", "thresholds", "output_dir", "resources",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise GovernanceConfigError(f"Missing governance fields: {missing}")
    run_type = str(raw["run_type"])
    if run_type not in {"smoke", "formal"}:
        raise GovernanceConfigError("run_type must be smoke or formal")
    seeds = _unique_tuple(raw["seeds"], "seeds", int)
    rates = tuple(float(x) for x in _unique_tuple(raw["prevalence_rates"], "prevalence_rates", (int, float)))
    if any(x < 0 or x > 1 for x in rates):
        raise GovernanceConfigError("prevalence_rates must be in [0, 1]")
    for mandatory in (0.05, 0.10):
        if mandatory not in rates:
            raise GovernanceConfigError("prevalence_rates must include 0.05 and 0.10")
    detectors = _unique_tuple(raw["detectors"], "detectors", str)
    unknown_detectors = sorted(set(detectors) - set(DETECTORS))
    if unknown_detectors:
        raise GovernanceConfigError(f"Unknown detectors: {unknown_detectors}")
    quantifiers = _unique_tuple(raw["quantifiers"], "quantifiers", str)
    unknown_quantifiers = sorted(set(quantifiers) - set(QUANTIFIERS))
    if unknown_quantifiers:
        raise GovernanceConfigError(f"Unknown quantifiers: {unknown_quantifiers}")
    primary = str(raw["primary_quantifier"])
    if primary not in quantifiers:
        raise GovernanceConfigError("primary_quantifier must be listed in quantifiers")

    data = raw["data"]
    if not isinstance(data, dict) or "mode" not in data:
        raise GovernanceConfigError("data must contain mode")
    mode = str(data["mode"])
    if mode not in DATA_MODES:
        raise GovernanceConfigError(f"data.mode must be one of {DATA_MODES}")
    registry = None
    if mode == "registry":
        if not data.get("registry_path"):
            raise GovernanceConfigError("registry mode requires data.registry_path")
        registry = (base_dir / str(data["registry_path"])).resolve()
    fixture_rows = int(data.get("rows_per_table", 1000))
    if fixture_rows < 400:
        raise GovernanceConfigError("data.rows_per_table must be at least 400")

    thresholds = raw["thresholds"]
    if not isinstance(thresholds, dict):
        raise GovernanceConfigError("thresholds must be a mapping")
    threshold_fields = {
        "detector_fpr_target", "artifact_auc_gate", "governance_prevalence", "harm_tolerance"
    }
    if set(thresholds) != threshold_fields:
        raise GovernanceConfigError(f"thresholds must contain exactly {sorted(threshold_fields)}")

    resources = raw["resources"]
    if not isinstance(resources, dict) or set(resources) != {"device", "max_cpu_threads"}:
        raise GovernanceConfigError("resources must contain exactly device and max_cpu_threads")
    device = str(resources["device"])
    if device not in {"cpu", "cuda"}:
        raise GovernanceConfigError("resources.device must be cpu or cuda")
    threads = int(resources["max_cpu_threads"])
    if threads < 1 or threads > 16:
        raise GovernanceConfigError("resources.max_cpu_threads must be in [1, 16]")

    bags = int(raw["bags_per_rate"])
    utility_bags = int(raw["utility_bags_per_rate"])
    bag_size = int(raw["bag_size"])
    if bags < 1 or bag_size < 40:
        raise GovernanceConfigError("bags_per_rate must be positive and bag_size at least 40")
    if utility_bags < 1 or utility_bags > bags:
        raise GovernanceConfigError("utility_bags_per_rate must be in [1, bags_per_rate]")

    return GovernanceConfig(
        experiment_id=str(raw["experiment_id"]), run_type=run_type, seeds=seeds,
        protocols=_protocol_specs(raw["protocols"]), prevalence_rates=rates,
        bags_per_rate=bags, utility_bags_per_rate=utility_bags,
        bag_size=bag_size, detectors=detectors,
        quantifiers=quantifiers, primary_quantifier=primary, data_mode=mode,
        registry_path=registry, fixture_rows_per_table=fixture_rows,
        detector_fpr_target=_bounded_number(thresholds["detector_fpr_target"], "detector_fpr_target", 0, 1),
        artifact_auc_gate=_bounded_number(thresholds["artifact_auc_gate"], "artifact_auc_gate", 0.5, 1),
        governance_prevalence_threshold=_bounded_number(thresholds["governance_prevalence"], "governance_prevalence", 0, 1),
        harm_tolerance=_bounded_number(thresholds["harm_tolerance"], "harm_tolerance", 0, 1),
        output_dir=(base_dir / str(raw["output_dir"])).resolve(),
        max_cpu_threads=threads, device=device,
    )


def load_governance_config(path: str | Path) -> GovernanceConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise GovernanceConfigError("Governance configuration must be a mapping")
    return validate_governance_config(raw, path.parent)
