"""Strict configuration contract for the governance benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROTOCOLS = ("P1", "P2", "P3", "P4")
DETECTORS = (
    "char3gram", "c2st_lr", "c2st_xgb", "flat_transformer",
    "table_transformer", "column_positional_ablation", "datum_transformer", "datum_ta",
)
QUANTIFIERS = ("cc", "pcc", "acc", "pacc", "emq", "hdy", "dys", "median_sweep", "kdey")
CONTAMINATION_MODES = ("replace", "append")
VALUATION_METHODS = ("knn_shapley", "data_oob")
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
    contamination_modes: tuple[str, ...]
    bags_per_rate: int
    utility_bags_per_rate: int
    bag_size: int
    detectors: tuple[str, ...]
    quantifiers: tuple[str, ...]
    primary_quantifier: str
    valuation_enabled: bool
    valuation_methods: tuple[str, ...]
    valuation_bags_per_rate: int
    valuation_sample_limit: int
    valuation_oob_estimators: int
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
    deep_dim: int
    deep_heads: int
    deep_layers: int
    deep_max_len: int
    deep_max_datum: int
    deep_max_columns: int
    deep_epochs: int
    deep_batch_size: int
    deep_learning_rate: float
    deep_weight_decay: float
    deep_gradient_clip_norm: float
    deep_early_stopping_patience: int
    deep_min_epochs: int


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
        "experiment_id", "run_type", "seeds", "protocols", "prevalence_rates", "contamination_modes",
        "bags_per_rate", "utility_bags_per_rate", "bag_size", "detectors", "quantifiers", "primary_quantifier",
        "valuation", "data", "thresholds", "output_dir", "resources", "deep_training",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise GovernanceConfigError(f"Missing governance fields: {missing}")
    run_type = str(raw["run_type"])
    if run_type not in {"smoke", "pilot", "formal"}:
        raise GovernanceConfigError("run_type must be smoke, pilot, or formal")
    seeds = _unique_tuple(raw["seeds"], "seeds", int)
    rates = tuple(float(x) for x in _unique_tuple(raw["prevalence_rates"], "prevalence_rates", (int, float)))
    if any(x < 0 or x > 1 for x in rates):
        raise GovernanceConfigError("prevalence_rates must be in [0, 1]")
    for mandatory in (0.05, 0.10):
        if mandatory not in rates:
            raise GovernanceConfigError("prevalence_rates must include 0.05 and 0.10")
    contamination_modes = _unique_tuple(raw["contamination_modes"], "contamination_modes", str)
    if set(contamination_modes) - set(CONTAMINATION_MODES):
        raise GovernanceConfigError(f"contamination_modes must be drawn from {CONTAMINATION_MODES}")
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

    valuation = raw["valuation"]
    valuation_fields = {"enabled", "methods", "bags_per_rate", "sample_limit", "oob_estimators"}
    if not isinstance(valuation, dict) or set(valuation) != valuation_fields:
        raise GovernanceConfigError(f"valuation must contain exactly {sorted(valuation_fields)}")
    valuation_enabled = bool(valuation["enabled"])
    methods_raw = valuation["methods"]
    if not isinstance(methods_raw, list) or not all(isinstance(x, str) for x in methods_raw):
        raise GovernanceConfigError("valuation.methods must be a list of strings")
    valuation_methods = tuple(methods_raw)
    if len(valuation_methods) != len(set(valuation_methods)) or set(valuation_methods) - set(VALUATION_METHODS):
        raise GovernanceConfigError(f"valuation.methods must be unique and drawn from {VALUATION_METHODS}")
    if valuation_enabled and not valuation_methods:
        raise GovernanceConfigError("enabled valuation requires at least one method")
    valuation_bags = int(valuation["bags_per_rate"])
    valuation_sample_limit = int(valuation["sample_limit"])
    valuation_oob_estimators = int(valuation["oob_estimators"])

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

    deep = raw["deep_training"]
    deep_fields = {
        "dim", "heads", "layers", "max_len", "max_datum", "max_columns",
        "epochs", "batch_size", "learning_rate", "weight_decay",
        "gradient_clip_norm", "early_stopping_patience", "min_epochs",
    }
    if not isinstance(deep, dict) or set(deep) != deep_fields:
        raise GovernanceConfigError(f"deep_training must contain exactly {sorted(deep_fields)}")
    deep_dim = int(deep["dim"])
    deep_heads = int(deep["heads"])
    deep_layers = int(deep["layers"])
    deep_max_len = int(deep["max_len"])
    deep_max_datum = int(deep["max_datum"])
    deep_max_columns = int(deep["max_columns"])
    deep_epochs = int(deep["epochs"])
    deep_batch_size = int(deep["batch_size"])
    deep_learning_rate = float(deep["learning_rate"])
    deep_weight_decay = float(deep["weight_decay"])
    deep_gradient_clip_norm = float(deep["gradient_clip_norm"])
    deep_patience = int(deep["early_stopping_patience"])
    deep_min_epochs = int(deep["min_epochs"])
    if deep_dim < 8 or deep_heads < 1 or deep_dim % deep_heads:
        raise GovernanceConfigError("deep_training dim must be >=8 and divisible by heads")
    if min(deep_layers, deep_max_len, deep_max_datum, deep_max_columns, deep_epochs, deep_batch_size) < 1:
        raise GovernanceConfigError("deep_training integer fields must be positive")
    if not 0 < deep_learning_rate <= .01 or not 0 <= deep_weight_decay <= 1:
        raise GovernanceConfigError("deep_training learning_rate/weight_decay are out of range")
    if deep_gradient_clip_norm <= 0 or deep_patience < 1 or not 1 <= deep_min_epochs <= deep_epochs:
        raise GovernanceConfigError("deep_training clipping/early-stopping settings are invalid")

    bags = int(raw["bags_per_rate"])
    utility_bags = int(raw["utility_bags_per_rate"])
    bag_size = int(raw["bag_size"])
    if bags < 1 or bag_size < 40:
        raise GovernanceConfigError("bags_per_rate must be positive and bag_size at least 40")
    if utility_bags < 1 or utility_bags > bags:
        raise GovernanceConfigError("utility_bags_per_rate must be in [1, bags_per_rate]")
    if valuation_bags < 0 or valuation_bags > utility_bags:
        raise GovernanceConfigError("valuation.bags_per_rate must be in [0, utility_bags_per_rate]")
    if valuation_sample_limit < 40 or valuation_oob_estimators < 10:
        raise GovernanceConfigError("valuation sample_limit must be >=40 and oob_estimators >=10")
    if run_type == "formal":
        if set(raw["protocols"]) != set(PROTOCOLS):
            raise GovernanceConfigError("formal runs must include P1-P4")
        if len(seeds) < 5 or bags < 100:
            raise GovernanceConfigError("formal runs require at least five seeds and 100 bags per rate")
        if set(contamination_modes) != set(CONTAMINATION_MODES):
            raise GovernanceConfigError("formal runs require replace and append contamination")
        if not valuation_enabled or set(valuation_methods) != set(VALUATION_METHODS):
            raise GovernanceConfigError("formal runs require KNN-Shapley and Data-OOB valuation")
        required_detectors = {"char3gram", "flat_transformer", "datum_transformer", "datum_ta"}
        if not required_detectors.issubset(detectors):
            raise GovernanceConfigError("formal runs are missing required direct detection baselines")
        if "table_transformer" in detectors:
            raise GovernanceConfigError(
                "table_transformer is a legacy approximation; use column_positional_ablation in formal runs"
            )
        if device != "cuda":
            raise GovernanceConfigError("formal deep-detector runs require resources.device=cuda")
        if deep_dim != 192 or deep_heads != 6 or deep_layers != 6 or deep_epochs < 20:
            raise GovernanceConfigError("formal runs require the frozen 192d/6-head/6-layer/20-epoch deep architecture")

    return GovernanceConfig(
        experiment_id=str(raw["experiment_id"]), run_type=run_type, seeds=seeds,
        protocols=_protocol_specs(raw["protocols"]), prevalence_rates=rates,
        contamination_modes=contamination_modes,
        bags_per_rate=bags, utility_bags_per_rate=utility_bags,
        bag_size=bag_size, detectors=detectors,
        quantifiers=quantifiers, primary_quantifier=primary,
        valuation_enabled=valuation_enabled, valuation_methods=valuation_methods,
        valuation_bags_per_rate=valuation_bags, valuation_sample_limit=valuation_sample_limit,
        valuation_oob_estimators=valuation_oob_estimators, data_mode=mode,
        registry_path=registry, fixture_rows_per_table=fixture_rows,
        detector_fpr_target=_bounded_number(thresholds["detector_fpr_target"], "detector_fpr_target", 0, 1),
        artifact_auc_gate=_bounded_number(thresholds["artifact_auc_gate"], "artifact_auc_gate", 0.5, 1),
        governance_prevalence_threshold=_bounded_number(thresholds["governance_prevalence"], "governance_prevalence", 0, 1),
        harm_tolerance=_bounded_number(thresholds["harm_tolerance"], "harm_tolerance", 0, 1),
        output_dir=(base_dir / str(raw["output_dir"])).resolve(),
        max_cpu_threads=threads, device=device,
        deep_dim=deep_dim, deep_heads=deep_heads, deep_layers=deep_layers,
        deep_max_len=deep_max_len, deep_max_datum=deep_max_datum,
        deep_max_columns=deep_max_columns, deep_epochs=deep_epochs,
        deep_batch_size=deep_batch_size, deep_learning_rate=deep_learning_rate,
        deep_weight_decay=deep_weight_decay,
        deep_gradient_clip_norm=deep_gradient_clip_norm,
        deep_early_stopping_patience=deep_patience, deep_min_epochs=deep_min_epochs,
    )


def load_governance_config(path: str | Path) -> GovernanceConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise GovernanceConfigError("Governance configuration must be a mapping")
    return validate_governance_config(raw, path.parent)
