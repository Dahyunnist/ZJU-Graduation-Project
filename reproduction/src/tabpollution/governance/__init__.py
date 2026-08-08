"""End-to-end synthetic tabular contamination governance benchmark."""

from .config import GovernanceConfig, load_governance_config
from .pipeline import run_governance_benchmark, validate_governance_setup
from .shards import (
    aggregate_governance_shards,
    build_shard_plan,
    run_governance_shard,
    run_governance_sharded,
    shard_status,
)

__all__ = [
    "GovernanceConfig",
    "load_governance_config",
    "run_governance_benchmark",
    "validate_governance_setup",
    "aggregate_governance_shards",
    "build_shard_plan",
    "run_governance_shard",
    "run_governance_sharded",
    "shard_status",
]
