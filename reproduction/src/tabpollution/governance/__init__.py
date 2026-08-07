"""End-to-end synthetic tabular contamination governance benchmark."""

from .config import GovernanceConfig, load_governance_config
from .pipeline import run_governance_benchmark, validate_governance_setup

__all__ = [
    "GovernanceConfig",
    "load_governance_config",
    "run_governance_benchmark",
    "validate_governance_setup",
]
