"""Sample-level synthetic tabular data detectors."""

from .classical import C2STDetector, Char3GramDetector
from .deep import DeepTextDetector

__all__ = ["C2STDetector", "Char3GramDetector", "DeepTextDetector"]
