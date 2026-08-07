"""SDV 1.x adapters, imported lazily so pure-data tests do not need SDV."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from tabpollution.generators.base import GeneratorError, model_frame, seed_everything


SUPPORTED_GENERATORS = {"GaussianCopula", "CTGAN", "TVAE"}


class SDVGenerator:
    def __init__(self, generator_name: str, config: dict[str, Any] | None = None):
        if generator_name not in SUPPORTED_GENERATORS:
            raise GeneratorError(f"Unknown generator: {generator_name}")
        self.generator_name = generator_name
        self.config = dict(config or {})
        self.model: Any | None = None
        self.generator_seed: int | None = None
        self.fit_seconds: float | None = None
        self.sample_times: dict[str, float] = {}

    @staticmethod
    def build_metadata(records: pd.DataFrame) -> Any:
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(model_frame(records))
        return metadata

    def _model_class(self) -> Any:
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

        return {
            "GaussianCopula": GaussianCopulaSynthesizer,
            "CTGAN": CTGANSynthesizer,
            "TVAE": TVAESynthesizer,
        }[self.generator_name]

    def fit(self, real_records: pd.DataFrame, metadata: Any, seed: int, **config: Any) -> None:
        if self.model is not None:
            raise GeneratorError("Generator instance is already fitted")
        resolved = {**self.config, **config}
        seed_everything(seed)
        model_class = self._model_class()
        self.model = model_class(metadata, **resolved)
        started = time.perf_counter()
        self.model.fit(model_frame(real_records))
        self.fit_seconds = time.perf_counter() - started
        self.generator_seed = seed
        self.config = resolved

    def sample(self, n: int, sample_seed: int, pool_name: str) -> pd.DataFrame:
        if self.model is None:
            raise GeneratorError("Generator is not fitted")
        if n <= 0:
            raise GeneratorError("Sample size must be positive")
        seed_everything(sample_seed)
        if hasattr(self.model, "_set_random_state"):
            self.model._set_random_state(sample_seed)
        started = time.perf_counter()
        result = self.model.sample(num_rows=n)
        self.sample_times[pool_name] = time.perf_counter() - started
        return result

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise GeneratorError("Cannot save an unfitted generator")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, generator_name: str, path: str | Path, config: dict[str, Any] | None = None) -> "SDVGenerator":
        instance = cls(generator_name, config)
        instance.model = load_sdv_synthesizer(generator_name, path, instance._model_class())
        return instance

    def get_provenance(self) -> dict[str, Any]:
        return {
            "implementation": "sdv.single_table",
            "generator_name": self.generator_name,
            "generator_seed": self.generator_seed,
            "config": self.config,
            "fit_seconds": self.fit_seconds,
            "sample_times": self.sample_times,
        }


def create_generator(generator_name: str, config: dict[str, Any] | None = None) -> SDVGenerator:
    return SDVGenerator(generator_name, config)


def load_sdv_synthesizer(generator_name: str, path: str | Path, model_class: Any | None = None) -> Any:
    """Load through the SDV 1.37+ utility with a compatibility fallback."""
    try:
        from sdv.utils import load_synthesizer

        return load_synthesizer(Path(path))
    except (ImportError, AttributeError):
        if model_class is None:
            model_class = SDVGenerator(generator_name)._model_class()
        return model_class.load(Path(path))
