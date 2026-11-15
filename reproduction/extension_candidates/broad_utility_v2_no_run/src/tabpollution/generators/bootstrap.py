"""Joint row-bootstrap baseline implementing the shared generator contract."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class BootstrapGenerator:
    def __init__(self):
        self._frame: pd.DataFrame | None = None
        self._seed: int | None = None
        self._input_sha256: str | None = None

    def fit(self, real_records: pd.DataFrame, metadata=None, seed: int = 1, **budget) -> None:
        if self._frame is not None:
            raise ValueError("BootstrapGenerator instance is already fitted")
        if real_records.empty:
            raise ValueError("BootstrapGenerator requires non-empty training data")
        self._frame = real_records.reset_index(drop=True).copy()
        self._seed = int(seed)
        canonical = self._frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        self._input_sha256 = hashlib.sha256(canonical).hexdigest()

    def sample(self, n: int, sample_seed: int, pool_name: str = "default") -> pd.DataFrame:
        if self._frame is None:
            raise ValueError("BootstrapGenerator is not fitted")
        if n <= 0:
            raise ValueError("sample size must be positive")
        indices = np.random.default_rng(int(sample_seed)).integers(0, len(self._frame), size=n)
        result = self._frame.iloc[indices].reset_index(drop=True).copy()
        result.insert(0, "record_id", [f"bootstrap:{pool_name}:{sample_seed}:{i}" for i in range(n)])
        return result

    def save(self, path: str | Path) -> None:
        if self._frame is None:
            raise ValueError("cannot save an unfitted BootstrapGenerator")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump({"frame": self._frame, "seed": self._seed,
                         "input_sha256": self._input_sha256}, stream)

    @classmethod
    def load(cls, generator_name: str, path: str | Path, config=None) -> "BootstrapGenerator":
        if generator_name != "Bootstrap":
            raise ValueError(f"unsupported generator {generator_name}")
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        instance = cls()
        instance._frame = payload["frame"]
        instance._seed = payload["seed"]
        instance._input_sha256 = payload["input_sha256"]
        return instance

    def metadata(self) -> dict:
        return {"generator_name": "Bootstrap", "generator_train_seed": self._seed,
                "input_sha256": self._input_sha256, "fit_seconds": 0.0,
                "budget": {"fit": "joint_row_resampling"}}

    def get_provenance(self) -> dict:
        return self.metadata()
