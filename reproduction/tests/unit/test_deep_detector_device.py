from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tabpollution.detectors.deep import DeepTextDetector


def test_deep_detector_cpu_roundtrip(tmp_path: Path) -> None:
    records = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(24)],
        "table_id": ["a"] * 12 + ["b"] * 12,
        "value": np.linspace(-1, 1, 24),
        "category": ["real"] * 12 + ["synthetic"] * 12,
    })
    labels = np.array([0] * 12 + [1] * 12)
    tables = np.array([0] * 12 + [1] * 12)
    detector = DeepTextDetector(
        "datum_ta", seed=2026, dim=8, heads=2, layers=1,
        max_datum=16, max_columns=4, epochs=1, batch_size=8,
        device="cpu", table_classes=4,
    )
    detector.fit(records, labels, table_labels=tables)
    before = detector.predict_score(records)
    assert np.ptp(before) > 1e-8
    assert detector.best_epoch == 1
    assert np.isfinite(detector.best_validation_score_ptp) or np.isnan(detector.best_validation_score_ptp)
    path = tmp_path / "detector.pt"
    detector.save(path)
    restored = DeepTextDetector.load(path)
    after = restored.predict_score(records)
    assert restored.get_provenance()["device"] == "cpu"
    assert np.allclose(before, after)
    assert restored.best_epoch == detector.best_epoch
