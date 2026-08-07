from __future__ import annotations

import pandas as pd
import pytest

from tabpollution.governance.data import exact_mixture


def _source(label: int, prefix: str, rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": [f"{prefix}-{i}" for i in range(rows)],
        "feature": range(rows),
        "source_label": label,
    })


def test_replace_contamination_preserves_budget_and_rate() -> None:
    bag = exact_mixture(_source(0, "r"), _source(1, "s"), 100, .10, 2026, mode="replace")
    assert len(bag) == 100
    assert bag["source_label"].mean() == pytest.approx(.10)


def test_append_contamination_preserves_real_budget_and_final_rate() -> None:
    bag = exact_mixture(_source(0, "r"), _source(1, "s"), 90, .10, 2026, mode="append")
    assert (bag["source_label"] == 0).sum() == 90
    assert (bag["source_label"] == 1).sum() == 10
    assert bag["source_label"].mean() == pytest.approx(.10)


def test_append_rejects_undefined_full_contamination() -> None:
    with pytest.raises(ValueError, match="undefined"):
        exact_mixture(_source(0, "r"), _source(1, "s"), 100, 1.0, 2026, mode="append")
