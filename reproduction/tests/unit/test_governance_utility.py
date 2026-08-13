from __future__ import annotations

import math

import pandas as pd

from tabpollution.governance.pipeline import _utility


def test_cleanup_single_class_train_has_undefined_utility() -> None:
    train = pd.DataFrame({"feature": [1, 2, 3] * 4, "target": ["0"] * 12})
    test = pd.DataFrame({"feature": list(range(20)), "target": ["0", "1"] * 10})
    assert math.isnan(_utility(train, test, "target", seed=2027, threads=1))


def test_malformed_multiclass_target_still_fails() -> None:
    train = pd.DataFrame({"feature": list(range(12)), "target": ["0", "1", "2"] * 4})
    test = pd.DataFrame({"feature": list(range(20)), "target": ["0", "1"] * 10})
    try:
        _utility(train, test, "target", seed=2027, threads=1)
    except ValueError as exc:
        assert "must be binary" in str(exc)
    else:
        raise AssertionError("Multiclass downstream targets must remain invalid")
