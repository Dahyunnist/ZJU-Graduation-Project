import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("e0", Path(__file__).parents[2] / "scripts" / "reanalyze_e0.py")
e0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0)


def test_dedup_rejects_conflicting_values_and_missingness():
    for values in [[0.5, 0.6], [0.5, np.nan]]:
        with pytest.raises(ValueError, match="Conflicting"):
            e0.unique_checked(pd.DataFrame({"id": [1, 1], "value": values}), ["id"], ["value"])


def test_dedup_preserves_zero_and_all_missing():
    result = e0.unique_checked(pd.DataFrame({"id": [1, 1, 2, 2], "value": [0, 0, np.nan, np.nan]}), ["id"], ["value"])
    assert len(result) == 2
    assert result.iloc[0].value == 0
    assert np.isnan(result.iloc[1].value)


def test_actual_and_nominal_mixture_validation():
    frame = pd.DataFrame({"nominal_prevalence": [.05, .1, .75], "contamination_mode": ["append"] * 3, "bag_size": [1053, 1111, 4000], "true_prevalence": [53/1053, 111/1111, .75]})
    e0.check_mixture(frame, 1000)
    assert not frame.true_prevalence.isin([.05, .1]).any()
    assert frame.nominal_prevalence.isin([.05, .1]).sum() == 2
    frame.loc[0, "true_prevalence"] = .05
    with pytest.raises(ValueError, match="prevalence"):
        e0.check_mixture(frame, 1000)


def test_metric_coverage_does_not_treat_missing_as_zero():
    frame = pd.DataFrame({"group": [1, 1, 1], "value": [0, 1, np.nan]})
    result = e0.summarize(frame, ["group"], ["value"]).iloc[0]
    assert (result.n_total, result.n_valid, result.n_missing, result["mean"]) == (3, 2, 1, .5)
