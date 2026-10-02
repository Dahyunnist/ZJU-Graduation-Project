"""Small executable checks supporting the 2026-10-02 research audit.

These characterize existing mixture semantics and mathematical negative
controls. They do not certify causal validity of the historical experiment.
"""

import numpy as np
import pandas as pd
import pytest

from tabpollution.governance.data import exact_mixture
from tabpollution.governance.metrics import analytical_positive_rate


def sources():
    real = pd.DataFrame({"record_id": [f"r{i}" for i in range(5000)], "source_label": 0})
    synthetic = pd.DataFrame({"record_id": [f"s{i}" for i in range(5000)], "source_label": 1})
    return real, synthetic


@pytest.mark.parametrize("rate,expected_size", [(0.05, 1053), (0.10, 1111), (0.75, 4000)])
def test_append_fixes_real_budget_not_total_budget(rate, expected_size):
    bag = exact_mixture(*sources(), 1000, rate, 2026, "append")
    assert len(bag) == expected_size
    assert (bag.source_label == 0).sum() == 1000
    if rate in (0.05, 0.10):
        # Filtering actual prevalence with exact .isin([.05, .10]) drops these.
        assert bag.source_label.mean() not in (0.05, 0.10)


def test_zero_prevalence_modes_share_the_same_real_bag():
    real, synthetic = sources()
    replace = exact_mixture(real, synthetic, 1000, 0, 2026, "replace")
    append = exact_mixture(real, synthetic, 1000, 0, 2026, "append")
    pd.testing.assert_frame_equal(replace, append)


def test_error_decomposition_is_identity_even_for_arbitrary_estimate():
    prevalence, observed, estimate = 0.1, 0.3, 0.9
    reference = analytical_positive_rate(prevalence, tpr=0.7, fpr=0.2)["expected_positive_rate"]
    terms = (reference - prevalence, observed - reference, estimate - observed)
    assert sum(terms) == pytest.approx(estimate - prevalence)


def test_strictly_increasing_map_with_transported_threshold_preserves_action():
    scores = np.array([0.01, 0.12, 0.5, 0.7, 0.99])
    threshold = 0.5
    transform = lambda x: 1 / (1 + np.exp(-(2 * x - 0.3)))
    np.testing.assert_array_equal(scores >= threshold, transform(scores) >= transform(threshold))
