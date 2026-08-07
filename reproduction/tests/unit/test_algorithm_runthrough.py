from __future__ import annotations

import numpy as np
import pandas as pd

from tabpollution.detectors.base import feature_frame, serialize_record
from tabpollution.detectors.deep import GradReverse
from tabpollution.quantification import ScoreQuantifier
from tabpollution.valuation import data_oob, exact_knn_shapley, knn_shapley


def test_provenance_is_removed():
    frame=pd.DataFrame({"row_id":["a"],"generator_name":["x"],"table_id":["t"],
                        "source_type":["real"],"p":[.25],"source_synth_row_id":[None],"value":[1]})
    assert feature_frame(frame).columns.tolist()==["value"]


def test_serialization_is_deterministic_and_shuffle_seeded():
    row=pd.Series({"a":1,"b":" x "})
    assert serialize_record(row)=="a:1 | b:x"
    assert serialize_record(row,shuffle=True,seed=3)==serialize_record(row,shuffle=True,seed=3)


def test_gradient_reversal_sign():
    import torch
    x=torch.tensor([2.],requires_grad=True); y=GradReverse.apply(x,.5); y.sum().backward()
    assert torch.allclose(x.grad,torch.tensor([-.5]))


def test_quantifier_formulas_and_bounds():
    scores=np.array([.05,.1,.2,.8,.9,.95]); labels=np.array([0,0,0,1,1,1])
    test=np.array([.1,.2,.85,.9])
    for method in ("cc","pcc","acc","pacc","emq","hdy","dys","median_sweep"):
        result=ScoreQuantifier(method).fit(scores,labels).predict_prevalence(test)
        assert 0 <= result["clipped"] <= 1


def test_quantifier_unstable_denominator_is_explicit():
    q=ScoreQuantifier("acc").fit(np.array([.6,.7,.6,.7]),np.array([0,0,1,1]))
    try: q.predict_prevalence(np.array([.6]))
    except ValueError as e: assert "unstable_denominator" in str(e)
    else: raise AssertionError("expected structured failure")


def test_knn_shapley_matches_exhaustive():
    x=np.array([[0.],[1.],[2.],[3.]]); y=np.array([0,0,1,1]); xv=np.array([[.2],[2.8]]); yv=np.array([0,1])
    fast=knn_shapley(x,y,xv,yv,k=3); exact=exact_knn_shapley(x,y,xv,yv,k=3)
    assert np.max(np.abs(fast-exact)) < 1e-10


def test_data_oob_has_coverage_and_only_oob_values():
    x=np.arange(80,dtype=float).reshape(40,2); y=np.array([0,1]*20)
    values,coverage=data_oob(x,y,n_estimators=100,seed=2)
    assert coverage.min()>0 and np.isfinite(values).all()
