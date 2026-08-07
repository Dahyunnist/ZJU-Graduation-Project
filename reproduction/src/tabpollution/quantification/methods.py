"""Classic binary quantification on frozen detector scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from scipy.optimize import minimize_scalar


METHODS = ("cc", "pcc", "acc", "pacc", "emq", "hdy", "dys", "median_sweep")


def available_quantifiers() -> tuple[str, ...]: return METHODS


def quapy_formula_crosscheck() -> dict[str, Any]:
    """Execute QuaPy 0.2.0 equivalents for the four elementary formulas."""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from quapy.method.aggregative import CC, PCC, ACC, PACC
    class Passthrough(BaseEstimator, ClassifierMixin):
        def fit(self, X, y): self.classes_ = np.array([0, 1]); return self
        def predict_proba(self, X):
            s=np.asarray(X).ravel(); return np.c_[1-s, s]
        def predict(self, X): return (np.asarray(X).ravel() >= .5).astype(int)
    calibration=np.array([.05,.1,.2,.8,.9,.95]); labels=np.array([0,0,0,1,1,1]); test=np.array([.1,.2,.85,.9])
    clf=Passthrough().fit(calibration[:,None],labels)
    qcc=CC(clf,fit_classifier=False); qpcc=PCC(clf,fit_classifier=False)
    qacc=ACC(clf,fit_classifier=False); qacc.aggregation_fit(clf.predict(calibration[:,None]),labels)
    qpacc=PACC(clf,fit_classifier=False); qpacc.aggregation_fit(clf.predict_proba(calibration[:,None]),labels)
    official={"cc":qcc.quantify(test[:,None])[1],"pcc":qpcc.quantify(test[:,None])[1],
              "acc":qacc.quantify(test[:,None])[1],"pacc":qpacc.quantify(test[:,None])[1]}
    local={m:ScoreQuantifier(m).fit(calibration,labels).predict_prevalence(test)["clipped"] for m in official}
    errors={m:abs(float(local[m])-float(official[m])) for m in official}
    return {"quapy_version":"0.2.0","official_positive_prevalence":official,"local_positive_prevalence":local,
            "absolute_errors":errors,"max_absolute_error":max(errors.values()),"tolerance":1e-7,
            "passed":max(errors.values())<1e-7}


class ScoreQuantifier:
    def __init__(self, method: str, bins: tuple[int, ...] = (8, 16, 32)):
        if method not in METHODS: raise ValueError(method)
        self.method, self.bins = method, bins
        self.state: dict[str, Any] = {}

    def fit(self, calibration_scores: np.ndarray, calibration_labels: np.ndarray, **context: Any) -> "ScoreQuantifier":
        s, y = np.asarray(calibration_scores, float), np.asarray(calibration_labels, int)
        if set(np.unique(y)) != {0, 1}: raise ValueError("calibration requires both labels")
        threshold = float(context.get("threshold", .5))
        hard = s >= threshold
        self.state = {
            "threshold": threshold, "train_prevalence": float(y.mean()),
            "tpr": float(hard[y == 1].mean()), "fpr": float(hard[y == 0].mean()),
            "soft_tpr": float(s[y == 1].mean()), "soft_fpr": float(s[y == 0].mean()),
            "positive_scores": s[y == 1].tolist(), "negative_scores": s[y == 0].tolist(),
        }
        return self

    @staticmethod
    def _adjust(value: float, fpr: float, tpr: float) -> float:
        denom = tpr - fpr
        if abs(denom) < 1e-8: raise ValueError("unstable_denominator")
        return (value - fpr) / denom

    def _distribution_match(self, test: np.ndarray, distance: str) -> float:
        pos, neg = np.asarray(self.state["positive_scores"]), np.asarray(self.state["negative_scores"])
        estimates = []
        for b in self.bins:
            hp, _ = np.histogram(pos, bins=b, range=(0, 1), density=False); hp = hp / max(1, hp.sum())
            hn, _ = np.histogram(neg, bins=b, range=(0, 1), density=False); hn = hn / max(1, hn.sum())
            ht, _ = np.histogram(test, bins=b, range=(0, 1), density=False); ht = ht / max(1, ht.sum())
            def objective(p: float) -> float:
                mix = p * hp + (1-p) * hn
                if distance == "hellinger": return float(np.sqrt(((np.sqrt(ht)-np.sqrt(mix))**2).sum()) / np.sqrt(2))
                return float(np.abs(ht-mix).sum())
            estimates.append(float(minimize_scalar(objective, bounds=(0, 1), method="bounded").x))
        self.state["last_bins"] = list(self.bins); self.state["last_bin_estimates"] = estimates
        return float(np.median(estimates))

    def _emq(self, test: np.ndarray) -> float:
        p0 = np.clip(self.state["train_prevalence"], 1e-6, 1-1e-6); p = p0
        for iteration in range(1, 501):
            numerator = (p / p0) * test
            denominator = numerator + ((1-p)/(1-p0)) * (1-test)
            posterior = numerator / np.clip(denominator, 1e-12, None)
            new = float(posterior.mean())
            if abs(new-p) < 1e-8: break
            p = new
        self.state["iterations"] = iteration; self.state["converged"] = iteration < 500
        self.state["initial_prior"] = p0; self.state["final_prior"] = new
        return new

    def _median_sweep(self, test: np.ndarray) -> float:
        pos, neg = np.asarray(self.state["positive_scores"]), np.asarray(self.state["negative_scores"])
        vals = []
        thresholds = np.linspace(.05, .95, 19)
        for t in thresholds:
            tpr, fpr, cc = (pos >= t).mean(), (neg >= t).mean(), (test >= t).mean()
            if abs(tpr-fpr) >= .25:
                vals.append(self._adjust(float(cc), float(fpr), float(tpr)))
        if not vals: raise ValueError("no_valid_median_sweep_threshold")
        self.state["thresholds"] = thresholds.tolist(); self.state["valid_thresholds"] = len(vals)
        return float(np.median(vals))

    def predict_prevalence(self, test_scores: np.ndarray, **context: Any) -> dict[str, Any]:
        s = np.asarray(test_scores, float)
        if self.method == "cc": raw = float((s >= self.state["threshold"]).mean())
        elif self.method == "pcc": raw = float(s.mean())
        elif self.method == "acc": raw = self._adjust(float((s >= self.state["threshold"]).mean()), self.state["fpr"], self.state["tpr"])
        elif self.method == "pacc": raw = self._adjust(float(s.mean()), self.state["soft_fpr"], self.state["soft_tpr"])
        elif self.method == "emq": raw = self._emq(s)
        elif self.method == "hdy": raw = self._distribution_match(s, "hellinger")
        elif self.method == "dys": raw = self._distribution_match(s, "tops")
        else: raw = self._median_sweep(s)
        return {"raw": raw, "clipped": float(np.clip(raw, 0, 1)), "out_of_range": bool(raw < 0 or raw > 1),
                "diagnostics": {k: v for k, v in self.state.items() if k not in {"positive_scores", "negative_scores"}}}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"method": self.method, "bins": self.bins, "state": self.state}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScoreQuantifier":
        data = json.loads(Path(path).read_text(encoding="utf-8")); obj = cls(data["method"], tuple(data["bins"])); obj.state = data["state"]; return obj
