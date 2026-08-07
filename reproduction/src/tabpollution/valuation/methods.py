"""KNN-Shapley and Data-OOB paper-aligned runthrough implementations."""

from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeClassifier


def knn_shapley(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                y_val: np.ndarray, k: int = 5) -> np.ndarray:
    """Exact O(N log N) unweighted KNN-Shapley recurrence from Jia et al."""
    n = len(y_train); values = np.zeros(n, dtype=float)
    distances = pairwise_distances(x_val, x_train)
    for d, target in zip(distances, y_val):
        order = np.argsort(d)
        contrib = np.zeros(n, dtype=float)
        contrib[order[-1]] = float(y_train[order[-1]] == target) / n
        for pos in range(n - 2, -1, -1):
            i, nxt = order[pos], order[pos + 1]
            rank = pos + 1
            label_delta = int(y_train[i] == target) - int(y_train[nxt] == target)
            contrib[i] = contrib[nxt] + label_delta * min(k, rank) / (k * rank)
        values += contrib
    return values / len(y_val)


def _knn_utility(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, indices: tuple[int, ...], k: int) -> float:
    if not indices: return 0.
    idx = np.asarray(indices); distances = pairwise_distances(x_val, x_train[idx])
    utility = []
    for row in distances:
        near = idx[np.argsort(row)[:min(k, len(idx))]]
        # Jia et al.'s recurrence values the soft KNN utility: matching
        # neighbours divided by the fixed K (also when |S| < K).
        utility.append(float(np.sum(y_train[near] == y_val[len(utility)])) / k)
    return float(np.mean(utility))


def exact_knn_shapley(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                      y_val: np.ndarray, k: int = 3) -> np.ndarray:
    n = len(y_train)
    if n > 12: raise ValueError("exact verification limited to N<=12")
    result = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n-size-1) / math.factorial(n)
            for subset in combinations(others, size):
                before = _knn_utility(x_train, y_train, x_val, y_val, subset, k)
                after = _knn_utility(x_train, y_train, x_val, y_val, subset + (i,), k)
                result[i] += weight * (after-before)
    return result


def data_oob(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 80,
             seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Value each point only with weak learners for which it is out-of-bag."""
    rng = np.random.default_rng(seed); n = len(y_train)
    value_sum = np.zeros(n); coverage = np.zeros(n, dtype=int)
    for b in range(n_estimators):
        bootstrap = rng.integers(0, n, size=n); in_bag = np.zeros(n, bool); in_bag[np.unique(bootstrap)] = True
        oob = np.flatnonzero(~in_bag)
        if not len(oob): continue
        model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=seed+b)
        model.fit(x_train[bootstrap], y_train[bootstrap]); pred = model.predict(x_train[oob])
        value_sum[oob] += (pred == y_train[oob]).astype(float); coverage[oob] += 1
    values = np.divide(value_sum, coverage, out=np.full(n, np.nan), where=coverage > 0)
    return values, coverage
