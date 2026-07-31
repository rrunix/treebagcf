"""Soft-voting correctness anchor: the engine vs an independent brute force.

Soft (probability-averaged) voting must agree with an exhaustive enumeration
of the threshold-grid cells, and its counterfactual must actually flip
sklearn's soft prediction (``rf.predict`` IS the soft vote for binary forests).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import calf
from calf.numba.kernels import refine_l1
from baselines.grid import feature_edges, iter_cells


def _forest(seed, n_estimators=9, max_depth=4, n_features=5):
    X, y = make_classification(
        n_samples=200, n_features=n_features, n_informative=n_features - 1,
        n_redundant=0, random_state=seed,
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed
    ).fit(X, y)
    return rf, X


def _soft_sum(parsed, xp, target):
    """Sum over trees of P(target | fired leaf) at point xp (parsed semantics)."""
    inside = np.all(xp > parsed.rules_lo_mat, axis=1) & np.all(xp <= parsed.rules_hi_mat, axis=1)
    total = 0.0
    seen = set()
    for rid in np.flatnonzero(inside):
        t = int(parsed.rules_tree_id[rid])
        if t in seen:
            continue
        seen.add(t)
        p1 = float(parsed.rules_proba1[rid])
        total += p1 if target == 1 else (1.0 - p1)
    return total


def _brute_soft_optimum(rf, X, x, target, scale):
    """Independent soft optimum using calf's own semantics: enumerate every
    threshold-grid cell, project with the identical float32-grid refinement, and
    keep the cheapest cell whose parsed soft vote reaches the target.  Same oracle
    and projection as the engine, just an independent exhaustive enumeration."""
    parsed = calf.parse_sklearn_rf(rf)
    x = x.astype(np.float32).astype(np.float64)
    box_lo = X.min(axis=0) - 1e-12
    box_hi = X.max(axis=0)
    edges = feature_edges(rf, box_lo, box_hi)
    thresh = 0.5 * len(rf.estimators_)
    best = np.inf
    for lo, hi in iter_cells(edges, max_cells=500_000):
        xp, _ = refine_l1(lo, hi, x, scale)
        s = _soft_sum(parsed, xp, target)
        ok = s > thresh if target == 1 else s >= thresh
        if ok:
            c = float(np.sum(scale * np.abs(x - xp)))
            if c < best:
                best = c
    return best


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("target", [0, 1])
def test_soft_matches_bruteforce(seed, target):
    # small forest keeps the brute-force grid tractable
    rf, X = _forest(seed, n_estimators=6, max_depth=3, n_features=4)
    scale = np.ones(X.shape[1])
    checked = 0
    for i in range(4):
        x = X[i]
        res = calf.extract_counterfactual(
            calf.parse_sklearn_rf(rf), calf.from_array(X), x, target, scale,
            voting="soft", max_iters=500_000,
        )
        brute = _brute_soft_optimum(rf, X, x, target, scale)
        if not np.isfinite(brute):
            assert not res.found, (seed, target, i)
            continue
        checked += 1
        assert res.found and res.proven_optimal, (seed, target, i)
        assert res.cost == pytest.approx(brute, abs=1e-5), (seed, target, i, res.cost, brute)
        # the returned point must actually flip sklearn's soft prediction
        assert int(rf.predict(res.x[None, :])[0]) == target
    assert checked > 0


def test_soft_voting_is_binary_only():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_classes=3, random_state=0
    )
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(X, y)
    with pytest.raises(NotImplementedError):
        calf.extract_counterfactual(
            calf.parse_sklearn_rf(rf), calf.from_array(X), X[0], 1,
            np.ones(5), voting="soft",
        )
