"""Tests for the exact LP dual solve (diagnostic) and the stall-dump hook.

The LP is the exact maximum of the additive dual family at a node, so it must
(a) dominate any ascent result and (b) stay admissible (<= true optimum).
In-search LP delivery was tried and pruned (2026-07-12: lp_stall_120,
lp_pool_120, deep_120 — no certified-outcome movement); the LP remains as the
offline diagnosis ceiling (research/stall_diag) and as an optional extra in
``AlphaLibrary.warmup(lp=True)``, a build-time step off the query path.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import calf
from calf.dual_lb import AdditiveDualPool
from calf.engine import _dataset_box, compile_rf
from calf.numba.kernels import root_active


@pytest.fixture(scope="module")
def forest():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=5, n_redundant=0, random_state=5
    )
    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0).fit(X, y)
    return rf, X, y


def test_lp_dominates_ascent_and_stays_admissible(forest):
    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    crf = compile_rf(parsed)
    box_lo, box_hi = _dataset_box(di)
    active = root_active(crf.rules_lo_mat, crf.rules_hi_mat, box_lo, box_hi)
    tau = 0.5 * parsed.n_trees
    n_checked = 0
    for i in range(4):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        opt = calf.solve(rf, X, x0, target, voting="soft", max_iters=1_000_000)
        if not opt.proven_optimal:
            continue
        n_checked += 1
        f32 = x0.astype(np.float32).astype(np.float64)
        p1 = parsed.rules_proba1.astype(np.float64)
        values = p1 if target == 1 else 1.0 - p1
        pool = AdditiveDualPool(parsed, f32, scale, values, tau)
        shim = SimpleNamespace(
            active_rules=active, box=SimpleNamespace(lo=box_lo, hi=box_hi)
        )
        b_ascent = pool.optimize_root(shim, incumbent=opt.cost, max_iters=500)
        b_lp = pool.lp_optimize_at(shim)
        # exact ceiling of the family: dominates ascent, stays admissible
        assert b_lp >= b_ascent - 1e-6, (i, b_lp, b_ascent)
        assert b_lp <= opt.cost + 1e-9, (i, b_lp, opt.cost)
        # the repaired LP entry entered the pool and is usable
        assert len(pool) == 2
    assert n_checked > 0


def test_warmup_lp_adds_admissible_entries(forest):
    """LP-augmented warm-up harvests more entries; seeding stays invariant."""
    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    lib_asc = calf.AlphaLibrary(parsed, scale, voting="soft")
    lib_lp = calf.AlphaLibrary(parsed, scale, voting="soft")
    preds = rf.predict(X)
    n_asc = lib_asc.warmup(X, 1, k=4, root_iters=200, preds=preds)
    n_lp = lib_lp.warmup(X, 1, k=4, root_iters=200, preds=preds, lp=True)
    assert n_lp > n_asc  # each centroid contributes the LP entry on top
    for ap, an, lam in lib_lp.entries_for(1):
        assert lam >= 0.0
        assert (ap >= 0).all() and (an >= 0).all()
        assert ap.sum(axis=0).max() <= 1.0 + 1e-9
        assert an.sum(axis=0).max() <= 1.0 + 1e-9
    # certified costs must not change when seeding from the LP-warmed library
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    base = calf.solve(rf, X, x0, target, voting="soft", max_iters=400_000)
    warm = calf.solve(rf, X, x0, target, voting="soft", max_iters=400_000,
                        alpha_library=lib_lp)
    assert base.proven_optimal and warm.proven_optimal
    assert warm.cost == pytest.approx(base.cost, abs=1e-9)


def test_stall_dump_collects_snapshots(forest):
    rf, X, _ = forest
    n_dumped = 0
    for i in range(6):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        dump: list = []
        res = calf.solve(
            rf, X, x0, target, voting="soft", dual_lb_stall_window=1,
            dual_lb_reopt_stride=1, stall_dump=dump, stall_dump_max=8,
            max_iters=400_000,
        )
        assert res.proven_optimal
        for snap in dump:
            n_dumped += 1
            assert snap["box_lo"].shape == x0.shape
            assert snap["box_hi"].shape == x0.shape
            assert snap["active"].dtype == np.int64
            assert snap["lb"] <= snap["best_cost"] + 1e-12
            assert len(dump) <= 8
    assert n_dumped > 0  # at least one query must have stalled once
