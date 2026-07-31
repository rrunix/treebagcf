"""Tests for the primal polish, dual-guided rounding, and the alpha library.

Three invariants matter:
- primal heuristics (polish, dual-guided rounding) must only ever produce
  genuinely feasible points at their true cost, never below the certified
  optimum, and never change what the search certifies;
- transferred dual entries (the alpha library) must stay admissible at
  queries they were NOT optimized for — the core transfer property;
- solve() with a library must certify the same optimum as without.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import calf
from calf.dual_lb import DualCostSplitPool
from calf.engine import _dataset_box, compile_rf
from calf.numba.kernels import (
    dual_guided_round_l1,
    greedy_round_l1,
    polish_l1,
    root_active,
)


@pytest.fixture(scope="module")
def forest():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=5, n_redundant=0, random_state=3
    )
    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0).fit(X, y)
    return rf, X, y


def _setup(rf, X):
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    crf = compile_rf(parsed)
    box_lo, box_hi = _dataset_box(di)
    active = root_active(crf.rules_lo_mat, crf.rules_hi_mat, box_lo, box_hi)
    return parsed, scale, crf, box_lo, box_hi, active


def _tree_votes(rf, x, target):
    return sum(int(est.predict(x[None, :])[0]) == target for est in rf.estimators_)


def test_polish_preserves_feasibility_and_reduces_cost(forest):
    rf, X, _ = forest
    parsed, scale, crf, box_lo, box_hi, active = _setup(rf, X)
    need = int(np.ceil(0.5 * parsed.n_trees))
    n_checked = 0
    for i in range(8):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        f32 = x0.astype(np.float32).astype(np.float64)
        rc, rx = greedy_round_l1(
            active, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id,
            crf.rules_class, box_lo, box_hi, f32, scale, crf.n_trees, need, target,
        )
        if not np.isfinite(rc):
            continue
        values = np.ascontiguousarray((crf.rules_class == target).astype(np.float64))
        pc, px = polish_l1(
            rx, f32, scale, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id,
            values, crf.n_trees, float(need), False,
            crf.threshold_offsets, crf.threshold_values, 24, 2,
        )
        n_checked += 1
        # never worse than its input, always at its true cost
        assert pc <= rc + 1e-12
        assert pc == pytest.approx(calf.l1_cost(f32, px, scale), abs=1e-9)
        # still a genuine counterfactual under sklearn's own trees
        assert _tree_votes(rf, px, target) >= need
        # a primal heuristic can never beat the certified optimum
        opt = calf.solve(rf, X, x0, target, max_iters=1_000_000)
        assert opt.proven_optimal
        assert pc >= opt.cost - 1e-9
    assert n_checked > 0


def test_polish_rejects_infeasible_input(forest):
    rf, X, _ = forest
    parsed, scale, crf, _, _, _ = _setup(rf, X)
    need = int(np.ceil(0.5 * parsed.n_trees))
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    f32 = x0.astype(np.float32).astype(np.float64)
    values = np.ascontiguousarray((crf.rules_class == target).astype(np.float64))
    # the factual itself does not reach the target: polish must refuse to
    # "improve" it rather than return a bogus low cost
    pc, px = polish_l1(
        f32, f32, scale, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id,
        values, crf.n_trees, float(need), False,
        crf.threshold_offsets, crf.threshold_values, 24, 2,
    )
    assert np.isinf(pc)
    assert np.array_equal(px, f32)


def test_dual_guided_round_is_valid(forest):
    rf, X, _ = forest
    parsed, scale, crf, box_lo, box_hi, active = _setup(rf, X)
    need = int(np.ceil(0.5 * parsed.n_trees))
    n_finite = 0
    for i in range(8):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        f32 = x0.astype(np.float32).astype(np.float64)
        pool = DualCostSplitPool(parsed, f32, scale, target, need)
        shim = SimpleNamespace(
            active_rules=active, box=SimpleNamespace(lo=box_lo, hi=box_hi)
        )
        bound = pool.optimize_root(shim, max_iters=300)
        if not len(pool):
            continue
        ap, an = pool.alphas[-1]
        rc, rx = dual_guided_round_l1(
            active, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id,
            crf.rules_class, box_lo, box_hi, f32, scale,
            np.ascontiguousarray(ap), np.ascontiguousarray(an),
            crf.n_trees, need, target,
        )
        if not np.isfinite(rc):
            continue
        n_finite += 1
        assert _tree_votes(rf, rx, target) >= need
        assert rc == pytest.approx(calf.l1_cost(f32, rx, scale), abs=1e-9)
        opt = calf.solve(rf, X, x0, target, max_iters=1_000_000)
        assert opt.proven_optimal
        assert rc >= opt.cost - 1e-9
        assert bound <= opt.cost + 1e-9  # the root dual bound stays admissible
    assert n_finite > 0


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_solve_polish_invariance(forest, voting):
    """Polish is primal-only: certified optima must match with it on or off."""
    rf, X, _ = forest
    for i in range(4):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        base = calf.solve(
            rf, X, x0, target, voting=voting, dual_round_polish=False,
            dual_lb_stall_window=1, max_iters=400_000,
        )
        pol = calf.solve(
            rf, X, x0, target, voting=voting, dual_round_polish=True,
            dual_lb_stall_window=1, max_iters=400_000,
        )
        assert base.proven_optimal and pol.proven_optimal
        assert pol.cost == pytest.approx(base.cost, abs=1e-9)


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_alpha_library_warmup_entries_transfer(forest, voting):
    """Warmed entries stay admissible at queries they were not optimized for."""
    rf, X, _ = forest
    parsed, scale, crf, box_lo, box_hi, active = _setup(rf, X)
    for target in (0, 1):
        lib = calf.AlphaLibrary(parsed, scale, voting=voting)
        added = lib.warmup(X, target, k=3, root_iters=200)
        assert added > 0
        entries = lib.entries_for(target)
        # evaluate every entry at OTHER factuals' root boxes
        qi = np.where(parsed.predict(X) != target)[0][:3]
        for i in qi:
            x0 = X[i]
            opt = calf.solve(
                rf, X, x0, target, voting=voting, max_iters=1_000_000
            )
            if not opt.proven_optimal:
                continue
            f32 = x0.astype(np.float32).astype(np.float64)
            shim = SimpleNamespace(
                active_rules=active, box=SimpleNamespace(lo=box_lo, hi=box_hi)
            )
            if voting == "hard":
                need = int(np.ceil(0.5 * parsed.n_trees))
                pool = DualCostSplitPool(parsed, f32, scale, target, need)
                got = pool._compacted(shim)
                assert got is not None
                req_pos, req_neg, tree_pos, starts, trees = got
                for ap, an in entries:
                    val = pool._entry_bound(
                        ap, an, req_pos, req_neg, tree_pos, starts, trees
                    )
                    assert val <= opt.cost + 1e-9
            else:
                from calf.dual_lb import AdditiveDualPool

                p1 = parsed.rules_proba1.astype(np.float64)
                values = p1 if target == 1 else 1.0 - p1
                pool = AdditiveDualPool(parsed, f32, scale, values, 0.5 * parsed.n_trees)
                # seed() itself scores each entry at this root; verify via the
                # pool's static evaluation of the seeded entries
                pool.seed(shim, entries, top_k=len(entries))
                for w, (_, _, lam) in zip(pool.static_w, pool.entries, strict=True):
                    tids = parsed.rules_tree_id[active]
                    min_w = np.full(parsed.n_trees, np.inf)
                    np.minimum.at(min_w, tids, w[active])
                    val = float(min_w.sum()) + lam * pool.tau
                    assert val <= opt.cost + 1e-9


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_solve_with_alpha_library_matches_and_harvests(forest, voting):
    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    lib = calf.AlphaLibrary(parsed, scale, voting=voting)
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    lib.warmup(X, target, k=2, root_iters=200)
    n0 = len(lib)
    assert n0 > 0
    base = calf.solve(
        rf, X, x0, target, voting=voting, dual_lb_stall_window=1,
        max_iters=400_000,
    )
    warm = calf.solve(
        rf, X, x0, target, voting=voting, dual_lb_stall_window=1,
        alpha_library=lib, max_iters=400_000,
    )
    assert base.proven_optimal and warm.proven_optimal
    assert warm.cost == pytest.approx(base.cost, abs=1e-9)
    # the engaged pool's fresh entries were harvested back
    assert len(lib) > n0


def test_dual_round_guided_flag_invariance(forest):
    """dual_round_guided only changes primal quality, never the certified cost."""
    rf, X, _ = forest
    for i in range(3):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        on = calf.solve(
            rf, X, x0, target, dual_lb_stall_window=1, max_iters=400_000,
        )
        off = calf.solve(
            rf, X, x0, target, dual_lb_stall_window=1, max_iters=400_000,
            dual_round_guided=False,
        )
        assert on.proven_optimal and off.proven_optimal
        assert on.cost == pytest.approx(off.cost, abs=1e-9)


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_alpha_library_save_load_roundtrip(forest, tmp_path, voting):
    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    scale = calf.l1_scale(calf.from_array(X))
    lib = calf.AlphaLibrary(parsed, scale, voting=voting)
    lib.warmup(X, 1, k=2, root_iters=100)
    assert len(lib) > 0
    path = tmp_path / f"alphas_{voting}.joblib"
    lib.save(path)
    loaded = calf.AlphaLibrary.load(path, parsed, scale)
    assert loaded.voting == voting and loaded.cap == lib.cap
    assert len(loaded) == len(lib)
    for a, b in zip(lib.entries_for(1), loaded.entries_for(1), strict=True):
        assert len(a) == len(b)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        if voting == "soft":
            assert a[2] == b[2]
    # the loaded library must be directly usable by solve
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    res = calf.solve(
        rf, X, x0, target, voting=voting, alpha_library=loaded,
        dual_lb_stall_window=1, max_iters=400_000,
    )
    assert res.proven_optimal
    # shape guard: loading against a different forest must fail loudly
    X2, y2 = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, random_state=9
    )
    rf2 = RandomForestClassifier(n_estimators=7, max_depth=3, random_state=9).fit(X2, y2)
    parsed2 = calf.parse_sklearn_rf(rf2)
    scale2 = calf.l1_scale(calf.from_array(X2))
    with pytest.raises(ValueError, match="alpha file"):
        calf.AlphaLibrary.load(path, parsed2, scale2)


def test_alpha_library_voting_mismatch(forest):
    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    scale = calf.l1_scale(calf.from_array(X))
    lib = calf.AlphaLibrary(parsed, scale, voting="soft")
    with pytest.raises(ValueError, match="voting"):
        calf.solve(rf, X, X[0], 1, voting="hard", alpha_library=lib)
