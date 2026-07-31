"""Smoke tests for the calf package.

Verify that the engine returns certified-optimal counterfactuals that actually
reach the target under the parsed forest semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import calf


@pytest.fixture(scope="module")
def forest():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0, random_state=0
    )
    rf = RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0).fit(X, y)
    return rf, X, y


def test_solve_returns_valid_certified_counterfactual(forest):
    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    for i in range(10):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        res = calf.solve(rf, X, x0, target, max_iters=1_000_000)
        assert res.found
        assert res.proven_optimal
        # the returned point must genuinely reach the target class
        assert int(parsed.predict(res.x)[0]) == target
        # ...and stay valid once sklearn casts it to float32 (its native grid):
        # a majority of the individual trees must vote the target class.
        tree_votes = sum(
            int(est.predict(res.x[None, :])[0]) == target for est in rf.estimators_
        )
        assert tree_votes >= int(np.ceil(0.5 * len(rf.estimators_)))
        # reported cost matches the scaled-L1 distance of the point.  The engine
        # operates on the float32 grid (sklearn's native precision), so it
        # measures the cost from the float32-snapped query.
        di = calf.from_array(X)
        scale = calf.l1_scale(di)
        x0_f32 = x0.astype(np.float32).astype(np.float64)
        assert res.cost == pytest.approx(calf.l1_cost(x0_f32, res.x, scale), abs=1e-9)


@pytest.mark.parametrize("seed", [2, 3, 4, 7, 10, 11])
def test_counterfactual_valid_under_sklearn_float32(seed):
    """Certified CFs must flip the forest under sklearn's own float32 evaluation.

    Regression test for the float32 boundary bug: sklearn casts queries to
    float32 and compares against float64 thresholds, so a float64 counterfactual
    a hair past a threshold could route back to the wrong leaf.  The engine now
    lands every coordinate on the float32 grid, on the correct side of the
    float64 faces, so a hard majority of the individual trees votes the target.
    """
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=5, n_redundant=0, random_state=seed
    )
    rf = RandomForestClassifier(n_estimators=9, max_depth=5, random_state=seed).fit(X, y)
    parsed = calf.parse_sklearn_rf(rf)
    target = 1
    need = int(np.ceil(0.5 * len(rf.estimators_)))
    checked = 0
    for qi in np.where(parsed.predict(X) != target)[0][:5]:
        res = calf.solve(rf, X, X[qi], target, max_iters=800_000)
        if not res.found:
            continue
        checked += 1
        # sklearn's individual trees evaluate in float32; a hard majority must
        # still land on the target class for the returned point.
        tree_votes = sum(
            int(est.predict(res.x[None, :])[0]) == target for est in rf.estimators_
        )
        assert tree_votes >= need, (seed, qi, tree_votes, need)
    assert checked > 0


def test_python_engine_pooled_frontier(forest):
    """The pooled frontier must not change the python engine's trajectory.

    Frontier nodes rehydrate their active sets at pop time (parking-lot hit or
    ``root_active`` recompute — see ``calf.engine._FrontierPool``).  The
    default (``active_cache_elems=None``) parks every array and always hits;
    with the parking lot disabled (``active_cache_elems=0``) every pop
    recomputes.  Identical iters/cost across both settings proves the
    recompute path is bit-equivalent to the incremental filter.  The certified
    cost must also match the njit facade's independent certificate.
    """
    rf, X, _ = forest
    for i in range(6):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        parked = calf.solve(rf, X, x0, target, engine="python", max_iters=200_000)
        recomputed = calf.solve(
            rf, X, x0, target, engine="python", max_iters=200_000,
            active_cache_elems=0,
        )
        assert parked.found and parked.proven_optimal
        assert recomputed.iters == parked.iters
        assert recomputed.cost == parked.cost
        facade = calf.solve(rf, X, x0, target, max_iters=1_000_000)
        assert facade.proven_optimal
        assert parked.cost == pytest.approx(facade.cost, abs=1e-9)


def test_python_engine_time_cap(forest):
    """``time_limit_s`` returns the anytime incumbent without changing proofs.

    A generous cap must not perturb the trajectory; a zero cap must return
    immediately with the anytime contract — nothing when nothing was found,
    or the initial UB (with a nonzero certified gap) when one is supplied.
    """
    rf, X, _ = forest
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    full = calf.solve(rf, X, x0, target, engine="python", max_iters=200_000)
    assert full.proven_optimal

    capped = calf.solve(
        rf, X, x0, target, engine="python", max_iters=200_000, time_limit_s=120.0
    )
    assert capped.iters == full.iters and capped.cost == full.cost
    assert capped.proven_optimal

    zero = calf.solve(
        rf, X, x0, target, engine="python", max_iters=200_000, time_limit_s=0.0
    )
    assert zero.iters == 0 and not zero.found and np.isinf(zero.cost)

    seeded = calf.solve(
        rf, X, x0, target, engine="python", max_iters=200_000, time_limit_s=0.0,
        initial_ub=(x0, 123.0),
    )
    assert seeded.iters == 0 and seeded.found and seeded.cost == 123.0
    assert seeded.optimality_gap > 0.0
    assert not seeded.proven_optimal


def test_solve_time_cap(forest):
    """``time_limit_s`` through ``calf.solve`` defaults: anytime contract.

    A generous cap changes nothing; a zero cap returns immediately —
    empty-handed, or with the initial UB and a nonzero gap.
    """
    rf, X, _ = forest
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    full = calf.solve(rf, X, x0, target, max_iters=1_000_000)
    assert full.proven_optimal

    capped = calf.solve(
        rf, X, x0, target, max_iters=1_000_000, time_limit_s=120.0
    )
    assert capped.iters == full.iters and capped.cost == full.cost
    assert capped.proven_optimal

    zero = calf.solve(rf, X, x0, target, max_iters=1_000_000, time_limit_s=0.0)
    assert zero.iters == 0 and not zero.found and np.isinf(zero.cost)

    seeded = calf.solve(
        rf, X, x0, target, max_iters=1_000_000, time_limit_s=0.0,
        initial_ub=(x0, 123.0),
    )
    assert seeded.iters == 0 and seeded.found and seeded.cost == 123.0
    assert not seeded.proven_optimal


def test_soft_voting_certifies_and_flips(forest):
    """Soft voting certifies and its point flips the real forest.

    The certified cost is cross-checked against an independent brute force in
    ``test_soft_bruteforce.py``; here we assert the engine proves and that,
    since sklearn's ``rf.predict`` IS the soft vote (argmax of averaged
    probabilities), the returned point flips the real forest's prediction.
    """
    rf, X, _ = forest
    for i in range(6):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        a = calf.solve(
            rf, X, x0, target, voting="soft", max_iters=400_000
        )
        assert a.found and a.proven_optimal
        assert int(rf.predict(a.x[None, :])[0]) == target


def test_soft_additive_dual_smoke(forest):
    """Soft + additive dual certifies the same optimum as geometric-only.

    ``dual_lb_stall_window=1`` forces engagement on the first stalled pop, so
    the run exercises engage / lazy static re-eval / box-clipped local eval /
    stride reopt on the additive pool.  An admissible dual can prune and
    reorder but never change the certified cost.
    """
    rf, X, _ = forest
    for i in range(4):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        base = calf.solve(
            rf, X, x0, target, engine="python", voting="soft",
            dual_lb="off", max_iters=400_000,
        )
        dual = calf.solve(
            rf, X, x0, target, engine="python", voting="soft",
            dual_lb="local", dual_lb_stall_window=1, max_iters=400_000,
        )
        assert base.proven_optimal and dual.proven_optimal
        assert dual.cost == pytest.approx(base.cost, abs=1e-9)


def test_dual_escalation_invariant(forest):
    """Forced strong-tier escalation never changes the certified optimum.

    ``dual_lb_max_reopts=1`` with an always-true gap threshold escalates on
    the first pop after the single cheap reopt, exercising the
    adaptive-strength path end to end (tiered reopt iters / pool cap / max
    reopts).  Escalation only re-tunes how hard the (admissible) dual is
    optimized, so cost must match the non-escalating run exactly.
    """
    rf, X, _ = forest
    for i in range(4):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        base = calf.solve(
            rf, X, x0, target, voting="soft", dual_lb_stall_window=1,
            dual_lb_escalate=False, max_iters=400_000,
        )
        esc = calf.solve(
            rf, X, x0, target, voting="soft", dual_lb_stall_window=1,
            dual_lb_escalate=True, dual_lb_max_reopts=1,
            dual_lb_escalate_gap=1e18, max_iters=400_000,
        )
        assert base.proven_optimal and esc.proven_optimal
        assert esc.cost == pytest.approx(base.cost, abs=1e-9)


def test_additive_dual_pool_admissible(forest):
    """Root additive-dual bound and kernel evals never exceed the true optimum."""
    from types import SimpleNamespace

    from calf.dual_lb import AdditiveDualPool
    from calf.engine import _dataset_box, compile_rf
    from calf.numba.kernels import (
        additive_dual_local_lb,
        additive_dual_static_lb,
        root_active,
    )

    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    crf = compile_rf(parsed)
    box_lo, box_hi = _dataset_box(di)
    active = root_active(crf.rules_lo_mat, crf.rules_hi_mat, box_lo, box_hi)
    tau = 0.5 * parsed.n_trees
    for i in range(3):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        opt = calf.solve(rf, X, x0, target, voting="soft", max_iters=1_000_000)
        assert opt.proven_optimal
        f32 = x0.astype(np.float32).astype(np.float64)
        p1 = parsed.rules_proba1.astype(np.float64)
        values = p1 if target == 1 else 1.0 - p1
        pool = AdditiveDualPool(parsed, f32, scale, values, tau)
        shim = SimpleNamespace(
            active_rules=active, box=SimpleNamespace(lo=box_lo, hi=box_hi)
        )
        bound = pool.optimize_root(shim, incumbent=opt.cost, max_iters=300)
        assert -1e-9 <= bound <= opt.cost + 1e-9
        dual_w = np.ascontiguousarray(np.stack(pool.static_w))
        dual_lam = np.array([lam for _, _, lam in pool.entries])
        ap = np.ascontiguousarray(np.stack([e[0] for e in pool.entries]))
        an = np.ascontiguousarray(np.stack([e[1] for e in pool.entries]))
        s = additive_dual_static_lb(
            active, crf.rules_tree_id, dual_w, dual_lam, tau, crf.n_trees
        )
        loc = additive_dual_local_lb(
            active, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id,
            np.ascontiguousarray(values), box_lo, box_hi, f32, scale,
            ap, an, dual_lam, tau, crf.n_trees,
        )
        assert s <= opt.cost + 1e-9
        assert loc <= opt.cost + 1e-9
        assert loc >= s - 1e-9  # clipping only inflates req


def test_soft_tree_lb_admissible(forest):
    """Soft per-tree order-statistic LB never exceeds the true soft optimum.

    Both the static (root-box costs) and node-local (box-clipped) forms must
    lower-bound the certified optimum, and the node-local form — measuring reach
    against the shrunken box — must dominate the static one.  This is the LB that
    lets soft best-first drill into the settled region on high-dim forests
    (before it, the geometric LB alone left the frontier stuck near 0).
    """
    from calf.engine import _dataset_box, compile_rf
    from calf.numba.kernels import (
        node_local_soft_tree_lb,
        root_active,
        soft_rule_costs_l1,
        static_soft_tree_lb,
    )

    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    crf = compile_rf(parsed)
    box_lo, box_hi = _dataset_box(di)
    active = root_active(crf.rules_lo_mat, crf.rules_hi_mat, box_lo, box_hi)
    tau = 0.5 * parsed.n_trees
    for i in range(4):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        strict = target == 1
        f32 = x0.astype(np.float32).astype(np.float64)
        p1 = crf.rules_proba1
        values = np.ascontiguousarray(p1 if target == 1 else 1.0 - p1, dtype=np.float64)
        soft_costs = soft_rule_costs_l1(crf.rules_lo_mat, crf.rules_hi_mat, f32, scale)
        s = static_soft_tree_lb(
            active, crf.rules_tree_id, values, soft_costs, crf.n_trees, tau, strict
        )
        loc = node_local_soft_tree_lb(
            active, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id, values,
            box_lo, box_hi, f32, scale, crf.n_trees, tau, strict,
        )
        opt = calf.solve(rf, X, x0, target, voting="soft", max_iters=1_000_000)
        assert opt.proven_optimal
        assert s <= opt.cost + 1e-9, (i, s, opt.cost)
        assert loc <= opt.cost + 1e-9, (i, loc, opt.cost)
        # at the root box the two forms coincide (no clipping); on any node the
        # local form only inflates reach costs, so it never drops below static.
        assert loc >= s - 1e-9, (i, loc, s)


def test_soft_greedy_rounding(forest):
    """Probability-repair rounding returns genuinely soft-feasible points.

    At the root box the rounding must either fail cleanly (inf) or return a
    point that (a) flips sklearn's soft-vote prediction, (b) reports its true
    weighted-L1 cost, and (c) never beats the certified optimum.
    """
    from types import SimpleNamespace

    from calf.engine import _dataset_box, compile_rf
    from calf.numba.kernels import greedy_round_soft, root_active

    rf, X, _ = forest
    parsed = calf.parse_sklearn_rf(rf)
    di = calf.from_array(X)
    scale = calf.l1_scale(di)
    crf = compile_rf(parsed)
    box_lo, box_hi = _dataset_box(di)
    active = root_active(crf.rules_lo_mat, crf.rules_hi_mat, box_lo, box_hi)
    tau = 0.5 * parsed.n_trees
    n_rounded = 0
    for i in range(6):
        x0 = X[i]
        target = 1 - int(rf.predict([x0])[0])
        f32 = x0.astype(np.float32).astype(np.float64)
        p1 = crf.rules_proba1
        target_proba = np.ascontiguousarray(
            p1 if target == 1 else 1.0 - p1, dtype=np.float64
        )
        rc, rx = greedy_round_soft(
            active, crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_tree_id,
            target_proba, box_lo, box_hi, f32, scale, crf.n_trees,
            tau, target == 1,
        )
        if not np.isfinite(rc):
            continue
        n_rounded += 1
        assert int(rf.predict(rx[None, :])[0]) == target
        assert rc == pytest.approx(calf.l1_cost(f32, rx, scale), abs=1e-9)
        opt = calf.solve(rf, X, x0, target, voting="soft", max_iters=1_000_000)
        assert opt.proven_optimal
        assert rc >= opt.cost - 1e-9
    assert n_rounded > 0  # the greedy must succeed on at least one easy row


def test_soft_training_ub_seed(forest):
    """solve() seeds soft runs with the cheapest target-predicted training row."""
    rf, X, _ = forest
    x0 = X[0]
    target = 1 - int(rf.predict([x0])[0])
    seeded = calf.solve(rf, X, x0, target, voting="soft", max_iters=1)
    assert seeded.found  # incumbent exists before any real search
    assert int(rf.predict(seeded.x[None, :])[0]) == target
    full = calf.solve(rf, X, x0, target, voting="soft", max_iters=1_000_000)
    assert full.proven_optimal
    assert full.cost <= seeded.cost + 1e-12


def test_additive_dual_local_lb_matches_dense_reference():
    """The sparse-delta kernel is bit-identical to the dense pricing loop.

    The 2026-07-12 rewrite clips each rule once and prices pool entries over
    the nonzero reach deltas only.  This reference replays the original dense
    loop (clip recomputed per pool entry, every feature visited) in pure
    Python; float op order is preserved on both sides, so results must match
    exactly — no tolerance.
    """
    from calf.numba.kernels import additive_dual_local_lb

    def dense_reference(active, rules_lo, rules_hi, tree_id, target_proba,
                        box_lo, box_hi, factual, scale, alpha_pos, alpha_neg,
                        dual_lam, tau, n_trees):
        n_pool = alpha_pos.shape[0]
        n_features = factual.size
        best = 0.0
        min_cost = np.full((n_pool, n_trees), np.inf)
        for rid in active:
            tid = tree_id[rid]
            compatible = True
            for f in range(n_features):
                lo = max(rules_lo[rid, f], box_lo[f])
                hi = min(rules_hi[rid, f], box_hi[f])
                if not (lo < hi):
                    compatible = False
                    break
            if not compatible:
                continue
            for k in range(n_pool):
                c = -dual_lam[k] * target_proba[rid]
                for f in range(n_features):
                    lo = max(rules_lo[rid, f], box_lo[f])
                    hi = min(rules_hi[rid, f], box_hi[f])
                    if lo > factual[f]:
                        c += alpha_pos[k, tid, f] * scale[f] * (lo - factual[f])
                    elif factual[f] > hi:
                        c += alpha_neg[k, tid, f] * scale[f] * (factual[f] - hi)
                if c < min_cost[k, tid]:
                    min_cost[k, tid] = c
        for k in range(n_pool):
            s = dual_lam[k] * tau
            for t in range(n_trees):
                if not np.isfinite(min_cost[k, t]):
                    return np.inf
                s += min_cost[k, t]  # sequential, same association as the kernel
            if s > best:
                best = s
        return best

    rng = np.random.default_rng(0)
    n_trees, n_features, n_pool = 4, 6, 3
    for trial in range(20):
        n_rules = int(rng.integers(8, 40))
        lo = rng.uniform(-2.0, 1.0, size=(n_rules, n_features))
        hi = lo + rng.uniform(0.05, 3.0, size=(n_rules, n_features))
        # sprinkle unbounded sides like real parsed rules, and a few empty
        # clips (rule entirely outside the box on one feature)
        lo[rng.random((n_rules, n_features)) < 0.3] = -np.inf
        hi[rng.random((n_rules, n_features)) < 0.3] = np.inf
        hi[rng.random((n_rules, n_features)) < 0.05] = -1.5
        tree_id = np.sort(rng.integers(0, n_trees, size=n_rules)).astype(np.int64)
        target_proba = rng.uniform(0.0, 1.0, size=n_rules)
        box_lo = np.full(n_features, -1.0)
        box_hi = np.full(n_features, 1.0)
        factual = rng.uniform(-1.0, 1.0, size=n_features)
        scale = rng.uniform(0.5, 2.0, size=n_features)
        alpha_pos = rng.uniform(0.0, 1.0, size=(n_pool, n_trees, n_features))
        alpha_neg = rng.uniform(0.0, 1.0, size=(n_pool, n_trees, n_features))
        dual_lam = rng.uniform(0.0, 2.0, size=n_pool)
        tau = 0.5 * n_trees
        # mix full and partial active sets; ensure every tree keeps a rule in
        # the full case so the feasible path is exercised too
        if trial % 2:
            active = rng.permutation(n_rules)[: max(4, n_rules // 2)].astype(np.int64)
        else:
            active = np.arange(n_rules, dtype=np.int64)

        got = additive_dual_local_lb(
            active, lo, hi, tree_id, target_proba, box_lo, box_hi,
            factual, scale, alpha_pos, alpha_neg, dual_lam, tau, n_trees,
        )
        want = dense_reference(
            active, lo, hi, tree_id, target_proba, box_lo, box_hi,
            factual, scale, alpha_pos, alpha_neg, dual_lam, tau, n_trees,
        )
        assert got == want, f"trial {trial}: {got!r} != {want!r}"


def test_optimize_additive_dual_kernel_matches_dense_reference():
    """The CSR-sparse ascent kernel is bit-identical to the dense ascent.

    The reference replays the pre-2026-07-12 dense loop in Python, reusing the
    kernels the old code called (`_additive_dual_eval` for evals, the same
    capped-simplex projection), so the only difference under test is the
    sparse compaction of the fixed req matrices.  Exact equality — bound,
    share matrices, and lam must match to the bit.
    """
    from calf.numba.kernels import (
        _additive_dual_eval,
        _project_columns_capped_simplex_inplace,
        optimize_additive_dual_kernel,
    )

    def dense_reference(req_pos, req_neg, values, tree_pos, n_rt, tau,
                        incumbent, alpha_pos, alpha_neg, lam,
                        max_iters, patience):
        n_leaves, n_features = req_pos.shape
        w = np.empty(n_leaves)
        mins = np.empty(n_rt)
        argmins = np.empty(n_rt, dtype=np.int64)
        g = _additive_dual_eval(req_pos, req_neg, values, tree_pos, n_rt, tau,
                                alpha_pos, alpha_neg, lam, w, mins, argmins)
        best = g
        best_ap = alpha_pos.copy()
        best_an = alpha_neg.copy()
        best_lam = lam
        if np.isfinite(incumbent):
            delta = 0.25 * max(incumbent - best, 0.05)
        else:
            delta = max(abs(best), 0.05)
        stall = 0
        grad_pos = np.zeros((n_rt, n_features))
        grad_neg = np.zeros((n_rt, n_features))
        for _ in range(max_iters):
            grad_pos[:] = 0.0
            grad_neg[:] = 0.0
            grad_lam = tau
            gnorm2 = 0.0
            for t in range(n_rt):
                l = argmins[t]
                if l < 0:
                    continue
                grad_lam -= values[l]
                for f in range(n_features):
                    grad_pos[t, f] = req_pos[l, f]
                    grad_neg[t, f] = req_neg[l, f]
                    gnorm2 += req_pos[l, f] * req_pos[l, f] + req_neg[l, f] * req_neg[l, f]
            gnorm2 += grad_lam * grad_lam
            if gnorm2 <= 1e-18:
                break
            step = (best + delta - g) / gnorm2
            for t in range(n_rt):
                for f in range(n_features):
                    alpha_pos[t, f] += step * grad_pos[t, f]
                    alpha_neg[t, f] += step * grad_neg[t, f]
            lam = lam + step * grad_lam
            if lam < 0.0:
                lam = 0.0
            _project_columns_capped_simplex_inplace(alpha_pos)
            _project_columns_capped_simplex_inplace(alpha_neg)
            g = _additive_dual_eval(req_pos, req_neg, values, tree_pos, n_rt, tau,
                                    alpha_pos, alpha_neg, lam, w, mins, argmins)
            if g > best + 1e-12:
                best = g
                best_ap = alpha_pos.copy()
                best_an = alpha_neg.copy()
                best_lam = lam
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    delta *= 0.5
                    stall = 0
                    if delta < 1e-6:
                        break
        return best, best_ap, best_an, best_lam

    rng = np.random.default_rng(1)
    for trial in range(10):
        n_rt, n_features = int(rng.integers(3, 7)), int(rng.integers(4, 10))
        leaves_per_tree = rng.integers(3, 12, size=n_rt)
        n_leaves = int(leaves_per_tree.sum())
        tree_pos = np.repeat(np.arange(n_rt), leaves_per_tree).astype(np.int64)
        # reqs the way production builds them: clipped leaf boxes vs a factual,
        # so per (leaf, feature) at most one side is nonzero and most are zero
        factual = rng.uniform(-1.0, 1.0, size=n_features)
        lo = rng.uniform(-2.0, 1.5, size=(n_leaves, n_features))
        hi = lo + rng.uniform(0.1, 2.5, size=(n_leaves, n_features))
        scale = rng.uniform(0.5, 2.0, size=n_features)
        req_pos = np.maximum(lo - factual, 0.0) * scale
        req_neg = np.maximum(factual - hi, 0.0) * scale
        values = rng.uniform(0.0, 1.0, size=n_leaves)
        tau = 0.5 * n_rt
        incumbent = float(rng.uniform(0.5, 3.0)) if trial % 2 else np.inf
        ap0 = np.full((n_rt, n_features), 1.0 / n_features)
        an0 = np.full((n_rt, n_features), 1.0 / n_features)
        lam0 = float(rng.uniform(0.0, 1.0))

        got = optimize_additive_dual_kernel(
            req_pos, req_neg, values, tree_pos, n_rt, tau, incumbent,
            ap0.copy(), an0.copy(), lam0, 40, 20,
        )
        want = dense_reference(
            req_pos, req_neg, values, tree_pos, n_rt, tau, incumbent,
            ap0.copy(), an0.copy(), lam0, 40, 20,
        )
        assert got[0] == want[0], f"trial {trial}: bound {got[0]!r} != {want[0]!r}"
        assert np.array_equal(got[1], want[1]), f"trial {trial}: alpha_pos differs"
        assert np.array_equal(got[2], want[2]), f"trial {trial}: alpha_neg differs"
        assert got[3] == want[3], f"trial {trial}: lam {got[3]!r} != {want[3]!r}"


def test_greedy_round_soft_matches_dense_reference():
    """The fused/live-list rounding kernel is bit-identical to the 3-pass loop.

    Reference replays the pre-2026-07-12 kernel in Python: full containment
    pass, full candidate pass, full reachability pass, every round over the
    whole active set.  Uses the same `refine_l1` kernel, so the only
    difference under test is the pass fusion and live-list compaction.
    Inputs draw rule bounds from a shared threshold grid so running boxes and
    rule faces coincide exactly (the boundary-containment regime that makes
    compaction of the containment pass unsafe — kept full precisely for it).
    """
    from calf.numba.kernels import greedy_round_soft, refine_l1

    def dense_reference(active, r_lo, r_hi, tree_id, target_proba, box_lo,
                        box_hi, factual, scale, n_trees, tau, strict_gt):
        n_features = factual.size
        run_lo = box_lo.copy()
        run_hi = box_hi.copy()
        v_cur = np.empty(n_trees)
        for _round in range(n_trees + 1):
            x, cost = refine_l1(run_lo, run_hi, factual, scale)
            v_cur[:] = -1.0
            score = 0.0
            for rid in active:
                tid = tree_id[rid]
                if v_cur[tid] >= 0.0:
                    continue
                if np.all((x > r_lo[rid]) & (x <= r_hi[rid])):
                    v_cur[tid] = target_proba[rid]
                    score += target_proba[rid]
            if (score > tau) if strict_gt else (score >= tau):
                return cost, x
            best_ratio, best_rid = -1.0, -1
            for rid in active:
                tid = tree_id[rid]
                compatible = True
                c = 0.0
                for f in range(n_features):
                    lo = max(r_lo[rid, f], run_lo[f])
                    hi = min(r_hi[rid, f], run_hi[f])
                    if not (lo < hi):
                        compatible = False
                        break
                    if lo > factual[f]:
                        c += scale[f] * (lo - factual[f])
                    elif factual[f] > hi:
                        c += scale[f] * (factual[f] - hi)
                if not compatible:
                    continue
                gain = target_proba[rid] - v_cur[tid]
                if gain > 0.0:
                    added = max(c - cost, 1e-12)
                    ratio = gain / added
                    if ratio > best_ratio:
                        best_ratio, best_rid = ratio, rid
            v_cur[:] = -1.0
            for rid in active:
                lo = np.maximum(r_lo[rid], run_lo)
                hi = np.minimum(r_hi[rid], run_hi)
                if np.all(lo < hi):
                    tid = tree_id[rid]
                    if target_proba[rid] > v_cur[tid]:
                        v_cur[tid] = target_proba[rid]
            ub = 0.0
            for t in range(n_trees):
                if v_cur[t] < 0.0:
                    return np.inf, factual.copy()
                ub += v_cur[t]
            if (ub <= tau) if strict_gt else (ub < tau):
                return np.inf, factual.copy()
            if best_rid < 0:
                return np.inf, factual.copy()
            run_lo = np.maximum(run_lo, r_lo[best_rid])
            run_hi = np.minimum(run_hi, r_hi[best_rid])
        return np.inf, factual.copy()

    rng = np.random.default_rng(2)
    for trial in range(20):
        n_trees, n_features = int(rng.integers(3, 8)), int(rng.integers(3, 7))
        # per-feature threshold grid; every rule face sits on it, like a parsed
        # forest, so commits create boxes whose faces coincide with rule faces
        grids = [np.sort(rng.uniform(-1.0, 1.0, size=6)) for _ in range(n_features)]
        rules = []
        tree_id = []
        for t in range(n_trees):
            for _ in range(int(rng.integers(2, 8))):
                lo = np.empty(n_features)
                hi = np.empty(n_features)
                for f in range(n_features):
                    a, b = sorted(rng.choice(7, size=2, replace=False))
                    lo[f] = -np.inf if a == 0 else grids[f][a - 1]
                    hi[f] = np.inf if b == 6 else grids[f][b - 1]
                rules.append((lo, hi))
                tree_id.append(t)
        n_rules = len(rules)
        r_lo = np.array([r[0] for r in rules])
        r_hi = np.array([r[1] for r in rules])
        tree_id = np.array(tree_id, dtype=np.int64)
        target_proba = rng.uniform(0.0, 1.0, size=n_rules)
        box_lo = np.full(n_features, -1.5)
        box_hi = np.full(n_features, 1.5)
        factual = rng.uniform(-1.5, 1.5, size=n_features).astype(np.float32).astype(np.float64)
        scale = rng.uniform(0.5, 2.0, size=n_features)
        tau = float(rng.uniform(0.3, 0.7)) * n_trees
        strict_gt = bool(trial % 2)
        active = np.arange(n_rules, dtype=np.int64)

        got_cost, got_x = greedy_round_soft(
            active, r_lo, r_hi, tree_id, target_proba, box_lo, box_hi,
            factual, scale, n_trees, tau, strict_gt,
        )
        want_cost, want_x = dense_reference(
            active, r_lo, r_hi, tree_id, target_proba, box_lo, box_hi,
            factual, scale, n_trees, tau, strict_gt,
        )
        assert (got_cost == want_cost or (np.isinf(got_cost) and np.isinf(want_cost))), (
            f"trial {trial}: cost {got_cost!r} != {want_cost!r}"
        )
        assert np.array_equal(got_x, want_x), f"trial {trial}: x differs"
