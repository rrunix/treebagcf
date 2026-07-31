"""Soft-voting MILP certifier (HiGHS via scipy, license-free) as a harness method.

Interval/cell formulation: binary interval selectors per feature, continuous
leaf variables forced integral by the per-tree partition constraint, and the
soft vote as one linear constraint

    sum_l  v_l * y_l  >=  tau (+eps if strict),   v_l = P(target | leaf),
    tau = threshold * n_trees,  strict iff target_class == 1  (sklearn argmax
    ties resolve to class 0).

Strictness cannot be modeled as a tiny epsilon — anything below HiGHS's
feasibility tolerance (~1e-7) is silently violated (caught by the wine
self-test in ``research/milp_soft_unproven.py``: a boundary point with vote ==
tau exactly).  :func:`solve_soft_milp` therefore solves the non-strict
relaxation first and re-solves with a real epsilon only when the optimum lands
exactly on the vote boundary.

Regime notes (2026-07-13, ``research/results/milp_soft_unproven``): cost scales
with leaf count — complementary to the engine, whose cost scales with the
LB-plateau width.  Proves all of digits_38 in <= 411 s where the engine stalls;
times out on easy-for-the-engine wide-plateau rows is not observed, but build +
solve on small datasets is far slower than the engine's milliseconds.  Its
role in the portfolio is the certifier of last resort, not the default.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

import calf

from .base import BaselineExplainer, BaselineUnavailable, CFResult


def build_soft_model(parsed, box_lo, box_hi, scale, target_proba, tau, strict_eps=0.0):
    """Interval/cell MILP with the soft vote constraint.  See module docstring."""
    lo_mat = parsed.rules_lo_mat
    hi_mat = parsed.rules_hi_mat
    n_leaves, n_features = lo_mat.shape
    n_trees = parsed.n_trees

    bin_edges: list[np.ndarray] = []
    feat_offsets = np.zeros(n_features + 1, dtype=np.int64)
    for f in range(n_features):
        ts = np.unique(np.concatenate([lo_mat[:, f], hi_mat[:, f]]))
        ts = ts[np.isfinite(ts)]
        ts = ts[(ts > box_lo[f]) & (ts < box_hi[f])]
        edges = np.concatenate(([box_lo[f]], ts, [box_hi[f]]))
        bin_edges.append(edges)
        feat_offsets[f + 1] = feat_offsets[f] + edges.size - 1
    n_lambda = int(feat_offsets[-1])
    n_vars = n_lambda + n_leaves

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    con_lb: list[float] = []
    con_ub: list[float] = []
    r = 0

    # one interval per feature
    for f in range(n_features):
        for j in range(feat_offsets[f], feat_offsets[f + 1]):
            rows.append(r)
            cols.append(int(j))
            vals.append(1.0)
        con_lb.append(1.0)
        con_ub.append(1.0)
        r += 1

    # one leaf per tree
    for t in range(n_trees):
        con_lb.append(1.0)
        con_ub.append(1.0)
    for l in range(n_leaves):
        rows.append(r + int(parsed.rules_tree_id[l]))
        cols.append(n_lambda + l)
        vals.append(1.0)
    r += n_trees

    # leaf-cell linking (leaf boxes are unions of whole intervals)
    for l in range(n_leaves):
        for f in range(n_features):
            lo = lo_mat[l, f]
            hi = hi_mat[l, f]
            if lo <= box_lo[f] and hi >= box_hi[f]:
                continue
            edges = bin_edges[f]
            a = edges[:-1]
            b = edges[1:]
            allowed = np.flatnonzero((a >= max(lo, box_lo[f])) & (b <= min(hi, box_hi[f])))
            if allowed.size == edges.size - 1:
                continue
            rows.append(r)
            cols.append(n_lambda + l)
            vals.append(1.0)
            for j in allowed:
                rows.append(r)
                cols.append(int(feat_offsets[f] + j))
                vals.append(-1.0)
            con_lb.append(-math.inf)
            con_ub.append(0.0)
            r += 1

    # soft vote: sum_l v_l y_l >= tau + eps (see module docstring on eps)
    for l in range(n_leaves):
        rows.append(r)
        cols.append(n_lambda + l)
        vals.append(float(target_proba[l]))
    con_lb.append(float(tau) + float(strict_eps))
    con_ub.append(math.inf)
    r += 1

    A = sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(r, n_vars)
    )
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[:n_lambda] = 1
    return {
        "A": A, "lb": np.asarray(con_lb), "ub": np.asarray(con_ub),
        "integrality": integrality, "n_lambda": n_lambda, "n_leaves": n_leaves,
        "feat_offsets": feat_offsets, "bin_edges": bin_edges,
        "scale": np.asarray(scale, dtype=np.float64),
    }


def interval_costs(model, factual):
    c = np.zeros(model["n_lambda"] + model["n_leaves"])
    n_features = model["feat_offsets"].size - 1
    for f in range(n_features):
        edges = model["bin_edges"][f]
        a = edges[:-1]
        b = edges[1:]
        x = factual[f]
        cost = np.where(x > b, x - b, np.where(x <= a, a - x, 0.0))
        c[model["feat_offsets"][f]: model["feat_offsets"][f + 1]] = model["scale"][f] * cost
    return c


def reconstruct_point(model, factual, x_var):
    """Cheapest point of the selected cell, on the float32 grid.

    sklearn casts queries to float32 before routing them, so a float64 point a
    sub-ulp nudge above a split threshold rounds back ONTO the threshold and
    is re-routed (observed 2026-07-13: every milp_soft witness failed
    rf.predict while passing the parsed float64 oracle).  Each coordinate is
    therefore snapped to the nearest float32 value inside its half-open
    interval (a, b]; coordinates already strictly inside stay untouched, so
    zero-move features keep exactly zero cost.
    """
    n_features = model["feat_offsets"].size - 1
    out = factual.copy()
    for f in range(n_features):
        lam = x_var[model["feat_offsets"][f]: model["feat_offsets"][f + 1]]
        j = int(np.argmax(lam))
        a = model["bin_edges"][f][j]
        b = model["bin_edges"][f][j + 1]
        v = min(max(out[f], a), b)
        v32 = np.float32(v)
        while float(v32) <= a:
            v32 = np.nextafter(v32, np.float32(np.inf))
        while float(v32) > b:
            v32 = np.nextafter(v32, np.float32(-np.inf))
        if float(v32) > a:
            v = float(v32)
        else:  # interval narrower than one float32 ulp: no valid grid point
            v = min(a + max(abs(a), 1.0) * 1e-12, b)
        out[f] = v
    return out


def soft_vote(parsed, target_proba, x):
    inside = (x > parsed.rules_lo_mat).all(axis=1) & (x <= parsed.rules_hi_mat).all(axis=1)
    hit = np.flatnonzero(inside)
    seen: dict[int, float] = {}
    for rid in hit:
        seen.setdefault(int(parsed.rules_tree_id[rid]), float(target_proba[rid]))
    return sum(seen.values())


def solve_soft_milp(model, parsed, box_lo, box_hi, scale, target_proba, tau,
                    strict, factual, time_limit_s):
    """Two-phase strictness handling.

    Phase 1 solves the non-strict relaxation (>= tau).  If the target is
    non-strict, or the optimum's reconstructed point already satisfies the
    strict vote, that optimum is also the strict optimum (the strictly
    feasible set is a subset; its argmin landing inside certifies both).
    Otherwise re-solve with the vote RHS bumped by an epsilon that is (a)
    well above HiGHS's feasibility tolerance and (b) at most half the
    observed boundary quantum, escalating once if the boundary repeats.
    Returns (result, n_boundary_resolves).
    """
    c = interval_costs(model, factual)

    def _run(m, budget):
        return milp(
            c=c,
            constraints=[LinearConstraint(m["A"], m["lb"], m["ub"])],
            integrality=m["integrality"],
            bounds=Bounds(0.0, 1.0),
            options={"time_limit": float(budget), "mip_rel_gap": 0.0, "presolve": True},
        )

    deadline = time.perf_counter() + float(time_limit_s)
    res = _run(model, time_limit_s)
    if not strict or res.x is None:
        return res, 0
    resolves = 0
    eps = 1e-5
    while True:
        x_prime = reconstruct_point(model, factual, res.x)
        vote = soft_vote(parsed, target_proba, x_prime)
        if vote > tau:
            return res, resolves          # strictly feasible: optimal for both
        budget = deadline - time.perf_counter()
        if budget < 10.0 or resolves >= 3:
            return res, resolves          # give up escalating; validity flags tell
        resolves += 1
        model2 = build_soft_model(parsed, box_lo, box_hi, scale, target_proba,
                                  tau, strict_eps=eps)
        res2 = _run(model2, budget)
        if res2.x is None:
            return res, resolves
        res = res2
        eps *= 10.0


class MilpSoftExplainer(BaselineExplainer):
    """Soft-vote MILP behind the common baseline interface.

    Certificate accounting mirrors the OCEAN wrapper: ``solver_optimal`` is
    HiGHS finishing with status 0; ``is_optimal`` additionally requires the
    reconstructed point to satisfy the strict soft vote on the parsed forest
    AND survive sklearn's float32 predict.  ``reaches_target`` reports the
    soft-vote check (the semantics this backend certifies), not the base
    class's hard-vote oracle.
    """

    name = "milp_soft"

    def _configure(self, **kw) -> None:
        super()._configure(**kw)
        if self.voting != "soft":
            raise BaselineUnavailable(
                "milp_soft encodes the soft vote only; run it under voting=soft"
            )
        if int(getattr(self.rf, "n_classes_", 2)) != 2:
            raise BaselineUnavailable("milp_soft supports binary forests only")
        if self.X is None:
            raise ValueError("MilpSoftExplainer requires X (root box) at construction")
        _tb = time.perf_counter()
        self.parsed = calf.parse_sklearn_rf(self.rf)
        self._build_time_s = time.perf_counter() - _tb
        self._tau = self.threshold * self.parsed.n_trees
        # One model per target class (the vote row bakes in target_proba); built
        # lazily on first use, then reused across the job's queries.  Model
        # construction is query-independent, so it is charged to build time.
        self._models: dict[int, dict] = {}

    def _target_proba(self, target_class: int) -> np.ndarray:
        p1 = self.parsed.rules_proba1.astype(np.float64)
        return p1 if int(target_class) == 1 else 1.0 - p1

    def _model_for(self, target_class: int) -> dict:
        tc = int(target_class)
        if tc not in self._models:
            _tb = time.perf_counter()
            self._models[tc] = build_soft_model(
                self.parsed, self.box_lo, self.box_hi, self.scale,
                self._target_proba(tc), self._tau,
            )
            self._build_time_s += time.perf_counter() - _tb
        return self._models[tc]

    def explain(
        self, x: np.ndarray, target_class: int, time_limit_s: float = 60.0
    ) -> CFResult:
        target_class = int(target_class)
        x = np.ascontiguousarray(x, dtype=np.float64)
        model = self._model_for(target_class)
        target_proba = self._target_proba(target_class)
        strict = target_class == 1  # sklearn argmax ties resolve to class 0

        t0 = time.perf_counter()
        try:
            res, boundary_resolves = solve_soft_milp(
                model, self.parsed, self.box_lo, self.box_hi, self.scale,
                target_proba, self._tau, strict, x, time_limit_s,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return CFResult(
                x_cf=None, cost=None, is_optimal=False, lower_bound=None,
                upper_bound=None, n_nodes=None,
                wall_time_s=time.perf_counter() - t0, status="error",
                method=self.name, error=f"{type(exc).__name__}: {exc}",
                build_time_s=self._build_time_s,
            )
        wall = time.perf_counter() - t0

        found = res.x is not None
        # HiGHS leaves mip_dual_bound / mip_node_count as None when it stops
        # before any B&B progress (huge-model timeouts: pima/abalone/seismic).
        dual_bound = getattr(res, "mip_dual_bound", None)
        lower_bound = (
            float(dual_bound)
            if dual_bound is not None and math.isfinite(float(dual_bound))
            else None
        )
        node_count = getattr(res, "mip_node_count", None)
        n_nodes = int(node_count) if node_count is not None else None
        extra = {
            "milp_status": int(res.status),
            "boundary_resolves": boundary_resolves,
            "n_binaries": model["n_lambda"],
            "n_constraints": int(model["A"].shape[0]),
        }

        if not found:
            # status 2 = proven infeasible; anything else without an incumbent
            # is a timeout/limit stop.
            infeasible = int(res.status) == 2
            return CFResult(
                x_cf=None, cost=None, is_optimal=infeasible,
                lower_bound=lower_bound, upper_bound=None, n_nodes=n_nodes,
                wall_time_s=wall,
                status="infeasible" if infeasible else "timeout",
                method=self.name,
                solver_optimal=infeasible,
                build_time_s=self._build_time_s, solve_time_s=wall,
                extra=extra,
            )

        x_cf = reconstruct_point(model, x, res.x)
        vote = soft_vote(self.parsed, target_proba, x_cf)
        valid_soft = bool(vote > self._tau if strict else vote >= self._tau)
        x32 = x_cf.astype(np.float32).reshape(1, -1)
        sklearn_valid = bool(int(self.rf.predict(x32)[0]) == target_class)
        cost = float(np.sum(self.scale * np.abs(x - x_cf)))

        solver_optimal = bool(int(res.status) == 0)
        reasons = []
        if not valid_soft:
            reasons.append("soft_vote_boundary")
        if not sklearn_valid:
            reasons.append("sklearn_invalid")
        is_optimal = solver_optimal and not reasons
        cert_note = (
            "solver_optimal_downgraded:" + ",".join(reasons)
            if solver_optimal and reasons else None
        )
        return CFResult(
            x_cf=x_cf,
            cost=cost,
            is_optimal=is_optimal,
            lower_bound=lower_bound,
            upper_bound=cost,
            n_nodes=n_nodes,
            wall_time_s=wall,
            status="optimal" if is_optimal else "feasible",
            method=self.name,
            reaches_target=valid_soft,
            solver_optimal=solver_optimal,
            cert_note=cert_note,
            build_time_s=self._build_time_s, solve_time_s=wall,
            extra=extra,
        )
