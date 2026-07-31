"""Cost-splitting Lagrangian dual lower bound for RF vote quorums.

Admissible bound family indexed by nonnegative share matrices
``alpha_pos[t, f]`` / ``alpha_neg[t, f]`` with column caps
``sum_t alpha_pos[t, f] <= 1`` and ``sum_t alpha_neg[t, f] <= 1``:

    cost(x, x') >= sum of the `need` smallest mu_t(alpha)

where ``mu_t(alpha)`` is tree ``t``'s minimum alpha_t-weighted one-sided
movement over its active target leaves (clipped to the current box).

Validity: any counterfactual x' lies in an active target leaf l_t of every
tree t in its quorum Q (|Q| >= need).  Entering l_t needs per-feature movement
at least ``req_pos[l_t, f]`` up and ``req_neg[l_t, f]`` down, so
``mu_t(alpha) <= sum_f alpha_pos[t,f]*dpos_f + alpha_neg[t,f]*dneg_f`` for x''s
actual movements dpos/dneg.  Summing over any `need` trees of Q and applying
the column caps bounds the sum by ``sum_f (dpos_f + dneg_f) = cost``.  The bound
holds for every fixed alpha; subgradient ascent only tightens it.  It does NOT
dominate the per-tree order statistic (node_local_tree_lb) — stack via max.

Lagrangian-decomposition bound (Guignard & Kim, 1987); per-tree subproblems are
solved integrally, removing the fractional within-tree leaf mixing that weakens
the plain LP relaxation.

Nodes are passed as duck-typed shims with ``.active_rules`` and ``.box.lo`` /
``.box.hi`` (the engine builds them as SimpleNamespace).
"""
from __future__ import annotations

import math

import numpy as np

from .parser import ParsedRF
from .numba.kernels import optimize_additive_dual_kernel, optimize_dual_shares_kernel

__all__ = ["AdditiveDualPool", "DualCostSplitPool"]


class DualCostSplitPool:
    """Pool of dual share matrices with precomputed static per-leaf weights.

    Every entry yields an admissible LB at every search node; evaluation uses
    only the node's active target leaves.  ``static_lb`` uses root-box leaf
    movements (precomputed).  ``local_lb`` clips leaf boxes to the node box and
    is meant for plateau-gated use.  ``optimize_root`` / ``reoptimize_at`` run
    the compiled subgradient ascent and append the resulting shares.
    """

    def __init__(
        self,
        parsed_rf: ParsedRF,
        factual: np.ndarray,
        scale: np.ndarray,
        target_class: int,
        need: int,
    ):
        self.parsed = parsed_rf
        self.factual = np.asarray(factual, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.target_class = target_class
        self.need = need
        self.n_trees = parsed_rf.n_trees
        self.target_rule_ids = np.where(parsed_rf.rules_class == target_class)[0].astype(np.int64)
        lo = parsed_rf.rules_lo_mat[self.target_rule_ids]
        hi = parsed_rf.rules_hi_mat[self.target_rule_ids]
        self.static_req_pos = np.maximum(lo - self.factual, 0.0) * self.scale
        self.static_req_neg = np.maximum(self.factual - hi, 0.0) * self.scale
        self.rule_row = np.full(parsed_rf.rules_class.size, -1, dtype=np.int64)
        self.rule_row[self.target_rule_ids] = np.arange(self.target_rule_ids.size)
        self.alphas: list[tuple[np.ndarray, np.ndarray]] = []
        self.static_w: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.alphas)

    def add(self, alpha_pos: np.ndarray, alpha_neg: np.ndarray, cap: int | None = None) -> None:
        trees = self.parsed.rules_tree_id[self.target_rule_ids]
        # w[l] = Σ_f static_req_pos[l,f] · alpha_pos[trees][l,f]
        w = np.einsum("lf,lf->l", self.static_req_pos, alpha_pos[trees]) + np.einsum(
            "lf,lf->l", self.static_req_neg, alpha_neg[trees]
        )
        self.alphas.append((alpha_pos, alpha_neg))
        self.static_w.append(w)
        if cap is not None and len(self.alphas) > cap:
            del self.alphas[1]
            del self.static_w[1]

    def _compacted(self, node):
        """Node-clipped reqs sorted by compacted tree index (kernel layout)."""
        got = self._clipped_reqs(node)
        if got is None:
            return None
        _, req_pos, req_neg, tids = got
        trees, tree_pos = np.unique(tids, return_inverse=True)
        order = np.argsort(tree_pos, kind="stable")
        req_pos = np.ascontiguousarray(req_pos[order])
        req_neg = np.ascontiguousarray(req_neg[order])
        tree_pos = tree_pos[order]
        starts = np.searchsorted(tree_pos, np.arange(trees.size))
        return req_pos, req_neg, tree_pos, starts, trees

    def _entry_bound(self, ap, an, req_pos, req_neg, tree_pos, starts, trees) -> float:
        """This entry's bound on the given compacted clipped reqs."""
        w = np.einsum("lf,lf->l", req_pos, ap[trees][tree_pos]) + np.einsum(
            "lf,lf->l", req_neg, an[trees][tree_pos]
        )
        min_w = np.minimum.reduceat(w, starts)
        if self.need >= trees.size:
            return float(min_w.sum())
        return float(np.partition(min_w, self.need - 1)[: self.need].sum())

    def seed(self, node, entries, top_k: int = 3) -> None:
        """Seed the pool with transferred share matrices (e.g. an AlphaLibrary).

        Every fixed feasible entry is admissible at any query — the factual
        enters the bound only through the movement requirements — so seeding is
        always safe; only tightness varies.  Entries are scored on THIS node's
        clipped reqs and the strongest ``top_k`` with a strictly positive bound
        are added (weak transfers would only pay per-node eval cost).
        """
        if not entries:
            return
        got = self._compacted(node)
        if got is None:
            return
        req_pos, req_neg, tree_pos, starts, trees = got
        if trees.size < self.need:
            return
        scored = sorted(
            ((self._entry_bound(ap, an, req_pos, req_neg, tree_pos, starts, trees),
              ap, an) for ap, an in entries),
            key=lambda e: e[0], reverse=True,
        )
        for val, ap, an in scored[: max(0, int(top_k))]:
            if val <= 0.0:
                break
            self.add(ap, an)

    def optimize_root(self, root, *, incumbent: float = math.inf, max_iters: int = 1000) -> float:
        got = self._compacted(root)
        if got is None:
            return math.inf
        req_pos, req_neg, tree_pos, starts, trees = got
        if trees.size < self.need:
            return math.inf
        n_features = req_pos.shape[1]
        if self.alphas:
            # Seeded pool: warm-start the ascent from the strongest entry.
            best_k = 0
            best_val = -math.inf
            for k, (ap, an) in enumerate(self.alphas):
                val = self._entry_bound(ap, an, req_pos, req_neg, tree_pos, starts, trees)
                if val > best_val:
                    best_val, best_k = val, k
            ap0 = self.alphas[best_k][0][trees].copy()
            an0 = self.alphas[best_k][1][trees].copy()
        else:
            ap0 = np.full((trees.size, n_features), 1.0 / trees.size)
            an0 = np.full((trees.size, n_features), 1.0 / trees.size)
        bound, best_ap, best_an = optimize_dual_shares_kernel(
            req_pos, req_neg, tree_pos, starts, trees.size, self.need,
            incumbent, ap0, an0, max_iters, 20,
        )
        full_ap = np.zeros((self.n_trees, self.scale.size))
        full_an = np.zeros((self.n_trees, self.scale.size))
        full_ap[trees] = best_ap
        full_an[trees] = best_an
        self.add(full_ap, full_an)
        return float(bound)

    def _sum_need_smallest_treemin(self, tids: np.ndarray, w: np.ndarray) -> float:
        min_w = np.full(self.n_trees, np.inf)
        np.minimum.at(min_w, tids, w)
        if self.need >= self.n_trees:
            return float(min_w.sum())
        return float(np.partition(min_w, self.need - 1)[: self.need].sum())

    def static_lb(self, node) -> float:
        af = node.active_rules[self.parsed.rules_class[node.active_rules] == self.target_class]
        if af.size == 0:
            return math.inf
        rows = self.rule_row[af]
        tids = self.parsed.rules_tree_id[af]
        best = 0.0
        for w in self.static_w:
            val = self._sum_need_smallest_treemin(tids, w[rows])
            if val > best:
                best = val
        return best

    def _clipped_reqs(self, node):
        af = node.active_rules[self.parsed.rules_class[node.active_rules] == self.target_class]
        if af.size == 0:
            return None
        lo = np.maximum(self.parsed.rules_lo_mat[af], node.box.lo)
        hi = np.minimum(self.parsed.rules_hi_mat[af], node.box.hi)
        ok = (lo < hi).all(axis=1)
        if not ok.any():
            return None
        af, lo, hi = af[ok], lo[ok], hi[ok]
        req_pos = np.maximum(lo - self.factual, 0.0) * self.scale
        req_neg = np.maximum(self.factual - hi, 0.0) * self.scale
        return af, req_pos, req_neg, self.parsed.rules_tree_id[af]

    def local_lb(self, node) -> float:
        got = self._clipped_reqs(node)
        if got is None:
            return math.inf
        _, req_pos, req_neg, tids = got
        best = 0.0
        for ap, an in self.alphas:
            w = np.einsum("lf,lf->l", req_pos, ap[tids]) + np.einsum(
                "lf,lf->l", req_neg, an[tids]
            )
            val = self._sum_need_smallest_treemin(tids, w)
            if val > best:
                best = val
        return best

    def reoptimize_at(
        self,
        node,
        *,
        incumbent: float = math.inf,
        max_iters: int = 150,
        cap: int = 4,
    ) -> float:
        got = self._compacted(node)
        if got is None:
            return math.inf
        req_pos, req_neg, tree_pos, starts, trees = got
        if trees.size < self.need:
            return math.inf
        # Warm start from the pool entry currently strongest for this node's
        # clipped movements.
        best_k = 0
        best_val = -math.inf
        for k, (ap, an) in enumerate(self.alphas):
            val = self._entry_bound(ap, an, req_pos, req_neg, tree_pos, starts, trees)
            if val > best_val:
                best_val = val
                best_k = k
        ap0, an0 = self.alphas[best_k]
        bound, best_ap, best_an = optimize_dual_shares_kernel(
            req_pos, req_neg, tree_pos, starts, trees.size, self.need,
            incumbent, ap0[trees].copy(), an0[trees].copy(), max_iters, 20,
        )
        full_ap = np.zeros((self.n_trees, self.scale.size))
        full_an = np.zeros((self.n_trees, self.scale.size))
        full_ap[trees] = best_ap
        full_an[trees] = best_an
        self.add(full_ap, full_an, cap=cap)
        return float(bound)


class AdditiveDualPool:
    """Dual pool for the additive-threshold (soft-voting) aggregation.

    Soft voting requires ``sum_t v(leaf_t(x')) >= tau`` with ``v = P(target |
    leaf)`` and ``tau = n_trees / 2`` (the bound for ``>=`` is valid for the
    strict ``>`` class-1 rule).  Each entry is ``(alpha_pos, alpha_neg, lam)``;
    its bound at a node is

        lam * tau + sum over ALL trees of min over the tree's active leaves of
        (alpha_t . req_l - lam * v_l)

    (see ``_additive_dual_eval``).  Shrinking the active set only raises each
    per-tree min, so per-node evaluation is admissible for every entry.  The
    quorum dual is the special case v in {0,1}, tau = need.

    Warm start matters (probed 2026-07-09): ascent from alpha=0, lam=0 barely
    moves lam.  ``optimize_root`` starts from uniform shares and picks lam0 by
    an exact 1-D grid (the bound is piecewise-linear in lam); ``reoptimize_at``
    warm-starts from the pool entry strongest for the node's clipped movements,
    whose lam is already on the right scale.
    """

    _LAM_GRID = np.concatenate([[0.0], np.logspace(-4, 1, 32)])

    def __init__(
        self,
        parsed_rf: ParsedRF,
        factual: np.ndarray,
        scale: np.ndarray,
        values: np.ndarray,
        tau: float,
    ):
        self.parsed = parsed_rf
        self.factual = np.asarray(factual, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.values = np.ascontiguousarray(values, dtype=np.float64)  # (n_rules,)
        self.tau = float(tau)
        self.n_trees = parsed_rf.n_trees
        lo = parsed_rf.rules_lo_mat
        hi = parsed_rf.rules_hi_mat
        self.static_req_pos = np.maximum(lo - self.factual, 0.0) * self.scale
        self.static_req_neg = np.maximum(self.factual - hi, 0.0) * self.scale
        self.entries: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.static_w: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, alpha_pos: np.ndarray, alpha_neg: np.ndarray, lam: float,
            cap: int | None = None) -> None:
        trees = self.parsed.rules_tree_id
        w = (
            np.einsum("lf,lf->l", self.static_req_pos, alpha_pos[trees])
            + np.einsum("lf,lf->l", self.static_req_neg, alpha_neg[trees])
            - lam * self.values
        )
        self.entries.append((alpha_pos, alpha_neg, float(lam)))
        self.static_w.append(w)
        if cap is not None and len(self.entries) > cap:
            del self.entries[1]
            del self.static_w[1]

    def _clipped_reqs(self, node):
        active = node.active_rules
        lo = np.maximum(self.parsed.rules_lo_mat[active], node.box.lo)
        hi = np.minimum(self.parsed.rules_hi_mat[active], node.box.hi)
        ok = (lo < hi).all(axis=1)
        if not ok.any():
            return None
        active, lo, hi = active[ok], lo[ok], hi[ok]
        req_pos = np.maximum(lo - self.factual, 0.0) * self.scale
        req_neg = np.maximum(self.factual - hi, 0.0) * self.scale
        return active, req_pos, req_neg, self.parsed.rules_tree_id[active]

    def seed(self, node, entries, top_k: int = 3) -> None:
        """Seed the pool with transferred (alpha_pos, alpha_neg, lam) entries.

        Same transfer argument as :meth:`DualCostSplitPool.seed`: the bound is
        valid for every fixed (alpha, lam >= 0), so entries optimized at other
        factuals stay admissible here.  Scored on this node's clipped reqs; the
        strongest ``top_k`` with a strictly positive bound are added.
        """
        if not entries:
            return
        got = self._clipped_reqs(node)
        if got is None:
            return
        act, req_pos, req_neg, tids = got
        v = self.values[act]
        trees, tree_pos = np.unique(tids, return_inverse=True)
        if trees.size < self.n_trees:
            return
        order = np.argsort(tree_pos, kind="stable")
        req_pos = np.ascontiguousarray(req_pos[order])
        req_neg = np.ascontiguousarray(req_neg[order])
        v = np.ascontiguousarray(v[order])
        tree_pos = np.ascontiguousarray(tree_pos[order])
        starts = np.searchsorted(tree_pos, np.arange(trees.size))
        scored = []
        for ap, an, lam in entries:
            w = (
                np.einsum("lf,lf->l", req_pos, ap[trees][tree_pos])
                + np.einsum("lf,lf->l", req_neg, an[trees][tree_pos])
                - lam * v
            )
            val = float(np.minimum.reduceat(w, starts).sum()) + lam * self.tau
            scored.append((val, ap, an, lam))
        scored.sort(key=lambda e: e[0], reverse=True)
        for val, ap, an, lam in scored[: max(0, int(top_k))]:
            if val <= 0.0:
                break
            self.add(ap, an, lam)

    def optimize_root(self, root, *, incumbent: float = math.inf,
                      max_iters: int = 1000) -> float:
        # warm="strongest" is a no-op on an empty pool (falls back to uniform
        # shares + the exact 1-D lam grid) and picks the best seeded entry
        # otherwise, so seeded roots skip the cold-start ascent entirely.
        return self._optimize(root, incumbent=incumbent, max_iters=max_iters,
                              warm="strongest", cap=None)

    def lp_optimize_at(self, node, *, cap: int | None = None,
                       time_limit_s: float | None = None) -> float:
        """Exact LP solve of the node's dual (see ``calf.lp_dual``).

        The exact ceiling of the whole bound family at this node.  DIAGNOSTIC
        ONLY: the 2026-07-11/12 ablations (lp_stall_120, lp_pool_120,
        deep_120) showed in-search LP delivery — node-local, at the root, or
        UB-box-tuned — never moves certified outcomes, so the engine does not
        call this; ``research/stall_diag`` and the tests do.  The (repaired,
        feasible) optimizer is added to the pool like any other entry.
        Returns the exact bound (or -inf on solver failure).
        """
        from .lp_dual import solve_additive_dual_lp

        got = self._clipped_reqs(node)
        if got is None:
            return math.inf
        act, req_pos, req_neg, tids = got
        v = self.values[act]
        trees, tree_pos = np.unique(tids, return_inverse=True)
        if trees.size < self.n_trees:
            return math.inf  # a tree lost all leaves: box infeasible
        order = np.argsort(tree_pos, kind="stable")
        req_pos = np.ascontiguousarray(req_pos[order])
        req_neg = np.ascontiguousarray(req_neg[order])
        v = np.ascontiguousarray(v[order])
        tree_pos = np.ascontiguousarray(tree_pos[order])
        starts = np.searchsorted(tree_pos, np.arange(trees.size))
        bound, ap, an, lam = solve_additive_dual_lp(
            req_pos, req_neg, v, tree_pos, starts, trees.size, self.tau,
            time_limit_s=time_limit_s,
        )
        if ap is None or not math.isfinite(bound):
            return -math.inf
        n_features = self.scale.size
        full_ap = np.zeros((self.n_trees, n_features))
        full_an = np.zeros((self.n_trees, n_features))
        full_ap[trees] = ap
        full_an[trees] = an
        self.add(full_ap, full_an, float(lam), cap=cap)
        return float(bound)

    def reoptimize_at(self, node, *, incumbent: float = math.inf,
                      max_iters: int = 150, cap: int = 4) -> float:
        return self._optimize(node, incumbent=incumbent, max_iters=max_iters,
                              warm="strongest", cap=cap)

    def _optimize(self, node, *, incumbent: float, max_iters: int,
                  warm: str | None, cap: int | None) -> float:
        got = self._clipped_reqs(node)
        if got is None:
            return math.inf
        act, req_pos, req_neg, tids = got
        v = self.values[act]
        trees, tree_pos = np.unique(tids, return_inverse=True)
        if trees.size < self.n_trees:
            # A tree lost all box-compatible leaves: RF leaves partition the
            # space, so the box is empty — infeasible, prune caller-side.
            return math.inf
        order = np.argsort(tree_pos, kind="stable")
        req_pos = np.ascontiguousarray(req_pos[order])
        req_neg = np.ascontiguousarray(req_neg[order])
        v = np.ascontiguousarray(v[order])
        tree_pos = np.ascontiguousarray(tree_pos[order])
        starts = np.searchsorted(tree_pos, np.arange(trees.size))
        n_features = self.scale.size

        if warm == "strongest" and self.entries:
            best_k, best_val = 0, -math.inf
            for k, (ap, an, lam) in enumerate(self.entries):
                w = (
                    np.einsum("lf,lf->l", req_pos, ap[trees][tree_pos])
                    + np.einsum("lf,lf->l", req_neg, an[trees][tree_pos])
                    - lam * v
                )
                val = float(np.minimum.reduceat(w, starts).sum()) + lam * self.tau
                if val > best_val:
                    best_val, best_k = val, k
            ap0, an0, lam0 = self.entries[best_k]
            ap0 = ap0[trees].copy()
            an0 = an0[trees].copy()
        else:
            ap0 = np.full((trees.size, n_features), 1.0 / trees.size)
            an0 = np.full((trees.size, n_features), 1.0 / trees.size)
            # Exact 1-D lam warm start: w0 is lam-independent, and the bound is
            # piecewise-linear in lam, so a log-grid scan lands near the peak.
            w0 = (
                np.einsum("lf,lf->l", req_pos, ap0[tree_pos])
                + np.einsum("lf,lf->l", req_neg, an0[tree_pos])
            )
            lam0, best_g = 0.0, -math.inf
            for lam in self._LAM_GRID:
                g = float(np.minimum.reduceat(w0 - lam * v, starts).sum()) + lam * self.tau
                if g > best_g:
                    best_g, lam0 = g, float(lam)

        bound, best_ap, best_an, best_lam = optimize_additive_dual_kernel(
            req_pos, req_neg, v, tree_pos, trees.size, self.tau,
            incumbent, ap0, an0, float(lam0), max_iters, 20,
        )
        full_ap = np.zeros((self.n_trees, n_features))
        full_an = np.zeros((self.n_trees, n_features))
        full_ap[trees] = best_ap
        full_an[trees] = best_an
        self.add(full_ap, full_an, float(best_lam), cap=cap)
        return float(bound)
