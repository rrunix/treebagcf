"""Exact LP solve of the additive (soft-voting) cost-splitting dual.

The additive dual bound (see ``calf.dual_lb.AdditiveDualPool``)

    B(alpha, lam) = lam*tau + sum_t min_{l in L_t} (alpha_t . req_l - lam*v_l)

is concave piecewise-linear in (alpha, lam), so its maximum over the feasible
set {alpha >= 0, column caps, lam >= 0} is a linear program:

    max  lam*tau + sum_t mu_t
    s.t. mu_t <= sum_f (ap[t,f]*req_pos[l,f] + an[t,f]*req_neg[l,f]) - lam*v_l
                                        for every tree t, leaf l in L_t
         sum_t ap[t,f] <= 1,  sum_t an[t,f] <= 1        for every feature f
         ap, an in [0, 1],  lam >= 0,  mu free

Subgradient ascent (the engine's reopts) only approximates this maximum; the
LP is the exact ceiling of the whole bound family at a node.  Two uses:

- diagnosis (``research/stall_diag``): how much of a stalled node's gap is
  ascent shortfall (LP - ascent) vs intrinsic relaxation gap (node optimum -
  LP);
- build-time warm-up (``AlphaLibrary.warmup(lp=True)``): one exact solve per
  k-means centroid, harvested into the cross-query library.

In-search LP delivery (node-local at stalls, at the root on engagement, and
UB-box-tuned variants) was tried and pruned 2026-07-12: the per-node bounds
improved 2-7x but certified outcomes never moved — the frontier at depth is
too wide for any small set of shared entries (lp_stall_120 / lp_pool_120 /
deep_120 results).

Admissibility: the returned bound is NOT the solver's objective — the solver's
iterate is repaired to strict feasibility (clip negatives, rescale overflowing
columns, clamp lam) and the bound is re-evaluated exactly from the repaired
(alpha, lam).  Any feasible point of the family is admissible, so solver
tolerances cannot void a certificate.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["solve_additive_dual_lp"]


def solve_additive_dual_lp(
    req_pos: np.ndarray,
    req_neg: np.ndarray,
    values: np.ndarray,
    tree_pos: np.ndarray,
    starts: np.ndarray,
    n_rt: int,
    tau: float,
    *,
    time_limit_s: float | None = None,
) -> tuple[float, np.ndarray | None, np.ndarray | None, float]:
    """Solve the node's additive dual LP exactly (HiGHS via scipy).

    Inputs are the compacted node arrays (leaves sorted by compacted tree
    index, per-tree segments given by ``starts``), exactly the layout of
    ``optimize_additive_dual_kernel``.  Returns ``(bound, alpha_pos,
    alpha_neg, lam)`` with the compacted (n_rt, n_features) share matrices,
    or ``(-inf, None, None, 0.0)`` when the solver fails.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    L, F = req_pos.shape
    T = int(n_rt)
    nap = T * F
    n_var = 1 + T + 2 * nap  # [lam, mu(T), ap(T*F), an(T*F)]

    c = np.zeros(n_var)
    c[0] = -float(tau)   # maximize lam*tau + sum mu  ->  minimize -(...)
    c[1 : 1 + T] = -1.0

    # Leaf rows: mu_t - ap_t.req_pos_l - an_t.req_neg_l + lam*v_l <= 0.
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    leaf_ids = np.arange(L)
    # mu coefficient (+1)
    rows.append(leaf_ids)
    cols.append(1 + tree_pos.astype(np.int64))
    data.append(np.ones(L))
    # lam coefficient (+v_l), only where nonzero
    nz = np.flatnonzero(values)
    rows.append(nz)
    cols.append(np.zeros(nz.size, dtype=np.int64))
    data.append(values[nz].astype(np.float64))
    # -req_pos on ap columns, -req_neg on an columns (sparse by construction:
    # req is nonzero only on features where the factual sits outside leaf∩box)
    lp, fp = np.nonzero(req_pos)
    rows.append(lp)
    cols.append(1 + T + tree_pos[lp] * F + fp)
    data.append(-req_pos[lp, fp])
    ln, fn = np.nonzero(req_neg)
    rows.append(ln)
    cols.append(1 + T + nap + tree_pos[ln] * F + fn)
    data.append(-req_neg[ln, fn])
    # Column-cap rows: sum_t ap[t,f] <= 1 and sum_t an[t,f] <= 1.
    t_grid, f_grid = np.divmod(np.arange(nap), F)
    rows.append(L + f_grid)
    cols.append(1 + T + np.arange(nap))
    data.append(np.ones(nap))
    rows.append(L + F + f_grid)
    cols.append(1 + T + nap + np.arange(nap))
    data.append(np.ones(nap))

    A = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(L + 2 * F, n_var),
    ).tocsr()
    b = np.concatenate([np.zeros(L), np.ones(2 * F)])
    bounds = [(0.0, None)] + [(None, None)] * T + [(0.0, 1.0)] * (2 * nap)

    options = {}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options=options)
    if res.x is None:
        return -math.inf, None, None, 0.0

    lam = max(0.0, float(res.x[0]))
    ap = np.clip(res.x[1 + T : 1 + T + nap].reshape(T, F), 0.0, None)
    an = np.clip(res.x[1 + T + nap :].reshape(T, F), 0.0, None)
    # Repair the column caps exactly (solver feasibility tolerance): scaling a
    # column down keeps every entry nonnegative and can only lower the bound,
    # which the exact re-evaluation below accounts for.
    for m in (ap, an):
        s = m.sum(axis=0)
        over = s > 1.0
        if over.any():
            m[:, over] /= s[over]

    # Exact bound at the repaired point — this, not res.fun, is what's returned.
    w = (
        np.einsum("lf,lf->l", req_pos, ap[tree_pos])
        + np.einsum("lf,lf->l", req_neg, an[tree_pos])
        - lam * values
    )
    bound = float(np.minimum.reduceat(w, starts).sum()) + lam * tau
    return bound, np.ascontiguousarray(ap), np.ascontiguousarray(an), lam
