"""Compiled kernels for the RF/L1 certified-counterfactual search.

This is the minimal kernel set for the method: geometric per-feature LB,
plateau node-local per-tree LB, the cost-splitting Lagrangian dual LB (static
and box-clipped forms plus the compiled subgradient ascent), baseline split
scoring, child construction, the RF vote stop test, and dual-guided greedy
primal rounding.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def lb_feature_vec(box_lo, box_hi, factual, scale):
    """Per-feature geometric LB: scaled L1 distance from factual to the box.

    Returns the per-feature vector (the engine sums it for the geometric slot).
    Distance is 0 on features where the factual already lies inside the box, and
    the one-sided gap otherwise -- a valid LB because any point in the box is at
    least this far from the factual on each axis independently.
    """
    out = np.empty(factual.size, dtype=np.float64)
    for f in range(factual.size):
        d = 0.0
        if box_lo[f] > factual[f]:      # box is entirely above the factual on f
            d += box_lo[f] - factual[f]
        if factual[f] > box_hi[f]:      # box is entirely below
            d += factual[f] - box_hi[f]
        out[f] = scale[f] * d
    return out


@njit(cache=True)
def lb_feature_scalar(lo_f, hi_f, x_f, scale_f):
    """Single-feature version of lb_feature_vec (used in split scoring)."""
    d = 0.0
    if lo_f > x_f:
        d += lo_f - x_f
    if x_f > hi_f:
        d += x_f - hi_f
    return scale_f * d


@njit(cache=True)
def kth_smallest_inf(values, need):
    """The `need`-th smallest finite entry; inf if fewer than `need` are finite.

    This is the tree-LB aggregator: the whole cost budget is charged to a single
    tree (see static_tree_lb), so a single order-statistic value is admissible.
    Selection keeps a sorted length-`need` buffer of the smallest values seen and
    insertion-sorts each new candidate in — O(n * need), allocation-free, and it
    short-circuits when a value can't beat the current `need`-th smallest.  The
    inf-initialised buffer doubles as the "fewer than need finite" detector.
    """
    if need == 0:
        return 0.0
    best = np.empty(need, dtype=np.float64)
    for i in range(need):
        best[i] = np.inf
    n_seen = 0
    for v in values:
        if not np.isfinite(v):
            continue
        n_seen += 1
        if v >= best[need - 1]:
            continue  # too big to enter the smallest-`need` set
        j = need - 1
        while j > 0 and v < best[j - 1]:  # shift larger entries right
            best[j] = best[j - 1]
            j -= 1
        best[j] = v
    if n_seen < need:
        return np.inf  # quorum unreachable -> box infeasible
    return best[need - 1]  # the largest of the `need` smallest = the need-th smallest


@njit(cache=True)
def sum_need_smallest_inf(values, need):
    """Sum of the `need` smallest finite entries; inf if fewer than `need` finite.

    Same selection as kth_smallest_inf, but returns the SUM of the buffer.  This
    is the dual-LB aggregator: the share matrices split each feature's cost
    across trees (column caps prevent double counting), so summing the `need`
    cheapest priced trees is valid and typically tighter than any single tree.
    """
    if need == 0:
        return 0.0
    best = np.empty(need, dtype=np.float64)
    for i in range(need):
        best[i] = np.inf
    n_seen = 0
    for v in values:
        if not np.isfinite(v):
            continue
        n_seen += 1
        if v >= best[need - 1]:
            continue
        j = need - 1
        while j > 0 and v < best[j - 1]:
            best[j] = best[j - 1]
            j -= 1
        best[j] = v
    if n_seen < need:
        return np.inf
    s = 0.0
    for i in range(need):
        s += best[i]
    return s


@njit(cache=True)
def foil_rule_costs_l1(rules_lo_mat, rules_hi_mat, rules_class, factual, scale, target_class):
    """Scaled L1 cost from factual to each target rule's root box; inf otherwise.

    Precomputed ONCE per query.  These fixed root-box costs feed the cheap
    static per-tree LB (static_tree_lb) at every node; non-target rules get inf
    so they never contribute to the target quorum.
    """
    n_rules = rules_class.size
    n_features = factual.size
    out = np.empty(n_rules, dtype=np.float64)
    for rid in range(n_rules):
        if rules_class[rid] != target_class:
            out[rid] = np.inf
            continue
        c = 0.0
        for f in range(n_features):
            d = 0.0
            if rules_lo_mat[rid, f] > factual[f]:
                d += rules_lo_mat[rid, f] - factual[f]
            if factual[f] > rules_hi_mat[rid, f]:
                d += factual[f] - rules_hi_mat[rid, f]
            c += scale[f] * d
        out[rid] = c
    return out


@njit(cache=True)
def root_active(rules_lo_mat, rules_hi_mat, box_lo, box_hi):
    """Rule indices whose box overlaps the root box (the root's active set).

    Two half-open boxes overlap on feature f iff rule_hi > box_lo AND
    box_hi > rule_lo; a rule is active iff that holds on every feature.  make_child
    narrows this set incrementally as the search branches.
    """
    n_rules, n_features = rules_lo_mat.shape
    tmp = np.empty(n_rules, dtype=np.int64)
    n = 0
    for rid in range(n_rules):
        ok = True
        for f in range(n_features):
            if not (rules_hi_mat[rid, f] > box_lo[f] and box_hi[f] > rules_lo_mat[rid, f]):
                ok = False
                break
        if ok:
            tmp[n] = rid
            n += 1
    return tmp[:n].copy()


@njit(cache=True)
def static_tree_lb(active, rules_tree_id, rules_class, foil_rule_costs, n_trees, need, target_class, best_cost):
    """Per-tree order-statistic LB using precomputed root-box foil costs.

    Per tree, the min cost over its active target rules; then the `need`-th
    smallest over trees.  Admissible: any counterfactual needs `need` trees to
    vote target, each at least its own min foil cost.
    """
    min_cost = np.empty(n_trees, dtype=np.float64)
    for t in range(n_trees):
        min_cost[t] = np.inf
    for i in range(active.size):
        rid = active[i]
        if rules_class[rid] != target_class:
            continue
        c = foil_rule_costs[rid]
        if c >= best_cost:
            continue
        tid = rules_tree_id[rid]
        if c < min_cost[tid]:
            min_cost[tid] = c
    return kth_smallest_inf(min_cost, need)


@njit(cache=True)
def node_local_tree_lb(active, rules_lo_mat, rules_hi_mat, rules_tree_id, rules_class,
                       box_lo, box_hi, factual, scale, n_trees, need, target_class):
    """Per-tree order-statistic LB with target-leaf boxes clipped to the node box.

    Tighter than the static form because the movement to reach each leaf is
    measured against the (shrunken) node box, not the root box.
    """
    min_cost = np.empty(n_trees, dtype=np.float64)
    n_features = factual.size
    for t in range(n_trees):
        min_cost[t] = np.inf
    for i in range(active.size):
        rid = active[i]
        if rules_class[rid] != target_class:
            continue
        c = 0.0
        compatible = True
        for f in range(n_features):
            lo = rules_lo_mat[rid, f]
            if box_lo[f] > lo:
                lo = box_lo[f]
            hi = rules_hi_mat[rid, f]
            if box_hi[f] < hi:
                hi = box_hi[f]
            if not (lo < hi):
                compatible = False
                break
            d = 0.0
            if lo > factual[f]:
                d += lo - factual[f]
            if factual[f] > hi:
                d += factual[f] - hi
            c += scale[f] * d
        if compatible:
            tid = rules_tree_id[rid]
            if c < min_cost[tid]:
                min_cost[tid] = c
    return kth_smallest_inf(min_cost, need)


@njit(cache=True)
def soft_rule_costs_l1(rules_lo_mat, rules_hi_mat, factual, scale):
    """Scaled L1 cost from factual to EVERY rule's box (soft foil costs).

    Precomputed ONCE per query, feeding the cheap static soft tree LB at every
    node.  Unlike ``foil_rule_costs_l1`` no rule gets inf: under soft voting
    every leaf carries probability mass, so every leaf needs a price.
    """
    n_rules, n_features = rules_lo_mat.shape
    out = np.empty(n_rules, dtype=np.float64)
    for rid in range(n_rules):
        c = 0.0
        for f in range(n_features):
            d = 0.0
            if rules_lo_mat[rid, f] > factual[f]:
                d += rules_lo_mat[rid, f] - factual[f]
            if factual[f] > rules_hi_mat[rid, f]:
                d += factual[f] - rules_hi_mat[rid, f]
            c += scale[f] * d
        out[rid] = c
    return out


@njit(cache=True)
def _soft_sweep_lb(costs, values, tids, n, n_trees, tau, strict_gt):
    """Soft order-statistic aggregator: min cost at which the probability sum
    can beat tau.

    ``costs/values/tids[:n]`` hold (min reach cost, target probability, tree id)
    for the candidate leaves.  For a point x with cost C, each tree's fired leaf
    is reachable within C, so  score(x) <= F(C) = sum_t max{v_l : c_l <= C}.
    F is a nondecreasing step function of C; the smallest C where F(C) beats tau
    (strictly for class 1, mirroring can_stop_soft) therefore lower-bounds every
    target point.  Returns inf when even F(inf) cannot beat tau — the box holds
    no target point at any cost (prune), the graded generalization of
    can_stop_soft's max-sum prune.  Hard voting is the v in {0,1} special case,
    where this reduces to the need-th smallest per-tree min (kth_smallest_inf).
    """
    order = np.argsort(costs[:n])
    cur = np.full(n_trees, -1.0)  # per-tree best value within budget (v >= 0)
    uncovered = n_trees
    s = 0.0
    for k in range(n):
        i = order[k]
        c = costs[i]
        if not np.isfinite(c):
            break
        v = values[i]
        t = tids[i]
        if cur[t] < 0.0:
            cur[t] = v
            uncovered -= 1
            s += v
        elif v > cur[t]:
            s += v - cur[t]
            cur[t] = v
        if uncovered == 0:
            # Testing inside a same-cost tie group is safe: F only grows within
            # the group, and every earlier prefix already failed the test.
            if strict_gt:
                if s > tau:
                    return c
            elif s >= tau:
                return c
    return np.inf


@njit(cache=True)
def static_soft_tree_lb(active, rules_tree_id, target_proba, soft_rule_costs,
                        n_trees, tau, strict_gt):
    """Soft per-tree order-statistic LB from precomputed root-box leaf costs.

    Admissible at any node: the root-box cost to each leaf lower-bounds the
    node-box cost.  The soft analogue of ``static_tree_lb``.
    """
    n = active.size
    costs = np.empty(n, dtype=np.float64)
    values = np.empty(n, dtype=np.float64)
    tids = np.empty(n, dtype=np.int64)
    for i in range(n):
        rid = active[i]
        costs[i] = soft_rule_costs[rid]
        values[i] = target_proba[rid]
        tids[i] = rules_tree_id[rid]
    return _soft_sweep_lb(costs, values, tids, n, n_trees, tau, strict_gt)


@njit(cache=True)
def node_local_soft_tree_lb(active, rules_lo_mat, rules_hi_mat, rules_tree_id,
                            target_proba, box_lo, box_hi, factual, scale,
                            n_trees, tau, strict_gt):
    """Soft per-tree order-statistic LB with leaf boxes clipped to the node box.

    Tighter than the static form because each leaf's reach cost is measured
    against the (shrunken) node box.  The soft analogue of
    ``node_local_tree_lb``.
    """
    n_features = factual.size
    n = active.size
    costs = np.empty(n, dtype=np.float64)
    values = np.empty(n, dtype=np.float64)
    tids = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        rid = active[i]
        c = 0.0
        compatible = True
        for f in range(n_features):
            lo = rules_lo_mat[rid, f]
            if box_lo[f] > lo:
                lo = box_lo[f]
            hi = rules_hi_mat[rid, f]
            if box_hi[f] < hi:
                hi = box_hi[f]
            if not (lo < hi):
                compatible = False
                break
            d = 0.0
            if lo > factual[f]:
                d += lo - factual[f]
            if factual[f] > hi:
                d += factual[f] - hi
            c += scale[f] * d
        if compatible:
            costs[m] = c
            values[m] = target_proba[rid]
            tids[m] = rules_tree_id[rid]
            m += 1
    return _soft_sweep_lb(costs, values, tids, m, n_trees, tau, strict_gt)


@njit(cache=True)
def dual_static_lb(active, rules_tree_id, dual_w, n_trees, need):
    """Cost-splitting dual LB from precomputed static per-rule weights.

    ``dual_w`` is (n_pool, n_rules); non-target rules hold +inf.  For each pool
    entry: per-tree min over active target rules, then sum of the `need`
    smallest.  Returns the max over pool entries (each entry is admissible).
    """
    n_pool = dual_w.shape[0]
    best = 0.0
    min_cost = np.empty(n_trees, dtype=np.float64)
    for k in range(n_pool):
        for t in range(n_trees):
            min_cost[t] = np.inf
        for i in range(active.size):
            rid = active[i]
            c = dual_w[k, rid]
            if not np.isfinite(c):
                continue
            tid = rules_tree_id[rid]
            if c < min_cost[tid]:
                min_cost[tid] = c
        val = sum_need_smallest_inf(min_cost, need)
        if val > best:
            best = val
    return best


@njit(cache=True)
def dual_local_lb(active, rules_lo_mat, rules_hi_mat, rules_tree_id, rules_class,
                  box_lo, box_hi, factual, scale, alpha_pos, alpha_neg,
                  n_trees, need, target_class):
    """Box-clipped cost-splitting dual LB.

    ``alpha_pos``/``alpha_neg`` are (n_pool, n_trees, n_features) share
    matrices.  Per active target rule the one-sided movement requirements are
    clipped to the node box, weighted by each pool entry's shares for the rule's
    tree, then reduced by per-tree min and sum-of-need-smallest.  Max over pool.
    """
    n_pool = alpha_pos.shape[0]
    n_features = factual.size
    best = 0.0
    min_cost = np.empty((n_pool, n_trees), dtype=np.float64)
    for k in range(n_pool):
        for t in range(n_trees):
            min_cost[k, t] = np.inf
    for i in range(active.size):
        rid = active[i]
        if rules_class[rid] != target_class:
            continue
        tid = rules_tree_id[rid]
        compatible = True
        for f in range(n_features):
            lo = rules_lo_mat[rid, f]
            if box_lo[f] > lo:
                lo = box_lo[f]
            hi = rules_hi_mat[rid, f]
            if box_hi[f] < hi:
                hi = box_hi[f]
            if not (lo < hi):
                compatible = False
                break
        if not compatible:
            continue
        for k in range(n_pool):
            c = 0.0
            for f in range(n_features):
                lo = rules_lo_mat[rid, f]
                if box_lo[f] > lo:
                    lo = box_lo[f]
                hi = rules_hi_mat[rid, f]
                if box_hi[f] < hi:
                    hi = box_hi[f]
                if lo > factual[f]:
                    c += alpha_pos[k, tid, f] * scale[f] * (lo - factual[f])
                elif factual[f] > hi:
                    c += alpha_neg[k, tid, f] * scale[f] * (factual[f] - hi)
            if c < min_cost[k, tid]:
                min_cost[k, tid] = c
    for k in range(n_pool):
        val = sum_need_smallest_inf(min_cost[k], need)
        if val > best:
            best = val
    return best


@njit(cache=True)
def can_stop_rf(active, rules_tree_id, rules_class, n_trees, need, target_class):
    """RF vote stop test.

    Returns target_class if `need` trees are settled onto the target class,
    -2 if the target quorum is unreachable (prune), -1 if undecided (split).

    Per tree, count active leaves and target-class active leaves in this box:
      - settled  (n_target == n_active > 0): EVERY leaf of the tree still
        reachable in this box votes target -> the tree votes target for sure.
      - reachable (n_target > 0): the tree CAN still vote target.
    `need` settled trees => whole box already reaches the target (accept).
    Fewer than `need` reachable trees => target quorum impossible (prune).
    Otherwise the verdict depends on where in the box you land (split).
    """
    n_active = np.zeros(n_trees, dtype=np.int64)
    n_target = np.zeros(n_trees, dtype=np.int64)
    for i in range(active.size):
        rid = active[i]
        tid = rules_tree_id[rid]
        n_active[tid] += 1
        if rules_class[rid] == target_class:
            n_target[tid] += 1
    settled_target = 0
    reachable_target = 0
    for t in range(n_trees):
        if n_active[t] == 0:
            return -2  # a tree with no reachable leaf: box is empty/infeasible
        if n_target[t] > 0:
            reachable_target += 1
            if n_target[t] == n_active[t]:
                settled_target += 1
    if settled_target >= need:
        return target_class  # accept: box is entirely target-voting
    if reachable_target < need:
        return -2            # prune: not enough trees can ever vote target
    return -1                # undecided: must branch


@njit(cache=True)
def can_stop_soft(active, rules_tree_id, target_proba, n_trees, thresh, strict_gt,
                  target_class):
    """Soft (probability-averaged) vote stop test.

    Each tree contributes ``target_proba`` (= P(target | active leaf)) for the
    leaf the point lands in; the forest predicts the target class iff the sum of
    those probabilities beats ``thresh`` (= 0.5 * n_trees), i.e. the averaged
    probability beats 0.5.  ``strict_gt`` selects ``>`` vs ``>=`` to mirror
    numpy.argmax's tie-break (target class 1 needs a strict majority of the
    probability mass; class 0 wins ties).

    Over a box, tree ``t``'s contribution ranges in ``[tmin_t, tmax_t]`` across
    its active leaves.  Accept (whole box target) when even the minimal sum wins;
    prune when even the maximal sum loses; otherwise split.
    """
    tmin = np.full(n_trees, np.inf)
    tmax = np.full(n_trees, -np.inf)
    seen = np.zeros(n_trees, dtype=np.int64)
    for i in range(active.size):
        rid = active[i]
        tid = rules_tree_id[rid]
        p = target_proba[rid]
        if p < tmin[tid]:
            tmin[tid] = p
        if p > tmax[tid]:
            tmax[tid] = p
        seen[tid] += 1
    min_sum = 0.0
    max_sum = 0.0
    for t in range(n_trees):
        if seen[t] == 0:
            return -2  # empty box on this tree
        min_sum += tmin[t]
        max_sum += tmax[t]
    if strict_gt:
        if min_sum > thresh:
            return target_class     # accept: every completion strictly wins
        if max_sum <= thresh:
            return -2               # prune: no completion can strictly win
    else:
        if min_sum >= thresh:
            return target_class
        if max_sum < thresh:
            return -2
    return -1                       # undecided: must branch


@njit(cache=True)
def choose_split_baseline(active, box_lo, box_hi, lb_per_feature, factual, scale,
                          rules_lo_bin_mat, rules_hi_bin_mat, threshold_offsets,
                          threshold_values, parent_lb_per_tree, accumulated):
    """Baseline split scoring.

    For every candidate threshold interior to the box, score the pair of
    children by the *worse* child's LB (max of geometric and per-tree bounds),
    picking the threshold that maximizes it.  Ties break toward the split that
    prunes more active rules (min of the two drops), then total drop.

    Rationale: raising the weaker child's LB is what tightens the frontier;
    breaking near-ties toward bigger active-set drops keeps the boxes shrinking.
    """
    parent_n_active = active.size
    n_features = factual.size
    if parent_n_active == 0:
        return -1, np.nan
    parent_lb_sum = 0.0
    for f in range(n_features):
        parent_lb_sum += lb_per_feature[f]

    best_f = -1
    best_t = np.nan
    best_lb = -np.inf
    best_progress = -1
    best_total = -1

    # Shared histogram scratch, sized for the widest feature grid: allocating
    # two fresh arrays per feature per expansion dominated the kernel's
    # allocation churn.  Each feature zeroes only the prefix it uses.
    max_bins = 1
    for f in range(n_features):
        nb = threshold_offsets[f + 1] - threshold_offsets[f] + 1
        if nb > max_bins:
            max_bins = nb
    lo_hist = np.empty(max_bins, dtype=np.int64)
    hi_hist = np.empty(max_bins, dtype=np.int64)

    for f in range(n_features):
        # Candidate thresholds for feature f live in threshold_values[start:stop].
        start = threshold_offsets[f]
        stop = threshold_offsets[f + 1]
        n_ts = stop - start
        if n_ts <= 0:
            continue
        # Restrict to thresholds strictly interior to the current box on f
        # (indices [lo_idx, hi_idx)); a split outside the box does nothing.
        lo_idx = 0
        while lo_idx < n_ts and threshold_values[start + lo_idx] <= box_lo[f]:
            lo_idx += 1
        hi_idx = lo_idx
        while hi_idx < n_ts and threshold_values[start + hi_idx] < box_hi[f]:
            hi_idx += 1
        if lo_idx >= hi_idx:
            continue

        # Histogram active rules by their lo/hi threshold-grid bin on f, so a
        # prefix sum gives, for each candidate threshold, how many rules survive
        # each child without re-scanning the active set per threshold.
        n_bins = n_ts + 1
        for j in range(n_bins):
            lo_hist[j] = 0
            hi_hist[j] = 0
        for i in range(parent_n_active):
            rid = active[i]
            lo_hist[rules_lo_bin_mat[rid, f]] += 1
            hi_hist[rules_hi_bin_mat[rid, f]] += 1

        cum_lo = 0
        cum_hi = 0
        for j in range(n_bins):
            cum_lo += lo_hist[j]
            cum_hi += hi_hist[j]
            if j < lo_idx or j >= hi_idx:
                continue
            # Rules kept by each child at threshold t:
            #   meet (x<=t): rules whose lo-bin is <= j  -> cum_lo
            #   not-meet (x>t): rules whose hi-bin is > j -> parent - cum_hi
            n_keep_meet = cum_lo
            n_keep_nmeet = parent_n_active - cum_hi
            if n_keep_meet <= 0 and n_keep_nmeet <= 0:
                continue

            # Geometric LB of each child = parent geom with feature f's term
            # swapped for the child's narrowed interval.
            t = threshold_values[start + j]
            meet_lb_f = lb_feature_scalar(box_lo[f], t, factual[f], scale[f])
            nmeet_lb_f = lb_feature_scalar(t, box_hi[f], factual[f], scale[f])
            meet_geom = parent_lb_sum - lb_per_feature[f] + meet_lb_f
            nmeet_geom = parent_lb_sum - lb_per_feature[f] + nmeet_lb_f

            # Stack with the (inherited) per-tree bound; empty child scores inf so
            # it never limits the min below.
            if np.isfinite(parent_lb_per_tree):
                meet_lb = accumulated + max(meet_geom, parent_lb_per_tree)
                nmeet_lb = accumulated + max(nmeet_geom, parent_lb_per_tree)
            else:
                meet_lb = np.inf
                nmeet_lb = np.inf
            if n_keep_meet <= 0:
                meet_lb = np.inf
            if n_keep_nmeet <= 0:
                nmeet_lb = np.inf
            # Primary score = the WEAKER (smaller-LB) child: that's the bound the
            # split is stuck with, so we want to make it as large as possible.
            lb_primary = meet_lb if meet_lb < nmeet_lb else nmeet_lb
            if not np.isfinite(lb_primary):
                continue

            # Tie-breakers: prefer the split that drops more rules from its
            # weaker side (active_progress), then more rules overall (total).
            drop_meet = parent_n_active - n_keep_meet
            drop_nmeet = parent_n_active - n_keep_nmeet
            active_progress = drop_meet if drop_meet < drop_nmeet else drop_nmeet
            total = drop_meet + drop_nmeet

            if (lb_primary > best_lb or
                    (lb_primary == best_lb and active_progress > best_progress) or
                    (lb_primary == best_lb and active_progress == best_progress and total > best_total)):
                best_lb = lb_primary
                best_progress = active_progress
                best_total = total
                best_f = f
                best_t = t

    return best_f, best_t


@njit(cache=True)
def make_child(active, box_lo, box_hi, lb_per_feature, factual, scale,
               rules_lo_mat, rules_hi_mat, feature, threshold, side):
    """Build one child box (side 0 = meet x<=t, side 1 = not-meet x>t).

    Returns (child_active, child_lo, child_hi, child_lb_per_feature, ok); ok=0
    means the child is empty.
    """
    # Copy the box and tighten feature `feature` on the chosen side.  side 0
    # (meet) lowers hi to the threshold (x<=t); side 1 (not-meet) raises lo
    # (x>t).  If the threshold is already outside the box, the child is empty.
    child_lo = box_lo.copy()
    child_hi = box_hi.copy()
    if side == 0:
        if threshold <= child_lo[feature]:
            return np.empty(0, dtype=np.int64), child_lo, child_hi, lb_per_feature.copy(), 0
        if threshold < child_hi[feature]:
            child_hi[feature] = threshold
    else:
        if threshold >= child_hi[feature]:
            return np.empty(0, dtype=np.int64), child_lo, child_hi, lb_per_feature.copy(), 0
        if threshold > child_lo[feature]:
            child_lo[feature] = threshold

    # Filter the parent's active rules to those still overlapping on `feature`
    # (only this feature changed, so only it can drop rules).  meet keeps rules
    # with lo < child_hi; not-meet keeps rules with hi > child_lo.
    tmp = np.empty(active.size, dtype=np.int64)
    n = 0
    for i in range(active.size):
        rid = active[i]
        keep = rules_lo_mat[rid, feature] < child_hi[feature]
        if side == 1:
            keep = rules_hi_mat[rid, feature] > child_lo[feature]
        if keep:
            tmp[n] = rid
            n += 1
    if n == 0:
        return np.empty(0, dtype=np.int64), child_lo, child_hi, lb_per_feature.copy(), 0
    # Only feature `feature`'s geometric term changed; patch it in place.
    child_lb = lb_per_feature.copy()
    child_lb[feature] = lb_feature_scalar(
        child_lo[feature], child_hi[feature], factual[feature], scale[feature]
    )
    return tmp[:n].copy(), child_lo, child_hi, child_lb, 1


@njit(cache=True)
def _dual_shares_eval(req_pos, req_neg, tree_pos, starts, n_rt, need,
                      alpha_pos, alpha_neg, w, mins, argmins, order):
    """Evaluate the cost-splitting dual: fills w/mins/argmins, returns the bound
    with the winning quorum stored in order[:need].

    Three reductions (see the paper, Eq. mu and Prop. Admissibility):
      1. price each leaf under the current shares  -> w[l]
      2. per tree, keep its cheapest leaf          -> mins[t], argmins[t]
      3. sum the `need` cheapest trees             -> the bound g
    The output buffers (w/mins/argmins/order) are caller-owned scratch so the
    ascent loop can call this every iteration without reallocating.
    """
    n_leaves, n_features = req_pos.shape
    # (1) Leaf pricing: w[l] = sum_f alpha_pos[t,f]*req_pos[l,f]
    #                              + alpha_neg[t,f]*req_neg[l,f],  t = leaf l's tree.
    # This is the alpha-weighted one-sided movement to enter leaf l (clipped
    # reqs are baked into req_pos/req_neg by the caller).
    for l in range(n_leaves):
        t = tree_pos[l]
        c = 0.0
        for f in range(n_features):
            c += req_pos[l, f] * alpha_pos[t, f] + req_neg[l, f] * alpha_neg[t, f]
        w[l] = c
    # (2) Per-tree minimum: each tree only needs to reach ONE target leaf, so it
    # takes its cheapest priced leaf.  mins[t] = mu_t(alpha); argmins[t] is the
    # winning leaf (its req vector is this tree's supergradient contribution).
    # A tree with no active target leaf keeps mins=inf, argmins=-1.
    for t in range(n_rt):
        mins[t] = np.inf
        argmins[t] = -1
    for l in range(n_leaves):
        t = tree_pos[l]
        if w[l] < mins[t]:
            mins[t] = w[l]
            argmins[t] = l
    # (3) Order statistic: a valid counterfactual needs `need` trees voting
    # target, so the bound is the sum of the `need` smallest mu_t.  order[:need]
    # records which trees won (the quorum S) for the supergradient step.  If
    # fewer than `need` trees are finite, g = inf -> box is infeasible.
    idx = np.argsort(mins)
    g = 0.0
    for i in range(need):
        order[i] = idx[i]
        g += mins[idx[i]]
    return g


@njit(cache=True)
def _project_columns_capped_simplex_inplace(alpha):
    """Project each column of alpha onto the capped simplex {v >= 0, sum(v) <= 1}.

    One column = one feature; the rows are the trees' shares of that feature.
    The cap sum_t alpha[t,f] <= 1 is what stops the bound from charging the same
    real feature move to several trees (Eq. cap).  After an unconstrained
    supergradient step the shares can leave the set, so we project them back;
    this keeps every iterate admissible.  Standard Euclidean projection onto the
    capped simplex (Held-Karp-style multiplier clamping), done per column.
    """
    n, m = alpha.shape
    for f in range(m):
        # Clamp negatives to 0 and sum the column.  If it already satisfies the
        # cap, no shift is needed -- the box constraint v >= 0 is all that bound.
        s = 0.0
        for t in range(n):
            if alpha[t, f] < 0.0:
                alpha[t, f] = 0.0
            s += alpha[t, f]
        if s <= 1.0:
            continue
        # Column overflows the cap: subtract the smallest threshold theta that
        # brings sum(max(v-theta, 0)) back to 1.  Found by scanning the sorted-
        # descending column and taking the largest prefix whose induced theta
        # still leaves its own entry positive (the classic simplex-projection
        # water-filling).
        col = np.sort(alpha[:, f])[::-1]
        css = 0.0
        theta = 0.0
        for k in range(n):
            css += col[k]
            tk = (css - 1.0) / (k + 1.0)
            if col[k] - tk > 0.0:
                theta = tk
        # Apply the shift: v <- max(v - theta, 0).  Now sum_t alpha[t,f] == 1.
        for t in range(n):
            v = alpha[t, f] - theta
            alpha[t, f] = v if v > 0.0 else 0.0


@njit(cache=True)
def _project_columns_capped_simplex_masked_inplace(alpha, stable):
    """`_project_columns_capped_simplex_inplace` skipping provably-no-op columns.

    ``stable[f] == 1`` records that column f's current values are invariant
    under the projection (all entries >= 0 and sum <= 1), so the dense pass
    would leave it untouched — skipping it is bit-identical.  Callers must
    clear ``stable[f]`` whenever they modify column f.  A clamp-only pass
    leaves the column invariant afterwards, so it sets ``stable[f] = 1``; a
    water-filled column keeps ``stable[f] = 0`` because its post-fill sum can
    still exceed 1.0 by rounding, in which case the dense pass would touch it
    again next iteration and we must too.
    """
    n, m = alpha.shape
    for f in range(m):
        if stable[f]:
            continue
        s = 0.0
        for t in range(n):
            if alpha[t, f] < 0.0:
                alpha[t, f] = 0.0
            s += alpha[t, f]
        if s <= 1.0:
            stable[f] = 1
            continue
        col = np.sort(alpha[:, f])[::-1]
        css = 0.0
        theta = 0.0
        for k in range(n):
            css += col[k]
            tk = (css - 1.0) / (k + 1.0)
            if col[k] - tk > 0.0:
                theta = tk
        for t in range(n):
            v = alpha[t, f] - theta
            alpha[t, f] = v if v > 0.0 else 0.0


@njit(cache=True)
def optimize_dual_shares_kernel(req_pos, req_neg, tree_pos, starts, n_rt, need,
                                incumbent, alpha_pos, alpha_neg, max_iters,
                                patience):
    """Projected supergradient ascent on the dual share matrices (compiled).

    Maximises the concave dual bound L(alpha) = sum of the `need` smallest priced
    trees over feasible share matrices.  Mutates alpha_pos/alpha_neg as working
    buffers and returns (best_bound, best_alpha_pos, best_alpha_neg).  The step
    is a Polyak step toward an adaptive target `best + delta`; because every
    iterate is projected back to feasibility, every L value is admissible, so we
    simply track and return the best one seen (safe to truncate at any budget).
    """
    n_leaves, n_features = req_pos.shape
    # Scratch buffers reused across iterations (no per-iter allocation): leaf
    # prices, per-tree minima + their argmin leaf, and the winning quorum order.
    w = np.empty(n_leaves, dtype=np.float64)
    mins = np.empty(n_rt, dtype=np.float64)
    argmins = np.empty(n_rt, dtype=np.int64)
    order = np.empty(n_rt, dtype=np.int64)
    # Evaluate the bound at the starting shares (uniform, or a warm start from
    # the pool); this also fills argmins/order for the first supergradient.
    g = _dual_shares_eval(req_pos, req_neg, tree_pos, starts, n_rt, need,
                          alpha_pos, alpha_neg, w, mins, argmins, order)
    best = g
    best_ap = alpha_pos.copy()
    best_an = alpha_neg.copy()
    # Polyak target gap `delta`.  With a finite incumbent, aim a quarter of the
    # way toward it (the ascent target is best+delta); otherwise fall back to a
    # scale-free guess.  delta shrinks on stall (below) so the step contracts as
    # the bound saturates.  NOTE: `incumbent` here is the SEARCH incumbent, never
    # a rounded one -- feeding rounded costs shrinks delta and weakens proofs.
    if np.isfinite(incumbent):
        delta = 0.25 * max(incumbent - best, 0.05)
    else:
        delta = max(best, 0.05)
    stall = 0
    grad_pos = np.zeros((n_rt, n_features), dtype=np.float64)
    grad_neg = np.zeros((n_rt, n_features), dtype=np.float64)
    for _ in range(max_iters):
        # Zero the supergradient buffers.  Only the `need` winning trees get a
        # nonzero row below; trees outside the quorum don't affect L, so their
        # supergradient component is 0.
        for t in range(n_rt):
            for f in range(n_features):
                grad_pos[t, f] = 0.0
                grad_neg[t, f] = 0.0
        # Build the supergradient (envelope theorem): for each winning tree, the
        # slope of its min-over-leaves is exactly the req vector of its argmin
        # leaf.  Accumulate ||g||^2 for the Polyak step denominator.
        gnorm2 = 0.0
        for i in range(need):
            t = order[i]
            l = argmins[t]
            if l < 0:
                continue
            for f in range(n_features):
                grad_pos[t, f] = req_pos[l, f]
                grad_neg[t, f] = req_neg[l, f]
                gnorm2 += req_pos[l, f] * req_pos[l, f] + req_neg[l, f] * req_neg[l, f]
        if gnorm2 <= 1e-18:
            break  # flat supergradient: the bound can't move, stop.
        # Polyak step size: (target - current) / ||g||^2, target = best + delta.
        step = (best + delta - g) / gnorm2
        # Ascend, then project each column back onto the capped simplex so the
        # new shares stay feasible (and the resulting bound stays admissible).
        for t in range(n_rt):
            for f in range(n_features):
                alpha_pos[t, f] += step * grad_pos[t, f]
                alpha_neg[t, f] += step * grad_neg[t, f]
        _project_columns_capped_simplex_inplace(alpha_pos)
        _project_columns_capped_simplex_inplace(alpha_neg)
        # Re-evaluate at the new shares; refresh argmins/order for next step.
        g = _dual_shares_eval(req_pos, req_neg, tree_pos, starts, n_rt, need,
                              alpha_pos, alpha_neg, w, mins, argmins, order)
        # Keep the best admissible bound; on repeated non-improvement, halve the
        # target gap (finer steps) and give up once it's negligible.
        if g > best + 1e-12:
            best = g
            best_ap = alpha_pos.copy()
            best_an = alpha_neg.copy()
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                delta *= 0.5
                stall = 0
                if delta < 1e-6:
                    break
    return best, best_ap, best_an


@njit(cache=True)
def _f32_gt(v):
    """Smallest float32 (as float64) strictly greater than the float64 value v.

    sklearn casts the query to float32 but keeps thresholds in float64, so it
    tests ``float32(x) > threshold``.  A point on a right-branch leaf's lower
    face must therefore be a float32 value strictly above the (float64) threshold
    ``v``.  Round v to float32, then step one float32 grid point up if it did not
    already clear v.
    """
    xf = np.float32(v)
    if np.float64(xf) <= v:
        xf = np.nextafter(xf, np.float32(np.inf))
    return np.float64(xf)


@njit(cache=True)
def _f32_le(v):
    """Largest float32 (as float64) that is <= the float64 value v.

    Dual of :func:`_f32_gt` for the inclusive upper face (``float32(x) <= t``):
    round v to float32 and step one grid point down if it overshot above v.
    """
    xf = np.float32(v)
    if np.float64(xf) > v:
        xf = np.nextafter(xf, -np.float32(np.inf))
    return np.float64(xf)


@njit(cache=True)
def refine_l1(box_lo, box_hi, factual, scale):
    """Project factual onto the box, landing on the float32 grid sklearn uses.

    The exact cheapest point of a box under separable L1 is the per-axis clip of
    the factual into (lo, hi] -- used both to accept a fully-target box and to
    finish greedy rounding.  Because sklearn evaluates ``float32(x)`` against
    float64 thresholds, every coordinate must be a float32 value on the correct
    side of the box faces: strictly above the exclusive lower face (``_f32_gt``)
    and at or below the inclusive upper face (``_f32_le``).  The factual is
    already float32-valued, so an interior coordinate is kept as-is.  Only the
    clipped faces are snapped, guaranteeing ``rf.predict`` agrees with the
    certified vote.
    """
    x = factual.copy()
    for f in range(factual.size):
        lo = box_lo[f]
        hi = box_hi[f]
        if x[f] <= lo:
            x[f] = _f32_gt(lo)      # exclusive lower face: smallest float32 > lo
        if x[f] > hi:
            x[f] = _f32_le(hi)      # inclusive upper face: largest float32 <= hi
    cost = 0.0
    for f in range(factual.size):
        d = factual[f] - x[f]
        if d < 0:
            d = -d
        cost += scale[f] * d
    return x, cost


@njit(cache=True)
def _group_target_leaves(active, rules_tree_id, rules_class, n_trees, target_class):
    """Group active target leaves by tree (counting sort).

    Returns (counts, grouped, n_target): tree t's leaves are
    ``grouped[counts[t]:counts[t+1]]``.
    """
    counts = np.zeros(n_trees + 1, dtype=np.int64)
    n_target = 0
    for i in range(active.size):
        rid = active[i]
        if rules_class[rid] == target_class:
            counts[rules_tree_id[rid] + 1] += 1
            n_target += 1
    for t in range(n_trees):
        counts[t + 1] += counts[t]
    grouped = np.empty(n_target, dtype=np.int64)
    fill = counts[:n_trees].copy()
    for i in range(active.size):
        rid = active[i]
        if rules_class[rid] == target_class:
            tid = rules_tree_id[rid]
            grouped[fill[tid]] = rid
            fill[tid] += 1
    return counts, grouped, n_target


@njit(cache=True)
def _repair_quorum(counts, grouped, order, tree_min, rules_lo_mat, rules_hi_mat,
                   box_lo, box_hi, factual, scale, n_trees, need):
    """Feasibility-repair pass shared by the rounding variants.

    Grow a running intersection box seeded with the node box, visiting trees in
    the given order (skipping inf-keyed trees).  Each tree contributes the
    compatible target leaf whose intersection with the running box costs least
    in TRUE projected L1; that intersection becomes the new box, so later trees
    are constrained by earlier picks (this is what makes the selected quorum
    mutually compatible instead of independently cheapest).
    """
    n_features = factual.size
    run_lo = box_lo.copy()
    run_hi = box_hi.copy()
    cand_lo = np.empty(n_features, dtype=np.float64)
    cand_hi = np.empty(n_features, dtype=np.float64)
    best_lo = np.empty(n_features, dtype=np.float64)
    best_hi = np.empty(n_features, dtype=np.float64)
    selected = 0
    for oi in range(n_trees):
        t = order[oi]
        if not np.isfinite(tree_min[t]):
            break  # remaining trees have no box-compatible target leaf
        # Pick this tree's best leaf against the RUNNING box (cand = leaf ∩ run).
        best_c = np.inf
        for j in range(counts[t], counts[t + 1]):
            rid = grouped[j]
            c = 0.0
            feasible = True
            for f in range(n_features):
                lo = rules_lo_mat[rid, f]
                if run_lo[f] > lo:
                    lo = run_lo[f]
                hi = rules_hi_mat[rid, f]
                if run_hi[f] < hi:
                    hi = run_hi[f]
                if not (lo < hi):
                    feasible = False
                    break
                cand_lo[f] = lo
                cand_hi[f] = hi
                if lo > factual[f]:
                    c += scale[f] * (lo - factual[f])
                elif factual[f] > hi:
                    c += scale[f] * (factual[f] - hi)
            if feasible and c < best_c:
                best_c = c
                for f in range(n_features):
                    best_lo[f] = cand_lo[f]
                    best_hi[f] = cand_hi[f]
        if not np.isfinite(best_c):
            continue  # no leaf of this tree fits the running box; skip it
        # Commit: shrink the running box to the chosen leaf's intersection.
        for f in range(n_features):
            run_lo[f] = best_lo[f]
            run_hi[f] = best_hi[f]
        selected += 1
        if selected >= need:
            break  # a full quorum is locked in; the running box is feasible
    if selected < need:
        return np.inf, factual.copy()  # couldn't assemble `need` compatible trees
    # The running box lies in a target leaf of each selected tree, so its factual
    # projection is a genuine counterfactual (>= need target votes).
    x, cost = refine_l1(run_lo, run_hi, factual, scale)
    return cost, x


@njit(cache=True)
def greedy_round_l1(active, rules_lo_mat, rules_hi_mat, rules_tree_id, rules_class,
                    box_lo, box_hi, factual, scale, n_trees, need, target_class):
    """Cost-greedy feasibility-repair primal rounding at a node.

    Ordering key: each tree's cheapest target leaf measured against the NODE
    box (not yet the running intersection), cheapest first; trees with no
    compatible leaf get inf.  The repair pass then intersects one leaf per
    visited tree (see ``_repair_quorum``).  Returns (cost, x); (inf, factual)
    if fewer than ``need`` trees are mutually compatible.
    """
    n_features = factual.size
    counts, grouped, n_target = _group_target_leaves(
        active, rules_tree_id, rules_class, n_trees, target_class
    )
    if n_target == 0:
        return np.inf, factual.copy()
    tree_min = np.empty(n_trees, dtype=np.float64)
    for t in range(n_trees):
        tree_min[t] = np.inf
        for j in range(counts[t], counts[t + 1]):
            rid = grouped[j]
            c = 0.0
            compatible = True
            for f in range(n_features):
                lo = rules_lo_mat[rid, f]
                if box_lo[f] > lo:
                    lo = box_lo[f]
                hi = rules_hi_mat[rid, f]
                if box_hi[f] < hi:
                    hi = box_hi[f]
                if not (lo < hi):
                    compatible = False
                    break
                if lo > factual[f]:
                    c += scale[f] * (lo - factual[f])
                elif factual[f] > hi:
                    c += scale[f] * (factual[f] - hi)
            if compatible and c < tree_min[t]:
                tree_min[t] = c
    order = np.argsort(tree_min)
    return _repair_quorum(counts, grouped, order, tree_min, rules_lo_mat,
                          rules_hi_mat, box_lo, box_hi, factual, scale,
                          n_trees, need)


@njit(cache=True)
def dual_guided_round_l1(active, rules_lo_mat, rules_hi_mat, rules_tree_id,
                         rules_class, box_lo, box_hi, factual, scale,
                         alpha_pos, alpha_neg, n_trees, need, target_class):
    """Dual-guided feasibility-repair primal rounding (Lagrangian heuristic).

    Same repair pass as ``greedy_round_l1`` but the tree visit order comes from
    the dual: each tree's key is mu_t(alpha) — its minimum alpha-weighted
    clipped movement over compatible target leaves (``alpha_pos``/``alpha_neg``
    are one pool entry, (n_trees, n_features)).  The `need` dual-cheapest trees
    are exactly the quorum the bound believes the optimum uses, so committing
    them first tends to assemble a cheaper compatible quorum than the raw
    cost order when the alphas have concentrated cost on the binding features.
    """
    n_features = factual.size
    counts, grouped, n_target = _group_target_leaves(
        active, rules_tree_id, rules_class, n_trees, target_class
    )
    if n_target == 0:
        return np.inf, factual.copy()
    # Ordering key: dual price mu_t(alpha) over NODE-box-clipped leaves.  A
    # tree with no compatible leaf keeps inf and is skipped by the repair.
    tree_min = np.empty(n_trees, dtype=np.float64)
    for t in range(n_trees):
        tree_min[t] = np.inf
        for j in range(counts[t], counts[t + 1]):
            rid = grouped[j]
            c = 0.0
            compatible = True
            for f in range(n_features):
                lo = rules_lo_mat[rid, f]
                if box_lo[f] > lo:
                    lo = box_lo[f]
                hi = rules_hi_mat[rid, f]
                if box_hi[f] < hi:
                    hi = box_hi[f]
                if not (lo < hi):
                    compatible = False
                    break
                if lo > factual[f]:
                    c += alpha_pos[t, f] * scale[f] * (lo - factual[f])
                elif factual[f] > hi:
                    c += alpha_neg[t, f] * scale[f] * (factual[f] - hi)
            if compatible and c < tree_min[t]:
                tree_min[t] = c
    order = np.argsort(tree_min)
    return _repair_quorum(counts, grouped, order, tree_min, rules_lo_mat,
                          rules_hi_mat, box_lo, box_hi, factual, scale,
                          n_trees, need)


@njit(cache=True)
def _point_score(x, rules_lo_mat, rules_hi_mat, rules_tree_id, values, n_trees):
    """Exact forest score at a point: sum over trees of `values` at the fired leaf.

    Scans the FULL rule set (a polished point may leave any node box).  Exactly
    one leaf per tree contains x (leaves partition the space; lo exclusive, hi
    inclusive — the convention shared with point evaluation), so the loop exits
    once every tree has fired.  Hard voting is the special case values in
    {0,1}; soft voting passes the per-leaf target probabilities.
    """
    n_rules = rules_tree_id.size
    n_features = x.size
    fired = np.zeros(n_trees, dtype=np.int8)
    remaining = n_trees
    s = 0.0
    for rid in range(n_rules):
        tid = rules_tree_id[rid]
        if fired[tid] != 0:
            continue
        contains = True
        for f in range(n_features):
            if not (x[f] > rules_lo_mat[rid, f] and x[f] <= rules_hi_mat[rid, f]):
                contains = False
                break
        if contains:
            fired[tid] = 1
            s += values[rid]
            remaining -= 1
            if remaining == 0:
                break
    return s


@njit(cache=True)
def _score_beats(score, tau, strict_gt):
    if strict_gt:
        return score > tau
    return score >= tau


@njit(cache=True)
def polish_l1(x0, factual, scale, rules_lo_mat, rules_hi_mat, rules_tree_id,
              values, n_trees, tau, strict_gt, threshold_offsets,
              threshold_values, max_checks, max_passes):
    """Per-feature pull-back polish of a feasible point (primal-only).

    The rounding boxes are conservative: they force membership in specific
    leaves, but the vote/probability sum often tolerates relaxing individual
    coordinates because other trees vote target anyway.  For each moved
    feature (largest cost contribution first) try, in order of decreasing
    saving: the full pull to the factual, then the value just past each split
    threshold between the factual and the current coordinate, nearest the
    factual first (capped at ``max_checks`` probes).  A candidate is kept iff
    the exact forest score still beats tau, so feasibility is invariant; up to
    ``max_passes`` sweeps.  Candidates sit on the float32 grid on the correct
    side of the (float64) thresholds, preserving sklearn-exact validity.

    Feasibility is defined by (values, tau, strict_gt): hard voting passes the
    0/1 target indicator with tau = need, soft voting the leaf probabilities
    with tau = 0.5 * n_trees.  Returns (cost, x); (inf, x0) when x0 itself does
    not beat tau (never "improves" an infeasible input).
    """
    n_features = factual.size
    s0 = _point_score(x0, rules_lo_mat, rules_hi_mat, rules_tree_id, values, n_trees)
    if not _score_beats(s0, tau, strict_gt):
        return np.inf, x0.copy()
    x = x0.copy()
    for _pass in range(max_passes):
        improved = False
        contrib = np.empty(n_features, dtype=np.float64)
        for f in range(n_features):
            d = x[f] - factual[f]
            if d < 0.0:
                d = -d
            contrib[f] = scale[f] * d
        order = np.argsort(-contrib)
        for oi in range(n_features):
            f = order[oi]
            old = x[f]
            if old == factual[f]:
                continue
            # Full pull first: if the factual coordinate already keeps the
            # score above tau, this feature's whole contribution vanishes.
            x[f] = factual[f]
            s = _point_score(x, rules_lo_mat, rules_hi_mat, rules_tree_id,
                             values, n_trees)
            if _score_beats(s, tau, strict_gt):
                improved = True
                continue
            start = threshold_offsets[f]
            stop = threshold_offsets[f + 1]
            found = False
            checks = 0
            if old > factual[f]:
                # Pull down: probe just above each threshold in
                # [factual, old), nearest the factual first.
                for j in range(start, stop):
                    t = threshold_values[j]
                    if t < factual[f]:
                        continue
                    if t >= old:
                        break
                    cand = _f32_gt(t)
                    if cand >= old:
                        break  # no saving left (thresholds only grow from here)
                    x[f] = cand
                    checks += 1
                    s = _point_score(x, rules_lo_mat, rules_hi_mat,
                                     rules_tree_id, values, n_trees)
                    if _score_beats(s, tau, strict_gt):
                        found = True
                        break
                    if checks >= max_checks:
                        break
            else:
                # Pull up: probe at/below each threshold in (old, factual],
                # nearest the factual first.
                for j in range(stop - 1, start - 1, -1):
                    t = threshold_values[j]
                    if t > factual[f]:
                        continue
                    if t <= old:
                        break
                    cand = _f32_le(t)
                    if cand <= old:
                        break
                    if cand >= factual[f]:
                        continue  # coincides with the full pull already tried
                    x[f] = cand
                    checks += 1
                    s = _point_score(x, rules_lo_mat, rules_hi_mat,
                                     rules_tree_id, values, n_trees)
                    if _score_beats(s, tau, strict_gt):
                        found = True
                        break
                    if checks >= max_checks:
                        break
            if found:
                improved = True
            else:
                x[f] = old
        if not improved:
            break
    cost = 0.0
    for f in range(n_features):
        d = x[f] - factual[f]
        if d < 0.0:
            d = -d
        cost += scale[f] * d
    return cost, x


@njit(cache=True)
def _additive_dual_eval(req_pos, req_neg, values, tree_pos, n_rt, tau,
                        alpha_pos, alpha_neg, lam, w, mins, argmins):
    """Evaluate the additive (soft-voting) dual: fills w/mins/argmins, returns g.

    Soft voting requires sum_t v(leaf_t(x')) >= tau with v = P(target|leaf), an
    additive-threshold constraint.  Lagrangian-relaxing it with multiplier lam
    and cost-splitting the movement with the same share matrices as the quorum
    dual gives (per fixed lam, alpha — every iterate is admissible):

        cost(x, x') >= lam*tau + sum_t min_{l in active leaves of t}
                                     (alpha_t . req_l  -  lam * v_l)

    Differences from the quorum eval: the per-leaf price carries the -lam*v_l
    reward, the min runs over ALL active leaves (every leaf has a probability),
    and the reduction is a plain sum over all trees plus lam*tau — no order
    statistic.  The quorum bound is the special case v in {0,1}, tau = need.
    """
    n_leaves, n_features = req_pos.shape
    for l in range(n_leaves):
        t = tree_pos[l]
        c = -lam * values[l]
        for f in range(n_features):
            c += req_pos[l, f] * alpha_pos[t, f] + req_neg[l, f] * alpha_neg[t, f]
        w[l] = c
    for t in range(n_rt):
        mins[t] = np.inf
        argmins[t] = -1
    for l in range(n_leaves):
        t = tree_pos[l]
        if w[l] < mins[t]:
            mins[t] = w[l]
            argmins[t] = l
    g = lam * tau
    for t in range(n_rt):
        g += mins[t]
    return g


@njit(cache=True)
def _additive_dual_eval_sparse(nz_ptr, nz_feat, nz_val, nz_pos, values, tree_pos,
                               n_rt, tau, alpha_pos, alpha_neg, lam,
                               w, mins, argmins):
    """`_additive_dual_eval` over a CSR view of the req matrices.

    Bit-identical to the dense eval: per (leaf, feature) at most one of
    req_pos/req_neg is nonzero (a clipped interval can't sit on both sides of
    the factual), the CSR stores features in the same ascending order, and the
    skipped entries were exact-zero addends.
    """
    n_leaves = w.size
    for l in range(n_leaves):
        t = tree_pos[l]
        c = -lam * values[l]
        for j in range(nz_ptr[l], nz_ptr[l + 1]):
            if nz_pos[j]:
                c += nz_val[j] * alpha_pos[t, nz_feat[j]]
            else:
                c += nz_val[j] * alpha_neg[t, nz_feat[j]]
        w[l] = c
    for t in range(n_rt):
        mins[t] = np.inf
        argmins[t] = -1
    for l in range(n_leaves):
        t = tree_pos[l]
        if w[l] < mins[t]:
            mins[t] = w[l]
            argmins[t] = l
    g = lam * tau
    for t in range(n_rt):
        g += mins[t]
    return g


@njit(cache=True)
def optimize_additive_dual_kernel(req_pos, req_neg, values, tree_pos, n_rt, tau,
                                  incumbent, alpha_pos, alpha_neg, lam,
                                  max_iters, patience):
    """Projected supergradient ascent on (alpha, lam) for the additive dual.

    Same Polyak machinery as `optimize_dual_shares_kernel` plus one extra
    ascent coordinate: d g / d lam = tau - sum_t v(argmin leaf of t), projected
    to lam >= 0.  IMPORTANT: callers must warm-start (uniform alpha, lam from a
    1-D grid — the bound is piecewise-linear in lam); from a cold alpha=0,
    lam=0 start the Polyak step barely moves lam (probed 2026-07-09).
    Returns (best_bound, best_alpha_pos, best_alpha_neg, best_lam).

    The req matrices are fixed for the whole ascent and sparse (a leaf pays
    only on features where its clipped box avoids the factual), so they are
    compacted to CSR once and the eval / gradient-norm / alpha-update loops
    touch nonzeros only — bit-identical to the dense loops, whose skipped
    terms were exact-zero additions in the same feature order.
    """
    n_leaves, n_features = req_pos.shape
    nz_ptr = np.empty(n_leaves + 1, dtype=np.int64)
    nz_ptr[0] = 0
    nnz = 0
    for l in range(n_leaves):
        for f in range(n_features):
            if req_pos[l, f] != 0.0 or req_neg[l, f] != 0.0:
                nnz += 1
        nz_ptr[l + 1] = nnz
    nz_feat = np.empty(nnz, dtype=np.int64)
    nz_val = np.empty(nnz, dtype=np.float64)
    nz_pos = np.empty(nnz, dtype=np.uint8)
    j = 0
    for l in range(n_leaves):
        for f in range(n_features):
            if req_pos[l, f] != 0.0:
                nz_feat[j] = f
                nz_val[j] = req_pos[l, f]
                nz_pos[j] = 1
                j += 1
            elif req_neg[l, f] != 0.0:
                nz_feat[j] = f
                nz_val[j] = req_neg[l, f]
                nz_pos[j] = 0
                j += 1
    w = np.empty(n_leaves, dtype=np.float64)
    mins = np.empty(n_rt, dtype=np.float64)
    argmins = np.empty(n_rt, dtype=np.int64)
    g = _additive_dual_eval_sparse(nz_ptr, nz_feat, nz_val, nz_pos, values,
                                   tree_pos, n_rt, tau, alpha_pos, alpha_neg,
                                   lam, w, mins, argmins)
    best = g
    best_ap = alpha_pos.copy()
    best_an = alpha_neg.copy()
    best_lam = lam
    if np.isfinite(incumbent):
        delta = 0.25 * max(incumbent - best, 0.05)
    else:
        delta = max(abs(best), 0.05)
    stall = 0
    stable_pos = np.zeros(n_features, dtype=np.uint8)
    stable_neg = np.zeros(n_features, dtype=np.uint8)
    for _ in range(max_iters):
        grad_lam = tau
        gnorm2 = 0.0
        for t in range(n_rt):
            l = argmins[t]
            if l < 0:
                continue
            grad_lam -= values[l]
            for j in range(nz_ptr[l], nz_ptr[l + 1]):
                gnorm2 += nz_val[j] * nz_val[j]
        gnorm2 += grad_lam * grad_lam
        if gnorm2 <= 1e-18:
            break
        step = (best + delta - g) / gnorm2
        for t in range(n_rt):
            l = argmins[t]
            if l < 0:
                continue
            for j in range(nz_ptr[l], nz_ptr[l + 1]):
                if nz_pos[j]:
                    alpha_pos[t, nz_feat[j]] += step * nz_val[j]
                    stable_pos[nz_feat[j]] = 0
                else:
                    alpha_neg[t, nz_feat[j]] += step * nz_val[j]
                    stable_neg[nz_feat[j]] = 0
        lam = lam + step * grad_lam
        if lam < 0.0:
            lam = 0.0
        _project_columns_capped_simplex_masked_inplace(alpha_pos, stable_pos)
        _project_columns_capped_simplex_masked_inplace(alpha_neg, stable_neg)
        g = _additive_dual_eval_sparse(nz_ptr, nz_feat, nz_val, nz_pos, values,
                                       tree_pos, n_rt, tau, alpha_pos, alpha_neg,
                                       lam, w, mins, argmins)
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


@njit(cache=True)
def additive_dual_static_lb(active, rules_tree_id, dual_w, dual_lam, tau, n_trees):
    """Additive (soft) dual LB from precomputed static per-rule weights.

    ``dual_w`` is (n_pool, n_rules) with w = alpha.req_root - lam*v (finite on
    every rule — all leaves participate, unlike the quorum dual's inf-masked
    non-target rules).  Per entry: per-tree min over active rules, then a plain
    sum over ALL trees plus lam*tau.  Shrinking the active set only raises each
    per-tree min, so evaluation at any node is admissible.  A tree with no
    active rule means the box is infeasible -> inf.
    """
    n_pool = dual_w.shape[0]
    best = 0.0
    min_cost = np.empty(n_trees, dtype=np.float64)
    for k in range(n_pool):
        for t in range(n_trees):
            min_cost[t] = np.inf
        for i in range(active.size):
            rid = active[i]
            c = dual_w[k, rid]
            tid = rules_tree_id[rid]
            if c < min_cost[tid]:
                min_cost[tid] = c
        s = dual_lam[k] * tau
        feasible = True
        for t in range(n_trees):
            if not np.isfinite(min_cost[t]):
                feasible = False
                break
            s += min_cost[t]
        if not feasible:
            return np.inf
        if s > best:
            best = s
    return best


@njit(cache=True)
def additive_dual_local_lb(active, rules_lo_mat, rules_hi_mat, rules_tree_id,
                           target_proba, box_lo, box_hi, factual, scale,
                           alpha_pos, alpha_neg, dual_lam, tau, n_trees):
    """Box-clipped additive (soft) dual LB.

    Same clipping as `dual_local_lb` but every active leaf participates with
    price alpha_t.req_clipped - lam*v_l, reduced by per-tree min and a plain
    sum over all trees plus lam*tau.  Clipping only inflates req, so this
    dominates the static entry pointwise (still stacked via max over the pool).

    The box clip is pool-independent and its reach deltas are sparse (a
    feature contributes only when the clipped rule interval sits strictly away
    from the factual), so the clip runs once per rule and the per-pool-entry
    loop touches only the nonzero deltas.  Bit-identical to the dense form:
    deltas are stored unscaled, the triple product keeps the original
    ``alpha * scale * delta`` association, and features are visited in the
    same increasing order (skipped terms were exact-zero additions).
    """
    n_pool = alpha_pos.shape[0]
    n_features = factual.size
    best = 0.0
    min_cost = np.empty((n_pool, n_trees), dtype=np.float64)
    for k in range(n_pool):
        for t in range(n_trees):
            min_cost[k, t] = np.inf
    d_feat = np.empty(n_features, dtype=np.int64)
    d_val = np.empty(n_features, dtype=np.float64)
    d_pos = np.empty(n_features, dtype=np.uint8)
    for i in range(active.size):
        rid = active[i]
        tid = rules_tree_id[rid]
        compatible = True
        nnz = 0
        for f in range(n_features):
            lo = rules_lo_mat[rid, f]
            if box_lo[f] > lo:
                lo = box_lo[f]
            hi = rules_hi_mat[rid, f]
            if box_hi[f] < hi:
                hi = box_hi[f]
            if not (lo < hi):
                compatible = False
                break
            if lo > factual[f]:
                d_feat[nnz] = f
                d_val[nnz] = lo - factual[f]
                d_pos[nnz] = 1
                nnz += 1
            elif factual[f] > hi:
                d_feat[nnz] = f
                d_val[nnz] = factual[f] - hi
                d_pos[nnz] = 0
                nnz += 1
        if not compatible:
            continue
        for k in range(n_pool):
            c = -dual_lam[k] * target_proba[rid]
            for j in range(nnz):
                f = d_feat[j]
                if d_pos[j]:
                    c += alpha_pos[k, tid, f] * scale[f] * d_val[j]
                else:
                    c += alpha_neg[k, tid, f] * scale[f] * d_val[j]
            if c < min_cost[k, tid]:
                min_cost[k, tid] = c
    for k in range(n_pool):
        s = dual_lam[k] * tau
        feasible = True
        for t in range(n_trees):
            if not np.isfinite(min_cost[k, t]):
                feasible = False
                break
            s += min_cost[k, t]
        if not feasible:
            return np.inf
        if s > best:
            best = s
    return best


@njit(cache=True)
def greedy_round_soft(active, rules_lo_mat, rules_hi_mat, rules_tree_id,
                      target_proba, box_lo, box_hi, factual, scale, n_trees,
                      tau, strict_gt):
    """Greedy probability-repair rounding for soft voting.

    The quorum rounding (`greedy_round_l1`) intersects one TARGET leaf per
    quorum tree — a hard-vote construction with no soft analog: under soft
    voting every tree's leaf contributes its probability, so feasibility is a
    property of the whole point, not of a picked subset.  This repairs the
    probability sum instead:

      1. x = min-cost point of the running box; score(x) = sum_t v(leaf_t(x))
         (exact — every leaf containing x is active, since leaves overlapping
         the box are active and x lies in the box).
      2. If score beats tau (strict for class 1), x is a real counterfactual.
      3. Otherwise commit the box-compatible leaf with the best probability
         gain per unit of added projected cost (intersect it into the running
         box) and repeat.  A committed tree is fixed forever — its other
         leaves become box-incompatible — so the loop takes <= n_trees rounds.

    Early exit: if even the per-tree max compatible probabilities cannot reach
    tau, no completion exists.  Returns (cost, x); (inf, factual) on failure.

    The candidate and reachability scans share one clip pass over a live-rule
    list that sheds box-incompatible rules as the running box shrinks —
    bit-identical to two full passes: incompatibility is monotone under
    intersection, dropped rules contributed nothing to either scan, and the
    relative rule order (which settles score/ratio ties) is preserved.  The
    containment pass must stay over the FULL active set: it tests x against
    the unclipped rule bounds, so a rule whose box clip is empty can still
    contain a boundary point of the box.
    """
    n_features = factual.size
    run_lo = box_lo.copy()
    run_hi = box_hi.copy()
    v_cur = np.empty(n_trees, dtype=np.float64)
    v_max = np.empty(n_trees, dtype=np.float64)
    live = active.copy()
    n_live = live.size

    for _round in range(n_trees + 1):
        x, cost = refine_l1(run_lo, run_hi, factual, scale)
        # Exact score at x: the unique containing leaf per tree (lo exclusive,
        # hi inclusive — the box convention shared with point evaluation).
        for t in range(n_trees):
            v_cur[t] = -1.0
        score = 0.0
        for i in range(active.size):
            rid = active[i]
            tid = rules_tree_id[rid]
            if v_cur[tid] >= 0.0:
                continue
            contains = True
            for f in range(n_features):
                if not (x[f] > rules_lo_mat[rid, f] and x[f] <= rules_hi_mat[rid, f]):
                    contains = False
                    break
            if contains:
                v_cur[tid] = target_proba[rid]
                score += target_proba[rid]
        if strict_gt:
            if score > tau:
                return cost, x
        else:
            if score >= tau:
                return cost, x

        # Fused candidate + reachability scan over the live rules: one clip
        # gives the projected cost (candidate ratio), the per-tree max
        # compatible v (reachability), and this round's survivors.
        best_ratio = -1.0
        best_rid = -1
        for t in range(n_trees):
            v_max[t] = -1.0
        n_keep = 0
        for i in range(n_live):
            rid = live[i]
            tid = rules_tree_id[rid]
            compatible = True
            c = 0.0
            for f in range(n_features):
                lo = rules_lo_mat[rid, f]
                if run_lo[f] > lo:
                    lo = run_lo[f]
                hi = rules_hi_mat[rid, f]
                if run_hi[f] < hi:
                    hi = run_hi[f]
                if not (lo < hi):
                    compatible = False
                    break
                if lo > factual[f]:
                    c += scale[f] * (lo - factual[f])
                elif factual[f] > hi:
                    c += scale[f] * (factual[f] - hi)
            if not compatible:
                continue
            live[n_keep] = rid
            n_keep += 1
            gain = target_proba[rid] - v_cur[tid]
            if gain > 0.0:
                added = c - cost
                if added < 1e-12:
                    added = 1e-12
                ratio = gain / added
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_rid = rid
            if target_proba[rid] > v_max[tid]:
                v_max[tid] = target_proba[rid]
        n_live = n_keep
        ub = 0.0
        for t in range(n_trees):
            if v_max[t] < 0.0:
                return np.inf, factual.copy()  # a tree lost all leaves: dead box
            ub += v_max[t]
        if strict_gt:
            if ub <= tau:
                return np.inf, factual.copy()
        else:
            if ub < tau:
                return np.inf, factual.copy()
        if best_rid < 0:
            return np.inf, factual.copy()
        # Commit: intersect the winning leaf into the running box.  Its tree is
        # fixed from now on (sibling leaves are disjoint from the new box).
        for f in range(n_features):
            lo = rules_lo_mat[best_rid, f]
            if lo > run_lo[f]:
                run_lo[f] = lo
            hi = rules_hi_mat[best_rid, f]
            if hi < run_hi[f]:
                run_hi[f] = hi
    return np.inf, factual.copy()
