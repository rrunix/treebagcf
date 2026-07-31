"""Numba-backed RF / weighted-L1 certified counterfactual search.

The method, minimal: A* best-first over axis-aligned boxes, with lower bounds
stacked as  max(geometric per-feature, plateau node-local per-tree,
cost-splitting Lagrangian dual)  and a dual-guided greedy rounding primal.

The dual engages only when the frontier minimum stalls (briskly-certifying
queries never pay for it), is re-optimized at popped plateau nodes on a stride,
and feeds its Polyak step ONLY search-derived incumbents (never rounded ones —
rounded incumbents shrink the ascent target and weaken proofs).
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import numpy as np

from .dataset_info import DatasetInfo
from .dual_lb import AdditiveDualPool, DualCostSplitPool
from .parser import ParsedRF
from .result import ExtractionResult
from .numba.kernels import (
    additive_dual_local_lb,
    additive_dual_static_lb,
    can_stop_rf,
    can_stop_soft,
    choose_split_baseline,
    dual_guided_round_l1,
    dual_local_lb,
    dual_static_lb,
    foil_rule_costs_l1,
    greedy_round_l1,
    greedy_round_soft,
    lb_feature_vec,
    make_child,
    node_local_soft_tree_lb,
    node_local_tree_lb as node_local_tree_lb_kernel,
    polish_l1,
    refine_l1,
    root_active,
    soft_rule_costs_l1,
    static_soft_tree_lb,
    static_tree_lb,
)


@dataclass(frozen=True)
class CompiledRF:
    """Contiguous arrays consumed by the compiled kernels."""

    rules_lo_mat: np.ndarray
    rules_hi_mat: np.ndarray
    rules_tree_id: np.ndarray
    rules_class: np.ndarray
    rules_proba1: np.ndarray
    rules_lo_bin_mat: np.ndarray
    rules_hi_bin_mat: np.ndarray
    threshold_offsets: np.ndarray
    threshold_values: np.ndarray
    n_trees: int
    n_features: int


@dataclass
class _Node:
    """One search box and its three stacked lower bounds.

    ``pool_ver`` records which dual-pool version this node was last evaluated
    against, so a grown pool triggers a cheap lazy re-check at pop (see engine).
    ``slot`` is the node's row in the ``_FrontierPool`` while it lives there;
    the object itself is transient (built at push time, rehydrated at pop time).
    """

    box_lo: np.ndarray
    box_hi: np.ndarray
    active: np.ndarray            # indices of rules still overlapping the box
    lb_per_feature: np.ndarray    # geometric LB per feature (summed once below)
    lb_per_tree: float            # per-tree order-statistic LB (static or local)
    lb_dual: float = 0.0          # cost-splitting dual LB (max over pool)
    pool_ver: int = 0
    slot: int = -1                # _FrontierPool row (-1 = not pooled)
    lb_geom: float = 0.0          # sum(lb_per_feature), cached: the array is
                                  # never mutated after construction (only
                                  # lb_per_tree/lb_dual are), and .lb is read
                                  # ~8x per iteration

    def __post_init__(self) -> None:
        self.lb_geom = float(self.lb_per_feature.sum())

    @property
    def lb(self) -> float:
        # The node key = the strongest of the three admissible bounds.
        return max(self.lb_geom, self.lb_per_tree, self.lb_dual)


class _FrontierPool:
    """Pooled best-first frontier: SoA node rows + a heap of (lb, counter, slot).

    A pool row costs 16·n_features bytes of box plus three scalars, instead of
    the ~1 KB of Python object/heap-tuple overhead of the previous
    dataclass-in-heap frontier.  Active-rule arrays — the term that dominates
    memory on big forests (8·|active| bytes per node, frontiers in the
    millions) — are not owned by the rows: they live in the parking lot below,
    which is the engine's memory knob.  The active set is a pure function of
    the box (``make_child``'s incremental one-feature filter applies exactly
    the strict inequalities of ``root_active``'s full rescan), so an unparked
    node recomputes it at pop time, bit-identical (same ascending int64 ids).
    ``lb_per_feature`` is likewise rehydrated with ``lb_feature_vec``.

    A "parking lot" keeps the active arrays of pushed nodes so pops can skip
    the recompute: at push time the array is already in hand (parking is a
    reference, not a copy), and each entry is consumed at most once (deleted on
    pop-hit).  It is keyed by the unique push counter — never the slot, which is
    recycled and would alias.  With ``cache_elems=None`` (the default) parking
    is unbounded: every pop hits and the engine's speed matches the pre-pool
    implementation exactly.  A finite budget (in total int64 elements — an
    entry-count cap would not bound memory, near-root arrays are huge) enables
    the memory-lean mode: FIFO eviction, evicted nodes recompute on pop.
    Measured on a 300-tree/51k-rule forest: budget 4M elems cut peak RSS
    1458→545 MB on a plateau hard row at ~1.9× wall time (recompute-bound);
    easy queries stay flat (children pop right after their push and hit).
    Hit or miss cannot change the search trajectory, only timing.
    """

    def __init__(
        self,
        crf: CompiledRF,
        factual: np.ndarray,
        scale: np.ndarray,
        cache_elems: int | None,
        cap: int = 4096,
    ) -> None:
        self._crf = crf
        self._factual = factual
        self._scale = scale
        n_features = crf.n_features
        self._box_lo = np.empty((cap, n_features), dtype=np.float64)
        self._box_hi = np.empty((cap, n_features), dtype=np.float64)
        self._lb_tree = np.empty(cap, dtype=np.float64)
        self._lb_dual = np.empty(cap, dtype=np.float64)
        self._pool_ver = np.empty(cap, dtype=np.int64)
        self._n_rows = 0
        self._free: list[int] = []
        self._heap: list[tuple[float, int, int]] = []
        self._counter = itertools.count()
        self._parked: dict[int, np.ndarray] = {}
        self._parked_elems = 0
        self._cache_elems = math.inf if cache_elems is None else max(0, int(cache_elems))

    def __len__(self) -> int:
        return len(self._heap)

    def min_lb(self) -> float:
        return self._heap[0][0] if self._heap else math.inf

    def push(self, node: _Node) -> None:
        """Store the node as a pool row and enqueue it; park its active array."""
        node.slot = self._alloc()
        self._box_lo[node.slot] = node.box_lo
        self._box_hi[node.slot] = node.box_hi
        self._write_scalars(node)
        key = next(self._counter)
        self._park(key, node.active)
        heapq.heappush(self._heap, (node.lb, key, node.slot))

    def pop(self) -> tuple[float, _Node]:
        """Pop the min-lb entry and rehydrate its node.

        The boxes are views into the pool rows — valid until the slot is
        released AND reused, i.e. for the whole expansion of this node.
        """
        lb, key, slot = heapq.heappop(self._heap)
        box_lo = self._box_lo[slot]
        box_hi = self._box_hi[slot]
        active = self._unpark(key)
        if active is None:
            active = root_active(
                self._crf.rules_lo_mat, self._crf.rules_hi_mat, box_lo, box_hi
            )
        node = _Node(
            box_lo=box_lo,
            box_hi=box_hi,
            active=active,
            lb_per_feature=lb_feature_vec(box_lo, box_hi, self._factual, self._scale),
            lb_per_tree=float(self._lb_tree[slot]),
            lb_dual=float(self._lb_dual[slot]),
            pool_ver=int(self._pool_ver[slot]),
            slot=slot,
        )
        return lb, node

    def requeue_or_drop(self, node: _Node, best_cost: float) -> None:
        """Re-enqueue a popped node whose dual LB was lifted at pop time; drop
        it when the lifted bound already meets the incumbent (never expandable)."""
        if node.lb < best_cost:
            self._write_scalars(node)
            key = next(self._counter)
            self._park(key, node.active)
            heapq.heappush(self._heap, (node.lb, key, node.slot))
        else:
            self.release(node)

    def release(self, node: _Node) -> None:
        """Return a popped node's row to the free list (node leaves the search)."""
        self._free.append(node.slot)
        node.slot = -1

    def _write_scalars(self, node: _Node) -> None:
        self._lb_tree[node.slot] = node.lb_per_tree
        self._lb_dual[node.slot] = node.lb_dual
        self._pool_ver[node.slot] = node.pool_ver

    def _alloc(self) -> int:
        if self._free:
            return self._free.pop()
        if self._n_rows == self._box_lo.shape[0]:
            self._grow()
        slot = self._n_rows
        self._n_rows += 1
        return slot

    def _grow(self) -> None:
        # Doubling reallocation.  Outstanding box views from pop() keep the old
        # arrays alive and are read-only from here on, so they stay valid.
        cap = 2 * self._box_lo.shape[0]
        for name in ("_box_lo", "_box_hi", "_lb_tree", "_lb_dual", "_pool_ver"):
            old = getattr(self, name)
            new = np.empty((cap,) + old.shape[1:], dtype=old.dtype)
            new[: self._n_rows] = old[: self._n_rows]
            setattr(self, name, new)

    def _park(self, key: int, active: np.ndarray) -> None:
        if active.size > self._cache_elems:
            return
        self._parked[key] = active
        self._parked_elems += active.size
        while self._parked_elems > self._cache_elems:
            oldest = next(iter(self._parked))
            self._parked_elems -= self._parked.pop(oldest).size

    def _unpark(self, key: int) -> np.ndarray | None:
        active = self._parked.pop(key, None)
        if active is not None:
            self._parked_elems -= active.size
        return active


def compile_rf(parsed_rf: ParsedRF) -> CompiledRF:
    """Pack a ParsedRF into contiguous arrays + a flattened threshold grid."""
    offsets = np.zeros(parsed_rf.n_features + 1, dtype=np.int64)
    values_parts = []
    cursor = 0
    for f, ts in enumerate(parsed_rf.feature_thresholds):
        ts = np.asarray(ts, dtype=np.float64)
        values_parts.append(ts)
        offsets[f] = cursor
        cursor += ts.size
    offsets[parsed_rf.n_features] = cursor
    values = (
        np.concatenate(values_parts).astype(np.float64, copy=False)
        if values_parts
        else np.empty(0, dtype=np.float64)
    )
    return CompiledRF(
        rules_lo_mat=np.ascontiguousarray(parsed_rf.rules_lo_mat, dtype=np.float64),
        rules_hi_mat=np.ascontiguousarray(parsed_rf.rules_hi_mat, dtype=np.float64),
        rules_tree_id=np.ascontiguousarray(parsed_rf.rules_tree_id, dtype=np.int64),
        rules_class=np.ascontiguousarray(parsed_rf.rules_class, dtype=np.int64),
        rules_proba1=np.ascontiguousarray(
            parsed_rf.rules_proba1 if parsed_rf.rules_proba1 is not None
            else np.zeros(parsed_rf.rules_class.size), dtype=np.float64
        ),
        rules_lo_bin_mat=np.ascontiguousarray(parsed_rf.rules_lo_bin_mat, dtype=np.int64),
        rules_hi_bin_mat=np.ascontiguousarray(parsed_rf.rules_hi_bin_mat, dtype=np.int64),
        threshold_offsets=offsets,
        threshold_values=np.ascontiguousarray(values, dtype=np.float64),
        n_trees=parsed_rf.n_trees,
        n_features=parsed_rf.n_features,
    )


def _dataset_box(dataset_info: DatasetInfo) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([f.lo - 1e-12 for f in dataset_info.features], dtype=np.float64)
    hi = np.array([f.hi for f in dataset_info.features], dtype=np.float64)
    return lo, hi


def extract_counterfactual(
    parsed_rf: ParsedRF,
    dataset_info: DatasetInfo,
    factual: np.ndarray,
    target_class: int,
    scale: np.ndarray,
    *,
    voting: Literal["hard", "soft"] = "hard",
    threshold: float = 0.5,
    max_iters: int = 1_000_000,
    time_limit_s: float | None = None,
    initial_ub: tuple[np.ndarray, float] | None = None,
    node_local_tree_lb: Literal["off", "always", "plateau"] = "plateau",
    node_local_tree_lb_eps: float = 1e-12,
    dual_lb: Literal["off", "local"] = "local",
    dual_lb_root_iters: int = 1000,
    dual_lb_stall_window: int = 500,
    dual_lb_start_iter: int | None = None,
    dual_lb_max_reopts: int = 64,
    dual_lb_reopt_iters: int = 30,
    dual_lb_reopt_stride: int = 4,
    dual_lb_pool_cap: int = 4,
    dual_lb_escalate: bool | None = None,
    dual_lb_escalate_gap: float = 0.55,
    dual_lb_escalate_lift_frac: float = 0.5,
    dual_lb_strong_reopt_iters: int = 200,
    dual_lb_strong_pool_cap: int = 16,
    dual_lb_strong_max_reopts: int = 1000,
    dual_round: bool = True,
    dual_round_polish: bool = True,
    dual_round_guided: bool = True,
    dual_warm_entries: list | None = None,
    dual_warm_seed_k: int = 3,
    dual_pool_out: list | None = None,
    stall_dump: list | None = None,
    stall_dump_max: int = 32,
    compiled_rf: CompiledRF | None = None,
    active_cache_elems: int | None = None,
) -> ExtractionResult:
    """Run the certified counterfactual search; returns the (proven) optimum.

    ``scale`` is the per-feature weighted-L1 scale (see ``calf.cost.l1_scale``).
    Defaults are the current research baseline (plateau tree-LB, local dual,
    stride-4 reopt, dual rounding on).

    ``time_limit_s`` caps wall time (clock starts on entry, checked once per
    expansion): on expiry the search stops and returns the best incumbent found
    so far with its certified ``optimality_gap`` to the frontier minimum — the
    same anytime contract as ``max_iters`` exhaustion.  ``proven_optimal``
    results returned before the cap are unaffected.

    ``voting='soft'`` uses probability-averaged aggregation (binary forests
    only, matching ``rf.predict``'s argmax over averaged probabilities;
    ``threshold`` is ignored).  The per-tree order-statistic LB switches to its
    soft analogue (the min cost at which the trees' best reachable leaf
    probabilities can sum past ``0.5*n_trees``; static + plateau node-local
    forms), and the dual switches to its additive form
    (:class:`calf.dual_lb.AdditiveDualPool`): one multiplier for the
    probability-sum constraint plus the same cost-splitting shares, engaged
    through the identical stall-gate / lazy re-eval / stride-reopt machinery,
    and ``dual_round`` switches to probability-repair rounding
    (``greedy_round_soft``).

    ``dual_lb_escalate`` (default ``None`` = on for soft, off for hard) is the
    adaptive dual-strength policy: every query starts at the cheap reopt tier
    (``dual_lb_reopt_iters`` / ``dual_lb_pool_cap`` / ``dual_lb_max_reopts``)
    and escalates — once, irreversibly — to the strong tier
    (``dual_lb_strong_*``) at the moment the cheap reopt budget exhausts, iff
    either escalation signal fires:

    - *gap*: the frontier minimum is still below ``dual_lb_escalate_gap *
      best_cost``.  Once the cheap budget is spent the dual freezes, so the
      question at exactly that point is "will the tree LB close the remaining
      gap alone?" — measured directly by lb/incumbent.  Probed 2026-07-10:
      rows the tree LB is about to prove sit at >= 0.62 (breast_cancer, which
      must stay cheap — a big pool makes EVERY pop's local eval and every
      child's static eval proportionally dearer, ~3x wall overhead, the
      2026-07 pilot's 7->5 breast_cancer regression); ionosphere sits at
      0.18-0.52 and there the strong tier is what certifies.  No incumbent at
      exhaustion (lb < inf) always escalates.
    - *lift*: at least ``dual_lb_escalate_lift_frac`` of the cheap reopts
      returned a node bound more than 1.25x the popped node's key — the dual
      is visibly the productive bound even though the frontier may not show
      it yet.  This catches the digits regime (lift fraction 0.73-0.86 vs
      <= 0.02 on breast_cancer/ionosphere), including rows whose gap ratio
      ties breast_cancer's.

    ``active_cache_elems`` bounds the frontier's active-array parking lot (in
    int64 elements) — the only per-search memory term that scales with forest
    size.  ``None`` (default) retains every active set: full speed, memory as
    the pre-pool engine.  A finite budget (e.g. ``4_000_000`` ≈ 32 MB) trades
    pop-time recomputes for a hard memory bound (~2.7× less peak RSS on big
    forests, up to ~2× wall time on plateau-hard rows) — never correctness or
    search order.  See :class:`_FrontierPool`.

    ``dual_round_guided`` (default on; hard voting only) runs the rounding
    repair a second time with the tree order priced by the newest dual pool
    entry and keeps the cheaper quorum; off restores the pure cost-greedy
    rounding for ablation.

    ``dual_round_polish`` (default on) post-processes every primal candidate —
    the initial UB and each successful rounding — with a per-feature pull-back
    polish (``polish_l1``): coordinates are pulled toward the factual across
    split thresholds as long as the exact forest score keeps beating the
    target.  Purely primal: it can only lower the incumbent, never touches the
    bounds, and (like rounding) never feeds the dual's Polyak step.

    ``dual_warm_entries`` transfers dual share matrices optimized at OTHER
    queries of the same forest (see :class:`calf.alpha_lib.AlphaLibrary`):
    at engagement the strongest ``dual_warm_seed_k`` entries for this query's
    root reqs seed the pool, and the root ascent warm-starts from the best of
    them instead of uniform shares.  Admissible for any factual by
    construction.  ``dual_pool_out`` (a caller-supplied list) receives the
    engaged pool object so callers can harvest its entries after the run.

    ``stall_dump`` (a caller-supplied list) collects up to ``stall_dump_max``
    snapshots of nodes whose reopt failed to lift them — the raw material for
    offline relaxation diagnosis (see ``research/stall_diag``); it never
    alters the search.

    Everything is float32.  sklearn stores split thresholds in float32 and casts
    every query to float32 before traversing the trees, so the forest lives on
    the float32 grid.  We match it exactly: the query is snapped to float32, the
    parsed thresholds already are float32-valued, and the returned point sits on
    the float32 grid — so ``rf.predict(x_cf)`` is guaranteed to agree with the
    certified vote (no float64/float32 boundary drift).  Only the LB / cost
    arithmetic runs in float64, which is exact over float32-valued data and keeps
    the admissibility of the bound (float32 accumulation could round it up and
    void the certificate).
    """
    deadline = None if time_limit_s is None else time.perf_counter() + float(time_limit_s)
    node_local_mode = node_local_tree_lb
    if node_local_mode not in ("off", "always", "plateau"):
        raise ValueError("node_local_tree_lb must be 'off', 'always', or 'plateau'")
    if dual_lb not in ("off", "local"):
        raise ValueError("dual_lb must be 'off' or 'local'")
    if voting not in ("hard", "soft"):
        raise ValueError("voting must be 'hard' or 'soft'")
    soft = voting == "soft"
    if soft:
        if parsed_rf.n_classes != 2:
            raise NotImplementedError("soft voting currently supports binary forests only")
        if target_class not in (0, 1):
            raise ValueError("soft voting target_class must be 0 or 1")
        # Stall-gate fallback for soft: on many soft rows the frontier LB does not
        # plateau, it *creeps* upward in float-dust increments, so the
        # >1e-12-per-pop stall test never accumulates a window and the dual never
        # engages — even as the row runs 0.5M+ iters unproven (found 2026-07-09:
        # ionosphere hard rows sat at cost 5-16 because the dual's rounding, which
        # fires only once engaged, never ran; forcing engagement cut them to
        # 0.9-2.3).  An iteration fallback guarantees the dual (and its rounding)
        # engage on any row still running past the budget; easy rows certify well
        # under it, so they never pay.  Hard voting keeps the pure stall gate
        # (its long-span rows genuinely plateau), unless the caller opts in.
        if dual_lb_start_iter is None:
            dual_lb_start_iter = 50_000
    # Adaptive dual strength: soft rows split into tree-LB-driven (escalation
    # would only add eval overhead) and dual-driven (the strong tier is what
    # certifies); hard voting keeps the validated fixed-tier baseline.
    if dual_lb_escalate is None:
        dual_lb_escalate = soft

    # Snap the query to the float32 grid sklearn will evaluate it on, so the
    # search reasons about exactly the points sklearn sees (the parsed thresholds
    # are already float32-valued).
    factual = np.ascontiguousarray(np.asarray(factual, dtype=np.float64).astype(np.float32), dtype=np.float64)
    if factual.shape != (parsed_rf.n_features,):
        raise ValueError(f"factual shape {factual.shape} != (n_features={parsed_rf.n_features},)")
    scale = np.ascontiguousarray(scale, dtype=np.float64)
    need = int(math.ceil(threshold * parsed_rf.n_trees))

    crf = compiled_rf if compiled_rf is not None else compile_rf(parsed_rf)
    box_lo, box_hi = _dataset_box(dataset_info)
    active = root_active(crf.rules_lo_mat, crf.rules_hi_mat, box_lo, box_hi)
    lb_per_feature = lb_feature_vec(box_lo, box_hi, factual, scale)
    if soft:
        p1 = crf.rules_proba1
        target_proba = np.ascontiguousarray(
            p1 if target_class == 1 else 1.0 - p1, dtype=np.float64
        )
        soft_thresh = 0.5 * crf.n_trees
        soft_strict = target_class == 1  # class 1 needs a strict prob majority
        # Soft per-tree order-statistic LB: the min cost at which each tree's
        # best reachable leaf-probability can sum past soft_thresh.  All leaves
        # carry mass (no target-class filter), so every rule gets a foil cost.
        soft_costs = soft_rule_costs_l1(
            crf.rules_lo_mat, crf.rules_hi_mat, factual, scale
        )
        foil_costs = None
    else:
        soft_costs = None
        foil_costs = foil_rule_costs_l1(
            crf.rules_lo_mat, crf.rules_hi_mat, crf.rules_class, factual, scale, target_class
        )

    # Primal feasibility oracle for the polish: hard voting scores the 0/1
    # target indicator against tau = need; soft scores the leaf probabilities
    # against tau = 0.5 * n_trees (strict for class 1).
    if soft:
        polish_values = target_proba
        polish_tau = soft_thresh
        polish_strict = soft_strict
    else:
        polish_values = np.ascontiguousarray(
            (crf.rules_class == target_class).astype(np.float64)
        )
        polish_tau = float(need)
        polish_strict = False

    def polish(x: np.ndarray) -> tuple[float, np.ndarray]:
        return polish_l1(
            x, factual, scale, crf.rules_lo_mat, crf.rules_hi_mat,
            crf.rules_tree_id, polish_values, crf.n_trees, polish_tau,
            polish_strict, crf.threshold_offsets, crf.threshold_values, 24, 2,
        )

    best_x: np.ndarray | None = None
    best_cost = math.inf
    if initial_ub is not None:
        ub_x, ub_cost = initial_ub
        best_x = np.asarray(ub_x, dtype=np.float64).copy()
        best_cost = float(ub_cost)
    # Incumbent fed to the dual's Polyak step sizing.  Rounded incumbents are
    # deliberately excluded: a sharply better incumbent shrinks the ascent
    # target and weakens short reopts (found 2026-07-02: equal-iters proofs
    # dropped 9/24 -> 1/24 when rounding fed the step).  Pruning/certification
    # still use the true best_cost below.
    search_incumbent = best_cost
    if dual_round_polish and best_x is not None:
        # Polish the initial UB once.  polish_l1 returns inf for an infeasible
        # seed (it never "improves" one), so this can only lower best_cost with
        # a verified-feasible point; search_incumbent keeps the unpolished cost
        # (the same discipline as rounding above).
        pc, px = polish(best_x)
        if pc < best_cost:
            best_x, best_cost = px, float(pc)

    if soft:
        root_tree_lb = static_soft_tree_lb(
            active, crf.rules_tree_id, target_proba, soft_costs,
            crf.n_trees, soft_thresh, soft_strict,
        )
    else:
        root_tree_lb = static_tree_lb(
            active, crf.rules_tree_id, crf.rules_class, foil_costs,
            crf.n_trees, need, target_class, best_cost,
        )

    dual_enabled = dual_lb != "off"
    use_dual = False
    dual_w = np.empty((0, 0), dtype=np.float64)
    dual_lam = np.empty(0, dtype=np.float64)
    dual_alpha_pos = np.empty((0, 0, 0), dtype=np.float64)
    dual_alpha_neg = np.empty((0, 0, 0), dtype=np.float64)
    pool: DualCostSplitPool | AdditiveDualPool | None = None
    pool_version = 0

    def refresh_dual_arrays() -> None:
        # Repack the pool's Python-side share matrices into contiguous arrays the
        # njit kernels consume, and bump the version so pushed nodes know the pool
        # changed (triggering their lazy re-check at pop).  dual_w holds the
        # precomputed root-box per-rule weights for the cheap static eval (hard:
        # alpha.req, inf on non-target rules; soft: alpha.req - lam*v, finite
        # everywhere); dual_alpha_pos/neg (+ dual_lam for soft) feed local eval.
        nonlocal dual_w, dual_lam, dual_alpha_pos, dual_alpha_neg, pool_version
        pool_version += 1
        if soft:
            dual_w = np.ascontiguousarray(np.stack(pool.static_w), dtype=np.float64)
            dual_lam = np.array([lam for _, _, lam in pool.entries], dtype=np.float64)
            dual_alpha_pos = np.ascontiguousarray(
                np.stack([ap for ap, _, _ in pool.entries]), dtype=np.float64
            )
            dual_alpha_neg = np.ascontiguousarray(
                np.stack([an for _, an, _ in pool.entries]), dtype=np.float64
            )
        else:
            n_rules = crf.rules_class.size
            dual_w = np.full((len(pool), n_rules), np.inf, dtype=np.float64)
            for k, w in enumerate(pool.static_w):
                dual_w[k, pool.target_rule_ids] = w
            dual_alpha_pos = np.ascontiguousarray(
                np.stack([ap for ap, _ in pool.alphas]), dtype=np.float64
            )
            dual_alpha_neg = np.ascontiguousarray(
                np.stack([an for _, an in pool.alphas]), dtype=np.float64
            )

    def engage_dual() -> None:
        """Turn the dual on: build the pool from a full root-box optimization.

        Fired once, when the frontier stalls.  Seeds the pool with a single α
        from a long ascent at the root; later reopts grow it and lifts cash it in.
        Soft voting uses the additive pool (one lam multiplier for the
        probability-sum constraint); hard voting the quorum pool.
        """
        nonlocal pool, use_dual
        if soft:
            pool = AdditiveDualPool(parsed_rf, factual, scale, target_proba, soft_thresh)
        else:
            pool = DualCostSplitPool(parsed_rf, factual, scale, target_class, need)
        if dual_pool_out is not None:
            dual_pool_out.append(pool)
        root_shim = SimpleNamespace(
            active_rules=active, box=SimpleNamespace(lo=box_lo, hi=box_hi)
        )
        if dual_warm_entries:
            # Transferred entries: seed the strongest few for THIS query's root
            # reqs; optimize_root then warm-starts from the best of them.
            pool.seed(root_shim, dual_warm_entries, top_k=dual_warm_seed_k)
        pool.optimize_root(root_shim, incumbent=search_incumbent, max_iters=dual_lb_root_iters)
        if len(pool):
            use_dual = True
            refresh_dual_arrays()

    def dual_static_eval(node_active: np.ndarray) -> float:
        """Pool static LB for an active set (admissible at any node)."""
        if soft:
            return additive_dual_static_lb(
                node_active, crf.rules_tree_id, dual_w, dual_lam,
                soft_thresh, crf.n_trees,
            )
        return dual_static_lb(node_active, crf.rules_tree_id, dual_w, crf.n_trees, need)

    def dual_local_eval(node_active: np.ndarray, nlo: np.ndarray, nhi: np.ndarray) -> float:
        """Pool box-clipped LB (dominates the static entry pointwise)."""
        if soft:
            return additive_dual_local_lb(
                node_active, crf.rules_lo_mat, crf.rules_hi_mat,
                crf.rules_tree_id, target_proba, nlo, nhi, factual, scale,
                dual_alpha_pos, dual_alpha_neg, dual_lam, soft_thresh, crf.n_trees,
            )
        return dual_local_lb(
            node_active, crf.rules_lo_mat, crf.rules_hi_mat,
            crf.rules_tree_id, crf.rules_class, nlo, nhi,
            factual, scale, dual_alpha_pos, dual_alpha_neg,
            crf.n_trees, need, target_class,
        )

    root = _Node(box_lo=box_lo, box_hi=box_hi, active=active,
                 lb_per_feature=lb_per_feature, lb_per_tree=root_tree_lb)

    frontier = _FrontierPool(crf, factual, scale, cache_elems=active_cache_elems)
    frontier.push(root)

    iters = 0
    n_reopts = 0
    unlifted_pops = 0
    last_frontier_lb = -math.inf
    stall_pops = 0
    dual_strong = False   # escalated to the strong reopt tier (one-way)
    esc_lifts = 0         # cheap reopts whose bound lifted the node > 1.25x

    while len(frontier) and iters < max_iters:
        if deadline is not None and time.perf_counter() >= deadline:
            break  # anytime exit: the tail below reports incumbent + gap
        iters += 1
        lb, node = frontier.pop()

        if lb >= best_cost:
            return ExtractionResult(
                x=best_x, cost=best_cost if best_x is not None else math.inf,
                optimality_gap=0.0, iters=iters, found=best_x is not None,
                target_class=target_class,
            )


        if soft:
            cls = can_stop_soft(
                node.active, crf.rules_tree_id, target_proba, crf.n_trees,
                soft_thresh, soft_strict, target_class,
            )
        else:
            cls = can_stop_rf(
                node.active, crf.rules_tree_id, crf.rules_class, crf.n_trees, need, target_class
            )
        if cls != -1:
            if cls == target_class:
                x_prime, cost_true = refine_l1(node.box_lo, node.box_hi, factual, scale)
                if cost_true < search_incumbent:
                    search_incumbent = float(cost_true)
                if cost_true < best_cost:
                    best_x = x_prime
                    best_cost = float(cost_true)
            frontier.release(node)
            continue

        # Stall-gated engagement: turn the dual on once the frontier minimum
        # plateaus — exactly the signature the dual bound attacks — OR once the
        # iteration fallback fires (dual_lb_start_iter), which catches the
        # slow-creep rows whose LB never plateaus hard enough to trip the stall
        # window (see the soft note above).
        if dual_enabled and not use_dual:
            if lb > last_frontier_lb + 1e-12:
                last_frontier_lb = lb
                stall_pops = 0
            else:
                stall_pops += 1
            hit_start = dual_lb_start_iter is not None and iters >= dual_lb_start_iter
            if stall_pops >= max(1, dual_lb_stall_window) or hit_start:
                engage_dual()
                stall_pops = 0
                last_frontier_lb = -math.inf

        # Lazy re-evaluation: the pool gained entries after this node was pushed.
        # One static eval lifts the whole stale plateau at pop time.
        if use_dual and node.pool_ver < pool_version:
            node.pool_ver = pool_version
            dval = dual_static_eval(node.active)
            if dval > node.lb_dual:
                node.lb_dual = float(dval)
                if node.lb > lb + 1e-15:
                    frontier.requeue_or_drop(node, best_cost)
                    continue

        # Adaptive strength: at the moment the cheap reopt budget exhausts the
        # dual would freeze for the rest of the run, so decide here, once —
        # escalate iff the frontier minimum says the tree LB won't close the
        # remaining gap alone (no incumbent counts as maximally far), or the
        # cheap reopts were visibly the productive bound (lift fraction).
        if (use_dual and dual_lb_escalate and not dual_strong
                and n_reopts >= dual_lb_max_reopts
                and (lb < dual_lb_escalate_gap * best_cost
                     or esc_lifts >= dual_lb_escalate_lift_frac * n_reopts)):
            dual_strong = True

        # Dual reopt at popped plateau nodes, throttled by stride.
        eff_max_reopts = dual_lb_strong_max_reopts if dual_strong else dual_lb_max_reopts
        if use_dual and n_reopts < eff_max_reopts:
            # Pool-first: a clipped eval of existing entries is cheaper than a
            # fresh subgradient run and often already lifts sibling plateaus.
            dloc = dual_local_eval(node.active, node.box_lo, node.box_hi)
            if dloc > node.lb_dual:
                node.lb_dual = float(dloc)
                if node.lb > lb + 1e-15:
                    frontier.requeue_or_drop(node, best_cost)
                    continue
            unlifted_pops += 1
            if dual_lb_reopt_stride <= 1 or unlifted_pops % dual_lb_reopt_stride == 0:
                if dual_round:
                    # Primal rounding at the reopt cadence.  Hard: grow a
                    # feasible quorum from this node's clipped target leaves.
                    # Soft: probability-repair — commit leaves with the best
                    # probability gain per cost until the sum beats tau.
                    # Either way a success is a real counterfactual.
                    if soft:
                        rc, rx = greedy_round_soft(
                            node.active, crf.rules_lo_mat, crf.rules_hi_mat,
                            crf.rules_tree_id, target_proba, node.box_lo,
                            node.box_hi, factual, scale, crf.n_trees,
                            soft_thresh, soft_strict,
                        )
                    else:
                        rc, rx = greedy_round_l1(
                            node.active, crf.rules_lo_mat, crf.rules_hi_mat,
                            crf.rules_tree_id, crf.rules_class, node.box_lo,
                            node.box_hi, factual, scale, crf.n_trees, need, target_class,
                        )
                        if dual_round_guided:
                            # Lagrangian heuristic: repeat the repair with the
                            # tree order priced by the newest pool entry (the
                            # alphas most recently optimized near this region)
                            # and keep the cheaper of the two quorums.
                            kg = dual_alpha_pos.shape[0] - 1
                            rc2, rx2 = dual_guided_round_l1(
                                node.active, crf.rules_lo_mat, crf.rules_hi_mat,
                                crf.rules_tree_id, crf.rules_class, node.box_lo,
                                node.box_hi, factual, scale,
                                dual_alpha_pos[kg], dual_alpha_neg[kg],
                                crf.n_trees, need, target_class,
                            )
                            if rc2 < rc:
                                rc, rx = rc2, rx2
                    if (
                        dual_round_polish
                        and math.isfinite(rc)
                        and rc < 1.25 * best_cost
                    ):
                        # Polish near-misses too (within 25% of the incumbent):
                        # the pull-back often crosses the incumbent line.
                        pc, px = polish(rx)
                        if pc < rc:
                            rc, rx = pc, px
                    if rc < best_cost:
                        best_x = rx
                        best_cost = float(rc)
                        if lb >= best_cost:
                            return ExtractionResult(
                                x=best_x, cost=best_cost, optimality_gap=0.0,
                                iters=iters, found=True, target_class=target_class,
                            )
                shim = SimpleNamespace(
                    active_rules=node.active,
                    box=SimpleNamespace(lo=node.box_lo, hi=node.box_hi),
                )
                nb = pool.reoptimize_at(
                    shim, incumbent=search_incumbent,
                    max_iters=(dual_lb_strong_reopt_iters if dual_strong
                               else dual_lb_reopt_iters),
                    cap=(dual_lb_strong_pool_cap if dual_strong
                         else dual_lb_pool_cap),
                )
                n_reopts += 1
                # Escalation lift accounting (cheap tier only): did this reopt
                # push the node's bound well past its popped key?
                if (not dual_strong and math.isfinite(nb)
                        and nb > 1.25 * max(lb, 1e-300)):
                    esc_lifts += 1
                refresh_dual_arrays()
                if math.isfinite(nb) and nb > node.lb_dual:
                    node.lb_dual = float(nb)
                    if node.lb > lb + 1e-15:
                        frontier.requeue_or_drop(node, best_cost)
                        continue
                # Reaching here means the reopt failed to lift this node past
                # its popped key — a genuinely stalled node.
                if stall_dump is not None and len(stall_dump) < stall_dump_max:
                    stall_dump.append({
                        "iters": int(iters),
                        "n_reopts": int(n_reopts),
                        "lb": float(lb),
                        "lb_geom": float(node.lb_per_feature.sum()),
                        "lb_tree": float(node.lb_per_tree),
                        "lb_dual": float(node.lb_dual),
                        "best_cost": float(best_cost),
                        "box_lo": np.array(node.box_lo, dtype=np.float64),
                        "box_hi": np.array(node.box_hi, dtype=np.float64),
                        "active": node.active.copy(),
                    })
        feature, thr = choose_split_baseline(
            node.active, node.box_lo, node.box_hi, node.lb_per_feature,
            factual, scale, crf.rules_lo_bin_mat, crf.rules_hi_bin_mat,
            crf.threshold_offsets, crf.threshold_values, node.lb_per_tree, 0.0,
        )
        if feature < 0:
            frontier.release(node)
            continue

        for side in (0, 1):
            child_active, child_lo, child_hi, child_lbf, ok = make_child(
                node.active, node.box_lo, node.box_hi, node.lb_per_feature,
                factual, scale, crf.rules_lo_mat, crf.rules_hi_mat,
                feature, thr, side,
            )
            if ok == 0:
                continue
            if soft:
                child_tree_lb = static_soft_tree_lb(
                    child_active, crf.rules_tree_id, target_proba, soft_costs,
                    crf.n_trees, soft_thresh, soft_strict,
                )
            else:
                child_tree_lb = static_tree_lb(
                    child_active, crf.rules_tree_id, crf.rules_class, foil_costs,
                    crf.n_trees, need, target_class, np.inf,
                )
            child = _Node(child_lo, child_hi, child_active, child_lbf, child_tree_lb)
            if use_dual:
                child.pool_ver = pool_version
                child.lb_dual = float(dual_static_eval(child_active))
            if (
                node_local_mode == "always"
                or (node_local_mode == "plateau" and child.lb <= node.lb + node_local_tree_lb_eps)
            ):
                if soft:
                    child.lb_per_tree = node_local_soft_tree_lb(
                        child.active, crf.rules_lo_mat, crf.rules_hi_mat,
                        crf.rules_tree_id, target_proba, child.box_lo, child.box_hi,
                        factual, scale, crf.n_trees, soft_thresh, soft_strict,
                    )
                else:
                    child.lb_per_tree = node_local_tree_lb_kernel(
                        child.active, crf.rules_lo_mat, crf.rules_hi_mat,
                        crf.rules_tree_id, crf.rules_class, child.box_lo, child.box_hi,
                        factual, scale, crf.n_trees, need, target_class,
                    )
                if use_dual and child.lb <= node.lb + node_local_tree_lb_eps:
                    local_dual = dual_local_eval(child.active, child.box_lo, child.box_hi)
                    if local_dual > child.lb_dual:
                        child.lb_dual = float(local_dual)
            if child.lb < best_cost:
                frontier.push(child)
        frontier.release(node)

    gap = 0.0
    if best_x is None:
        cost_out = math.inf
    else:
        cost_out = best_cost
        if len(frontier):
            gap = max(0.0, best_cost - frontier.min_lb())
    return ExtractionResult(
        x=best_x, cost=cost_out, optimality_gap=gap, iters=iters,
        found=best_x is not None, target_class=target_class,
    )

