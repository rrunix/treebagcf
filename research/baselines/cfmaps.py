"""Counterfactual Maps (Khouna, Ferry & Vidal, arXiv:2602.09128) — reimplemented.

No public code was located for this method (see STATUS.md), so this is a
from-the-paper reimplementation of its self-contained algorithm:

  1. Preprocessing (once per target class): extract the labelled hyperrectangle
     partition equivalent to the forest and keep the regions whose hard vote is
     the target class.
  2. Index those target regions in a **volumetric KD-tree** (median split on the
     widest axis of the region centres).
  3. Query: **branch-and-bound nearest-region** search over the KD-tree.  The
     distance from the factual to a region box is the exact scaled-L1 lower
     bound; projecting the factual onto the nearest target box yields the
     globally optimal counterfactual, with an explicit optimality certificate.

Complexity caveat reproduced honestly: the region count grows *exponentially*
with ensemble depth (the paper's experiments stop at depth 7).  ``max_cells``
guards against blow-up; this is precisely the axis on which our unlimited-depth
method is meant to win.

The partition here is the threshold grid (see :mod:`baselines.grid`): a refinement
of the coarsest forest-equivalent partition.  Refinement changes region *count*,
not the optimum — the nearest-region projection is identical — so the returned
CF and certificate match the brute-force oracle exactly (asserted in tests).
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from . import metrics
from .base import BaselineExplainer, CFResult
from .grid import feature_edges, iter_cells


class _KDNode:
    __slots__ = ("lo", "hi", "left", "right", "region_ids")

    def __init__(self, lo, hi, left=None, right=None, region_ids=None):
        self.lo = lo              # bounding-box lower corner over contained regions
        self.hi = hi              # bounding-box upper corner
        self.left = left
        self.right = right
        self.region_ids = region_ids  # non-None only at leaves


class VolumetricKDTree:
    """KD-tree over region boxes, split on the widest axis of region centres."""

    def __init__(self, region_lo: np.ndarray, region_hi: np.ndarray, leaf_size: int = 8):
        self.region_lo = region_lo
        self.region_hi = region_hi
        self.centres = 0.5 * (region_lo + region_hi)
        self.leaf_size = int(leaf_size)
        ids = np.arange(region_lo.shape[0])
        self.root = self._build(ids)

    def _bbox(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.region_lo[ids].min(axis=0), self.region_hi[ids].max(axis=0)

    def _build(self, ids: np.ndarray) -> _KDNode:
        lo, hi = self._bbox(ids)
        if ids.size <= self.leaf_size:
            return _KDNode(lo, hi, region_ids=ids)
        centres = self.centres[ids]
        spread = centres.max(axis=0) - centres.min(axis=0)
        axis = int(np.argmax(spread))
        if spread[axis] <= 0:  # all centres coincide; stop splitting
            return _KDNode(lo, hi, region_ids=ids)
        order = np.argsort(centres[:, axis], kind="stable")
        ids = ids[order]
        mid = ids.size // 2
        left = self._build(ids[:mid])
        right = self._build(ids[mid:])
        return _KDNode(lo, hi, left=left, right=right)

    def nearest(
        self, factual: np.ndarray, scale: np.ndarray
    ) -> tuple[float, np.ndarray | None, int]:
        """Branch-and-bound nearest region.

        Returns ``(cost, x_cf, n_nodes_visited)``.  ``cost`` is the exact scaled-L1
        distance to the closest region and ``x_cf`` the projection realising it.
        """
        best_cost = math.inf
        best_x: np.ndarray | None = None
        n_visited = 0
        # Priority queue keyed by each node's bounding-box distance lower bound:
        # visiting the most promising subtree first tightens ``best_cost`` early
        # so far subtrees are pruned without descent.
        root_lb = metrics.box_distance(factual, self.root.lo, self.root.hi, scale)
        counter = 0
        pq: list[tuple[float, int, _KDNode]] = [(root_lb, counter, self.root)]
        while pq:
            node_lb, _, node = heapq.heappop(pq)
            if node_lb >= best_cost:
                break  # every remaining node is at least this far — done
            n_visited += 1
            if node.region_ids is not None:
                for rid in node.region_ids:
                    lo = self.region_lo[rid]
                    hi = self.region_hi[rid]
                    d = metrics.box_distance(factual, lo, hi, scale)
                    if d < best_cost:
                        x = metrics.project_onto_box(factual, lo, hi)
                        c = metrics.scaled_l1(factual, x, scale)
                        if c < best_cost:
                            best_cost = c
                            best_x = x
                continue
            for child in (node.left, node.right):
                if child is None:
                    continue
                child_lb = metrics.box_distance(factual, child.lo, child.hi, scale)
                if child_lb < best_cost:
                    counter += 1
                    heapq.heappush(pq, (child_lb, counter, child))
        return best_cost, best_x, n_visited


class CounterfactualMapsExplainer(BaselineExplainer):
    """Nearest opposite-region projection over a KD-tree of target regions."""

    name = "counterfactual_maps"

    def _configure(self, max_cells: int = 200_000, leaf_size: int = 8) -> None:
        if self.box_lo is None:
            raise ValueError("CounterfactualMapsExplainer requires X (root box) at construction")
        self.max_cells = int(max_cells)
        self.leaf_size = int(leaf_size)
        # Cache the extracted map per target class (the expensive preprocessing).
        self._maps: dict[int, VolumetricKDTree | None] = {}

    def _build_map(self, target_class: int) -> tuple[VolumetricKDTree | None, float, int]:
        t0 = time.perf_counter()
        need = self._need()
        edges = feature_edges(self.rf, self.box_lo, self.box_hi)
        los: list[np.ndarray] = []
        his: list[np.ndarray] = []
        for lo, hi in iter_cells(edges, max_cells=self.max_cells):
            # Label the region by the hard vote at its interior representative.
            rep = metrics.project_onto_box(0.5 * (lo + hi), lo, hi)
            votes = int(metrics.target_vote_count(self.rf, rep[None, :], target_class)[0])
            if votes >= need:
                los.append(lo)
                his.append(hi)
        n_regions = len(los)
        tree = None
        if n_regions:
            tree = VolumetricKDTree(
                np.stack(los), np.stack(his), leaf_size=self.leaf_size
            )
        return tree, time.perf_counter() - t0, n_regions

    def preprocess(self, target_class: int) -> None:
        """Build and cache the target-class map (call once before batched queries)."""
        if target_class not in self._maps:
            tree, build_s, n_regions = self._build_map(target_class)
            self._maps[target_class] = tree
            self._last_build = {"target": target_class, "build_s": build_s, "n_regions": n_regions}

    def explain(
        self, x: np.ndarray, target_class: int, time_limit_s: float = 60.0
    ) -> CFResult:
        x = np.asarray(x, dtype=np.float64)
        t0 = time.perf_counter()
        build_s = 0.0
        n_regions = None
        try:
            if target_class not in self._maps:
                tree, build_s, n_regions = self._build_map(target_class)
                self._maps[target_class] = tree
            tree = self._maps[target_class]
        except ValueError as exc:
            return CFResult(
                x_cf=None, cost=None, is_optimal=False, lower_bound=None,
                upper_bound=None, n_nodes=None,
                wall_time_s=time.perf_counter() - t0, status="error",
                method=self.name, error=str(exc),
            )

        if tree is None:  # no target region exists in the root box
            return CFResult(
                x_cf=None, cost=None, is_optimal=True, lower_bound=math.inf,
                upper_bound=None, n_nodes=0, wall_time_s=time.perf_counter() - t0,
                status="infeasible", method=self.name,
                extra={"build_s": build_s, "n_regions": n_regions},
            )

        cost, x_cf, n_visited = tree.nearest(x, self.scale)
        wall = time.perf_counter() - t0
        return CFResult(
            x_cf=x_cf, cost=cost, is_optimal=True, lower_bound=cost,
            upper_bound=cost, n_nodes=n_visited, wall_time_s=wall,
            status="optimal", method=self.name,
            reaches_target=self.verify(x_cf, target_class),
            extra={"build_s": build_s, "n_regions": n_regions},
        )
