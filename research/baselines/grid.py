"""Threshold-grid enumeration shared by the exact geometry baselines.

The forest partitions feature space into axis-aligned cells whose boundaries are
the union of every tree's split thresholds.  Inside one cell the hard vote is
constant, so the *global* optimal counterfactual is the cheapest projection of
the factual onto any target-voting cell.  Both the brute-force oracle and the
Counterfactual-Maps reimplementation build on this decomposition.

The cell count is ``prod_f (#thresholds_f + 1)`` — exponential in ensemble
depth, exactly the scaling caveat both methods share (their experiments stop at
depth 7).  Callers pass ``max_cells`` to fail loudly rather than hang.
"""
from __future__ import annotations

import itertools

import numpy as np


def collect_thresholds(rf) -> list[np.ndarray]:
    """Sorted unique split thresholds per feature, gathered from every tree."""
    n_features = int(rf.n_features_in_)
    sets: list[set[float]] = [set() for _ in range(n_features)]
    for est in rf.estimators_:
        tree = est.tree_
        feat = tree.feature
        thr = tree.threshold
        for node in range(tree.node_count):
            f = int(feat[node])
            if f >= 0:  # internal node (leaves have feature == -2)
                sets[f].add(float(thr[node]))
    return [np.array(sorted(s), dtype=np.float64) for s in sets]


def feature_edges(
    rf, box_lo: np.ndarray, box_hi: np.ndarray
) -> list[np.ndarray]:
    """Per-feature cell edges: box bounds plus in-box thresholds.

    Feature ``f``'s cells are the half-open intervals ``(edges[j], edges[j+1]]``.
    """
    thresholds = collect_thresholds(rf)
    edges: list[np.ndarray] = []
    for f, ts in enumerate(thresholds):
        inside = ts[(ts > box_lo[f]) & (ts < box_hi[f])]
        edges.append(np.concatenate(([box_lo[f]], inside, [box_hi[f]])))
    return edges


def n_cells(edges: list[np.ndarray]) -> int:
    total = 1
    for e in edges:
        total *= max(e.size - 1, 1)
    return total


def iter_cells(edges: list[np.ndarray], max_cells: int | None = None):
    """Yield ``(lo, hi)`` arrays for every cell in the grid.

    ``lo`` is the (exclusive) lower corner, ``hi`` the (inclusive) upper corner.
    """
    total = n_cells(edges)
    if max_cells is not None and total > max_cells:
        raise ValueError(
            f"grid has {total} cells (> max_cells={max_cells}); "
            "geometry baselines are only intended for shallow/small forests"
        )
    per_feature = [range(max(e.size - 1, 1)) for e in edges]
    n_features = len(edges)
    for combo in itertools.product(*per_feature):
        lo = np.empty(n_features, dtype=np.float64)
        hi = np.empty(n_features, dtype=np.float64)
        for f, j in enumerate(combo):
            lo[f] = edges[f][j]
            hi[f] = edges[f][j + 1]
        yield lo, hi
