"""Brute-force exact optimum — the verification oracle for tiny forests.

Enumerate every threshold-grid cell, project the factual onto it, keep the
cheapest cell whose projection actually reaches the target under hard voting.
This is provably the global optimum (the objective is constant-per-cell and the
projection minimises it inside the cell), so it is the ground truth that every
"optimal" backend must agree with on small instances.

Not for scale — the grid is exponential in depth.  Guarded by ``max_cells``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import metrics
from .base import BaselineExplainer, CFResult
from .grid import feature_edges, iter_cells


class BruteForceExplainer(BaselineExplainer):
    name = "brute_force"

    def _configure(self, max_cells: int = 200_000) -> None:
        if self.box_lo is None:
            raise ValueError("BruteForceExplainer requires X (for the root box) at construction")
        self.max_cells = int(max_cells)

    def explain(
        self, x: np.ndarray, target_class: int, time_limit_s: float = 60.0
    ) -> CFResult:
        x = np.asarray(x, dtype=np.float64)
        t0 = time.perf_counter()
        need = self._need()
        edges = feature_edges(self.rf, self.box_lo, self.box_hi)

        best_cost = math.inf
        best_x: np.ndarray | None = None
        n_visited = 0
        try:
            for lo, hi in iter_cells(edges, max_cells=self.max_cells):
                n_visited += 1
                # Cheapest lower bound to this cell; skip if it can't beat best.
                if metrics.box_distance(x, lo, hi, self.scale) >= best_cost:
                    continue
                x_prime = metrics.project_onto_box(x, lo, hi)
                votes = int(metrics.target_vote_count(self.rf, x_prime[None, :], target_class)[0])
                if votes < need:
                    continue
                cost = metrics.scaled_l1(x, x_prime, self.scale)
                if cost < best_cost:
                    best_cost = cost
                    best_x = x_prime
        except ValueError as exc:
            return CFResult(
                x_cf=None, cost=None, is_optimal=False, lower_bound=None,
                upper_bound=None, n_nodes=n_visited,
                wall_time_s=time.perf_counter() - t0, status="error",
                method=self.name, error=str(exc),
            )

        wall = time.perf_counter() - t0
        if best_x is None:
            return CFResult(
                x_cf=None, cost=None, is_optimal=True, lower_bound=math.inf,
                upper_bound=None, n_nodes=n_visited, wall_time_s=wall,
                status="infeasible", method=self.name,
            )
        return CFResult(
            x_cf=best_x, cost=best_cost, is_optimal=True, lower_bound=best_cost,
            upper_bound=best_cost, n_nodes=n_visited, wall_time_s=wall,
            status="optimal", method=self.name,
            reaches_target=self.verify(best_x, target_class),
        )
