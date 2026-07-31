"""calf — certified-optimal counterfactuals for random forests.

Minimal implementation of the cost-splitting Lagrangian dual bound + A* box
search + dual-guided greedy rounding.  See ``example.md`` for the method.

Quick start:

    from sklearn.ensemble import RandomForestClassifier
    import calf

    rf = RandomForestClassifier(...).fit(X, y)
    res = calf.solve(rf, X, factual=X[0], target_class=1)
    print(res.cost, res.proven_optimal, res.x)
"""
from __future__ import annotations

import numpy as np

from .alpha_lib import AlphaLibrary
from .cost import l1_cost, l1_scale
from .dataset_info import DatasetInfo, FeatureSpec, from_array
from .engine import (
    CompiledRF,
    compile_rf,
    extract_counterfactual,
)
from .parser import ParsedRF, Rule, parse_sklearn_rf
from .result import ExtractionResult

__all__ = [
    "solve",
    "extract_counterfactual",
    "parse_sklearn_rf",
    "ParsedRF",
    "Rule",
    "DatasetInfo",
    "FeatureSpec",
    "from_array",
    "l1_scale",
    "l1_cost",
    "compile_rf",
    "CompiledRF",
    "ExtractionResult",
    "AlphaLibrary",
]


def solve(
    model,
    X,
    factual,
    target_class: int,
    *,
    weights=None,
    feature_names: list[str] | None = None,
    voting: str = "hard",
    alpha_library: AlphaLibrary | None = None,
    **engine_kwargs,
) -> ExtractionResult:
    """Certified-optimal counterfactual for a fitted sklearn RandomForestClassifier.

    Parameters
    ----------
    model : fitted RandomForestClassifier
    X : 2D array used only to infer per-feature [min, max] bounds (root box)
    factual : the point to explain (length n_features)
    target_class : the class the counterfactual must reach
    weights : optional per-feature cost weights (default all ones; use a large
        weight to discourage moving a feature, effectively immutable)
    feature_names : optional names for the DatasetInfo
    voting : "hard" (majority) or "soft" (probability-averaged, binary only).
        Soft solves are seeded with an initial upper bound from the cheapest
        training row predicted as the target (pass ``initial_ub=None`` to
        disable, or your own ``initial_ub=(x, cost)`` to override).
    alpha_library : optional :class:`AlphaLibrary` for cross-query dual
        warm-starts.  Its entries seed this query's dual pool (strongest-first
        at this query's root box), and the pool's optimized entries are
        harvested back after the run, so repeated solves over one dataset
        keep sharing dual progress.  Must have been built with the same
        ``voting`` mode.
    **engine_kwargs : forwarded to the chosen engine (e.g. max_iters,
        node_local_tree_lb, threshold; dual_* for the python engine).
    """
    parsed = parse_sklearn_rf(model)
    dataset_info = from_array(X, names=feature_names)
    scale = l1_scale(dataset_info, weights=weights)
    factual = np.asarray(factual, dtype=np.float64)
    if voting == "soft" and "initial_ub" not in engine_kwargs:
        # Seed the incumbent with the cheapest training row the forest already
        # predicts as the target: sklearn's predict IS the soft vote, so the
        # seed is soft-feasible by construction, and it gives hard rows a
        # reportable anytime gap from iteration 0 (probed 2026-07-09: without
        # it, plateau rows run tens of thousands of iters incumbent-free).
        # Hard voting is NOT seeded this way — rf.predict does not imply a
        # tree-majority, so the seed could be hard-infeasible.
        preds = model.predict(np.asarray(X, dtype=np.float64))
        mask = preds == target_class
        if np.any(mask):
            from .numba.kernels import refine_l1

            x32 = factual.astype(np.float32).astype(np.float64)
            cand = np.asarray(X, dtype=np.float64)[mask]
            cand = cand.astype(np.float32).astype(np.float64)  # sklearn's grid
            costs = (np.abs(cand - x32) * scale).sum(axis=1)
            j = int(np.argmin(costs))
            seed_x, seed_cost = cand[j], float(costs[j])
            # Box-refine the seed: the intersection of its containing leaves
            # (one per tree) is a constant-prediction box, so pulling the seed
            # toward the factual inside it keeps the exact same soft vote at a
            # never-worse (usually much lower) cost.
            inside = (
                (seed_x > parsed.rules_lo_mat) & (seed_x <= parsed.rules_hi_mat)
            ).all(axis=1)
            if np.any(inside):
                b_lo = parsed.rules_lo_mat[inside].max(axis=0)
                b_hi = parsed.rules_hi_mat[inside].min(axis=0)
                ref_x, ref_cost = refine_l1(
                    np.ascontiguousarray(b_lo), np.ascontiguousarray(b_hi),
                    x32, scale,
                )
                # Defensive: keep the refinement only if sklearn agrees.
                if ref_cost < seed_cost and int(
                    model.predict(ref_x[None, :])[0]
                ) == target_class:
                    seed_x, seed_cost = ref_x, float(ref_cost)
            engine_kwargs["initial_ub"] = (seed_x, seed_cost)
    # Single engine since 2026-07-09: the python A* loop over numba kernels
    # (the whole-search-in-numba core was removed to avoid divergence).
    engine_kwargs.pop("engine", None)
    pool_out: list = []
    if alpha_library is not None:
        if alpha_library.voting != voting:
            raise ValueError(
                f"alpha_library was built for voting={alpha_library.voting!r}, "
                f"this solve uses voting={voting!r}"
            )
        engine_kwargs.setdefault(
            "dual_warm_entries", alpha_library.entries_for(target_class)
        )
        engine_kwargs.setdefault("dual_pool_out", pool_out)
    res = extract_counterfactual(
        parsed, dataset_info, factual, target_class, scale,
        voting=voting, **engine_kwargs,
    )
    if alpha_library is not None:
        for pool in pool_out:
            alpha_library.harvest(pool, target_class)
    return res
