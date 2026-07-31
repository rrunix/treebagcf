"""Common interface for every counterfactual (CF) baseline.

The experiment harness treats all baselines interchangeably: it hands each one
the *same* trained ``sklearn.ensemble.RandomForestClassifier``, the *same* query
point ``x``, the *same* ``target_class``, and the *same* distance (weighted /
range-normalised L1, matching our own ``calf`` method) and time limit.  Every
backend returns a :class:`CFResult`.

Distance convention (kept identical across all methods):

    cost(x, x') = sum_f scale[f] * |x[f] - x'[f]|,   scale[f] = weight[f] / range[f]

``range[f]`` is the per-feature data range (max - min over the training frame),
so a unit of cost is "one full feature range moved".  This is exactly what
``calf.cost.l1_scale`` computes, so costs reported here are directly
comparable to our method's ``ExtractionResult.cost``.

Voting convention: **hard voting** (per-tree argmax leaf, then majority over
trees) — our paper's regime.  See :func:`baselines.metrics.target_vote_count`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics


class BaselineUnavailable(RuntimeError):
    """Raised when a backend's dependency (solver / package) is not installed.

    The registry catches this so an unavailable backend is *skipped* rather than
    crashing the whole comparison.  Never raise this for a genuine solve failure
    — use ``CFResult(status="error")`` for that.
    """


@dataclass
class CFResult:
    """Uniform result record for one (method, query) solve.

    Fields marked "anytime" are populated only by backends that expose a running
    lower/upper bracket (our method and CP-SAT); one-shot solvers (MaxSAT/SAT)
    leave ``lower_bound=None`` mid-search and set it equal to ``cost`` on a
    proven optimum.
    """

    x_cf: Optional[np.ndarray]        # counterfactual point, or None if none found
    cost: Optional[float]             # objective value under the shared scaled-L1
    is_optimal: bool                  # True only if the backend *certifies* optimality
    lower_bound: Optional[float]      # best certified LB at return (anytime)
    upper_bound: Optional[float]      # = cost if a feasible CF was found
    n_nodes: Optional[int]            # search-effort proxy, if the backend exposes one
    wall_time_s: float                # per-query INFERENCE time (== solve_time_s); the headline
    status: str                       # optimal | feasible | timeout | infeasible | error
    method: str = ""                  # backend label, e.g. "ocean_milp"
    metric: str = "weighted_l1"       # objective metric, for auditing
    reaches_target: Optional[bool] = None  # harness re-check that x_cf hits target under hard voting
    # Build/solve split so the per-query timing is apples-to-apples: build is the
    # one-time forest->solver encoding (query-independent, amortised across a
    # forest's queries); solve is the per-query inference.  wall_time_s == solve.
    build_time_s: Optional[float] = None
    solve_time_s: Optional[float] = None
    error: Optional[str] = None       # message when status == "error"
    # Certificate accounting (exact-solver backends).  ``solver_optimal`` is True
    # when the backend proved optimality *for its own encoding* (finished under
    # the cap); ``is_optimal`` additionally requires that the certificate
    # transfers to OUR problem (metric argmin match, voting-semantics match, and
    # the point survives sklearn's float32 evaluation).  ``cert_note`` flags WHY
    # a solver-optimal run was downgraded (empty when nothing was).
    solver_optimal: Optional[bool] = None
    cert_note: Optional[str] = None
    extra: dict = field(default_factory=dict)  # backend-specific diagnostics

    def as_row(self) -> dict:
        """Flat dict for CSV aggregation (arrays dropped)."""
        return {
            "method": self.method,
            "status": self.status,
            "is_optimal": self.is_optimal,
            "cost": self.cost,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "n_nodes": self.n_nodes,
            "wall_time_s": self.wall_time_s,
            "build_time_s": self.build_time_s,
            "solve_time_s": self.solve_time_s,
            "metric": self.metric,
            "reaches_target": self.reaches_target,
            "solver_optimal": self.solver_optimal,
            "cert_note": self.cert_note,
            "error": self.error,
        }


class BaselineExplainer:
    """Base class every backend subclasses.

    Subclasses implement :meth:`explain`.  The base constructor resolves the
    shared cost scale (so all methods optimise the identical objective) and
    stores the forest and voting parameters.

    Parameters
    ----------
    rf : fitted RandomForestClassifier
    X : optional 2-D training frame; used to infer per-feature ranges (and, for
        the geometry-based backends, the root box) when ``feature_ranges`` is not
        given explicitly.
    feature_ranges : optional per-feature ``max - min`` used for L1 normalisation;
        overrides ``X`` if both are given.
    weights : optional per-feature cost weights (default all-ones).  A very large
        weight makes a feature effectively immutable.
    mapper : optional backend-specific feature descriptor (OCEAN mapper).
    voting : "hard" (majority vote) — the only supported mode; kept explicit so
        the harness records it.
    norm : distance norm; only ``1`` (L1) is supported.
    threshold : quorum fraction; ``need = ceil(threshold * n_trees)`` trees must
        vote the target class (0.5 => strict majority).
    """

    name: str = "baseline"

    def __init__(
        self,
        rf,
        *,
        X: Optional[np.ndarray] = None,
        feature_ranges: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        mapper=None,
        voting: str = "hard",
        norm: int = 1,
        threshold: float = 0.5,
        **kw,
    ):
        if voting not in ("hard", "soft"):
            raise ValueError(
                f"voting must be 'hard' (majority) or 'soft' (probability-averaged); got {voting!r}"
            )
        if int(norm) != 1:
            raise ValueError(f"only norm=1 (L1) is supported; got {norm!r}")
        self.rf = rf
        self.voting = voting
        self.norm = int(norm)
        self.threshold = float(threshold)
        self.mapper = mapper
        self.n_features = int(rf.n_features_in_)
        self.n_trees = len(rf.estimators_)
        self.classes_ = np.asarray(rf.classes_)

        if feature_ranges is not None:
            ranges = np.asarray(feature_ranges, dtype=np.float64)
        elif X is not None:
            ranges = metrics.feature_ranges_from_X(X)
        else:
            # Plain (unnormalised) L1 fallback; the harness should always pass X
            # or feature_ranges so methods share the same normalised objective.
            ranges = np.ones(self.n_features, dtype=np.float64)
        if ranges.shape != (self.n_features,):
            raise ValueError(
                f"feature_ranges shape {ranges.shape} != (n_features={self.n_features},)"
            )
        self.feature_ranges = ranges
        self.weights = (
            np.ones(self.n_features, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        self.scale = metrics.l1_scale(ranges, self.weights)

        # Keep the raw training frame around: geometry backends need the box and
        # OCEAN needs it to build a feature mapper.
        if X is not None:
            self.X = np.asarray(X, dtype=np.float64)
            # Root box (used by geometry backends): [min - eps, max] per feature,
            # matching calf's _dataset_box so certified optima line up exactly.
            self.box_lo = self.X.min(axis=0) - 1e-12
            self.box_hi = self.X.max(axis=0)
        else:
            self.X = None
            self.box_lo = None
            self.box_hi = None
        self.feature_names = kw.pop("feature_names", None)

        self._configure(**kw)

    # --- hooks -----------------------------------------------------------
    def _configure(self, **kw) -> None:
        """Optional per-backend setup (override in subclasses)."""
        if kw:
            raise TypeError(f"unexpected kwargs for {type(self).__name__}: {sorted(kw)}")

    def explain(
        self, x: np.ndarray, target_class: int, time_limit_s: float = 60.0
    ) -> CFResult:
        raise NotImplementedError

    # --- shared helpers --------------------------------------------------
    def _need(self) -> int:
        return metrics.need(self.n_trees, self.threshold)

    def scaled_cost(self, x: np.ndarray, x_cf: np.ndarray) -> float:
        return metrics.scaled_l1(x, x_cf, self.scale)

    def verify(self, x_cf: Optional[np.ndarray], target_class: int) -> Optional[bool]:
        """Re-check under hard voting that ``x_cf`` truly reaches the target.

        Independent of whatever the backend claims — the harness's own oracle.
        """
        if x_cf is None:
            return None
        return metrics.reaches_target(
            self.rf, np.asarray(x_cf, dtype=np.float64), target_class, self.threshold
        )
