"""OCEAN-backed baselines: MILP, CP (CPCF), MaxSAT, and SAT.

`vidalt/OCEAN` is a unified library exposing several exact/anytime CF backends
over one parsed sklearn tree ensemble.  Installing it gives four baselines at
once (see BASELINES.md).  This module wraps each backend behind the common
:class:`~baselines.base.BaselineExplainer` interface.

**Install:** ``pip install oceanpy`` (imports as ``ocean``).  Core API verified
against the oceanpy README (2026-07): ``MixedIntegerProgramExplainer``,
``ConstraintProgrammingExplainer``, ``MaxSATExplainer``, each with
``explain(x, y=<target>, norm=1)``, and mappers built by
``ocean.feature.parse_features``.  Only the MIP backend needs Gurobi; CP and
MaxSAT are licence-free, so the suite runs without a Gurobi licence.  OCEAN has
no public SAT explainer, so ``ocean_sat`` stays SKIPPED.  Result-extraction
attribute names (cost / bounds / optimality flags) are still probed defensively
via ``_extract_*`` since the README does not document the result object; run
``python -m baselines.smoke_test`` after install and adjust if a probe misses.

Design decisions that keep the comparison apples-to-apples:

* **Objective.** Our yardstick is range-normalised weighted L1 (see
  :mod:`baselines.metrics`).  We pass ``self.weights`` to OCEAN when the backend
  accepts per-feature weights; the reported ``cost`` is ALWAYS recomputed from
  the returned point under our scale, so it is comparable no matter what the
  backend minimised internally.  If we cannot confirm the backend optimised the
  same metric, ``is_optimal`` is downgraded to ``False`` and the mismatch is
  recorded in ``extra["metric_confirmed"]``.
* **Voting.** MILP/CP encode the RF's soft (probability-averaged) vote; MaxSAT
  is the hard-majority backend (unimplemented in oceanpy 2.0.3).  A backend's
  certificate counts (``is_optimal``) only when its semantics match the suite's
  ``voting`` mode AND the point survives sklearn's float32 evaluation; a
  solver-proved-but-downgraded run is flagged in ``cert_note``/``solver_optimal``
  (2026-07-12 — previously soft backends were hard-coded to never certify,
  which was correct for hard-voting suites only).
* **Solver licence.** MILP needs Gurobi (free academic).  CP uses OR-Tools
  CP-SAT (licence-free) and is the reproducible exact/anytime fallback.
"""
from __future__ import annotations

import importlib
import math
import time
from typing import Any

import numpy as np

from .base import BaselineExplainer, BaselineUnavailable, CFResult


# --- API name candidates (edit here if the installed build differs) -------
# For each backend, the first attribute name that exists on the ``ocean`` module
# is used.  Extend these lists rather than sprinkling names through the code.
_CANDIDATES: dict[str, list[str]] = {
    "milp": ["MixedIntegerProgramExplainer", "MILPExplainer", "OceanMILP"],
    "cp": ["ConstraintProgrammingExplainer", "CPExplainer", "CPCFExplainer"],
    "maxsat": ["MaxSATExplainer", "WeightedMaxSATExplainer", "MaxSatExplainer"],
    "sat": ["SATExplainer", "MACEExplainer", "SatExplainer"],
}

# Module names to try when importing the package.
_MODULE_CANDIDATES = ["ocean", "OCEAN"]


def _import_ocean():
    """Import the OCEAN package or raise :class:`BaselineUnavailable`.

    Distinguishes "not installed" from "installed but its import chain is broken"
    (e.g. xgboost failing to load libomp) so the real cause is not hidden behind
    a generic message.
    """
    from importlib.util import find_spec

    last: Exception | None = None
    installed = False
    for name in _MODULE_CANDIDATES:
        try:
            if find_spec(name) is None:
                continue
        except Exception:
            pass
        installed = True
        try:
            return importlib.import_module(name)
        except Exception as exc:  # a broken transitive import (installed but unusable)
            last = exc
    if installed:
        raise BaselineUnavailable(
            f"OCEAN is installed but failed to import ({type(last).__name__}): {last}"
        )
    raise BaselineUnavailable(
        f"OCEAN is not installed (tried {_MODULE_CANDIDATES}). Install with: pip install oceanpy"
    )


_GUROBI_FEASTOL_PATCHED = False


def _patch_gurobi_feastol_floor() -> None:
    """Clamp Gurobi ``FeasibilityTol`` requests to its 1e-9 floor (idempotent).

    OCEAN's MILP builder derives a per-feature big-M ``epsilon`` from the
    smallest gap between a feature's split thresholds and shrinks
    ``FeasibilityTol`` to fit it.  On fully-grown forests a feature can have two
    thresholds only ~1e-9 apart (e.g. pima's ``diabetes_pedigree_function``,
    gap 3.7e-9), driving the tol below Gurobi's hard 1e-9 minimum so ``setParam``
    raises at model-build time.  OCEAN's own ``delta <= 2*min_tol`` branch
    already clamps to the floor and proceeds; this just completes that clamp for
    the boundary case it misses.  1e-9 stays below the offending gap, so the
    encoding remains valid.  No-op if gurobipy is not importable (CP/MaxSAT).
    """
    global _GUROBI_FEASTOL_PATCHED
    if _GUROBI_FEASTOL_PATCHED:
        return
    try:
        import gurobipy as gp
    except Exception:
        return
    _orig = gp.Model.setParam
    floor = 1e-9

    def _clamped(self, paramname, newval=None):
        if isinstance(paramname, str) and paramname.lower() == "feasibilitytol":
            try:
                if newval is not None and float(newval) < floor:
                    newval = floor
            except (TypeError, ValueError):
                pass
        return _orig(self, paramname, newval)

    gp.Model.setParam = _clamped
    _GUROBI_FEASTOL_PATCHED = True


def _resolve_explainer(module, backend: str):
    for attr in _CANDIDATES[backend]:
        cls = getattr(module, attr, None)
        if cls is not None:
            return cls, attr
    raise BaselineUnavailable(
        f"OCEAN backend {backend!r} not found on the installed package "
        f"(tried {_CANDIDATES[backend]}). Update baselines/ocean.py::_CANDIDATES "
        "to the name in this build."
    )


def build_mapper(module, X, feature_names=None, discretes=(), encoded=(), scale=False):
    """Build OCEAN's feature ``mapper`` from a raw frame via ``parse_features``.

    OCEAN describes feature types with a ``Mapper`` returned by
    ``ocean.feature.parse_features(df, ...)`` alongside a processed frame.  We
    default ``scale=False`` so the mapper matches a forest trained on the *raw*
    features (our harness trains one RF on raw data and shares it across every
    method); with ``scale=True`` OCEAN normalises continuous columns to
    [-0.5, 0.5] and the forest would have to be trained on that processed frame
    instead.  ``discretes`` / ``encoded`` name ordinal and one-hot columns.

    Returns the ``Mapper`` object, or ``None`` if the API is not as expected.
    """
    import pandas as pd

    if not isinstance(X, pd.DataFrame):
        cols = list(feature_names) if feature_names else [f"f{i}" for i in range(np.asarray(X).shape[1])]
        X = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=cols)
    # OCEAN's parse_features DROPS zero-range (constant) columns, which shortens
    # mapper.names so the forest's original feature indices overrun it
    # (mapper.names[idx] -> IndexError in the tree parser; hits ionosphere's
    # all-zero column and the digit border pixels).  The forest never splits on a
    # constant feature, so give each such column a tiny artificial spread on one
    # row so parse_features keeps it and the indices stay aligned.  The
    # counterfactual is unaffected — no split means no incentive to move it.
    const_cols = [c for c in X.columns if X[c].to_numpy().max() == X[c].to_numpy().min()]
    if const_cols:
        X = X.copy()
        for c in const_cols:
            v = float(X[c].iloc[0])
            X.iloc[0, X.columns.get_loc(c)] = v + (abs(v) + 1.0) * 1e-6
    parse_features = getattr(getattr(module, "feature", None), "parse_features", None)
    if parse_features is None:
        return None
    _processed, mapper = parse_features(
        X, discretes=tuple(discretes), encoded=tuple(encoded), scale=scale
    )
    return mapper


class _OceanExplainer(BaselineExplainer):
    """Shared machinery for all four OCEAN backends."""

    backend: str = ""
    #: whether this backend exposes an anytime lower/upper bracket (informational)
    anytime: bool = False
    #: aggregation the backend optimises. OCEAN 2.0.3: MILP/CP encode the RF's
    #: *soft* (probability-averaged) vote; MaxSAT is the hard-majority backend
    #: (but unimplemented in 2.0.3).  The certificate transfers only when this
    #: matches the suite's requested ``voting`` mode (see ``explain``).
    voting_semantics: str = "soft"

    def _configure(self, discretes=(), encoded=(), mapper_scale=False,
                   threads: int | None = 1, explainer_kwargs: dict | None = None) -> None:
        self._module = _import_ocean()
        self._cls, self._cls_name = _resolve_explainer(self._module, self.backend)
        if self.backend == "milp":
            # Let the MILP model build on fully-grown forests with tightly-spaced
            # split thresholds (see _patch_gurobi_feastol_floor).
            _patch_gurobi_feastol_floor()
        # Solver threads.  Default 1 for a thread-fair comparison with our
        # single-threaded method: OCEAN CP (OR-Tools CP-SAT) and MILP (Gurobi)
        # both otherwise use ALL cores by default.  Pass ``threads=None`` to let
        # the solver use its own default (multi-core).  CP maps it to
        # ``parameters.num_workers``, MILP to Gurobi ``Threads``.
        self._threads = None if threads is None else int(threads)
        self._explainer_kwargs = dict(explainer_kwargs or {})
        # Prefer an explicit mapper; else build one from the training frame the
        # base stored (self.X).  All OCEAN explainers require a mapper.
        if self.mapper is None and self.X is not None:
            self.mapper = build_mapper(
                self._module, self.X, self.feature_names,
                discretes=discretes, encoded=encoded, scale=mapper_scale,
            )
        self._mapper_scale = bool(mapper_scale)
        self._model = None  # built lazily per forest on first explain
        self._build_time_s = None  # set on first build; reused across queries

    # -- backend construction / invocation (verified against oceanpy 2.0.3) --
    def _build_model(self, target_class: int):
        """Instantiate the OCEAN explainer: ``cls(rf, mapper=..., **kwargs)``.

        ``weights`` in the OCEAN constructor is *per-tree* ensemble voting weight,
        not a per-feature objective weight — left at its ``None`` default, which
        is uniform hard-majority voting (our regime).  Extra backend options
        (``epsilon``, ``env``, ...) can be passed via ``explainer_kwargs``.
        """
        if self.mapper is None:
            raise BaselineUnavailable(
                f"{self.name}: no mapper (pass mapper= or X= so one can be built "
                "via ocean.feature.parse_features)"
            )
        return self._cls(self.rf, mapper=self.mapper, **self._explainer_kwargs)

    def _invoke(self, model, x: np.ndarray, target_class: int, time_limit_s: float):
        """Call ``explain(x, y=<target>, norm=1, max_time=<seconds>, num_workers=<threads>)``."""
        kw = {}
        if self._threads is not None:  # pin solver threads (CP num_workers / Gurobi Threads)
            kw["num_workers"] = self._threads
        return model.explain(
            x, y=int(target_class), norm=int(self.norm),
            max_time=max(1, int(round(time_limit_s))), **kw,
        )

    @staticmethod
    def _extract_point(result) -> np.ndarray | None:
        """Pull the counterfactual vector out of an OCEAN ``Explanation``."""
        if result is None:
            return None
        to_numpy = getattr(result, "to_numpy", None)
        if callable(to_numpy):
            arr = np.asarray(to_numpy(), dtype=np.float64).ravel()
            return arr
        for attr in ("x", "values"):
            val = getattr(result, attr, None)
            if val is not None:
                return np.asarray(val, dtype=np.float64).ravel()
        return None

    def _metric_matches(self) -> bool:
        """Does OCEAN's objective share an argmin with our weighted-L1 metric?

        OCEAN minimises plain L1 in the mapper's feature space.  With an
        unscaled mapper that is raw-unit L1 (``sum |Δ_f|``), which is a positive
        multiple of our ``sum scale_f·|Δ_f|`` iff our per-feature scale is
        uniform.  With a scaled mapper OCEAN minimises ``sum |Δ_f|/range_f``,
        matching our default (``scale_f = weight_f/range_f``) iff the *weights*
        are uniform.  Only when they share an argmin is OCEAN's optimum also
        optimal under our metric, so only then do we surface ``is_optimal=True``.
        """
        key = self.weights if self._mapper_scale else self.scale
        return bool(np.allclose(key, key.flat[0]))

    def explain(
        self, x: np.ndarray, target_class: int, time_limit_s: float = 60.0
    ) -> CFResult:
        x = np.asarray(x, dtype=np.float64)
        try:
            # Build the forest->MILP/CP encoding once (query-independent) and
            # reuse it across queries; cleanup() resets the previous query's
            # majority-class constraint (verified to match a fresh model).  Build
            # time is charged separately from per-query solve time.
            if self._model is None:
                tb = time.perf_counter()
                self._model = self._build_model(target_class)
                self._build_time_s = time.perf_counter() - tb
            else:
                self._model.cleanup()
            ts = time.perf_counter()
            raw = self._invoke(self._model, x, target_class, time_limit_s)
            solve = time.perf_counter() - ts
        except BaselineUnavailable:
            raise
        except Exception as exc:
            return CFResult(
                x_cf=None, cost=None, is_optimal=False, lower_bound=None,
                upper_bound=None, n_nodes=None,
                wall_time_s=0.0, status="error",
                method=self.name, error=f"{type(exc).__name__}: {exc}",
                build_time_s=getattr(self, "_build_time_s", None),
                extra={"cls": self._cls_name},
            )
        wall = solve  # headline per-query time = inference only
        build_time_s = getattr(self, "_build_time_s", None)

        x_cf = self._extract_point(raw)
        if x_cf is None:
            return CFResult(
                x_cf=None, cost=None, is_optimal=False,
                lower_bound=None, upper_bound=None, n_nodes=None,
                wall_time_s=wall, status="infeasible", method=self.name,
                build_time_s=build_time_s, solve_time_s=solve,
                extra={"cls": self._cls_name},
            )

        # Recompute the objective under OUR scaled-L1 so the number is comparable
        # across every method regardless of what OCEAN minimised internally.
        cost = self.scaled_cost(x, x_cf)
        reaches = self.verify(x_cf, target_class)  # hard-voting re-check
        # OCEAN's exact solvers return the optimum *of their own encoding*.
        # ``solver_optimal``: the solver finished under its budget, i.e. it
        # PROVED that optimum (an exact solver that hits max_time returns its
        # incumbent unproven; finishing early is the proof signal we have —
        # oceanpy 2.0.3 exposes no status on the Explanation).
        # The certificate transfers to OUR problem (``is_optimal``) only if:
        #   (a) the objective shares an argmin with our weighted-L1 metric,
        #   (b) the backend's vote aggregation matches the suite's voting mode
        #       (MILP/CP encode the soft probability-averaged vote, so they
        #       certify soft-voting suites and never hard-voting ones), and
        #   (c) the point survives sklearn's float32 evaluation — OCEAN solves
        #       a continuous encoding, and a CF on a split boundary can be
        #       re-routed by the real forest's float32 cast (then its optimum
        #       is for a slightly different model).
        # ``cert_note`` records why a solver-proved optimum was downgraded.
        solver_optimal = bool(solve < 0.95 * float(time_limit_s))
        metric_matches = self._metric_matches()
        voting_matches = self.voting_semantics == self.voting
        try:
            x32 = np.asarray(x_cf, dtype=np.float64).astype(np.float32).reshape(1, -1)
            sklearn_ok = bool(int(self.rf.predict(x32)[0]) == int(target_class))
        except Exception:
            sklearn_ok = False
        is_optimal = bool(
            solver_optimal and metric_matches and voting_matches and sklearn_ok
        )
        cert_note = ""
        if solver_optimal and not is_optimal:
            reasons = []
            if not metric_matches:
                reasons.append("metric_mismatch")
            if not voting_matches:
                reasons.append(f"voting_semantics_{self.voting_semantics}_vs_suite_{self.voting}")
            if not sklearn_ok:
                reasons.append("sklearn_invalid")
            cert_note = "solver_optimal_downgraded:" + "+".join(reasons)
        # Dual bound from the solver, converted to OUR units.  OCEAN minimises
        # (possibly range-normalised) plain L1; our cost multiplies each |Δ_f| by
        # scale_f.  When metric_matches (that multiplier is uniform) the bound is
        # comparable up to that constant, so we can report a real gap on a
        # timed-out anytime run.  Best-effort: never let bound extraction break
        # the result.
        raw_bound = self._solver_lower_bound()
        lower_bound = cost if is_optimal else None
        if lower_bound is None and raw_bound is not None and metric_matches:
            unit = float((self.weights if self._mapper_scale else self.scale).flat[0])
            lb = raw_bound * unit
            # a valid LB can't exceed the incumbent; clamp float noise
            lower_bound = min(lb, cost) if math.isfinite(lb) else None
        return CFResult(
            x_cf=x_cf,
            cost=cost,
            is_optimal=is_optimal,
            lower_bound=lower_bound,
            upper_bound=cost,
            n_nodes=None,
            wall_time_s=wall,
            build_time_s=build_time_s,
            solve_time_s=solve,
            status="optimal" if is_optimal else "feasible",
            method=self.name,
            reaches_target=reaches,
            solver_optimal=solver_optimal,
            cert_note=cert_note or None,
            extra={
                "cls": self._cls_name,
                "voting_semantics": self.voting_semantics,
                "mapper_scale": self._mapper_scale,
                "metric_matches_ours": metric_matches,
                "solver_lower_bound_raw": raw_bound,
                # a soft-voting optimum certifies OUR (hard) problem only when it
                # also clears the hard majority — noted for auditing.
                "certifies_own_optimum": True,
            },
        )

    def _solver_lower_bound(self) -> float | None:
        """Best-effort dual bound off the solved model, in OCEAN's own units.

        Gurobi exposes ``ObjBound``; OR-Tools/CP and others may expose a
        differently-named attribute or none.  Returns None if unavailable — the
        caller then simply reports no gap for that run.
        """
        model = self._model
        if model is None:
            return None
        for attr in ("ObjBound", "objective_bound", "best_bound", "BestBd"):
            try:
                val = getattr(model, attr, None)
                if val is not None and math.isfinite(float(val)):
                    return float(val)
            except Exception:
                continue
        return None


class OceanMILPExplainer(_OceanExplainer):
    """OCEAN MILP backend (Parmentier & Vidal 2021).  Needs Gurobi."""

    name = "ocean_milp"
    backend = "milp"
    anytime = False
    voting_semantics = "soft"  # encodes the RF probability-averaged vote


class OceanCPExplainer(_OceanExplainer):
    """OCEAN CP / CPCF backend (OR-Tools CP-SAT).  Licence-free, anytime.

    Soft voting (like MILP): not directly comparable to our hard-voting method
    except when leaves are pure (e.g. unlimited depth), where soft == hard.
    """

    name = "ocean_cp"
    backend = "cp"
    anytime = True
    voting_semantics = "soft"


class OceanMaxSATExplainer(_OceanExplainer):
    """OCEAN weighted-MaxSAT backend (Raevskaya & Lehtonen 2025).

    The hard-majority backend — exactly our regime.  **Not implemented in
    oceanpy 2.0.3** (``ocean.maxsat._model.build`` raises ``NotImplementedError``),
    so it currently errors; kept wired for when upstream ships it.
    """

    name = "ocean_maxsat"
    backend = "maxsat"
    anytime = False
    voting_semantics = "hard"


class OceanSATExplainer(_OceanExplainer):
    """OCEAN SAT / MACE-style backend (Karimi et al. 2020).  Optional.

    OCEAN 2.0.3 exposes no public SAT explainer, so this stays SKIPPED unless a
    future build adds one under a name in ``_CANDIDATES["sat"]``.
    """

    name = "ocean_sat"
    backend = "sat"
    anytime = False
    voting_semantics = "hard"
