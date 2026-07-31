"""Run one task -> one result record, applying the time-cap semantics.

A *task* is one ``(dataset, rf_variant, query, method)`` tuple.  ``run_task``
loads the cached forest and data, builds the method via the baselines registry
on the shared objective, runs it under the wall-clock cap, and returns a flat
record (spec + result) ready to be written as a shard.

Time-cap contract (as specified):
- proven optimal within cap                  -> status="optimal"
- anytime method, feasible but unproven       -> status="feasible" + cost + gap + CF
- non-anytime method, not finished within cap -> status="timeout", no cost/CF (it failed)
- no counterfactual exists                     -> status="infeasible"
- backend raised                               -> status="error"

The cooperative cap (passed to each method) is primary; a genuine hang is caught
by the runner's hard kill, which writes a ``status="timeout"`` shard itself.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# Which methods expose a trustworthy running incumbent+bound (so a timed-out run
# still reports a useful CF + gap).  Everything else is treated as one-shot: if
# it doesn't finish within the cap, it simply failed.
ANYTIME = {
    "calf": True,
    "calf_prev": True,
    "calf_polish": True,
    "calf_warm": True,
    "calf_warm_early": True,
    "calf_warmlp": True,
    "ocean_cp": True,
    "ocean_milp": True,   # branch-and-bound: incumbent + ObjBound on timeout
    "ocean_cp_t4": True,      # 4-thread ablation arms of the OCEAN backends
    "ocean_milp_t4": True,
    "milp_soft": True,    # HiGHS branch-and-bound: incumbent + dual bound on timeout
    "ocean_maxsat": False,
    "ocean_sat": False,
    "brute_force": False,
    "counterfactual_maps": False,
}

# Identity fields carried by a job (shared across its queries); query_idx and
# target_class are filled in per query to form the full task spec.
_JOB_IDENTITY = (
    "dataset", "rf_variant", "rf_n_estimators", "rf_max_depth", "rf_seed",
    "method", "voting", "threshold", "time_cap_s", "metric", "norm",
)


def _feature_ranges(X: np.ndarray, metric: str) -> np.ndarray:
    """Per-feature L1 scale basis.

    ``plain_l1`` -> ranges of 1 so ``scale`` is uniform (raw-unit L1), matching
    OCEAN's unscaled mapper exactly (the validated apples-to-apples setup, see
    the pima baseline).  ``range_normalized`` -> data ranges (a unit of cost is
    one full feature range).
    """
    if metric == "plain_l1":
        return np.ones(X.shape[1], dtype=np.float64)
    if metric == "range_normalized":
        from baselines import metrics
        return metrics.feature_ranges_from_X(X)
    raise ValueError(f"unknown metric {metric!r} (plain_l1 | range_normalized)")


def build_explainer(method: str, rf, X: np.ndarray, spec: dict,
                    pool_idx: np.ndarray | None = None):
    """Construct one method on the shared objective via the baselines registry.

    ``pool_idx`` (holdout protocol) restricts the rows a method may use to tune
    its query-independent params — calf's alpha warm-up and incumbent seed —
    to the train split, while the feature-domain box stays over the full frame
    (it defines the search space, not a learned quantity, and must contain the
    held-out test factuals).  Baselines without such params ignore it.
    """
    from baselines import registry

    ranges = _feature_ranges(X, spec["metric"])
    mapper_scale = spec["metric"] == "range_normalized"
    kw = dict(spec.get("method_kwargs", {}))
    if method.startswith("calf") and pool_idx is not None:
        kw["pool_idx"] = np.asarray(pool_idx, dtype=np.int64)
    # calf warm arms cache their alpha library next to the cached forest
    # (same joblib convention): <model>__alphas_<voting>_k<k>i<iters>.joblib.
    # The warm settings are part of the name so arms with different warm-up
    # budgets never share (and silently reuse) each other's libraries.  First
    # job to build one pays the warm-up; resumed/re-run jobs load it.
    if method.startswith("calf") and kw.get("alpha_warm") and "alpha_cache" not in kw:
        model_path = spec.get("model_path")
        if model_path:
            wk = int(kw.get("alpha_warm_k", 16))
            wi = int(kw.get("alpha_warm_iters", 1000))
            wl = "lp" if kw.get("alpha_warm_lp") else ""
            kw["alpha_cache"] = str(Path(model_path).with_suffix("")) + (
                f"__alphas_{spec['voting']}_k{wk}i{wi}{wl}.joblib"
            )
    # OCEAN backends take mapper_scale; ignore it for non-OCEAN via try/except.
    common = dict(
        X=X,
        feature_ranges=ranges,
        voting=spec["voting"],
        norm=spec["norm"],
        threshold=spec["threshold"],
    )
    if method.startswith("ocean_"):
        common["mapper_scale"] = mapper_scale
    return registry.build(method, rf, **common, **kw)


def run_job(job: dict, run_dir) -> None:
    """Build one explainer per (dataset, variant, method) and solve every query.

    The forest->solver encoding (OCEAN's model / calf's compile) is built ONCE
    and reused across the job's queries — this is what makes the per-query solve
    time apples-to-apples and avoids rebuilding big models per query.  Each query
    writes its own shard immediately, so a hard-kill mid-job keeps the finished
    ones and a rerun redoes only the rest.
    """
    import joblib
    from pathlib import Path
    from calf.datasets import load
    from . import store

    run_dir = Path(run_dir)
    method = job["method"]
    rf = joblib.load(job["model_path"])
    # numeric_only must match _train_forest (categoricals dropped for the L1 search)
    X, _y, _di = load(job["dataset"], root=job["data_root"], numeric_only=True)

    # Holdout protocol: restrict param tuning (warm-up/seed) to the train split.
    pool_idx = None
    split_path = job.get("split_path")
    if split_path:
        import json
        pool_idx = np.asarray(
            json.loads(Path(split_path).read_text())["train_idx"], dtype=np.int64)

    explainer = None
    build_err = None
    try:
        explainer = build_explainer(method, rf, X, job, pool_idx=pool_idx)
    except Exception as exc:  # construction failed -> every query is an error
        build_err = f"{type(exc).__name__}: {exc}"

    for q_idx, target in job["queries"]:
        spec = {**{k: job[k] for k in _JOB_IDENTITY},
                "query_idx": int(q_idx), "target_class": int(target)}
        tid = store.task_id(spec)
        if store.is_done(run_dir, tid):
            continue
        if build_err is not None:
            store.write_shard(run_dir, tid, _record(spec, status="error", error=build_err))
            continue
        try:
            x = np.asarray(X[q_idx], dtype=np.float64)
            res = explainer.explain(x, int(target), time_limit_s=float(job["time_cap_s"]))
            rec = _result_record(spec, res, rf, method)
        except Exception as exc:  # per-query solve failure -> error, keep going
            rec = _record(spec, status="error", error=f"{type(exc).__name__}: {exc}")
        store.write_shard(run_dir, tid, rec)


def _result_record(spec: dict, res, rf, method: str) -> dict:
    """Convert a CFResult (+ real-forest validity + cap semantics) to a record."""
    # Ground-truth validity under the ACTUAL sklearn forest on the float32 grid
    # it evaluates queries on.  The parsed (lo,hi] float64 oracle that backends'
    # verify() uses is fooled by boundary-nudged points exactly as sklearn's
    # float32 cast is not — a CF on a split threshold can pass the parsed oracle
    # yet be re-routed by the real forest.  rf.predict is sklearn's native verdict.
    sklearn_valid = None
    if res.x_cf is not None:
        try:
            x32 = np.asarray(res.x_cf, dtype=np.float64).astype(np.float32).reshape(1, -1)
            sklearn_valid = bool(int(rf.predict(x32)[0]) == int(spec["target_class"]))
        except Exception:
            sklearn_valid = None

    anytime = ANYTIME.get(method, True)
    timed_out = (not res.is_optimal) and res.wall_time_s >= 0.9 * float(spec["time_cap_s"])

    # Non-anytime method that did not prove optimality within the cap: it failed.
    if not anytime and not res.is_optimal and res.status not in ("infeasible", "error"):
        return _record(
            spec, status="timeout", wall_time_s=res.wall_time_s,
            solve_time_s=res.solve_time_s, build_time_s=res.build_time_s,
            timed_out=True, error=res.error, n_nodes=res.n_nodes,
        )

    cost = res.cost
    gap = None
    if cost is not None and res.lower_bound is not None:
        gap = max(0.0, cost - res.lower_bound)
    return _record(
        spec,
        status=res.status,
        is_optimal=bool(res.is_optimal),
        cost=cost,
        lower_bound=res.lower_bound,
        upper_bound=res.upper_bound,
        gap=gap,
        n_nodes=res.n_nodes,
        reaches_target=res.reaches_target,
        sklearn_valid=sklearn_valid,
        solver_optimal=getattr(res, "solver_optimal", None),
        cert_note=getattr(res, "cert_note", None),
        wall_time_s=res.wall_time_s,
        solve_time_s=res.solve_time_s,
        build_time_s=res.build_time_s,
        timed_out=bool(timed_out),
        x_cf=None if res.x_cf is None else np.asarray(res.x_cf, dtype=np.float64).tolist(),
        error=res.error,
        extra=res.extra,
    )


def _record(spec: dict, **fields) -> dict:
    """Merge the task spec identity with result fields into one flat record."""
    from .store import IDENTITY_KEYS

    rec = {k: spec.get(k) for k in IDENTITY_KEYS}
    rec.setdefault("is_optimal", False)
    rec.setdefault("cost", None)
    rec.setdefault("lower_bound", None)
    rec.setdefault("upper_bound", None)
    rec.setdefault("gap", None)
    rec.setdefault("n_nodes", None)
    rec.setdefault("reaches_target", None)
    rec.setdefault("sklearn_valid", None)
    rec.setdefault("solver_optimal", None)
    rec.setdefault("cert_note", None)
    rec.setdefault("wall_time_s", None)
    rec.setdefault("solve_time_s", None)
    rec.setdefault("build_time_s", None)
    rec.setdefault("timed_out", False)
    rec.setdefault("x_cf", None)
    rec.setdefault("error", None)
    rec.update(fields)
    return rec
