"""Smoke test: run every available backend on one instance and cross-check.

Trains a tiny hard-voting RF, runs each available baseline on a single query,
prints a ``CFResult`` table, and asserts that every method claiming optimality
agrees on the objective (within tolerance) with the brute-force oracle.

Run:  python -m baselines.smoke_test
      python baselines/smoke_test.py

Unavailable backends (no Gurobi / OCEAN not installed) are reported as SKIPPED,
not failed — the license-free path (calf, brute_force, counterfactual_maps,
and OCEAN CP if installed) must still agree.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow running as a bare script (python baselines/smoke_test.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from baselines import OPTIMAL_METHODS, build_available
from baselines import metrics


def tiny_forest(seed: int = 0):
    """A small forest and one query the *hard-voting* forest classifies as 0."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0, random_state=seed
    )
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=seed).fit(X, y)
    preds = metrics.hard_vote_predict(rf, X)
    idx = np.where(preds == 0)[0]
    if idx.size == 0:
        raise RuntimeError("no class-0 instance under hard voting; try another seed")
    return rf, X, X[idx[0]], 1  # target_class = 1


def _fmt(v, width=10):
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        if math.isinf(v):
            return "inf".rjust(width)
        return f"{v:.6f}".rjust(width)
    return str(v).rjust(width)


def run(seed: int = 0, time_limit_s: float = 30.0) -> int:
    rf, X, x, target = tiny_forest(seed)
    print(
        f"tiny hard-voting RF: n_trees={len(rf.estimators_)} "
        f"n_features={rf.n_features_in_} target_class={target} "
        f"need={metrics.need(len(rf.estimators_))}\n"
    )

    # Compare all methods under plain L1 (unit feature ranges).  OCEAN minimises
    # L1 in its (unscaled) mapper space, i.e. raw-unit L1, so a shared plain-L1
    # metric lets its exact optimum be cross-checked against the oracle; with
    # non-uniform range normalisation OCEAN would optimise a different objective
    # and be (correctly) excluded from the agreement assertion.
    feature_ranges = np.ones(rf.n_features_in_, dtype=float)
    statuses = build_available(
        rf, X=X, feature_ranges=feature_ranges, voting="hard", norm=1,
        per_backend_kwargs={"calf": {"max_iters": 500_000}},
    )

    header = f"{'method':22} {'status':12} {'optimal':8} {'cost':>10} {'reaches':8} {'time_s':>9}  notes"
    print(header)
    print("-" * len(header))

    results = {}
    for st in statuses:
        if not st.available:
            reason = " ".join(st.reason.split())[:110]
            print(f"{st.key:22} {'SKIPPED':12} {'-':8} {'-':>10} {'-':8} {'-':>9}  {reason}")
            continue
        res = st.explainer.explain(x, target, time_limit_s=time_limit_s)
        results[st.key] = res
        notes = res.error or ""
        if not notes and res.extra.get("voting_semantics") == "soft":
            notes = "soft vote — excluded from hard-voting agreement check"
        print(
            f"{st.key:22} {res.status:12} {str(res.is_optimal):8} "
            f"{_fmt(res.cost)} {str(res.reaches_target):8} {_fmt(res.wall_time_s, 9)}  {notes[:60]}"
        )

    print()
    return _check(results)


def _check(results: dict) -> int:
    """Assert optimal methods agree with the brute-force oracle; return exit code."""
    if "brute_force" not in results:
        print("FAIL: brute_force oracle did not run")
        return 1
    oracle = results["brute_force"]
    if oracle.status not in ("optimal", "infeasible"):
        print(f"FAIL: brute_force did not solve (status={oracle.status})")
        return 1

    ok = True
    tol = 1e-6
    for key, res in results.items():
        if key not in OPTIMAL_METHODS or not res.is_optimal:
            continue
        if res.reaches_target is False:
            print(f"FAIL: {key} returned a point that does not reach the target under hard voting")
            ok = False
        if oracle.status == "infeasible":
            if res.status != "infeasible":
                print(f"FAIL: {key} found a CF but the oracle proved infeasible")
                ok = False
            continue
        if res.cost is None or abs(res.cost - oracle.cost) > tol:
            print(
                f"FAIL: {key} optimal cost {res.cost} disagrees with oracle {oracle.cost} "
                f"(|Δ|={abs((res.cost or math.inf) - oracle.cost):.2e} > {tol})"
            )
            ok = False

    if ok:
        n_opt = sum(1 for k, r in results.items() if k in OPTIMAL_METHODS and r.is_optimal)
        print(f"OK: {n_opt} optimal method(s) agree with the brute-force oracle "
              f"(cost={_fmt(oracle.cost).strip()}).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
