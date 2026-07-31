"""Our own method (``calf``) exposed through the common baseline interface.

This is not a "baseline" — it is the paper's method — but wrapping it here lets
the smoke test and the harness run it side by side with the competitors on the
identical objective, and it is the one certified-optimal method available
without any solver install (so it anchors the cross-check).

The engine enforces both ``max_iters`` and the wall-clock ``time_limit_s``
(anytime: on expiry it returns the best incumbent with its certified gap).
The reported bracket is exact: ``lower_bound = cost - optimality_gap`` and
``is_optimal`` iff the gap is certified zero.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

import calf

from .base import BaselineExplainer, CFResult


class CALFExplainer(BaselineExplainer):
    name = "calf"

    def _configure(self, max_iters: int = 1_000_000, engine: str = "python",
                   alpha_warm: bool = False, alpha_warm_k: int = 16,
                   alpha_warm_iters: int = 1000, alpha_warm_lp: bool = False,
                   alpha_cache: str | None = None, pool_idx=None,
                   **engine_kwargs) -> None:
        self.max_iters = int(max_iters)
        # Holdout protocol: rows the method may use to build query-independent
        # params (alpha warm-up library, incumbent seed).  None -> the whole
        # frame (legacy).  The feature-domain box (box_lo/box_hi from the full X
        # in the base ctor) is deliberately NOT restricted: it is the search
        # space and must contain held-out test factuals, not a learned quantity.
        self._pool_idx = None if pool_idx is None else np.asarray(pool_idx, dtype=np.int64)
        # `engine` is accepted for config compatibility but ignored: since
        # 2026-07-09 there is a single engine (python A* over numba kernels).
        self.engine_kwargs = engine_kwargs
        self._alpha_warm = bool(alpha_warm)
        self._alpha_warm_k = int(alpha_warm_k)
        self._alpha_warm_iters = int(alpha_warm_iters)
        self._alpha_warm_lp = bool(alpha_warm_lp)
        self._alpha_cache = alpha_cache
        self._alpha_lib = None
        # Parse + compile once (our forest->arrays "build", query-independent);
        # reused across queries on the same forest.  Timed so it is reported as
        # build_time_s, analogous to OCEAN's model build.
        _tb = time.perf_counter()
        self.parsed = calf.parse_sklearn_rf(self.rf)
        self.compiled = calf.compile_rf(self.parsed)
        self._build_time_s = time.perf_counter() - _tb
        self._dataset_info = calf.DatasetInfo(
            features=tuple(
                calf.FeatureSpec(name=f"f{i}", lo=float(self.box_lo[i] + 1e-12), hi=float(self.box_hi[i]))
                for i in range(self.n_features)
            )
        ) if self.box_lo is not None else None
        # For soft-voting incumbent seeding (query-independent part): the training
        # rows the forest already predicts as each class, snapped to sklearn's
        # float32 grid.  Without a seed the soft search can run millions of iters
        # incumbent-free (see calf.solve); the seed is a soft-feasible CF by
        # construction (rf.predict IS the soft vote) and is box-refined per query.
        self._pred = None
        if self.X is not None:
            # Seed/warm-up pool: train rows only under the holdout protocol.
            pool = (self._pool_idx if self._pool_idx is not None
                    else np.arange(self.X.shape[0]))
            self._pool = pool
            self._Xpool = np.asarray(self.X[pool], dtype=np.float64)
            self._pred = self.rf.predict(self._Xpool)
            self._X32 = self._Xpool.astype(np.float32).astype(np.float64)
        # Alpha warm-up (opt-in): a library of dual share matrices shared across
        # this forest's queries, cached on disk next to the model file (same
        # joblib convention).  Building it is a query-independent forest
        # artifact, so its wall time is charged to build_time_s like the parse.
        if self._alpha_warm:
            if self.X is None:
                raise ValueError("alpha_warm requires X at construction")
            _tw = time.perf_counter()
            cache = Path(self._alpha_cache) if self._alpha_cache else None
            if cache is not None and cache.is_file():
                self._alpha_lib = calf.AlphaLibrary.load(cache, self.parsed, self.scale)
            else:
                self._alpha_lib = calf.AlphaLibrary(
                    self.parsed, self.scale, voting=self.voting
                )
                for tc in range(int(self.rf.n_classes_)):
                    self._alpha_lib.warmup(
                        self._Xpool, tc, k=self._alpha_warm_k,
                        threshold=self.threshold,
                        root_iters=self._alpha_warm_iters,
                        lp=self._alpha_warm_lp,
                        preds=self._pred,  # rf.predict over the pool; the parsed
                                           # fallback is too slow on big X
                    )
                if cache is not None:
                    self._alpha_lib.save(cache)
            self._build_time_s += time.perf_counter() - _tw

    def _soft_initial_ub(self, x: np.ndarray, target_class: int):
        """Cheapest target-predicted training row toward x, box-refined (or None)."""
        if self._pred is None:
            return None
        from calf.numba.kernels import refine_l1

        mask = self._pred == target_class
        if not np.any(mask):
            return None
        x32 = x.astype(np.float32).astype(np.float64)
        cand = self._X32[mask]
        costs = (np.abs(cand - x32) * self.scale).sum(axis=1)
        j = int(np.argmin(costs))
        seed_x, seed_cost = cand[j], float(costs[j])
        inside = ((seed_x > self.parsed.rules_lo_mat)
                  & (seed_x <= self.parsed.rules_hi_mat)).all(axis=1)
        if np.any(inside):
            b_lo = self.parsed.rules_lo_mat[inside].max(axis=0)
            b_hi = self.parsed.rules_hi_mat[inside].min(axis=0)
            ref_x, ref_cost = refine_l1(np.ascontiguousarray(b_lo),
                                        np.ascontiguousarray(b_hi), x32, self.scale)
            if ref_cost < seed_cost and int(self.rf.predict(ref_x[None, :])[0]) == target_class:
                seed_x, seed_cost = ref_x, float(ref_cost)
        return seed_x, seed_cost

    def explain(
        self, x: np.ndarray, target_class: int, time_limit_s: float = 60.0
    ) -> CFResult:
        if self._dataset_info is None:
            raise ValueError("CALFExplainer requires X (root box) at construction")
        x = np.asarray(x, dtype=np.float64)
        kw = dict(self.engine_kwargs)
        # Seed the soft search with an initial upper bound (as calf.solve does);
        # without it the soft search can run incumbent-free for millions of iters.
        if self.voting == "soft" and "initial_ub" not in kw:
            seed = self._soft_initial_ub(x, int(target_class))
            if seed is not None:
                kw["initial_ub"] = seed
        pool_out: list = []
        if self._alpha_lib is not None:
            # Seed this query's dual pool from the library and harvest its
            # optimized entries back afterward (queries of a job run
            # sequentially in one process, so the library accumulates).
            kw.setdefault(
                "dual_warm_entries", self._alpha_lib.entries_for(int(target_class))
            )
            kw.setdefault("dual_pool_out", pool_out)
        t0 = time.perf_counter()
        try:
            res = calf.extract_counterfactual(
                self.parsed, self._dataset_info, x, int(target_class), self.scale,
                voting=self.voting, threshold=self.threshold, max_iters=self.max_iters,
                time_limit_s=time_limit_s, compiled_rf=self.compiled, **kw,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return CFResult(
                x_cf=None, cost=None, is_optimal=False, lower_bound=None,
                upper_bound=None, n_nodes=None,
                wall_time_s=time.perf_counter() - t0, status="error",
                method=self.name, error=f"{type(exc).__name__}: {exc}",
                build_time_s=self._build_time_s,
            )
        wall = time.perf_counter() - t0  # solve/inference only (compile is in the ctor)
        if self._alpha_lib is not None:
            for pool in pool_out:
                self._alpha_lib.harvest(pool, int(target_class))

        if not res.found:
            # No incumbent found.  found=False means one of: (a) the frontier
            # was exhausted -> certified infeasible (no CF exists); (b) the
            # iteration budget ran out; (c) the wall-clock deadline fired.  Only
            # (a) is a proof.  It is distinguished by having stopped BEFORE both
            # limits: a deadline stop consumes ~the whole cap, a budget stop
            # reaches max_iters, and only a genuine exhaustion returns early
            # under both.
            budget_hit = res.iters >= self.max_iters
            time_hit = time_limit_s is not None and wall >= 0.9 * float(time_limit_s)
            proved_infeasible = not (budget_hit or time_hit)
            return CFResult(
                x_cf=None, cost=None, is_optimal=proved_infeasible,
                lower_bound=None, upper_bound=None, n_nodes=res.iters,
                wall_time_s=wall,
                status="infeasible" if proved_infeasible else "timeout",
                method=self.name,
                build_time_s=self._build_time_s, solve_time_s=wall,
            )

        lb = res.cost - res.optimality_gap
        return CFResult(
            x_cf=np.asarray(res.x, dtype=np.float64),
            cost=float(res.cost),
            is_optimal=res.proven_optimal,
            lower_bound=float(lb),
            upper_bound=float(res.cost),
            n_nodes=res.iters,
            wall_time_s=wall,
            build_time_s=self._build_time_s, solve_time_s=wall,
            status="optimal" if res.proven_optimal else "feasible",
            method=self.name,
            reaches_target=self.verify(res.x, target_class),
            extra={"optimality_gap": res.optimality_gap},
        )
