"""Cross-query warm-start library of dual share matrices (the "alpha library").

The cost-splitting dual bound is valid for EVERY fixed feasible share matrix,
regardless of the factual — the factual enters the bound only through the
movement requirements, never through the feasibility of the shares (and the
soft entries' lam >= 0 is likewise factual-independent).  Share matrices
optimized at one query are therefore admissible at any other query of the same
forest; only their tightness varies, and tightness transfers well between
nearby factuals (similar movement directions -> similar binding leaves).

The library exploits that transfer within a dataset epoch:

- ``warmup`` pre-populates it from k-means centroids of the rows the forest
  currently predicts away from the target class (the population future queries
  are drawn from), one root-level dual ascent per centroid — so even the first
  query starts warm.
- ``calf.solve(..., alpha_library=lib)`` seeds each query's pool with the
  entries strongest at THAT query's root box (see ``DualCostSplitPool.seed``)
  and harvests the query's own optimized entries back afterward, so the
  library keeps adapting to the actual query distribution.

Entries are keyed by target class (the target rule set, and hence the optimal
shares, differ per class).  They transfer across ``threshold``/``need`` and
even across cost ``weights``: the column caps ``sum_t alpha[t, f] <= 1`` are
all that admissibility needs.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .dual_lb import AdditiveDualPool, DualCostSplitPool
from .parser import ParsedRF

__all__ = ["AlphaLibrary"]


class AlphaLibrary:
    """FIFO-capped store of transferred dual entries for one (forest, voting).

    Hard entries are ``(alpha_pos, alpha_neg)`` pairs, soft entries
    ``(alpha_pos, alpha_neg, lam)`` triples, both with full
    ``(n_trees, n_features)`` share matrices — exactly the pool's own format.
    """

    def __init__(self, parsed_rf: ParsedRF, scale: np.ndarray, *,
                 voting: str = "hard", cap: int = 64):
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        self.parsed = parsed_rf
        self.scale = np.ascontiguousarray(scale, dtype=np.float64)
        self.voting = voting
        self.cap = int(cap)
        self._entries: dict[int, list[tuple]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def entries_for(self, target_class: int) -> list[tuple]:
        """Entries for one target class (a copy; safe to hand to the engine)."""
        return list(self._entries.get(int(target_class), ()))

    def save(self, path) -> None:
        """Persist the entries to ``path`` (joblib, like the cached forests).

        Only the share matrices travel — the forest itself is NOT stored (the
        model file already is); ``load`` re-binds them to a parsed forest.  The
        write is atomic (tmp + rename) so concurrent readers never see a
        partial file.
        """
        import joblib

        path = Path(path)
        payload = {
            "format": 1,
            "voting": self.voting,
            "cap": self.cap,
            "n_trees": self.parsed.n_trees,
            "n_features": self.parsed.n_features,
            "entries": {tc: list(entries) for tc, entries in self._entries.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        joblib.dump(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, parsed_rf: ParsedRF, scale: np.ndarray) -> "AlphaLibrary":
        """Rebuild a library from :meth:`save` output, bound to this forest.

        The entries are only meaningful for the forest they were optimized on;
        the (n_trees, n_features) shape is checked as a cheap guard against
        pointing at the wrong file.
        """
        import joblib

        payload = joblib.load(path)
        if (payload["n_trees"], payload["n_features"]) != (
            parsed_rf.n_trees, parsed_rf.n_features
        ):
            raise ValueError(
                f"alpha file {path} was built for a "
                f"{payload['n_trees']}-tree/{payload['n_features']}-feature "
                f"forest; got {parsed_rf.n_trees}/{parsed_rf.n_features}"
            )
        lib = cls(parsed_rf, scale, voting=payload["voting"], cap=payload["cap"])
        lib._entries = {
            int(tc): [tuple(e) for e in entries]
            for tc, entries in payload["entries"].items()
        }
        return lib

    def harvest(self, pool, target_class: int) -> None:
        """Absorb a query pool's entries (skipping ones seeded from here)."""
        bucket = self._entries.setdefault(int(target_class), [])
        fresh = pool.entries if self.voting == "soft" else pool.alphas
        for entry in fresh:
            seeded_back = False
            for old in bucket:
                if old[0] is entry[0]:  # identity: this entry came from us
                    seeded_back = True
                    break
            if not seeded_back:
                bucket.append(tuple(entry))
        while len(bucket) > self.cap:
            bucket.pop(0)

    def warmup(self, X, target_class: int, *, k: int = 16, threshold: float = 0.5,
               root_iters: int = 1000, random_state: int = 0,
               preds: np.ndarray | None = None, lp: bool = False) -> int:
        """Pre-populate from k-means centroids of the non-target-predicted rows.

        Each centroid (snapped to the float32 grid the engine queries on) gets
        one root-box dual ascent; the resulting entries are harvested.  The
        per-centroid incumbent — the cheapest row already predicted as the
        target — only sizes the Polyak step, so an imperfect one is harmless.
        Returns the number of entries added.

        ``lp`` (soft voting only): additionally solve the exact root LP at
        each centroid and harvest its (repaired, admissible) optimizer too —
        measured 2026-07-11 at 4-6x the ascent's root bound for well under a
        second per centroid.  Warm-up is a build-time step, so the extra cost
        is off the query path; note the 120s ablations could not attribute a
        certified-outcome gain to it beyond the warm-up itself (in-engine LP
        delivery was pruned outright for that reason, 2026-07-12).

        ``preds``: per-row predicted classes.  Pass ``model.predict(X)`` when
        the sklearn model is at hand — the parsed-forest fallback is a pure-
        Python oracle that is prohibitively slow on large X × big forests
        (found 2026-07-11: it hard-killed the abalone warm arm; the class
        split only steers centroid placement, so sklearn's float32 verdict is
        more than good enough).
        """
        X = np.asarray(X, dtype=np.float64)
        parsed = self.parsed
        preds = np.asarray(preds) if preds is not None else parsed.predict(X)
        rows = X[preds != target_class]
        if rows.shape[0] == 0:
            rows = X
        k = max(1, min(int(k), rows.shape[0]))
        if k == rows.shape[0]:
            centroids = rows.copy()
        else:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, n_init=4, random_state=random_state)
            km.fit(rows * self.scale)
            centroids = km.cluster_centers_ / self.scale
        centroids = centroids.astype(np.float32).astype(np.float64)

        # Root box + active set, matching engine._dataset_box / root_active.
        box_lo = X.min(axis=0) - 1e-12
        box_hi = X.max(axis=0)
        compat = (
            (parsed.rules_hi_mat > box_lo) & (box_hi > parsed.rules_lo_mat)
        ).all(axis=1)
        active = np.where(compat)[0].astype(np.int64)
        shim = SimpleNamespace(
            active_rules=active, box=SimpleNamespace(lo=box_lo, hi=box_hi)
        )
        targets = X[preds == target_class].astype(np.float32).astype(np.float64)

        n_before = len(self)
        for c in centroids:
            c = np.ascontiguousarray(c)
            incumbent = math.inf
            if targets.shape[0]:
                incumbent = float(
                    (np.abs(targets - c) * self.scale).sum(axis=1).min()
                )
            if self.voting == "soft":
                p1 = parsed.rules_proba1.astype(np.float64)
                values = p1 if target_class == 1 else 1.0 - p1
                pool = AdditiveDualPool(
                    parsed, c, self.scale, values, 0.5 * parsed.n_trees
                )
            else:
                need = int(math.ceil(threshold * parsed.n_trees))
                pool = DualCostSplitPool(parsed, c, self.scale, target_class, need)
            bound = pool.optimize_root(shim, incumbent=incumbent, max_iters=root_iters)
            if lp and self.voting == "soft":
                pool.lp_optimize_at(shim, time_limit_s=30.0)
            if len(pool) and math.isfinite(bound):
                self.harvest(pool, target_class)
        return len(self) - n_before
