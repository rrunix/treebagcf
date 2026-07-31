"""Shared distance, projection, and hard-voting primitives.

Every baseline is scored on the *same* objective and the *same* voting rule so
the comparison is apples-to-apples.  These helpers are deliberately free of any
dependency on our search engine — they are the neutral yardstick.
"""
from __future__ import annotations

import math

import numpy as np


# --- cost / distance -----------------------------------------------------
def feature_ranges_from_X(X) -> np.ndarray:
    """Per-feature ``max - min``; constant columns widened to 1.0.

    Mirrors ``calf.dataset_info.from_array`` so the normalisation is identical
    to the one our own method uses.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {X.shape}")
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = hi - lo
    rng[rng <= 0] = 1.0  # constant column: avoid zero-division, match from_array
    return rng


def l1_scale(feature_ranges: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """scale[f] = weight[f] / range[f]  (weights default to all ones)."""
    ranges = np.asarray(feature_ranges, dtype=np.float64)
    if np.any(ranges <= 0):
        raise ValueError("all feature_ranges must be positive")
    if weights is None:
        weights = np.ones_like(ranges)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != ranges.shape:
            raise ValueError("weights and feature_ranges must have the same shape")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
    return np.ascontiguousarray(weights / ranges, dtype=np.float64)


def scaled_l1(factual: np.ndarray, x: np.ndarray, scale: np.ndarray) -> float:
    """Weighted / range-normalised L1 distance."""
    return float(np.sum(scale * np.abs(np.asarray(factual) - np.asarray(x))))


def box_distance(
    factual: np.ndarray, lo: np.ndarray, hi: np.ndarray, scale: np.ndarray
) -> float:
    """Scaled-L1 distance from ``factual`` to the axis-aligned box ``(lo, hi]``.

    This is the exact minimum of :func:`scaled_l1` over the box (the continuous
    relaxation at the open lower endpoint — a measure-zero difference, matching
    the engine's LB convention).
    """
    d_below = np.maximum(lo - factual, 0.0)
    d_above = np.maximum(factual - hi, 0.0)
    return float(np.sum(scale * (d_below + d_above)))


def project_onto_box(
    factual: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Closest point of ``(lo, hi]`` to ``factual`` (L1-optimal, per-axis clip).

    The box is half-open on the left (x > lo); if the clip lands on ``lo`` we
    nudge up by a tiny epsilon to stay strictly inside, exactly like
    ``calf.numba.kernels.refine_l1`` / the MILP reconstruction.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    x_prime = np.minimum(np.maximum(np.asarray(factual, dtype=np.float64), lo), hi)
    on_lo = x_prime <= lo
    if np.any(on_lo):
        eps = np.maximum(np.abs(lo[on_lo]), 1.0) * 1e-12
        x_prime[on_lo] = lo[on_lo] + eps
        x_prime = np.minimum(x_prime, hi)
    return x_prime


# --- hard voting ---------------------------------------------------------
# Ground-truth voting uses the PARSED rule ensemble, not sklearn's ``predict``.
# The whole project (search engine, MILP baseline) treats the parsed forest —
# half-open ``(lo, hi]`` boxes in float64 — as the oracle.  sklearn evaluates
# split thresholds in float32, so at a decision boundary the two disagree by a
# hair; using parsed semantics keeps every baseline consistent with each other
# and with the certified optimum (see research/milp_baseline.py::point_votes).
def need(n_trees: int, threshold: float = 0.5) -> int:
    """Trees that must vote the target class: ceil(threshold * n_trees)."""
    return int(math.ceil(threshold * n_trees))


def _parsed_of(rf):
    """Return a cached parsed view of ``rf`` (calf semantics)."""
    cached = getattr(rf, "_baselines_parsed", None)
    if cached is None:
        import calf

        cached = calf.parse_sklearn_rf(rf)
        try:
            rf._baselines_parsed = cached
        except Exception:  # pragma: no cover - read-only estimator
            pass
    return cached


def _fired_class_per_tree(parsed, x: np.ndarray) -> dict[int, int]:
    """Map tree_id -> fired leaf's class for a single point under (lo, hi]."""
    inside = np.all(x > parsed.rules_lo_mat, axis=1) & np.all(x <= parsed.rules_hi_mat, axis=1)
    seen: dict[int, int] = {}
    for rid in np.flatnonzero(inside):
        t = int(parsed.rules_tree_id[rid])
        if t not in seen:  # exactly one rule fires per tree; guard boundary dups
            seen[t] = int(parsed.rules_class[rid])
    return seen


def target_vote_count(rf, X: np.ndarray, target_class: int) -> np.ndarray:
    """Number of trees (per row of ``X``) whose hard vote is ``target_class``."""
    parsed = _parsed_of(rf)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    out = np.empty(X.shape[0], dtype=np.int64)
    for i, x in enumerate(X):
        seen = _fired_class_per_tree(parsed, x)
        out[i] = sum(1 for c in seen.values() if c == int(target_class))
    return out


def hard_vote_predict(rf, X: np.ndarray) -> np.ndarray:
    """Forest hard-voting prediction (argmax of per-tree vote counts).

    Uses parsed ``(lo, hi]`` rule semantics — the project's oracle — not
    sklearn's soft-voting ``rf.predict``.  Equal to ``parsed.predict``.
    """
    parsed = _parsed_of(rf)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n_classes = parsed.n_classes
    out = np.empty(X.shape[0], dtype=np.int64)
    for i, x in enumerate(X):
        seen = _fired_class_per_tree(parsed, x)
        votes = np.zeros(n_classes, dtype=np.int64)
        for c in seen.values():
            votes[c] += 1
        out[i] = int(votes.argmax())
    return out


def reaches_target(rf, x: np.ndarray, target_class: int, threshold: float = 0.5) -> bool:
    """True iff ``x`` gets >= need target votes under hard voting."""
    votes = int(target_vote_count(rf, np.atleast_2d(x), target_class)[0])
    return votes >= need(len(rf.estimators_), threshold)
