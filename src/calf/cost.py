"""Weighted-L1 cost scaling.

The search minimizes  cost(x) = sum_f scale[f] * |x[f] - factual[f]|.
``scale`` folds together a per-feature weight and a normalizer (default: the
feature's data range, so movement is measured in fraction-of-range units).
Immutable features are expressed as a very large weight.
"""
from __future__ import annotations

import numpy as np

from .dataset_info import DatasetInfo


def l1_scale(dataset_info: DatasetInfo, weights: np.ndarray | None = None) -> np.ndarray:
    """Per-feature cost scale = weight / range.  weights defaults to all ones."""
    ranges = np.array([f.range for f in dataset_info.features], dtype=np.float64)
    if np.any(ranges <= 0):
        raise ValueError("all feature ranges must be positive")
    if weights is None:
        weights = np.ones_like(ranges)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != ranges.shape:
            raise ValueError("weights and features must have the same length")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
    return np.ascontiguousarray(weights / ranges, dtype=np.float64)


def l1_cost(factual: np.ndarray, x: np.ndarray, scale: np.ndarray) -> float:
    """Scaled L1 distance between factual and x."""
    return float(np.sum(scale * np.abs(np.asarray(factual) - np.asarray(x))))
