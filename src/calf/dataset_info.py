"""Feature metadata: per-feature numerical bounds used to seed the root box.

Numerical features only — the search branches on axis-aligned thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    lo: float
    hi: float

    @property
    def range(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True)
class DatasetInfo:
    features: tuple[FeatureSpec, ...]

    @property
    def n_features(self) -> int:
        return len(self.features)

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


def from_array(X, names: list[str] | None = None) -> DatasetInfo:
    """Infer per-feature [min, max] bounds from a 2D numpy array."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {X.shape}")
    n_features = X.shape[1]
    if names is None:
        names = [f"f{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError(f"names has {len(names)} entries, X has {n_features} columns")
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    feats = []
    for i, name in enumerate(names):
        a, b = float(lo[i]), float(hi[i])
        if b <= a:  # constant column: widen so the root box is non-degenerate
            b = a + 1.0
        feats.append(FeatureSpec(name=str(name), lo=a, hi=b))
    return DatasetInfo(features=tuple(feats))
