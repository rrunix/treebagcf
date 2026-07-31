"""Read bundled datasets from <root>/<name>.parquet + <root>/<name>.yaml.

Default `<root>` is `./data` relative to the current working directory.
Override via the `root` argument.

The yaml schema:

    name: pima
    target: y                       # name of the target column in the parquet
    n_classes: 2
    features:
      - {name: pregnancies, kind: numerical, lo: 0.0, hi: 17.0}
      - {name: sex,        kind: categorical, categories: [M, F, I]}
      ...
    notes: optional human-readable provenance string

The parquet file holds the design matrix and the target column. Categorical
columns are stored as integer codes (0..len(categories)-1); the yaml's
`categories` list maps codes back to original labels (positionally).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..dataset_info import DatasetInfo, FeatureSpec


_DEFAULT_ROOT = Path("data")


def _resolve_root(root) -> Path:
    if root is None:
        return _DEFAULT_ROOT
    return Path(root)


def available(root=None) -> list[str]:
    """List dataset names discoverable under `root` (looks for paired
    <name>.parquet + <name>.yaml)."""
    base = _resolve_root(root)
    if not base.exists():
        return []
    names = set()
    for p in base.glob("*.yaml"):
        if (base / f"{p.stem}.parquet").exists():
            names.add(p.stem)
    return sorted(names)


def load(
    name: str, root=None, *, numeric_only: bool = False
) -> tuple[np.ndarray, np.ndarray, DatasetInfo]:
    """Load `<root>/<name>.parquet` + `<root>/<name>.yaml`.

    Returns `(X, y, dataset_info)` where:
      - X has shape (n_samples, n_features) and dtype float64.
      - y has shape (n_samples,) and dtype int64.
      - dataset_info carries the per-feature numerical bounds from the yaml.

    ``numeric_only`` drops the categorical feature columns and keeps only the
    numerical ones.  The L1 counterfactual search treats every feature as a
    continuous axis, which is meaningless for a categorical (it would return a
    fractional "0.5 of a category"); stripping them keeps the benchmark to
    features the method handles honestly.  Datasets left with too few numerical
    features after stripping should simply be excluded upstream.
    """
    base = _resolve_root(root)
    parquet_path = base / f"{name}.parquet"
    yaml_path = base / f"{name}.yaml"
    if not parquet_path.exists():
        raise FileNotFoundError(f"missing parquet: {parquet_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"missing metadata: {yaml_path}")

    with yaml_path.open("r") as f:
        meta = yaml.safe_load(f)

    target = meta["target"]
    feature_specs = meta["features"]
    if numeric_only:
        feature_specs = [s for s in feature_specs if s["kind"] == "numerical"]
        if not feature_specs:
            raise ValueError(
                f"{name}: numeric_only left no features (all categorical)"
            )

    df = pd.read_parquet(parquet_path)

    feature_names = [s["name"] for s in feature_specs]
    missing = set(feature_names + [target]) - set(df.columns)
    if missing:
        raise ValueError(
            f"{parquet_path} missing columns required by metadata: {sorted(missing)}"
        )

    X = df[feature_names].to_numpy(dtype=np.float64, copy=True)
    y = df[target].to_numpy(dtype=np.int64, copy=True)

    # DatasetInfo is numeric-only (the search branches on axis-aligned
    # thresholds); categorical columns survive in X only under
    # numeric_only=False and carry no metadata here.
    dataset_info = DatasetInfo(
        features=tuple(
            FeatureSpec(name=s["name"], lo=float(s["lo"]), hi=float(s["hi"]))
            for s in feature_specs
            if s["kind"] == "numerical"
        )
    )

    return X, y, dataset_info
