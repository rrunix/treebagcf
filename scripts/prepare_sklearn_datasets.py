"""Write additional offline sklearn datasets into data/.

These datasets are binary classification tasks with numerical features only,
so they can be used directly by the current splitter and benchmark.

Usage:
    uv run python scripts/prepare_sklearn_datasets.py
    uv run python scripts/prepare_sklearn_datasets.py --out data
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer, load_digits


def _clean_name(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError("empty feature name")
    if name[0].isdigit():
        name = f"f_{name}"
    return name


def _write_numerical_dataset(
    *,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    notes: str,
) -> None:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError(f"{name}: expected 2D X, got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"{name}: inconsistent X/y shapes {X.shape}, {y.shape}")
    classes = sorted(np.unique(y).tolist())
    if classes != [0, 1]:
        raise ValueError(f"{name}: expected binary labels 0/1, got {classes}")
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"{name}: {len(feature_names)} names for {X.shape[1]} features")

    clean_names = []
    seen: dict[str, int] = {}
    for raw in feature_names:
        base = _clean_name(raw)
        count = seen.get(base, 0)
        seen[base] = count + 1
        clean_names.append(base if count == 0 else f"{base}_{count}")

    df = pd.DataFrame(X, columns=clean_names)
    df["y"] = y

    features = []
    for j, col_name in enumerate(clean_names):
        col = X[:, j]
        lo = float(np.min(col))
        hi = float(np.max(col))
        if lo == hi:
            hi = lo + 1.0
        features.append({"name": col_name, "kind": "numerical", "lo": lo, "hi": hi})

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    meta = {
        "name": name,
        "target": "y",
        "n_classes": 2,
        "n_samples": int(X.shape[0]),
        "features": features,
        "notes": notes,
    }
    with (out_dir / f"{name}.yaml").open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"{name}: {X.shape[0]} rows, {X.shape[1]} numerical features")


def _breast_cancer(out_dir: Path) -> None:
    ds = load_breast_cancer()
    _write_numerical_dataset(
        name="breast_cancer",
        X=ds.data,
        y=ds.target,
        feature_names=[str(v) for v in ds.feature_names],
        out_dir=out_dir,
        notes="sklearn.datasets.load_breast_cancer; binary target 0=malignant, 1=benign.",
    )


def _digits_pair(out_dir: Path, a: int, b: int) -> None:
    ds = load_digits()
    mask = np.isin(ds.target, [a, b])
    X = ds.data[mask]
    y = (ds.target[mask] == b).astype(np.int64)
    feature_names = [f"pixel_{i:02d}" for i in range(X.shape[1])]
    _write_numerical_dataset(
        name=f"digits_{a}{b}",
        X=X,
        y=y,
        feature_names=feature_names,
        out_dir=out_dir,
        notes=f"sklearn.datasets.load_digits restricted to digits {a} and {b}; target 1 means digit {b}.",
    )


def _digits_even(out_dir: Path) -> None:
    ds = load_digits()
    y = (ds.target % 2 == 0).astype(np.int64)
    feature_names = [f"pixel_{i:02d}" for i in range(ds.data.shape[1])]
    _write_numerical_dataset(
        name="digits_even",
        X=ds.data,
        y=y,
        feature_names=feature_names,
        out_dir=out_dir,
        notes="sklearn.datasets.load_digits; binary target 1 means even digit.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    _breast_cancer(out_dir)
    _digits_pair(out_dir, 3, 8)
    _digits_even(out_dir)


if __name__ == "__main__":
    main()
