"""Result storage: one JSON shard per task, aggregated to parquet/CSV on demand.

The store is the memoization layer.  Each ``(dataset, rf_variant, query, method)``
task hashes to a stable ``task_id``; its result lives at ``tasks/<task_id>.json``.
"Shard exists" == "task done", so resume is a directory check and aggregation is
a pure fold over the shards you can re-run any time — even mid-run.

Shards are written temp-then-rename so a crash or a kill can never leave a
half-written file that a resume would mistake for a completed task.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Keys that define a task's identity — the memoization key.  Changing any of
# these (a bigger forest, a longer cap, a different metric) yields a new
# task_id, so old results are preserved rather than silently overwritten.
# Deliberately excludes code version: editing a method does NOT invalidate its
# shards; re-run those explicitly with `--force-method`.
IDENTITY_KEYS = (
    "dataset",
    "rf_variant",
    "rf_n_estimators",
    "rf_max_depth",
    "rf_seed",
    "query_idx",
    "target_class",
    "method",
    "voting",
    "threshold",
    "time_cap_s",
    "metric",
    "norm",
)


def task_id(spec: dict) -> str:
    """Stable short id for a task spec (sha1 over the identity keys)."""
    payload = {k: spec.get(k) for k in IDENTITY_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def tasks_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "tasks"


def shard_path(run_dir: Path, tid: str) -> Path:
    return tasks_dir(run_dir) / f"{tid}.json"


def is_done(run_dir: Path, tid: str) -> bool:
    return shard_path(run_dir, tid).is_file()


def write_shard(run_dir: Path, tid: str, record: dict) -> None:
    """Atomically write one task's result record."""
    d = tasks_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{tid}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, default=_json_default)
        os.replace(tmp, shard_path(run_dir, tid))  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_shard(run_dir: Path, tid: str) -> dict | None:
    p = shard_path(run_dir, tid)
    if not p.is_file():
        return None
    with open(p) as fh:
        return json.load(fh)


def iter_shards(run_dir: Path) -> Iterable[dict]:
    d = tasks_dir(run_dir)
    if not d.is_dir():
        return
    for p in sorted(d.glob("*.json")):
        try:
            with open(p) as fh:
                yield json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # skip a torn/partial file; a rerun will replace it


def done_ids(run_dir: Path) -> set[str]:
    d = tasks_dir(run_dir)
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.json")}


# --- aggregation ---------------------------------------------------------

# Columns kept in the flat table (x_cf stays in the shard, not the CSV).
_TABLE_COLUMNS = (
    "dataset", "rf_variant", "rf_n_estimators", "rf_max_depth", "rf_seed",
    "query_idx", "target_class", "method", "voting", "threshold",
    "time_cap_s", "metric", "norm",
    "status", "is_optimal", "reaches_target", "sklearn_valid", "found",
    "solver_optimal", "cert_note",
    "cost", "lower_bound", "upper_bound", "gap", "n_nodes",
    "solve_time_s", "build_time_s", "wall_time_s", "timed_out", "error",
)


def _row_from_record(rec: dict) -> dict:
    out = {k: rec.get(k) for k in _TABLE_COLUMNS}
    cost = rec.get("cost")
    lb = rec.get("lower_bound")
    out["found"] = rec.get("x_cf") is not None or (cost is not None)
    if out.get("gap") is None and cost is not None and lb is not None:
        out["gap"] = max(0.0, cost - lb)
    return out


def aggregate(run_dir: Path, write: bool = True):
    """Fold all shards into a tidy DataFrame; optionally write parquet + CSV."""
    import pandas as pd

    rows = [_row_from_record(r) for r in iter_shards(run_dir)]
    df = pd.DataFrame(rows, columns=list(_TABLE_COLUMNS))
    if write and len(df):
        run_dir = Path(run_dir)
        df.to_parquet(run_dir / "aggregate.parquet", index=False)
        df.to_csv(run_dir / "aggregate.csv", index=False)
    return df


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)
