"""Config-driven, parallel, resumable experiment runner.

Flow:
1. Read the sweep config (YAML).
2. Train + cache one RF per (dataset, rf_variant); resolve query indices.
3. Freeze the fully-resolved config into the run dir (reproducibility).
4. Enumerate (dataset, variant, query, method) tasks; skip any whose shard
   already exists (resume).
5. Run the rest with a process-per-task scheduler: each task runs in its own
   spawned child (crash isolation) and is hard-killed if it blows past
   ``time_cap * kill_slack + grace`` — the safety net for a genuine solver hang.
   Cooperative caps (passed to each method) are primary and usually return well
   before the kill.

Each task writes its own shard, so Ctrl-C at any point is safe: rerun to
continue exactly where it stopped.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from . import methods, store


def _worker(job: dict, run_dir_str: str) -> None:
    """Child-process entry: build one explainer and solve all the job's queries."""
    from .methods import run_job
    run_job(job, Path(run_dir_str))


# --- config + preparation -------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("metric", "plain_l1")
    cfg.setdefault("norm", 1)
    cfg.setdefault("threshold", 0.5)
    cfg.setdefault("seed", 0)
    cfg.setdefault("kill_slack", 1.5)
    cfg.setdefault("kill_grace_s", 30.0)
    # Train/test protocol.  ``test_frac == 0`` (default) means no holdout: the
    # forest trains on the whole frame and queries are drawn from it — the
    # original behaviour, so pre-existing configs are unchanged.  ``test_frac >
    # 0`` fits the forest on a stratified train split, and the ``test_sample``
    # query strategy draws factuals from the held-out test split only; method
    # params (calf's warm-up + incumbent seed) then see train rows only.
    cfg.setdefault("test_frac", 0.0)
    cfg.setdefault("split_seed", cfg["seed"])
    return cfg


def _split_indices(y: np.ndarray, test_frac: float, split_seed: int):
    """(train_idx, test_idx), stratified by label.  test_frac<=0 -> both = all."""
    idx = np.arange(len(y))
    if test_frac <= 0.0:
        return idx, idx  # no holdout: train on all, query pool is all (legacy)
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        idx, test_size=float(test_frac), random_state=int(split_seed), stratify=y
    )
    return np.sort(train_idx), np.sort(test_idx)


def _train_forest(dataset: str, variant: str, rf_kwargs: dict, seed: int,
                  data_root: str, models_dir: Path, test_frac: float,
                  split_seed: int):
    """Train (or load cached) one RF on the train split.

    Returns ``(model_path, split_path, rf_params, X, preds, train_idx,
    test_idx)``.  The forest is fit on ``X[train_idx]`` only; ``preds`` are over
    the whole frame (used to classify test factuals).  The split is cached next
    to the model so a resume reuses the identical partition.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from calf.datasets import load

    # numeric_only: the L1 search treats every feature as continuous, so drop
    # categorical columns (a fractional category is meaningless).  Both the
    # forest here and the query eval in methods.py must use the SAME feature set.
    X, y, _di = load(dataset, root=data_root, numeric_only=True)
    # Split params in the filename so a frac/seed change never reuses a stale
    # (e.g. full-data) model within a reused run dir.
    tag = f"tf{test_frac:g}s{split_seed}" if test_frac > 0 else "full"
    path = models_dir / f"{dataset}__{variant}__{tag}.joblib"
    split_path = models_dir / f"{dataset}__{variant}__{tag}.split.json"
    if path.is_file() and split_path.is_file():
        rf = joblib.load(path)
        sp = json.loads(split_path.read_text())
        train_idx = np.asarray(sp["train_idx"], dtype=np.int64)
        test_idx = np.asarray(sp["test_idx"], dtype=np.int64)
    else:
        train_idx, test_idx = _split_indices(y, test_frac, split_seed)
        rf = RandomForestClassifier(random_state=seed, **rf_kwargs).fit(
            X[train_idx], y[train_idx]
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, path)
        split_path.write_text(json.dumps({
            "test_frac": test_frac, "split_seed": int(split_seed),
            "n_train": int(train_idx.size), "n_test": int(test_idx.size),
            "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(),
        }))
    params = {
        "rf_n_estimators": int(rf.n_estimators),
        "rf_max_depth": rf_kwargs.get("max_depth"),
        "rf_seed": int(seed),
    }
    return path, split_path, params, X, rf.predict(X), train_idx, test_idx


def _resolve_queries(preds: np.ndarray, strategy: str, qcfg: dict,
                     pool: np.ndarray, split_seed: int) -> list[tuple[int, int]]:
    """Return [(query_idx, target_class), ...], selected from ``pool``.

    ``first_per_pred_class`` (legacy): first ``n_per_class`` of each predicted
    class in natural index order — with ``pool = all`` this reproduces the
    pre-split behaviour exactly.  ``test_sample`` (holdout protocol): up to
    ``n_queries`` factuals drawn from the (test) pool, balanced across the two
    predicted classes, in a seeded shuffle so the subsample is unbiased.  The
    target is always the flip of the model's current prediction.
    """
    pool = np.asarray(pool, dtype=np.int64)
    if strategy == "first_per_pred_class":
        n_per_class = int(qcfg["n_per_class"])
        out = []
        for pred_cls in (0, 1):
            sel = [int(i) for i in pool if preds[i] == pred_cls][:n_per_class]
            out.extend((i, 1 - pred_cls) for i in sel)
        return out
    if strategy == "test_sample":
        n = int(qcfg["n_queries"])
        order = pool.copy()
        np.random.default_rng(int(split_seed)).shuffle(order)
        by = {0: [], 1: []}
        for i in order:
            by[int(preds[i])].append(int(i))
        half = n // 2
        chosen = by[0][:half] + by[1][:half]
        rest = by[0][half:] + by[1][half:]
        chosen += rest[: max(0, n - len(chosen))]
        return [(int(i), 1 - int(preds[i])) for i in chosen[:n]]
    raise ValueError(f"unknown query strategy {strategy!r}")


def enumerate_jobs(cfg: dict, run_dir: Path) -> list[dict]:
    """Prepare forests, resolve queries, build one job per (dataset, variant, method).

    A *job* bundles all of a forest's queries for one method so the worker builds
    the explainer once and reuses it (amortised build).  Each query still writes
    its own shard, so resume/task_id stay per-query.
    """
    data_root = cfg["data_root"]
    models_dir = run_dir / "models"
    qcfg = cfg["queries"]
    test_frac = float(cfg["test_frac"])
    split_seed = int(cfg["split_seed"])
    jobs: list[dict] = []
    resolved_queries: dict[str, list] = {}
    split_summary: dict[str, dict] = {}
    for dataset in cfg["datasets"]:
        for variant, rf_kwargs in cfg["rf"].items():
            path, split_path, params, _X, preds, train_idx, test_idx = _train_forest(
                dataset, variant, dict(rf_kwargs), cfg["seed"], data_root,
                models_dir, test_frac, split_seed,
            )
            # holdout protocol draws from test only; legacy draws from all
            pool = test_idx if test_frac > 0 else np.arange(len(preds))
            queries = _resolve_queries(preds, qcfg["strategy"], qcfg, pool, split_seed)
            key = f"{dataset}__{variant}"
            resolved_queries[key] = queries
            split_summary[key] = {
                "n_train": int(train_idx.size), "n_test": int(test_idx.size),
                "n_queries": len(queries),
            }
            for method, mkw in cfg["methods"].items():
                jobs.append({
                    "dataset": dataset,
                    "rf_variant": variant,
                    **params,
                    "method": method,
                    "voting": mkw.get("voting", "hard"),
                    "threshold": float(cfg["threshold"]),
                    "time_cap_s": float(cfg["time_cap_s"]),
                    "metric": cfg["metric"],
                    "norm": int(cfg["norm"]),
                    "queries": [(int(q), int(t)) for q, t in queries],
                    # non-identity fields (not hashed into task_id):
                    "model_path": str(path),
                    "split_path": str(split_path) if test_frac > 0 else None,
                    "data_root": data_root,
                    "method_kwargs": {k: v for k, v in mkw.items() if k != "voting"},
                })
    # freeze the resolved config for reproducibility
    frozen = dict(cfg)
    frozen["_resolved_queries"] = resolved_queries
    frozen["_split_summary"] = split_summary
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(frozen, fh, sort_keys=False)
    return jobs


def _job_query_ids(job: dict) -> list[str]:
    """task_ids of every query in a job (for resume/skip accounting)."""
    ids = []
    for q, t in job["queries"]:
        spec = {**{k: job[k] for k in methods._JOB_IDENTITY},
                "query_idx": q, "target_class": t}
        ids.append(store.task_id(spec))
    return ids


# --- scheduling -----------------------------------------------------------

def run(cfg: dict, run_dir: Path, workers: int, force: bool = False,
        force_methods: set[str] | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = enumerate_jobs(cfg, run_dir)

    force_methods = force_methods or set()
    total_q = sum(len(j["queries"]) for j in jobs)
    todo, skipped = [], 0
    for job in jobs:
        ids = _job_query_ids(job)
        forced = force or job["method"] in force_methods
        done = 0 if forced else sum(store.is_done(run_dir, i) for i in ids)
        skipped += done
        if done < len(ids):
            todo.append((job, forced))

    print(f"[{cfg.get('experiment','exp')}] {total_q} query-tasks across {len(jobs)} jobs, "
          f"{skipped} already done, {len(todo)} jobs to run on {workers} workers")
    if not todo:
        store.aggregate(run_dir)
        print("nothing to run; aggregate refreshed")
        return

    ctx = mp.get_context("spawn")
    kill_slack = float(cfg["kill_slack"])
    grace = float(cfg["kill_grace_s"])
    pending = deque(todo)
    active: dict = {}
    jobs_left = len(todo)
    jobs_done = 0

    def _wall(job):
        # build allowance + one cap per (remaining) query, with slack
        return len(job["queries"]) * job["time_cap_s"] * kill_slack + grace

    while pending or active:
        while pending and len(active) < workers:
            job, forced = pending.popleft()
            j = dict(job)
            if forced:  # redo every query: drop existing shards for this job
                for i in _job_query_ids(job):
                    store.shard_path(run_dir, i).unlink(missing_ok=True)
            p = ctx.Process(target=_worker, args=(j, str(run_dir)))
            p.start()
            active[p] = (job, time.time() + _wall(job))
        for p in [p for p in active if not p.is_alive()]:
            job, _ = active.pop(p)
            p.join()
            _finalize_job(run_dir, job, crashed=(p.exitcode not in (0, None)))
            jobs_done += 1
            _log_job(run_dir, job, jobs_done, jobs_left)
        now = time.time()
        for p in list(active):
            job, deadline = active[p]
            if now > deadline:
                p.terminate(); p.join()
                active.pop(p)
                _finalize_job(run_dir, job, killed=True)
                jobs_done += 1
                _log_job(run_dir, job, jobs_done, jobs_left, killed=True)
        time.sleep(0.05)

    df = store.aggregate(run_dir)
    print(f"done. wrote {run_dir/'aggregate.csv'} ({len(df)} rows)")
    from . import report
    print()
    print(report.render(df))


def _finalize_job(run_dir, job, crashed=False, killed=False):
    """Write error/timeout shards for any of a job's queries left unwritten."""
    for q, t in job["queries"]:
        spec = {**{k: job[k] for k in methods._JOB_IDENTITY},
                "query_idx": q, "target_class": t}
        tid = store.task_id(spec)
        if store.is_done(run_dir, tid):
            continue
        if killed:
            store.write_shard(run_dir, tid, methods._record(
                spec, status="timeout", timed_out=True, error="job hard-killed"))
        else:
            store.write_shard(run_dir, tid, methods._record(
                spec, status="error", error="worker crashed before writing"))


def _log_job(run_dir, job, n, total, killed=False):
    stats = {}
    for q, t in job["queries"]:
        spec = {**{k: job[k] for k in methods._JOB_IDENTITY},
                "query_idx": q, "target_class": t}
        rec = store.read_shard(run_dir, store.task_id(spec)) or {}
        stats[rec.get("status", "?")] = stats.get(rec.get("status", "?"), 0) + 1
    tag = "KILLED " if killed else ""
    summ = " ".join(f"{k}:{v}" for k, v in sorted(stats.items()))
    print(f"  [{n}/{total} jobs] {tag}{job['dataset']}/{job['method']} "
          f"({len(job['queries'])}q): {summ}")
