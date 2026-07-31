"""CLI for the experiment harness.

    # run a sweep (resumable — safe to Ctrl-C and rerun)
    uv run python research/run_suite.py run --config research/suite_configs/holdout_soft_120.yaml
    uv run python research/run_suite.py run --config … --workers 8 --force-method ocean_milp

    # (re)build the parquet + CSV table from shards, any time (even mid-run)
    uv run python research/run_suite.py aggregate --experiment holdout_soft_120
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent          # research/
_RESULTS = _ROOT / "results"


def _run_dir(experiment: str) -> Path:
    return _RESULTS / experiment


def cmd_run(args: argparse.Namespace) -> None:
    from exp_suite import runner, store  # noqa: F401

    cfg = runner.load_config(Path(args.config))
    # absolutize data_root (relative to the repo root) so workers resolve it
    # regardless of cwd
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = (_ROOT.parent / data_root).resolve()
    cfg["data_root"] = str(data_root)
    experiment = cfg.get("experiment") or Path(args.config).stem
    run_dir = _run_dir(experiment)
    workers = args.workers or cfg.get("workers") or max(1, (os.cpu_count() or 4) - 2)
    force_methods = set(args.force_method or [])
    runner.run(cfg, run_dir, workers=workers, force=args.force,
               force_methods=force_methods)


def cmd_aggregate(args: argparse.Namespace) -> None:
    from exp_suite import store

    from exp_suite import report

    run_dir = _run_dir(args.experiment)
    df = store.aggregate(run_dir)
    print(f"{len(df)} rows -> {run_dir/'aggregate.csv'}")
    if len(df):
        # a compact per-(method,status) tally so you can eyeball coverage
        tally = df.groupby(["method", "status"]).size().unstack(fill_value=0)
        print(tally.to_string())
        print()
        print(report.render(df))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run/resume a sweep")
    r.add_argument("--config", required=True)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--force", action="store_true", help="ignore existing shards (redo all)")
    r.add_argument("--force-method", action="append", help="redo only these methods")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("aggregate", help="(re)build aggregate.parquet/csv from shards")
    a.add_argument("--experiment", required=True)
    a.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
