# CALF — Certified Anytime Lagrangian Forest

Certified-optimal counterfactuals for random forests under a weighted-L1 cost.
CALF runs a best-first (A\*) search over axis-aligned boxes; each box carries a
lower bound stacked as the max of three admissible bounds, and the search
branches until the frontier minimum meets the incumbent — at which point the
incumbent is **provably optimal**. When the time budget runs out first, the
incumbent/frontier pair still brackets the true optimum, so every answer ships
with a finite certified gap (anytime certificates).

The three stacked bounds:

1. **geometric** per-feature LB (scaled L1 to the box),
2. **plateau node-local per-tree** LB (per-tree order statistic, leaf boxes
   clipped to the node box), and
3. the **cost-splitting Lagrangian dual** LB (`src/calf/dual_lb.py`) — the
   novelty — engaged only when the frontier stalls, re-optimized at popped
   nodes on a stride, warm-started from a small share pool.

The primal side is dual-guided greedy feasibility-repair rounding, which turns
the dual's per-tree leaf preferences into real counterfactuals.

## Install

```sh
uv sync                                          # core library + tests
uv sync --package calf-research                  # + experiment harness
uv sync --package calf-research --extra baselines  # + OCEAN/Gurobi/OR-Tools baselines
```

The `baselines` extra is needed to reproduce the comparison suites; the OCEAN
MILP arm additionally needs a Gurobi licence at run time (`GRB_LICENSE_FILE`).
See `research/baselines/README.md` for known install issues.

## Quickstart

```python
from sklearn.ensemble import RandomForestClassifier
import calf
from calf.datasets import load

X, y, _info = load("pima", root="research/data", numeric_only=True)
rf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0).fit(X, y)

target = 1 - int(rf.predict(X[:1])[0])  # flip the model's prediction
res = calf.solve(rf, X, factual=X[0], target_class=target)
print(res.cost, res.proven_optimal, res.x)
```

`calf.solve` parses the forest, infers the root box from `X`, builds the L1
scale (pass `weights=` for per-feature costs), and runs the search.

## Repository structure

```
src/calf/                  the method: parser, A* engine, dual bound, numba kernels
src/calf/datasets/         parquet + yaml dataset loader
tests/                     pytest suite for the core method
scripts/                   one-shot dataset preparation CLIs
research/                  experiment workspace member (calf-research)
  exp_suite/               suite harness: config -> parallel resumable runs -> tables
  baselines/               OCEAN CP/MILP, HiGHS interval-MILP, brute force, ... (common interface)
  suite_configs/           the paper's suite configs (holdout_soft_120, holdout_soft_120_t4)
  results/                 completed runs (per-task shards + aggregates) and figure/table outputs
  ideal/                   scripts that produce the paper's tables and figures
  tests/                   harness pytest suite + benchmark scripts
  data/                    bundled datasets (*.parquet + *.yaml)
```

## Running the experiment suite

The paper's numbers come from two suites, both defined in
`research/suite_configs/`. Runs are resumable: each (dataset, query, method)
task writes its own shard, so Ctrl-C and rerun is always safe. Run from the
repo root:

```sh
# main holdout suite: CALF vs OCEAN-CP vs OCEAN-MILP vs interval-MILP,
# soft voting, 120 s per-query cap, stratified 80/20 holdout
uv run python research/run_suite.py run --config research/suite_configs/holdout_soft_120.yaml

# 4-thread solver variant (reuses the cached train-only forests and splits
# from holdout_soft_120, so run that one first)
uv run python research/run_suite.py run --config research/suite_configs/holdout_soft_120_t4.yaml

# (re)build aggregate.parquet / aggregate.csv from the shards, any time
uv run python research/run_suite.py aggregate --experiment holdout_soft_120
```

In configs and result tables our method appears under the arm name `calf`
(`calf_warm_early` etc. are ablation aliases of the same engine).

Completed runs for both suites ship in `research/results/holdout_soft_120*`,
so the analysis below works without re-running anything.

## Reproducing the paper's tables and figures

The scripts in `research/ideal/` read the completed runs in
`research/results/` and write markdown/LaTeX tables and PDF figures:

```sh
uv run python research/ideal/conference_table.py   # main comparison table (cert counts, median wall)
uv run python research/ideal/conference_figure.py  # certified-rate vs wall-time scatter
uv run python research/ideal/gap_figure.py         # optimality-gap distribution on unproven queries
uv run python research/ideal/speed_scaling.py      # wall-time vs forest-structure scaling
uv run python research/ideal/dataset_table.py      # dataset / forest characteristics table
```

Each writes into its own directory under `research/results/` (override with
`--out`).

## Datasets

`research/data/` ships the benchmark datasets as `<name>.parquet` plus a
sidecar `<name>.yaml` carrying feature metadata. The loader
(`calf.datasets.load`) only ever reads parquet + yaml; the CLIs to
(re-)generate the bundle are in `scripts/`.

## Tests

```sh
uv run pytest                    # core method (tests/)
cd research && uv run pytest     # harness + baselines (research/tests/)
```