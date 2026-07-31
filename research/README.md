# calf-research

Experiment harness, benchmarks, and SOTA baselines for the CALF certified
counterfactual method. Workspace member of `calf`.

## Install

From the repo root (single shared workspace venv):

```bash
uv sync --package calf-research                    # harness only
uv sync --package calf-research --extra baselines  # + OCEAN/Gurobi/OR-Tools baselines
```

The `baselines` extra pulls the solver backends (OCEAN MILP/CP/MaxSAT). The MILP
backend additionally needs a Gurobi licence at run time (`GRB_LICENSE_FILE`);
CP/MaxSAT do not. See `baselines/README.md` for known install issues.

## Layout

- `exp_suite/` — the experiment harness (config → parallel, resumable runs → parquet/CSV tables).
- `baselines/` — SOTA counterfactual baselines behind a common `CFResult` interface.
- `suite_configs/` — the paper's suite configs (`holdout_soft_120`, `holdout_soft_120_t4`).
- `ideal/` — scripts producing the paper's tables and figures from completed runs.
- `tests/` — the research pytest suite plus benchmark scripts.
- `data/` — bundled datasets (`*.parquet` + `*.yaml`).
- `results/` — completed per-experiment run directories and figure/table outputs.

See the repo-root `README.md` for how to run the suites and regenerate the
paper's tables and figures.
