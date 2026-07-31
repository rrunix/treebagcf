# Baselines

SOTA optimal / near-optimal counterfactual (CF) methods for tree ensembles,
wrapped behind one interface so the experiment harness can call them
interchangeably on the **same** trained `RandomForestClassifier`, the same query
instances, the same distance, the same voting rule, and the same time limit.

See `../BASELINES.md` for the sourcing brief and `../STATUS.md` for per-method
provenance (sourced-from-repo vs reimplemented, versions, skips).

## Common interface

Every backend subclasses `baselines.base.BaselineExplainer` and returns a
`baselines.base.CFResult`:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from baselines import CALFExplainer, BruteForceExplainer, CounterfactualMapsExplainer

rf = RandomForestClassifier(n_estimators=50, max_depth=6).fit(X, y)

expl = CounterfactualMapsExplainer(rf, X=X, voting="hard", norm=1)
res = expl.explain(x, target_class=1, time_limit_s=60.0)
print(res.status, res.cost, res.is_optimal, res.lower_bound, res.upper_bound)
```

`CFResult` fields: `x_cf, cost, is_optimal, lower_bound, upper_bound, n_nodes,
wall_time_s, status, method, metric, reaches_target, error, extra`.

### Discovering available backends

```python
from baselines import build_available
for st in build_available(rf, X=X, voting="hard"):
    if st.available:
        print(st.key, st.explainer.explain(x, 1))
    else:
        print(st.key, "SKIPPED:", st.reason)   # e.g. OCEAN / Gurobi not installed
```

Unavailable backends (missing OCEAN, Gurobi, or OR-Tools) are *skipped* with a
reason, never crash the run.

## Distance & voting conventions (identical across all methods)

* **Distance — range-normalised weighted L1**
  `cost(x, x') = Σ_f (weight_f / range_f) · |x_f − x'_f|`, where `range_f` is the
  per-feature data range. This is exactly `calf.cost.l1_scale`, so costs are
  directly comparable to our own method. Pass `X=` (to infer ranges) or
  `feature_ranges=`, and optional `weights=` (large weight ⇒ effectively
  immutable). The reported `cost` is **always recomputed** from the returned
  point under this metric, regardless of what a backend optimised internally.
* **Voting — hard (majority)** with `need = ceil(threshold · n_trees)` trees
  voting the target class (`threshold=0.5` ⇒ strict majority).
* **Ground-truth oracle = the parsed rule ensemble** (half-open `(lo, hi]` boxes
  in float64), *not* sklearn's `predict`. sklearn compares split thresholds in
  float32, so at a decision boundary the two differ by a hair; the whole project
  (search engine, MILP baseline) treats the parsed forest as the oracle, and so
  do these baselines. `CFResult.reaches_target` re-checks the returned point
  under this oracle.

## Methods

| key | method | source | licence | anytime | notes |
|---|---|---|---|---|---|
| `calf` | **our method** | this repo (`src/calf`) | free | ✅ | budgeted by `max_iters`, not wall-clock |
| `brute_force` | grid-cell enumeration | this repo | free | n/a | exact oracle for tiny forests only |
| `counterfactual_maps` | Counterfactual Maps | **reimplemented** (Khouna et al. 2026) | free | no | exact; region count ∝ exp(depth) |
| `ocean_milp` | OCEAN MILP | OCEAN pkg | **Gurobi** | no | Parmentier & Vidal 2021 |
| `ocean_cp` | CPCF | OCEAN pkg | free (OR-Tools) | ✅ | licence-free exact/anytime baseline |
| `ocean_maxsat` | weighted-MaxSAT | OCEAN pkg | free | no | hard-voting only — our regime |
| `ocean_sat` | SAT / MACE | OCEAN pkg | free | no | optional |

## Install

The license-free methods (`calf`, `brute_force`, `counterfactual_maps`) need
nothing beyond the repo's own deps (`uv sync`).

### OCEAN — MILP + CP + MaxSAT (one package)

```bash
uv pip install oceanpy          # imports as `ocean`
# MILP backend only — needs Gurobi (free academic licence at gurobi.com):
uv pip install gurobipy         # then activate a licence
```

`oceanpy` pulls OR-Tools (for CP) and a MaxSAT solver as dependencies; **no
Gurobi is needed for CP or MaxSAT**. Verified API (2026-07):
`MixedIntegerProgramExplainer`, `ConstraintProgrammingExplainer`,
`MaxSATExplainer`, each `explain(x, y=<target>, norm=1)`. OCEAN exposes no public
SAT explainer, so `ocean_sat` stays SKIPPED.

**Mapper.** OCEAN needs a feature `mapper`, built here with
`ocean.feature.parse_features(df, scale=False)` (via `baselines.ocean.build_mapper`),
using `scale=False` so the mapper matches a forest trained on the **raw**
features — the same forest we hand every other method. (OCEAN's own
`parse_features(scale=True)` normalises continuous columns to [-0.5, 0.5]; only
use that if you also train the RF on the processed frame.) The wrapper builds the
mapper automatically from the `X` passed at construction; pass `discretes=`/
`encoded=` for ordinal / one-hot columns, or a ready-made `mapper=`.

> The README does not document OCEAN's result object, so cost / bound /
> optimality-flag attribute names are probed defensively (`_extract_*` in
> `baselines/ocean.py`). After install, run `python -m baselines.smoke_test`; if a
> backend returns a CF but reports `cost=None`, add the real attribute name to
> the relevant `_extract_*` probe.
>
> Also confirm and record here once installed: how each backend selects **hard
> vs soft voting** (MaxSAT is hard-only; MILP/CP may default to soft — the
> resolved mode is surfaced in `CFResult.extra["voting_mode"]`), and whether the
> backend accepts **per-feature L1 weights** (only then does it optimise our
> exact metric; otherwise `is_optimal` is downgraded and
> `extra["metric_confirmed"]=False`).

### Running without Gurobi (fully open source)

CP (OR-Tools CP-SAT) and MaxSAT are licence-free, so:

```bash
uv pip install oceanpy
uv run python -m baselines.smoke_test         # runs calf + brute_force + cfmaps + ocean_cp + ocean_maxsat
uv run pytest tests/test_baselines.py         # license-free correctness (never touches OCEAN)
```

`ocean_milp` will error without a Gurobi licence (recorded as `status="error"`,
never crashes). Our own MILP reference `research/milp_baseline.py` uses HiGHS via
scipy (also licence-free) if you want an exact MILP baseline without Gurobi.

> ⚠️ Python version: this repo's venv is 3.14. If `oceanpy` (or a transitive dep)
> does not yet support 3.14, install the OCEAN backends in a side env (3.11/3.12)
> and point the harness at it; the license-free geometry backends and our method
> run fine on 3.14.

#### Known install issues

1. **`XGBoostError: libxgboost.dylib could not be loaded … libomp.dylib`** —
   `ocean` imports `xgboost`, which needs the OpenMP runtime. On macOS:
   `brew install libomp`.
2. **`TypeError: metaclass conflict` importing `ocean.cp`** — recent `ortools`
   (≥ ~9.14, e.g. 9.15) rebuild `CpModel` with pybind11 3.0, whose metaclass
   conflicts with `abc.ABCMeta` in oceanpy's `class BaseModel(ABC, cp.CpModel)`.
   oceanpy floors `ortools>=9.12` with no upper bound, so pip pulls a too-new one.
   The ABC-compatible builds are `ortools` 9.12 / 9.13 — **but those ship no
   Python 3.14 wheels** (they stop at `cp313`), and the only 3.14 wheels are
   `ortools>9.14`, which reintroduce the conflict. **Net: OCEAN cannot run on
   Python 3.14.** Run the OCEAN backends in a Python 3.13 side env:
   ```bash
   uv venv --python 3.13 .venv-ocean
   source .venv-ocean/bin/activate
   uv pip install -e . -r baselines/requirements-ocean.txt   # pinned oceanpy + ortools 9.13
   python -m baselines.smoke_test
   deactivate
   ```
   Versions are pinned in `baselines/requirements-ocean.txt` (kept out of the
   main `uv.lock` — see the note at the top of that file). `gurobipy` installs
   automatically as an oceanpy dep; the MILP backend additionally needs a Gurobi
   licence at run time.
   Our method + the geometry baselines run in the main 3.14 venv; merge the two
   result CSVs by `dataset,query_idx` (the harness already matches on those keys),
   so mixing envs is fine for the comparison.

## Verify

```bash
uv run pytest tests/test_baselines.py     # license-free correctness (no Gurobi/OCEAN)
uv run python -m baselines.smoke_test      # table + cross-check across available backends
```

`test_baselines.py` pins the reimplemented Counterfactual Maps and our own
method against the brute-force optimum on tiny forests (2–3 trees, depth ≤ 3),
per the BASELINES.md checklist.
