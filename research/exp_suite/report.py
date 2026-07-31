"""Post-run summary tables (shared by run_suite and compare_methods_report).

Renders, for one aggregate DataFrame:

- per-method summary with the three-tier certificate accounting:
  ``is_optimal_flag`` (the recorded harness flag, as-is), ``solver_opt``
  (the backend proved its OWN encoding's optimum), ``cert_real`` (the
  certificate actually transfers to the float32 sklearn model);
- per-method downgrade reasons (``cert_note``) for solver-optimal rows whose
  certificate does not transfer;
- a per-dataset table (real certificates / recorded flag / walls / gaps);
- a portfolio section (multi-method frames): rows certified by ANY method,
  matched on (dataset, query_idx, target_class), with the parallel
  time-to-certificate (min wall among certifying methods) and the sequential
  total cost (sum of walls) per row.

Recorded flags (``solver_optimal`` / ``cert_note``, written since the
2026-07-12 voting-aware ocean.py fix) are used when present; on aggregates
predating them the tiers are derived: finished under the cap => solver-side
optimal (exact solver), plus sklearn-valid => transferable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_BOOL_COLS = ("found", "sklearn_valid", "timed_out", "is_optimal")
_ROW_KEY = ["dataset", "query_idx", "target_class"]


def coerce_bools(df: pd.DataFrame) -> pd.DataFrame:
    """CSV round-trips turn bool columns into float (NaN on no-CF rows)."""
    df = df.copy()
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def certificate_tiers(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(solver_opt, cert_real) masks for one method's rows."""
    if "solver_optimal" in g.columns and g["solver_optimal"].notna().any():
        return g["solver_optimal"].fillna(False).astype(bool), g.is_optimal
    if g.method.str.startswith("ocean").all():
        # pre-fix aggregate: derive (see module docstring)
        solver_opt = ~g.timed_out & g.found
        return solver_opt, solver_opt & g.sklearn_valid
    return g.is_optimal, g.is_optimal


def method_summary(df: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for meth, g in df.groupby("method", sort=False):
        solver_opt, cert = certificate_tiers(g)
        rows.append({
            "method": (labels or {}).get(meth, meth),
            "n": len(g),
            "is_optimal_flag": int(g.is_optimal.sum()),
            "solver_opt": int(solver_opt.sum()),
            "cert_real": int(cert.sum()),
            "sklearn_valid": int(g.sklearn_valid.sum()),
            "found": int(g.found.sum()),
            "med_wall_s": round(g.wall_time_s.median(), 3),
            "mean_wall_s": round(g.wall_time_s.mean(), 1),
        })
    return pd.DataFrame(rows)


def downgrade_reasons(df: pd.DataFrame) -> list[str]:
    """Lines describing solver-optimal rows whose certificate doesn't transfer."""
    lines = []
    for meth, g in df.groupby("method", sort=False):
        solver_opt, cert = certificate_tiers(g)
        down = g[solver_opt & ~cert]
        if not len(down):
            continue
        if "cert_note" in down.columns and down.cert_note.notna().any():
            reasons = down.cert_note.fillna("unrecorded").value_counts()
        else:
            reasons = pd.Series({"sklearn_invalid (derived — pre-fix aggregate)": len(down)})
        lines.append(f"{meth}: {len(down)} solver-optimal rows do NOT transfer:")
        lines += [f"   {cnt:3d}  {reason}" for reason, cnt in reasons.items()]
    return lines


def per_dataset(df: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for (ds, meth), g in df.groupby(["dataset", "method"], sort=False):
        _, cert_mask = certificate_tiers(g)
        unproven = g[~cert_mask]
        rows.append({
            "dataset": ds, "method": (labels or {}).get(meth, meth),
            "cert_real": int(cert_mask.sum()),
            "opt_flag": int(g.is_optimal.sum()),
            "med_wall": round(g.wall_time_s.median(), 2),
            "med_gap_unpr": round(unproven.gap.median(), 3)
            if len(unproven) and unproven.gap.notna().any() else np.nan,
        })
    t = pd.DataFrame(rows).set_index(["dataset", "method"]).sort_index().unstack("method")
    return t[["cert_real", "opt_flag", "med_wall", "med_gap_unpr"]]


def portfolio(df: pd.DataFrame, methods: list[str] | None = None) -> tuple[dict, pd.DataFrame]:
    """Either-certifies accounting across methods, matched per query row.

    Returns ``(overall, per_dataset)``.  ``overall`` has the matched row
    count, per-method and portfolio certificate counts, and two wall models
    over portfolio-certified rows: ``med_wall_parallel`` (min wall among the
    methods that certified the row — run both, take the first certificate)
    and ``med_wall_sequential`` (sum of all methods' walls on that row — the
    total compute cost of running the portfolio).  ``per_dataset`` has one
    certified-count column per method plus the portfolio column.
    """
    df = coerce_bools(df)
    methods = methods or list(df.method.unique())
    parts = []
    for meth in methods:
        g = df[df.method == meth]
        _, cert = certificate_tiers(g)
        parts.append(g[_ROW_KEY + ["wall_time_s"]].assign(cert=cert.values, method=meth))
    c = pd.concat(parts, ignore_index=True)
    cert_p = c.pivot_table(index=_ROW_KEY, columns="method", values="cert",
                           aggfunc="any").fillna(False).astype(bool)
    wall_p = c.pivot_table(index=_ROW_KEY, columns="method", values="wall_time_s")
    any_cert = cert_p.any(axis=1)
    certified_walls = wall_p.where(cert_p)
    overall = {
        "rows": int(len(cert_p)),
        "per_method": {m: int(cert_p[m].sum()) for m in cert_p.columns},
        "portfolio": int(any_cert.sum()),
        "med_wall_parallel": float(certified_walls[any_cert].min(axis=1).median())
        if any_cert.any() else np.nan,
        "med_wall_sequential": float(wall_p[any_cert].sum(axis=1).median())
        if any_cert.any() else np.nan,
    }
    per_ds = cert_p.assign(portfolio=any_cert).groupby(level="dataset").sum()
    per_ds["rows"] = cert_p.groupby(level="dataset").size()
    return overall, per_ds


def render_portfolio(df: pd.DataFrame, methods: list[str] | None = None) -> str:
    overall, per_ds = portfolio(df, methods)
    best = max(overall["per_method"].items(), key=lambda kv: kv[1])
    head = (f"matched rows {overall['rows']} | portfolio cert "
            f"{overall['portfolio']} | best single {best[1]} ({best[0]}) | "
            f"med wall: parallel {overall['med_wall_parallel']:.2f} s, "
            f"sequential {overall['med_wall_sequential']:.2f} s")
    return "\n".join([
        "-- portfolio (certificate from ANY method, matched rows) --",
        head, per_ds.to_string(),
    ])


def render(df: pd.DataFrame, labels: dict[str, str] | None = None,
           show_portfolio: bool = True) -> str:
    """Full text summary for one aggregate frame."""
    if not len(df):
        return "(no rows)"
    df = coerce_bools(df)
    pd.set_option("display.width", 250)
    parts = ["-- method summary --",
             method_summary(df, labels).to_string(index=False)]
    reasons = downgrade_reasons(df)
    if reasons:
        parts += ["", "-- certificate downgrades (solver-optimal, not transferable) --",
                  *reasons]
    parts += ["", "-- per dataset (real cert / recorded is_optimal / med wall / med unproven gap) --",
              per_dataset(df, labels).to_string()]
    if show_portfolio and df.method.nunique() > 1:
        parts += ["", render_portfolio(df)]
    return "\n".join(parts)
