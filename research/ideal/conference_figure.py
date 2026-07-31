"""Conference figure: certified-rate vs. median wall-time scatter (log-x).

One marker per (method, dataset): x = median per-query wall (log scale, ms->s),
y = % of that dataset's held-out queries the method certifies (real transferable
certificates via ``report.certificate_tiers``).  A large ringed marker per method
sits at its aggregate (all 1333 rows): overall cert-% and overall median wall.

The story the figure carries in one glance: CALF sits top-left (high cert,
millisecond time); the OCEAN backends sprawl bottom-right (low cert, tens of
seconds); milp_soft is high-cert but slower -- a complementary partner, not a
competitor.  Reads the holdout run's shards directly (fresh, so it can be
regenerated mid-run), matched 1:1 on (dataset, query_idx, target_class).

    uv run python research/ideal/conference_figure.py [--out research/results/conference_figure]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent  # research/ — exp_suite, results/, data/ live one level up
sys.path.insert(0, str(_ROOT))

from exp_suite import report  # noqa: E402

EXPERIMENT = "holdout_soft_120"
# (label, method, colour, marker) -- drawn back-to-front so CALF sits on top.
METHODS = [
    ("OCEAN-CP", "ocean_cp", "#c44e52", "v"),
    ("OCEAN-MILP", "ocean_milp", "#dd8452", "^"),
    ("MILP-soft", "milp_soft", "#8172b3", "s"),
    ("CALF (ours)", "calf", "#4c72b0", "o"),
]
# per-query wall floor for the log axis (sub-ms medians -> 1 ms so they plot)
TIME_FLOOR_S = 1e-3


def load_shards(experiment: str) -> pd.DataFrame:
    files = glob.glob(str(_ROOT / "results" / experiment / "tasks" / "*.json"))
    rows = [json.load(open(f)) for f in files]
    if not rows:
        raise SystemExit(f"no shards under {experiment}/tasks -- has the run started?")
    return report.coerce_bools(pd.DataFrame(rows))


def method_points(df: pd.DataFrame, method: str):
    """Per-dataset (n, cert_pct, median_wall) plus the pooled aggregate."""
    g = df[df.method == method]
    if g.empty:
        return [], None
    _, cert = report.certificate_tiers(g)
    g = g.assign(cert=cert.values)
    pts = []
    for ds, gg in g.groupby("dataset", sort=True):
        n = len(gg)
        med = float(gg["wall_time_s"].astype(float).median())
        pts.append((ds, n, 100.0 * int(gg["cert"].sum()) / n, med))
    n = len(g)
    agg = (100.0 * int(g["cert"].sum()) / n, float(g["wall_time_s"].astype(float).median()), n)
    return pts, agg


def clamp_t(t: float) -> float:
    return max(TIME_FLOOR_S, t) if np.isfinite(t) else TIME_FLOOR_S


def build_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    legend_handles = []
    notes = []
    for label, method, colour, marker in METHODS:
        pts, agg = method_points(df, method)
        if agg is None:
            notes.append(f"{label}: no data")
            continue
        # per-dataset cloud (small, semi-transparent)
        xs = [clamp_t(t) for *_, t in pts]
        ys = [y for _, _, y, _ in pts]
        ax.scatter(xs, ys, s=34, marker=marker, facecolor=colour, edgecolor="white",
                   linewidth=0.5, alpha=0.55, zorder=2)
        # pooled aggregate: large ringed marker
        ax.scatter([clamp_t(agg[1])], [agg[0]], s=190, marker=marker,
                   facecolor=colour, edgecolor="black", linewidth=1.4, zorder=4)
        legend_handles.append(
            plt.Line2D([], [], marker=marker, color="none", markerfacecolor=colour,
                       markeredgecolor="black", markersize=10,
                       label=f"{label}  ({agg[0]:.0f}%, {_fmt_t(agg[1])})"))

    ax.set_xscale("log")
    ax.set_xlim(TIME_FLOOR_S * 0.7, 200)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("median per-query wall time (log scale)")
    ax.set_ylabel("certified optimal (%)")
    # human tick labels: 1 ms .. 100 s
    ticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"])
    ax.axvline(120, color="grey", ls=":", lw=1, zorder=1)
    ax.text(120, -1.5, "120 s cap", color="grey", fontsize=7, ha="right", va="bottom", rotation=90)
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
              title="aggregate (cert %, median t)", title_fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return fig, notes


def _fmt_t(s: float) -> str:
    if not np.isfinite(s):
        return "--"
    if s < 0.095:
        return f"{s * 1000:.0f} ms"
    return f"{s:.1f} s" if s < 100 else f"{s:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default=EXPERIMENT)
    ap.add_argument("--out", default=str(_ROOT / "results" / "conference_figure"))
    args = ap.parse_args()
    df = load_shards(args.experiment)
    fig, notes = build_figure(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure.pdf")
    fig.savefig(out / "figure.png", dpi=200)
    print(f"wrote {out}/figure.pdf and figure.png")
    # per-method completeness (so a mid-run figure is not read as final)
    for label, method, *_ in METHODS:
        g = df[df.method == method]
        print(f"  {label:16s} {len(g):>4} rows")
    for n in notes:
        print("  NOTE:", n)


if __name__ == "__main__":
    main()
