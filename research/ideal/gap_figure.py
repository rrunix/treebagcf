"""Anytime-quality figure: distribution of the certified optimality gap on the
queries each method leaves *unproven* within the time cap.

For a query a method does not certify, the relative gap is
``(incumbent - bound) / incumbent`` -- how far the method can still guarantee its
best counterfactual is from the optimum.  A tight gap is a usable anytime
bracket; a gap near 100% means the bound is worthless; no bound at all means the
method is not anytime.  We draw one violin per method over its unproven queries:

    CALF        clusters low (Lagrangian dual gives a real bracket)
    OCEAN-MILP    piles near 100% (LP bound is near-useless)
    MILP-soft     piles near 100% (HiGHS bound is near-useless)
    OCEAN-CP      has NO bound on any query -> shown as an annotation, no violin

Reads the holdout shards fresh.  Certified queries are excluded (their gap is 0).

    uv run python research/ideal/gap_figure.py [--out research/results/gap_figure]
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
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from exp_suite import report  # noqa: E402

EXPERIMENT = "holdout_soft_120"
# (label, method, colour) -- order left..right
METHODS = [
    ("OCEAN-MILP", "ocean_milp", "#dd8452"),
    ("OCEAN-CP", "ocean_cp", "#c44e52"),
    ("MILP (HiGHS)", "milp_soft", "#8172b3"),
    ("CALF (ours)", "calf", "#4c72b0"),
]


def load_shards() -> pd.DataFrame:
    files = glob.glob(str(_ROOT / "results" / EXPERIMENT / "tasks" / "*.json"))
    rows = [json.load(open(f)) for f in files]
    return report.coerce_bools(pd.DataFrame(rows))


def unproven_gaps(df: pd.DataFrame, method: str):
    """Relative gaps (%) on the method's unproven queries; (gaps, n_unproven)."""
    g = df[df.method == method]
    solver, _ = report.certificate_tiers(g)
    unp = g[~solver.values].copy()
    n_unproven = len(unp)
    unp["cost"] = pd.to_numeric(unp["cost"], errors="coerce")
    unp["lb"] = pd.to_numeric(unp["lower_bound"], errors="coerce")
    unp = unp[unp["cost"].notna() & unp["lb"].notna() & (unp["cost"] > 0)]
    gaps = ((unp["cost"] - unp["lb"]) / unp["cost"]).clip(lower=0, upper=1) * 100.0
    return gaps.to_numpy(), n_unproven


def build_figure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xticklabels = []
    for i, (label, method, colour) in enumerate(METHODS, start=1):
        gaps, n_unproven = unproven_gaps(df, method)
        xticklabels.append(f"{label}\n($n_{{unp}}={n_unproven}$)")
        if len(gaps) == 0:
            # no bound on any unproven query (OCEAN-CP): annotate, no violin
            ax.text(i, 50, "no anytime\nbound", ha="center", va="center",
                    fontsize=8, style="italic", color=colour,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, lw=1))
            continue
        vp = ax.violinplot([gaps], positions=[i], widths=0.75,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colour); body.set_edgecolor(colour)
            body.set_alpha(0.45)
        med = float(np.median(gaps))
        ax.hlines(med, i - 0.34, i + 0.34, color=colour, lw=2.2, zorder=4)
        ax.text(i + 0.40, med, f"{med:.0f}\\%".replace("\\%", "%"),
                va="center", ha="left", fontsize=8.5, color=colour)
        # faint jittered points for the actual sample (varies by index, no RNG)
        jitter = ((np.arange(len(gaps)) % 21) - 10) / 10.0 * 0.16
        ax.scatter(i + jitter, gaps, s=5, color=colour, alpha=0.18, zorder=2)

    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels(xticklabels, fontsize=8.5)
    ax.set_ylim(-4, 104)
    ax.set_ylabel("relative optimality gap (%)")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", ls="-", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title("Relative gap on non-certified queries", fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_ROOT / "results" / "gap_figure"))
    args = ap.parse_args()
    df = load_shards()
    fig = build_figure(df)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure.pdf")
    fig.savefig(out / "figure.png", dpi=200)
    print(f"wrote {out}/figure.pdf and figure.png")
    for label, method, _c in METHODS:
        gaps, n = unproven_gaps(df, method)
        med = f"{np.median(gaps):.0f}%" if len(gaps) else "no bound"
        print(f"  {label:16s} unproven={n:4d}  with-bound={len(gaps):4d}  median gap={med}")


if __name__ == "__main__":
    main()
