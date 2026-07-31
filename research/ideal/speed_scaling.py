"""What governs each method's speed? -- scaling of median wall vs. forest structure.

For every (method, dataset) we take the median per-query wall time (over solved /
timed-out rows, so errors don't masquerade as "fast") and plot it against five
structural properties of the *train-only* forest that produced those timings:

    leaves        total leaves in the forest (|R|, the rule-set size)
    thr_per_feat  mean distinct split thresholds per feature (how finely cut)
    d             number of numeric features
    n_train       training rows the forest was fit on
    mean_depth    mean root-to-leaf depth

The ranking metric is the **Spearman rank correlation** rho between each property
and a method's median wall across the 11 datasets: rank-based, so it is robust
to the 120 s cap censoring (a capped dataset still ranks "slowest").  The
property with the largest |rho| for a method is the best predictor of -- i.e.
the knob that governs -- that method's speed.  The hypothesis under test: the
MILP/CP encodings scale with *leaves*, whereas our A* does not (its cost tracks
LB-plateau width, which leaf count does not capture -- e.g. digits_even is
low-leaf yet our hardest, abalone is high-leaf yet trivial for us).

Structural stats come from the cached train forests of ``holdout_soft_120`` so
the x-axis matches the timings exactly.  Reads shards fresh (regenerable
mid-run).  Forest-stat helpers are shared with ``dataset_table``.

    uv run python research/ideal/speed_scaling.py [--out research/results/speed_scaling]
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
from scipy.stats import spearmanr, linregress  # noqa: E402

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent  # research/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from exp_suite import report  # noqa: E402
from dataset_table import forest_stats  # noqa: E402  (same dir)

EXPERIMENT = "holdout_soft_120"
MODEL_TAG = "tf0.2s0"  # matches holdout_soft_120.yaml (test_frac 0.2, split_seed 0)

# (label, method, colour, marker)
METHODS = [
    ("CALF (ours)", "calf", "#4c72b0", "o"),
    ("MILP-soft", "milp_soft", "#8172b3", "s"),
    ("OCEAN-MILP", "ocean_milp", "#dd8452", "^"),
    ("OCEAN-CP", "ocean_cp", "#c44e52", "v"),
]
# (key, axis label, log-x?)
CHARACTERISTICS = [
    ("leaves", "forest leaves", True),
    ("thr_per_feat", "split levels / feature", True),
    ("d", "features", True),
    ("n_train", "training samples", True),
    ("mean_depth", "mean leaf depth", False),
]
# rows that reflect genuine solve effort (exclude errors / infeasible / no-timing)
_EFFORT_STATUS = {"optimal", "feasible", "timeout"}
TIME_FLOOR_S = 1e-3


def forest_characteristics() -> dict[str, dict]:
    """Per-dataset structural stats from the cached train-only forests."""
    import joblib

    models = _ROOT / "results" / EXPERIMENT / "models"
    out: dict[str, dict] = {}
    for mp in sorted(models.glob(f"*__default__{MODEL_TAG}.joblib")):
        ds = mp.name.split("__")[0]
        rf = joblib.load(mp)
        s = forest_stats(rf, rf.n_features_in_)
        split = mp.with_suffix(".split.json")
        n_train = len(json.loads(split.read_text())["train_idx"]) if split.exists() else np.nan
        out[ds] = {"leaves": s["leaves"], "thr_per_feat": s["thr_per_feat"],
                   "mean_depth": s["mean_depth"], "d": int(rf.n_features_in_),
                   "n_train": n_train}
    return out


def median_speed(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(method, dataset) -> median per-query wall over genuine-effort rows."""
    g = df[df["status"].isin(_EFFORT_STATUS)].copy()
    g["wall_time_s"] = g["wall_time_s"].astype(float)
    g = g[np.isfinite(g["wall_time_s"])]
    med = g.groupby(["method", "dataset"])["wall_time_s"].median()
    return {(m, d): float(v) for (m, d), v in med.items()}


def load_shards() -> pd.DataFrame:
    files = glob.glob(str(_ROOT / "results" / EXPERIMENT / "tasks" / "*.json"))
    rows = [json.load(open(f)) for f in files]
    if not rows:
        raise SystemExit(f"no shards under {EXPERIMENT}/tasks")
    return report.coerce_bools(pd.DataFrame(rows))


def _aligned(chars, speed, method, key):
    """(x, y) arrays over datasets where both the char and a speed exist."""
    xs, ys = [], []
    for ds, c in chars.items():
        v = c.get(key)
        s = speed.get((method, ds))
        if v is not None and np.isfinite(v) and s is not None and np.isfinite(s):
            xs.append(v); ys.append(s)
    return np.array(xs, float), np.array(ys, float)


def fit_one(x, y, logx: bool):
    """OLS in log-space. logx=True -> power law cost ~ x^slope (log-log);
    logx=False -> exponential cost ~ 10**(slope*x) (semilog, for depth).

    NOTE: capped (timed-out) datasets sit at ~120 s, so for cap-hitting methods
    the fitted slope is a LOWER BOUND on the true growth (y is right-censored).
    """
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    X = np.log10(x) if logx else x
    lr = linregress(X, np.log10(y))
    return {"slope": float(lr.slope), "r2": float(lr.rvalue ** 2),
            "intercept": float(lr.intercept), "logx": logx, "n": int(len(x))}


def scaling_fits(chars, speed):
    """fits[method_label][key] -> fit dict (or None); plus a slope DataFrame."""
    fits, rows = {}, []
    for label, method, *_ in METHODS:
        fits[label] = {}
        row = {"method": label}
        for key, _lab, logx in CHARACTERISTICS:
            x, y = _aligned(chars, speed, method, key)
            f = fit_one(x, y, logx)
            fits[label][key] = f
            row[key] = f["slope"] if f else np.nan
        rows.append(row)
    return fits, pd.DataFrame(rows).set_index("method")


def _fit_label(label, f, logx) -> str:
    if f is None:
        return label
    if logx:  # power-law exponent
        return f"{label}  $k$={f['slope']:+.2f} ($R^2$={f['r2']:.2f})"
    factor = 10.0 ** f["slope"]  # exponential: multiplier per +1 depth
    return f"{label}  $\\times${factor:.1f}/lvl ($R^2$={f['r2']:.2f})"


# short column headers for the paper table
_COL_TEX = {"leaves": "Leaves", "thr_per_feat": "Splits/feat.", "d": "$d$",
            "n_train": "Train $n$", "mean_depth": "Depth"}


def spearman_table(chars, speed):
    """(rho, pvalue) DataFrames -- rows methods, cols structural properties."""
    rrows, prows = [], []
    for label, method, *_ in METHODS:
        rr, pr = {"method": label}, {"method": label}
        for key, _lab, _log in CHARACTERISTICS:
            x, y = _aligned(chars, speed, method, key)
            if len(x) >= 3:
                res = spearmanr(x, y)
                rr[key], pr[key] = res.statistic, res.pvalue
            else:
                rr[key] = pr[key] = np.nan
        rrows.append(rr); prows.append(pr)
    rho = pd.DataFrame(rrows).set_index("method")
    pval = pd.DataFrame(prows).set_index("method")
    return rho, pval


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "^{***}" if p < 0.001 else "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""


def corr_to_latex(rho: pd.DataFrame, pval: pd.DataFrame) -> str:
    keys = [k for k, _l, _lg in CHARACTERISTICS]
    head = "Method & " + " & ".join(_COL_TEX[k] for k in keys) + r" \\"
    body = []
    for label in rho.index:
        cells = []
        for k in keys:
            r, p = rho.loc[label, k], pval.loc[label, k]
            cells.append("--" if not np.isfinite(r) else f"${r:+.2f}{_stars(p)}$")
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule", head, r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}",
        r"% Spearman rank correlation of median per-query wall vs. forest structure",
        r"% across the 11 benchmark datasets. $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.",
    ])


def corr_to_markdown(rho: pd.DataFrame, pval: pd.DataFrame) -> str:
    keys = [k for k, _l, _lg in CHARACTERISTICS]
    star = {"^{***}": "***", "^{**}": "**", "^{*}": "*", "": ""}
    out = ["| Method | " + " | ".join(k for k in keys) + " |",
           "|" + "---|" * (len(keys) + 1)]
    for label in rho.index:
        cells = []
        for k in keys:
            r, p = rho.loc[label, k], pval.loc[label, k]
            cells.append("--" if not np.isfinite(r) else f"{r:+.2f}{star[_stars(p)]}")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("\n\\* p<.05, ** p<.01, *** p<.001 (Spearman, n=11)")
    return "\n".join(out)


def build_figure(chars, speed, corr: pd.DataFrame, fits: dict):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    axes = axes.ravel()
    for ax, (key, xlabel, logx) in zip(axes, CHARACTERISTICS):
        for label, method, colour, marker in METHODS:
            x, y = _aligned(chars, speed, method, key)
            if not len(x):
                continue
            ax.scatter(x, np.maximum(y, TIME_FLOOR_S), s=42, marker=marker,
                       facecolor=colour, edgecolor="white", linewidth=0.5,
                       alpha=0.85, zorder=3, label=_fit_label(label, fits[label][key], logx))
            f = fits[label][key]
            if f is not None:  # overlay the fitted line (straight in these axes)
                xs = np.linspace(x.min(), x.max(), 50)
                Xs = np.log10(xs) if logx else xs
                ax.plot(xs, 10.0 ** (f["slope"] * Xs + f["intercept"]),
                        color=colour, lw=1.1, ls="--", alpha=0.8, zorder=2)
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel + ("" if logx else "   (semilog fit)"))
        ax.set_ylabel("median wall (s)")
        ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=6.4, loc="upper left", framealpha=0.9)

    # 6th panel: Spearman heatmap (the ranking metric) -- rows methods, cols chars
    ax = axes[5]
    M = corr[[k for k, _l, _lg in CHARACTERISTICS]].to_numpy(float)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CHARACTERISTICS)))
    ax.set_xticklabels([k for k, _l, _lg in CHARACTERISTICS], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([lab for lab, *_ in METHODS], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.55 else "black")
    ax.set_title("Spearman $\\rho$ (structure vs. speed)", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("What governs each method's speed? median per-query wall vs. forest structure",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_ROOT / "results" / "speed_scaling"))
    args = ap.parse_args()

    chars = forest_characteristics()
    df = load_shards()
    speed = median_speed(df)
    corr, pval = spearman_table(chars, speed)
    fits, slopes = scaling_fits(chars, speed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig = build_figure(chars, speed, corr, fits)
    fig.savefig(out / "figure.pdf")
    fig.savefig(out / "figure.png", dpi=180)
    corr.round(3).to_csv(out / "spearman.csv")
    pval.round(4).to_csv(out / "spearman_pval.csv")
    slopes.round(3).to_csv(out / "scaling_slopes.csv")
    (out / "spearman_table.tex").write_text(corr_to_latex(corr, pval) + "\n")
    (out / "spearman_table.md").write_text(corr_to_markdown(corr, pval) + "\n")

    pd.set_option("display.width", 140)
    # critical |rho| for n=11 so significance is legible without a table lookup
    print("Spearman: with n=11, |rho|>0.62 => p<.05, >0.76 => p<.01, >0.85 => p<.001\n")
    print(corr_to_markdown(corr, pval))
    print()
    print("Spearman rho (median wall vs. structural property), by method:\n")
    print(corr.round(2).to_string())
    print("\nOLS scaling exponents -- power-law k for size metrics (cost ~ x^k),")
    print("exponential for depth (cost ~ 10^(slope*depth)); R^2 in parentheses:\n")
    hdr = f"{'method':16s}" + "".join(
        f"{k:>16s}" for k, _l, _lg in CHARACTERISTICS)
    print(hdr)
    for label, *_ in METHODS:
        cells = []
        for key, _l, logx in CHARACTERISTICS:
            f = fits[label][key]
            cells.append("--".rjust(16) if f is None
                         else f"{f['slope']:+.2f}({f['r2']:.2f})".rjust(16))
        print(f"{label:16s}" + "".join(cells))
    print("\n(depth column: 10^slope = cost multiplier per +1 mean depth)")
    # completeness (mid-run guard)
    print("\nrows per method:", {m: int((df.method == m).sum()) for _l, m, *_ in METHODS})
    print(f"\nwrote {out}/figure.pdf, figure.png, spearman.csv")


if __name__ == "__main__":
    main()
