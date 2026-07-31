"""Emit the conference comparison table (markdown + LaTeX booktabs).

One row per dataset, one column pair per method: certified-optimal count
(real, transferable certificates via ``report.certificate_tiers``) and median
per-query solve wall.

Source: the holdout protocol run ``holdout_soft_120`` (stratified 80/20 split;
forests fit on train, explanations generated for held-out TEST factuals).  All
three methods live in the same run, so rows match 1:1 with no cross-run merge.
No portfolio column: the methods are reported independently (milp_soft is not
part of this suite).

Arms (matched on dataset, query_idx, target_class; 1333 rows each):
- calf      : holdout_soft_120  (our A* + dual-LB engine, warm_early)
- ocean_cp    : holdout_soft_120  (OCEAN 2.0.3, OR-Tools CP-SAT, 1 thread)
- ocean_milp  : holdout_soft_120  (OCEAN 2.0.3, Gurobi, 1 thread)

All arms: same train-only forests (100 trees, seed 0), same held-out test
queries, plain L1, soft voting, 120 s per-query cap.

    uv run python research/ideal/conference_table.py [--out research/results/conference_table]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent  # research/ — exp_suite, results/, data/ live one level up
sys.path.insert(0, str(_ROOT))

from exp_suite import report  # noqa: E402

KEY = ["dataset", "query_idx", "target_class"]
ARMS = [
    # (column label, experiment, method)
    ("OCEAN-MILP", "holdout_soft_120", "ocean_milp"),
    ("OCEAN-CP", "holdout_soft_120", "ocean_cp"),
    ("CALF (ours)", "holdout_soft_120", "calf"),
]
# No portfolio: methods reported independently (set to a (label, label) pair of
# two present arms to re-enable an either-certifies column).
PORTFOLIO_OF = None


def load_arm(experiment: str, method: str) -> pd.DataFrame:
    df = report.coerce_bools(
        pd.read_csv(_ROOT / "results" / experiment / "aggregate.csv"))
    g = df[df.method == method]
    _, cert = report.certificate_tiers(g)
    return g.assign(cert=cert.values)[KEY + ["cert", "wall_time_s"]]


def fmt_pct(c: int, n: int) -> str:
    return f"{100.0 * c / n:.1f}%" if n else "--"


def fmt_wall_single(s: float) -> str:
    if not np.isfinite(s):
        return "--"
    if s < 0.095:
        return f"{s * 1000:.0f} ms"
    return f"{s:.1f} s" if s < 100 else f"{s:.0f} s"


def fmt_pair(median: float, p90: float) -> str:
    """``median / p90`` per-query wall, each in its own natural unit (ms or s)."""
    if not np.isfinite(median):
        return "--"
    return f"{fmt_wall_single(median)} / {fmt_wall_single(p90)}"


def build() -> tuple[pd.DataFrame, dict]:
    arms = {label: load_arm(exp, meth) for label, exp, meth in ARMS}
    base = arms[ARMS[0][0]][KEY].copy()
    for label, g in arms.items():
        base = base.merge(
            g.rename(columns={"cert": f"cert__{label}", "wall_time_s": f"wall__{label}"}),
            on=KEY, how="left",
        )
    for label in arms:
        base[f"cert__{label}"] = base[f"cert__{label}"].fillna(False).astype(bool)
    labels = [label for label, _, _ in ARMS]
    if PORTFOLIO_OF is not None:
        a, b = PORTFOLIO_OF
        base["cert__Portfolio"] = base[f"cert__{a}"] | base[f"cert__{b}"]
        # parallel model: run both arms; wall = first certificate when one lands,
        # both-exhausted (max of arm walls) when neither does — consistent with
        # the single-arm medians, which also include capped rows.
        walls = base[[f"wall__{a}", f"wall__{b}"]]
        first_cert = walls.where(base[[f"cert__{a}", f"cert__{b}"]].to_numpy()).min(axis=1)
        base["wall__Portfolio"] = first_cert.where(base["cert__Portfolio"], walls.max(axis=1))
        labels = labels + ["Portfolio"]
    def cell(g: pd.DataFrame, label: str) -> tuple[int, float, float]:
        # certified count over all queries; median / p90 wall over the CERTIFIED
        # subset only -> "when it proves optimal, how slow is the slowest 10%".
        # Complements cert% (how many) without the 120 s cap pile-up that would
        # pin p90 to the cap wherever cert < 90%.
        mask = g[f"cert__{label}"].astype(bool)
        w = g.loc[mask, f"wall__{label}"].astype(float)
        w = w[np.isfinite(w)]
        if len(w) == 0:
            return int(mask.sum()), float("nan"), float("nan")
        return int(mask.sum()), float(w.median()), float(w.quantile(0.90))

    rows = []
    for ds, g in base.groupby("dataset", sort=True):
        row: dict = {"dataset": ds, "n": len(g)}
        for label in labels:
            row[label] = cell(g, label)
        rows.append(row)
    total = {"dataset": "TOTAL", "n": len(base)}
    for label in labels:
        total[label] = cell(base, label)
    rows.append(total)
    meta = {"labels": labels, "n_rows": len(base)}
    return pd.DataFrame(rows), meta


def _name(dataset: str, n: int) -> str:
    """Dataset label with its instance count folded in: ``abalone (150)``."""
    return f"{dataset} ({n})"


def to_markdown(t: pd.DataFrame, labels: list[str]) -> str:
    header = "| dataset (n) | " + " | ".join(f"{l} cert. / t" for l in labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    lines = [header, sep]
    for _, r in t.iterrows():
        cells = []
        for l in labels:
            c, med, p90 = r[l]
            cells.append(f"**{fmt_pct(c, r['n'])}** · {fmt_pair(med, p90)}")
        ds = "TOTAL" if r["dataset"] == "TOTAL" else r["dataset"]
        bold = "**" if r["dataset"] == "TOTAL" else ""
        lines.append(f"| {bold}{_name(ds, r['n'])}{bold} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tex_wall(median: float, p90: float) -> str:
    return (fmt_pair(median, p90)
            .replace(" ms", r"\,ms").replace(" s", r"\,s"))


def to_latex(t: pd.DataFrame, labels: list[str]) -> str:
    cols = "l" + "rr" * len(labels)
    head1 = "Dataset ($n$) & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{l}}}" for l in labels) + r" \\"
    head2 = " & " + " & ".join([r"cert.\ & $\tilde t\,/\,t_{90}$"] * len(labels)) + r" \\"
    body = []
    for _, r in t.iterrows():
        if r["dataset"] == "TOTAL":
            name = r"\midrule Total (%d)" % r["n"]
        else:
            name = _name(r["dataset"].replace("_", r"\_"), r["n"])
        cells = []
        for l in labels:
            c, med, p90 = r[l]
            cells.append(f"{fmt_pct(c, r['n']).replace('%', r'\%')} & {_tex_wall(med, p90)}")
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{tabular}{" + cols + "}", r"\toprule",
        head1, head2, r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_ROOT / "results" / "conference_table"))
    args = ap.parse_args()
    t, meta = build()
    md = to_markdown(t, meta["labels"])
    tex = to_latex(t, meta["labels"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(md + "\n")
    (out / "table.tex").write_text(tex + "\n")
    print(md)
    print(f"\nwrote {out}/table.md and table.tex ({meta['n_rows']} matched rows)")


if __name__ == "__main__":
    main()
