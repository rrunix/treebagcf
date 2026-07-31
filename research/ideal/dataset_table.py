"""Dataset / forest characteristics table for the soft-voting suite.

One row per dataset (markdown + LaTeX booktabs to
``results/dataset_table/table.tex``):

- ``d``          : number of (numeric) features, as the harness loads them;
- ``n``          : number of observations (instances);
- ``lvl/feat``   : mean number of split levels (distinct split thresholds) per
                   feature across the whole forest — how finely each feature is
                   cut;
- ``leaves``     : total leaves in the forest ($= |\\mathcal{R}|$, the rule-set
                   size that every inner-loop operation scales with);
- ``depth``      : mean leaf depth (average root-to-leaf path length).

Forests are the suite's: 100 trees, unlimited depth, scikit-learn seed 0
(soft voting does not affect structure).  Datasets are listed in the same
order as the Setup paragraph of the paper.

    uv run python research/ideal/dataset_table.py [--out research/results/dataset_table]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent.resolve()
_ROOT = _HERE.parent  # research/ — exp_suite, results/, data/ live one level up
sys.path.insert(0, str(_ROOT))

# Same order as the Setup paragraph in paper.tex.
DATASETS = [
    "banknote", "breast_cancer", "digits_38", "digits_even", "ionosphere",
    "mammographic_masses", "occupancy", "pima", "wine", "abalone", "seismic",
]
SEED = 0


def _tree_leaf_stats(tree) -> tuple[int, int]:
    """(#leaves, sum of leaf depths) for one sklearn tree."""
    cl, cr = tree.children_left, tree.children_right
    leaves = 0
    depth_sum = 0
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        if cl[node] == -1:  # leaf
            leaves += 1
            depth_sum += d
        else:
            stack.append((cl[node], d + 1))
            stack.append((cr[node], d + 1))
    return leaves, depth_sum


def forest_stats(rf, n_features: int) -> dict:
    thresholds: list[set] = [set() for _ in range(n_features)]
    total_leaves = 0
    total_depth = 0
    for est in rf.estimators_:
        t = est.tree_
        feat, thr = t.feature, t.threshold
        for f, th in zip(feat, thr):
            if f >= 0:  # internal node
                thresholds[f].add(float(th))
        lv, ds = _tree_leaf_stats(t)
        total_leaves += lv
        total_depth += ds
    thr_per_feat = np.mean([len(s) for s in thresholds])
    return {
        "leaves": total_leaves,
        "thr_per_feat": float(thr_per_feat),
        "mean_depth": total_depth / total_leaves,
    }


def _warmup_seconds(rf, X: np.ndarray, train_idx: np.ndarray) -> float:
    """Wall time of the alpha warm-up alone, on the train forest.

    Measured as build(with warm-up) - build(parse only) so the shared parse cost
    cancels, leaving the warm-up.  The warm-up sees train rows only (k=16, 1000
    root iters), matching the experiment; ``alpha_cache=None`` forces a fresh
    compute (no cache load) and writes nothing, so existing caches are untouched.
    """
    from baselines import registry

    common = dict(X=X, feature_ranges=np.ones(X.shape[1]), voting="soft",
                  norm=1, threshold=0.5, pool_idx=train_idx)
    warm = registry.build("calf", rf, alpha_warm=True, alpha_warm_k=16,
                          alpha_warm_iters=1000, alpha_cache=None, **common)
    base = registry.build("calf", rf, alpha_warm=False, **common)
    return max(0.0, float(warm._build_time_s) - float(base._build_time_s))


def summarize(data_root: str, models_dir: str, model_tag: str) -> list[dict]:
    """Per-dataset stats from the cached TRAIN-only forests of the holdout run
    (so leaves/depth match the models actually evaluated), plus the one-time
    alpha warm-up cost our method pays per forest."""
    import joblib
    import json
    from calf.datasets import load

    rows = []
    for ds in DATASETS:
        X, _y, _di = load(ds, root=data_root, numeric_only=True)
        mp = Path(models_dir) / f"{ds}__default__{model_tag}.joblib"
        rf = joblib.load(mp)
        train_idx = np.asarray(
            json.loads(mp.with_suffix(".split.json").read_text())["train_idx"],
            dtype=np.int64)
        s = forest_stats(rf, X.shape[1])
        warm = _warmup_seconds(rf, X, train_idx)
        rows.append({"dataset": ds, "d": X.shape[1], "n": X.shape[0],
                     "warm": warm, **s})
    return rows


def _fmt_warm(w: float) -> str:
    return f"{w:.2f}" if w < 10 else f"{w:.1f}"


def to_markdown(rows: list[dict]) -> str:
    out = ["| Dataset | Features | Obs. | Split levels/feat. | Leaves | Avg. depth | alpha warm-up (s) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['dataset']} | {r['d']} | {r['n']} | "
                   f"{r['thr_per_feat']:.0f} | {r['leaves']} | {r['mean_depth']:.1f} | "
                   f"{_fmt_warm(r['warm'])} |")
    return "\n".join(out)


def to_latex(rows: list[dict]) -> str:
    body = []
    for r in rows:
        name = r["dataset"].replace("_", r"\_")
        body.append(f"\\texttt{{{name}}} & {r['d']} & {r['n']} & "
                    f"{r['thr_per_feat']:.0f} & {r['leaves']} & {r['mean_depth']:.1f} & "
                    f"{_fmt_warm(r['warm'])}"
                    r" \\")
    return "\n".join([
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Feat. & Obs. & Split lvl./feat. & Leaves & Avg.\ depth & $\alpha$ warm-up (s) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_ROOT / "results" / "dataset_table"))
    ap.add_argument("--data-root", default=str(_ROOT / "data"))
    ap.add_argument("--models-dir",
                    default=str(_ROOT / "results" / "holdout_soft_120" / "models"))
    ap.add_argument("--model-tag", default="tf0.2s0")
    args = ap.parse_args()
    rows = summarize(args.data_root, args.models_dir, args.model_tag)
    md, tex = to_markdown(rows), to_latex(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(md + "\n")
    (out / "table.tex").write_text(tex + "\n")
    print(md)
    print(f"\nwrote {out}/table.md and table.tex")


if __name__ == "__main__":
    main()
