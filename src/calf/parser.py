"""Parse a fitted sklearn RandomForestClassifier into a flat rule list.

Each leaf becomes one Rule with a per-feature axis-aligned box: the tightest
interval implied by the root-to-leaf path.  Box convention matches sklearn's
splits: lo is exclusive (x[f] > lo[f]), hi is inclusive (x[f] <= hi[f]);
features not on the path get lo=-inf, hi=+inf.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TREE_LEAF = -1  # sklearn: children_left[node] == -1 marks a leaf


@dataclass(frozen=True)
class Rule:
    tree_id: int
    leaf_id: int
    lo: np.ndarray  # exclusive lower bound (x[f] > lo[f])
    hi: np.ndarray  # inclusive upper bound (x[f] <= hi[f])
    predicted_class: int
    proba1: float = 0.0  # P(class 1 | leaf) — the leaf's class-1 fraction (soft voting)

    def fires(self, x: np.ndarray) -> bool:
        return bool(np.all(x > self.lo) and np.all(x <= self.hi))


@dataclass(frozen=True)
class ParsedRF:
    n_trees: int
    n_classes: int
    n_features: int
    rules: tuple[Rule, ...]
    rules_by_tree: tuple[tuple[int, ...], ...]
    # For each feature, the sorted unique thresholds appearing in any rule.
    feature_thresholds: tuple[np.ndarray, ...]
    # Stacked rule bounds and per-rule scalar lookups (hot-path arrays).
    rules_lo_mat: np.ndarray   # (n_rules, n_features) float64
    rules_hi_mat: np.ndarray   # (n_rules, n_features) float64
    rules_tree_id: np.ndarray  # (n_rules,) int64
    rules_class: np.ndarray    # (n_rules,) int64
    # Per-rule leaf class-1 probability (leaf class-1 fraction).  Used by soft
    # (probability-averaged) voting; ignored by hard majority voting.
    rules_proba1: np.ndarray   # (n_rules,) float64
    # Per-rule threshold-grid bins for split scoring (see parse_sklearn_rf).
    rules_lo_bin_mat: np.ndarray  # (n_rules, n_features) int64
    rules_hi_bin_mat: np.ndarray  # (n_rules, n_features) int64

    def predict(self, X: np.ndarray) -> np.ndarray:
        """RF majority vote under the parsed (float64 hard-vote) semantics.

        The ground-truth oracle for the search: exactly one leaf fires per tree,
        each casts one vote for its class, and the argmax class wins.  Used by
        tests to confirm a returned counterfactual really reaches the target.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.zeros(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            votes = np.zeros(self.n_classes, dtype=np.int64)
            for rule_ids in self.rules_by_tree:
                for rid in rule_ids:  # exactly one rule per tree fires
                    r = self.rules[rid]
                    if r.fires(x):
                        votes[r.predicted_class] += 1
                        break
            out[i] = int(votes.argmax())
        return out


def parse_sklearn_rf(model) -> ParsedRF:
    """Parse a fitted sklearn RandomForestClassifier (numerical features only)."""
    if not hasattr(model, "estimators_"):
        raise TypeError(
            "expected a fitted sklearn RandomForestClassifier (has .estimators_); "
            f"got {type(model).__name__}"
        )
    n_features = int(model.n_features_in_)
    n_classes = int(model.n_classes_)
    n_trees = len(model.estimators_)

    all_rules: list[Rule] = []
    rules_by_tree: list[list[int]] = []
    threshold_sets: list[set[float]] = [set() for _ in range(n_features)]

    for tid, est in enumerate(model.estimators_):
        rule_ids: list[int] = []
        path_lo = np.full(n_features, -np.inf, dtype=np.float64)
        path_hi = np.full(n_features, np.inf, dtype=np.float64)
        _walk(est.tree_, 0, path_lo, path_hi, tid, all_rules, rule_ids, threshold_sets)
        rules_by_tree.append(rule_ids)

    feature_thresholds = tuple(np.array(sorted(s), dtype=np.float64) for s in threshold_sets)
    if all_rules:
        rules_lo_mat = np.stack([r.lo for r in all_rules])
        rules_hi_mat = np.stack([r.hi for r in all_rules])
        rules_tree_id = np.array([r.tree_id for r in all_rules], dtype=np.int64)
        rules_class = np.array([r.predicted_class for r in all_rules], dtype=np.int64)
        rules_proba1 = np.array([r.proba1 for r in all_rules], dtype=np.float64)
        # Map each rule's real-valued box bounds onto the per-feature sorted
        # threshold grid, so the split-scoring kernel can histogram rules by bin
        # index (integer) instead of comparing floats per candidate threshold.
        rules_lo_bin_mat = np.empty((len(all_rules), n_features), dtype=np.int64)
        rules_hi_bin_mat = np.empty((len(all_rules), n_features), dtype=np.int64)
        for f, ts in enumerate(feature_thresholds):
            n_ts = ts.size
            lo_f = rules_lo_mat[:, f]
            hi_f = rules_hi_mat[:, f]
            # lo_bin: 0 for -inf, k+1 for lo == ts[k]; hi_bin: k for hi == ts[k], n for +inf.
            # The +1/left-searchsorted convention lets choose_split_baseline read
            # "rules kept by the meet child" as a prefix count (cum_lo) and the
            # not-meet child as a suffix (parent - cum_hi).
            lo_bins = np.searchsorted(ts, lo_f, side="left").astype(np.int64) + 1
            lo_bins[np.isneginf(lo_f)] = 0
            hi_bins = np.searchsorted(ts, hi_f, side="left").astype(np.int64)
            hi_bins[np.isposinf(hi_f)] = n_ts
            rules_lo_bin_mat[:, f] = lo_bins
            rules_hi_bin_mat[:, f] = hi_bins
    else:
        rules_lo_mat = np.empty((0, n_features), dtype=np.float64)
        rules_hi_mat = np.empty((0, n_features), dtype=np.float64)
        rules_tree_id = np.empty(0, dtype=np.int64)
        rules_class = np.empty(0, dtype=np.int64)
        rules_proba1 = np.empty(0, dtype=np.float64)
        rules_lo_bin_mat = np.empty((0, n_features), dtype=np.int64)
        rules_hi_bin_mat = np.empty((0, n_features), dtype=np.int64)

    return ParsedRF(
        n_trees=n_trees,
        n_classes=n_classes,
        n_features=n_features,
        rules=tuple(all_rules),
        rules_by_tree=tuple(tuple(rs) for rs in rules_by_tree),
        feature_thresholds=feature_thresholds,
        rules_lo_mat=rules_lo_mat,
        rules_hi_mat=rules_hi_mat,
        rules_tree_id=rules_tree_id,
        rules_class=rules_class,
        rules_proba1=rules_proba1,
        rules_lo_bin_mat=rules_lo_bin_mat,
        rules_hi_bin_mat=rules_hi_bin_mat,
    )


def _walk(tree, node, path_lo, path_hi, tid, out_rules, out_rule_ids, threshold_sets):
    """DFS a single sklearn tree, emitting one Rule per leaf.

    path_lo/path_hi hold the running box implied by the splits above `node`;
    they are mutated on the way down and restored on the way up (so one buffer
    serves the whole traversal).  At a leaf the current box is snapshotted.
    """
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # (n_nodes, 1, n_classes)

    if children_left[node] == _TREE_LEAF:
        counts = value[node, 0, :].astype(np.float64)  # class vote counts at leaf
        total = counts.sum()
        # P(class 1 | leaf) — the leaf's class-1 fraction, for soft voting.
        proba1 = float(counts[1] / total) if (total > 0 and counts.shape[0] > 1) else 0.0
        rid = len(out_rules)
        out_rules.append(
            Rule(tree_id=tid, leaf_id=int(node), lo=path_lo.copy(), hi=path_hi.copy(),
                 predicted_class=int(counts.argmax()), proba1=proba1)
        )
        out_rule_ids.append(rid)
        return

    f = int(feature[node])
    t = float(threshold[node])
    threshold_sets[f].add(t)  # collect the global per-feature threshold grid

    # sklearn: left = (x[f] <= t), right = (x[f] > t).
    # Descend left with hi tightened to t, then restore; then right with lo=t.
    old_hi = path_hi[f]
    if t < old_hi:
        path_hi[f] = t
    _walk(tree, int(children_left[node]), path_lo, path_hi, tid, out_rules, out_rule_ids, threshold_sets)
    path_hi[f] = old_hi

    old_lo = path_lo[f]
    if t > old_lo:
        path_lo[f] = t
    _walk(tree, int(children_right[node]), path_lo, path_hi, tid, out_rules, out_rule_ids, threshold_sets)
    path_lo[f] = old_lo
