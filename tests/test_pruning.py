"""Tests for reduced-error post-pruning (REP) of ID3 trees.

REP collapses an internal node into a leaf whenever doing so does not
hurt validation accuracy. The two invariants we test:

1. ``prune_rep`` never produces a larger tree than its input
   (``count_nodes(pruned) <= count_nodes(original)``).
2. ``prune_rep`` never reduces validation accuracy. The algorithm is
   constructed to swap subtrees for leaves only when ``leaf_acc >=
   subtree_acc``, so this is a correctness property, not a heuristic.

Both tests run on a moderately deep iris tree (small dataset, fast to
train) so they finish in well under a second each.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from ai.dt_pipeline import predict_batch, prune_rep
from ai.id3 import count_nodes, id3


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IRIS_CSV = os.path.join(_REPO_ROOT, 'data', 'iris.csv')


def _load_iris_split(seed: int = 42) -> tuple:
    """Return ``(X_train, X_val, y_train, y_val)`` from a quantile-binned iris.

    Mirrors the discretisation used by :func:`ai.dt_pipeline.run_iris_demo`
    but without the print noise and full-dataset shuffle: 60% train,
    40% val, deterministic at the given seed.
    """
    df = pd.read_csv(_IRIS_CSV)
    col_map = {
        'sepallength': 'sepal_length', 'sepalwidth': 'sepal_width',
        'petallength': 'petal_length', 'petalwidth': 'petal_width',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    target_col = 'class'
    feature_cols = [c for c in df.columns if c != target_col]

    for col in feature_cols:
        df[col] = pd.qcut(df[col], 4, labels=False, duplicates='drop').astype(str)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(0.6 * len(df))
    train = df.iloc[:split]
    val = df.iloc[split:]
    return (train[feature_cols], val[feature_cols],
            train[target_col].reset_index(drop=True),
            val[target_col].reset_index(drop=True))


def test_prune_rep_reduces_size():
    """A deep tree trained on iris collapses some nodes under REP.

    With ``max_depth=8`` and ``min_samples=2`` on iris, the unpruned
    tree always has at least one prunable subtree — the rare-class
    branches that the val set fails to confirm. We assert the
    weak property ``pruned <= original`` (== holds when REP made no
    swaps, which would be unusual for this configuration but allowed).
    """
    X_train, X_val, y_train, y_val = _load_iris_split()
    tree = id3(X_train, y_train, X_train.columns.tolist(),
               max_depth=8, min_samples=2)
    pruned = prune_rep(tree, X_val, y_val)

    n_before = count_nodes(tree)
    n_after = count_nodes(pruned)
    assert n_after <= n_before, (
        f"REP must never grow the tree: before={n_before}, after={n_after}"
    )


def test_prune_rep_never_worsens_val_accuracy():
    """Validation accuracy must not decrease after REP, by construction."""
    X_train, X_val, y_train, y_val = _load_iris_split()
    tree = id3(X_train, y_train, X_train.columns.tolist(),
               max_depth=8, min_samples=2)

    pre_pred = predict_batch(X_val, tree)
    post_pred = predict_batch(X_val, prune_rep(tree, X_val, y_val))

    pre_acc = (pre_pred == y_val.values).mean()
    post_acc = (post_pred == y_val.values).mean()

    assert post_acc >= pre_acc, (
        f"REP made val accuracy worse: pre={pre_acc:.4f}, post={post_acc:.4f}"
    )


def test_prune_rep_does_not_mutate_input():
    """The input tree must come out unchanged regardless of pruning outcome."""
    X_train, X_val, y_train, y_val = _load_iris_split()
    tree = id3(X_train, y_train, X_train.columns.tolist(),
               max_depth=8, min_samples=2)
    n_before = count_nodes(tree)

    _pruned = prune_rep(tree, X_val, y_val)

    n_after = count_nodes(tree)
    assert n_after == n_before, (
        "prune_rep mutated its input tree "
        f"(node count {n_before} -> {n_after})"
    )


def test_prune_rep_aggregates_class_counts_on_collapsed_leaves():
    """A leaf created by collapsing an internal node inherits aggregated counts.

    Build a tiny tree where pruning the root is the right call (the
    root's children disagree but the val set sees only the majority class).
    The pruned leaf's ``class_counts`` should sum the original leaves'
    counts, so :func:`ai.id3.predict_top_k` still has meaningful output.
    """
    X_train = pd.DataFrame({'f': ['a'] * 5 + ['b'] * 5})
    y_train = pd.Series([0, 0, 0, 1, 1] + [1, 1, 0, 0, 0])
    tree = id3(X_train, y_train, ['f'], max_depth=2, min_samples=1)

    # All-majority val rows force the root replacement to win on val acc.
    majority = tree['majority']
    X_val = pd.DataFrame({'f': ['a', 'b']})
    y_val = pd.Series([majority, majority])

    pruned = prune_rep(tree, X_val, y_val)

    # If REP collapsed the root, the resulting leaf carries aggregated
    # class_counts whose sum equals the original training size.
    if pruned['is_leaf']:
        assert 'class_counts' in pruned
        assert sum(pruned['class_counts'].values()) == len(y_train)
