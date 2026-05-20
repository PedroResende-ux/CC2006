"""Smoke tests for the ID3 algorithm and Iris warm-up pipeline."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from ai.id3 import entropy, id3, predict, predict_top_k


def test_entropy_pure_and_uniform_binary():
    """Entropy is 0 for a pure series and 1 for a 50/50 binary series."""
    assert entropy(pd.Series([1, 1, 1])) == 0.0
    assert math.isclose(entropy(pd.Series([1, 2])), 1.0, abs_tol=1e-9)


def test_id3_perfectly_separable_dataset():
    """ID3 trained on a separable 2-class set reaches 100% training accuracy."""
    X = pd.DataFrame({
        'colour': ['red', 'red', 'blue', 'blue', 'red', 'blue'],
        'shape':  ['square', 'circle', 'square', 'circle', 'circle', 'square'],
    })
    y = pd.Series(['A', 'A', 'B', 'B', 'A', 'B'])

    tree = id3(X, y, features=['colour', 'shape'], max_depth=5, min_samples=1)

    preds = predict(X, tree)
    assert (preds == y.values).all()


def test_run_iris_demo_after_fix(capsys):
    """Iris demo reaches >= 90% after switching to quantile discretisation.

    The previous single-IG-threshold-per-feature scheme capped accuracy at
    ~66.7%: it perfectly isolated setosa but reduced every other feature
    to a binary indicator that could not distinguish versicolor from
    virginica. After replacing it with 4-bucket equal-frequency quantile
    binning (post-refactor), the demo clears the canonical >= 90%
    sanity-check threshold consistently — empirically ~93%.
    """
    # Import lazily so a broken pipeline doesn't sink the other tests.
    from ai.dt_pipeline import run_iris_demo

    _tree, acc = run_iris_demo()
    # Suppress the verbose pretty-printed tree from cluttering pytest output.
    capsys.readouterr()

    assert acc >= 90.0, f"Expected iris test accuracy >= 90%, got {acc:.2f}%"


# ---------------------------------------------------------------------------
# predict_top_k
# ---------------------------------------------------------------------------


def test_predict_top_k_basic():
    """Top-k returns leaf classes ordered by ``class_counts`` desc, k cap honoured."""
    leaf = {
        'is_leaf': True,
        'label': 'A',
        'majority': 'A',
        'class_counts': {'A': 10, 'B': 5, 'C': 3, 'D': 1},
    }
    row = pd.Series({'unused': 0})

    assert predict_top_k(row, leaf, k=1) == ['A']
    assert predict_top_k(row, leaf, k=2) == ['A', 'B']
    assert predict_top_k(row, leaf, k=3) == ['A', 'B', 'C']
    # Asking for more classes than the leaf has returns everything available.
    assert predict_top_k(row, leaf, k=10) == ['A', 'B', 'C', 'D']


def test_predict_top_k_walks_internal_node():
    """A row whose feature value matches a child descends to that child."""
    tree = {
        'is_leaf': False,
        'feature': 'colour',
        'majority': 'X',
        'n_samples': 6,
        'children': {
            'red': {
                'is_leaf': True, 'label': 'A', 'majority': 'A',
                'class_counts': {'A': 4, 'B': 1},
            },
            'blue': {
                'is_leaf': True, 'label': 'B', 'majority': 'B',
                'class_counts': {'B': 3, 'A': 2},
            },
        },
    }
    assert predict_top_k(pd.Series({'colour': 'red'}), tree, k=2) == ['A', 'B']
    assert predict_top_k(pd.Series({'colour': 'blue'}), tree, k=2) == ['B', 'A']


def test_predict_top_k_unseen_value_falls_back_to_majority():
    """Unseen feature value returns ``[majority]`` — matches predict_one."""
    tree = {
        'is_leaf': False,
        'feature': 'colour',
        'majority': 'X',
        'n_samples': 4,
        'children': {
            'red': {'is_leaf': True, 'label': 'A', 'majority': 'A',
                    'class_counts': {'A': 4}},
        },
    }
    assert predict_top_k(pd.Series({'colour': 'green'}), tree, k=3) == ['X']


def test_predict_top_k_backwards_compat():
    """A leaf without ``class_counts`` (old pickle) collapses to ``[label]``."""
    leaf = {
        'is_leaf': True,
        'label': 'X',
        'majority': 'X',
        # No ``class_counts`` — represents a pre-Prompt-E2 pickle.
    }
    row = pd.Series({'unused': 0})
    assert predict_top_k(row, leaf, k=3) == ['X']


def test_id3_leaves_carry_class_counts():
    """every leaf has ``class_counts`` matching its partition."""
    X = pd.DataFrame({
        'colour': ['red', 'red', 'blue', 'blue', 'blue'],
    })
    y = pd.Series(['A', 'A', 'B', 'B', 'A'])
    tree = id3(X, y, features=['colour'], max_depth=2, min_samples=1)

    # Root splits on colour; both children should be leaves with class_counts.
    assert not tree['is_leaf']
    assert 'n_samples' in tree
    assert tree['n_samples'] == 5
    for val, child in tree['children'].items():
        assert child['is_leaf']
        assert 'class_counts' in child
        cc = child['class_counts']
        # class_counts sums to the partition size
        assert sum(cc.values()) == int((X['colour'] == val).sum())
