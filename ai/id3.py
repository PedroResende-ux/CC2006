"""Pure ID3 decision-tree algorithm.

Domain-agnostic primitives used by :mod:`ai.dt_pipeline` (PopOut + Iris)
and any other consumer that wants a classical multi-way categorical ID3.

The tree is represented as a nested ``dict``::

    leaf  = {'is_leaf': True,  'label': <class>, 'majority': <class>}
    inner = {'is_leaf': False, 'feature': <name>, 'majority': <class>,
             'children': {<feature_value>: <subtree>, ...}}

The ``majority`` field on inner nodes is the fallback used by :func:`predict_one`
when a test row carries a value that was never seen for the split feature
during training.

This module deliberately depends only on :mod:`numpy` and :mod:`pandas`;
it knows nothing about PopOut, Iris, or how the input was discretised.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Information theory
# ---------------------------------------------------------------------------


def entropy(y: pd.Series) -> float:
    """Shannon entropy (base 2) of the class distribution in ``y``.

    Returns ``0.0`` for an empty series or a pure series.
    """
    if len(y) == 0:
        return 0.0
    counts = y.value_counts().values
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(X: pd.DataFrame, y: pd.Series, feature: str) -> float:
    """Information gain obtained by splitting ``X``/``y`` on ``feature``.

    Multi-way split over the unique values of ``feature`` — the classical
    ID3 formulation.
    """
    total_n = len(y)
    if total_n == 0:
        return 0.0
    weighted = sum(
        (len(subset) / total_n) * entropy(subset)
        for val in X[feature].unique()
        for subset in [y[X[feature] == val]]
    )
    return entropy(y) - weighted


def best_feature(X: pd.DataFrame, y: pd.Series, features: List[str]) -> str:
    """Return the feature in ``features`` with the highest information gain."""
    gains = {f: information_gain(X, y, f) for f in features}
    return max(gains, key=gains.get)


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def id3(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    max_depth: int = 10,
    min_samples: int = 20,
    depth: int = 0,
    parent_majority: Optional[Any] = None,
) -> dict:
    """Recursively build a categorical ID3 decision tree.

    Pre-pruning is governed by ``max_depth`` and ``min_samples``; no
    post-pruning is applied here (see :mod:`ai.dt_pipeline` for the
    pre-pruning grid search).
    """
    majority = y.value_counts().idxmax() if len(y) > 0 else parent_majority

    if len(y) == 0:
        return {'is_leaf': True, 'label': parent_majority, 'majority': parent_majority}
    if y.nunique() == 1:
        return {'is_leaf': True, 'label': y.iloc[0], 'majority': majority}
    if not features:
        return {'is_leaf': True, 'label': majority, 'majority': majority}
    if depth >= max_depth:
        return {'is_leaf': True, 'label': majority, 'majority': majority}
    if len(y) < min_samples:
        return {'is_leaf': True, 'label': majority, 'majority': majority}

    split_feature = best_feature(X, y, features)
    remaining = [f for f in features if f != split_feature]

    node = {
        'is_leaf': False,
        'feature': split_feature,
        'majority': majority,
        'children': {},
    }

    for val in X[split_feature].unique():
        mask = X[split_feature] == val
        node['children'][val] = id3(
            X[mask], y[mask], remaining,
            max_depth, min_samples,
            depth + 1, majority,
        )

    return node


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_one(row: pd.Series, tree: dict) -> Any:
    """Classify a single row by walking the tree.

    Falls back to the node's ``majority`` class when the row carries a
    feature value that was never observed during training.
    """
    if tree['is_leaf']:
        return tree['label']
    value = row[tree['feature']]
    if value not in tree['children']:
        return tree['majority']
    return predict_one(row, tree['children'][value])


def predict(X: pd.DataFrame, tree: dict) -> np.ndarray:
    """Classify every row in ``X`` with the trained ``tree``."""
    cols = X.columns.tolist()
    return np.array([
        predict_one(pd.Series(row, index=cols), tree)
        for row in X.itertuples(index=False)
    ])


# ---------------------------------------------------------------------------
# Tree introspection
# ---------------------------------------------------------------------------


def count_nodes(tree: dict) -> int:
    """Total nodes (internal + leaves) in the subtree rooted at ``tree``."""
    if tree['is_leaf']:
        return 1
    return 1 + sum(count_nodes(c) for c in tree['children'].values())


def count_leaves(tree: dict) -> int:
    """Leaf-only count for the subtree rooted at ``tree``."""
    if tree['is_leaf']:
        return 1
    return sum(count_leaves(c) for c in tree['children'].values())


def tree_depth(tree: dict, current: int = 0) -> int:
    """Maximum depth (longest root-to-leaf path) of ``tree``."""
    if tree['is_leaf']:
        return current
    return max(tree_depth(c, current + 1) for c in tree['children'].values())
