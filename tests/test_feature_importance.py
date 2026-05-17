"""Tests for :func:`ai.dt_pipeline.feature_importance`.

``feature_importance`` walks a trained tree and ranks the split features
by the number of training samples that flowed through nodes splitting on
them (with a tie-breaker on the count of splits). The root sits above
the entire training set, so its split feature is always at the top of
the ranking for trees where the root is the only place the feature
appears.
"""

from __future__ import annotations

import pandas as pd

from ai.dt_pipeline import feature_importance
from ai.id3 import id3


def test_feature_importance_root_is_top():
    """The root's split feature heads the importance table.

    Feature ``A`` perfectly separates ``y`` so the root splits on it
    with 6 samples. Feature ``B`` carries no signal beyond what ``A``
    already explains, so the resulting tree has no further internal
    nodes and ``A`` is the sole entry — but still must be reported at
    the top with ``n_samples_total == 6``.
    """
    X = pd.DataFrame({
        'A': ['a', 'a', 'b', 'b', 'a', 'b'],
        'B': ['x', 'y', 'x', 'y', 'x', 'y'],
    })
    y = pd.Series([0, 0, 1, 1, 0, 1])  # perfectly separable by A
    tree = id3(X, y, features=['A', 'B'], max_depth=3, min_samples=1)

    fi = feature_importance(tree)

    assert len(fi) >= 1
    assert fi.iloc[0]['feature'] == 'A'
    assert fi.iloc[0]['n_samples_total'] == 6


def test_feature_importance_sample_weighted_ranks_above_split_count():
    """A feature near the root outranks a deeper, more-frequent feature."""
    # Two features. ``A`` splits the root (12 samples). ``B`` splits the
    # two children (6 samples each = 12 total). Pure counts: both have
    # 1 vs 2 splits, so ``B`` would win on n_splits alone — but our
    # sample-weighted ranking gives them the same n_samples_total (12),
    # which is the right framing: both are equally "important" by
    # partition-size weight.
    X = pd.DataFrame({
        'A': ['a'] * 6 + ['b'] * 6,
        'B': (['x', 'y'] * 3) + (['x', 'y'] * 3),
    })
    # y depends jointly on (A, B): A picks the side, B disambiguates within.
    y = pd.Series([0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3])
    tree = id3(X, y, features=['A', 'B'], max_depth=3, min_samples=1)

    fi = feature_importance(tree)

    # Both features should be present.
    assert set(fi['feature']) == {'A', 'B'}

    # ``A`` is at the root (n_samples=12). ``B`` splits each child
    # (6 + 6 = 12 cumulative). Both rank equally on n_samples_total but
    # ``A`` should still appear because it's the first split.
    a_row = fi[fi['feature'] == 'A'].iloc[0]
    b_row = fi[fi['feature'] == 'B'].iloc[0]
    assert a_row['n_samples_total'] == 12
    assert a_row['n_splits'] == 1
    assert b_row['n_samples_total'] == 12
    assert b_row['n_splits'] == 2


def test_feature_importance_empty_for_leaf_tree():
    """A tree that's just a leaf has no splits and returns an empty frame."""
    leaf = {
        'is_leaf': True, 'label': 0, 'majority': 0, 'class_counts': {0: 1},
    }
    fi = feature_importance(leaf)
    assert list(fi.columns) == ['feature', 'n_splits', 'n_samples_total']
    assert len(fi) == 0
