"""Unit tests for the additive metric helpers in :mod:`ai.dt_pipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.dt_pipeline import (
    POPOUT_BIN_DEFINITIONS,
    bin_features,
    confusion_matrix,
    macro_f1,
    top_k_accuracy,
)
from ai.id3 import id3


def test_macro_f1_perfect_prediction():
    """Macro-F1 is 1.0 when every prediction matches the truth."""
    y_true = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert macro_f1(y_true, y_pred) == 1.0


def test_macro_f1_majority_class_baseline():
    """Always predicting the majority class collapses macro-F1 below 0.4.

    For a 3-class imbalanced set where the majority dominates, the
    classifier scores F1=1 on the majority's recall × precision mix but
    F1=0 on the other two classes. Macro-F1 must therefore be well below
    a 'reasonable' threshold; we use 0.4 as the bound.
    """
    # 3 classes, heavily imbalanced toward class 0
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 2])
    y_pred = np.zeros(len(y_true), dtype=int)
    score = macro_f1(y_true, y_pred)
    assert score < 0.4, f"expected macro_f1 < 0.4, got {score:.4f}"


def test_confusion_matrix_diagonal():
    """Perfect predictions place all mass on the diagonal (trace == total)."""
    y_true = pd.Series([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    cm = confusion_matrix(y_true, y_pred)
    trace = int(np.trace(cm.values))
    assert trace == len(y_true), (
        f"trace ({trace}) should equal total ({len(y_true)})"
    )
    # And off-diagonal must be zero
    off_diag = int(cm.values.sum() - trace)
    assert off_diag == 0


# ---------------------------------------------------------------------------
# Part 0 — POPOUT_BIN_DEFINITIONS schema fix
# ---------------------------------------------------------------------------


def test_bin_features_no_nan_strings_across_full_domain():
    """After Prompt E2 Part 0, ``bin_features`` covers the full count range.

    The pre-fix bins had upper bound ``6`` on ``own/opp_pieces_bottom_row``
    (board width is 7 → counts in [0, 7]) and ``60`` on ``move_count``
    (dataset has rows up to 68). Values above those caps came through
    ``pd.cut`` as ``NaN`` and then as the literal ``"nan"`` string via
    ``.astype(str)``, which the tree picked up as a spurious fourth bucket.
    This test guards against regressions of either bin edge.
    """
    # Cover the full legitimate domain of each continuous column.
    domain = pd.DataFrame({
        'move_count': list(range(0, 80)),                   # 0..79 incl. > 60
        'own_pieces_bottom_row': list(range(0, 8))          # 0..7 incl. = 7
                                 + [0] * 72,
        'opp_pieces_bottom_row': list(range(0, 8))
                                 + [0] * 72,
    })
    continuous_cols = list(POPOUT_BIN_DEFINITIONS.keys())

    binned_train, binned_val, binned_test = bin_features(
        domain.copy(), domain.copy(), domain.copy(), continuous_cols,
    )

    for col in continuous_cols:
        for binned in (binned_train, binned_val, binned_test):
            uniques = set(binned[col].unique().tolist())
            assert 'nan' not in uniques, (
                f"column {col!r} produced 'nan' strings after binning: "
                f"{sorted(uniques)}"
            )


# ---------------------------------------------------------------------------
# top_k_accuracy
# ---------------------------------------------------------------------------


def test_top_k_accuracy_perfect():
    """A tree that perfectly classifies its training set scores 1.0 for any k.

    On the training set the leaves are pure (one class each), so the top-k
    list always contains the true class — top-1, top-2 and top-3 all
    reach 1.0.
    """
    X = pd.DataFrame({
        'colour': ['red', 'red', 'blue', 'blue'],
    })
    y = pd.Series([0, 0, 1, 1])
    tree = id3(X, y, features=['colour'], max_depth=2, min_samples=1)

    for k in (1, 2, 3):
        score = top_k_accuracy(y, X, tree, k=k)
        assert score == 1.0, f"k={k}: expected 1.0, got {score}"


def test_top_k_accuracy_top1_lower_bound():
    """When the leaf's top-1 is wrong but the true class is in the top-3,
    ``top_k_accuracy(k=3) > top_k_accuracy(k=1)``.

    A trivial 'leaf-only' tree where the leaf has three classes in its
    ``class_counts`` distribution. The leaf's majority is class 0, so
    rows with labels 1 or 2 are wrong under top-1 but correct under
    top-3 (all three classes appear in ``class_counts``).
    """
    leaf = {
        'is_leaf': True,
        'label': 0,
        'majority': 0,
        'class_counts': {0: 10, 1: 8, 2: 5},
    }
    # Three rows, one per class. The tree doesn't look at the feature
    # because the root is a leaf — any payload column works.
    X = pd.DataFrame({'unused': [0, 0, 0]})
    y = pd.Series([0, 1, 2])

    top1 = top_k_accuracy(y, X, leaf, k=1)
    top3 = top_k_accuracy(y, X, leaf, k=3)

    # Only row with label 0 is correct under top-1.
    assert top1 == pytest.approx(1 / 3)
    # All three classes are in the top-3 of the leaf, so all rows pass.
    assert top3 == 1.0
    assert top3 > top1


def test_top_k_accuracy_empty_X_returns_zero():
    """Empty input yields 0.0 rather than crashing on a zero-divide."""
    X = pd.DataFrame({'colour': pd.Series(dtype=object)})
    y = pd.Series(dtype=int)
    leaf = {'is_leaf': True, 'label': 0, 'majority': 0, 'class_counts': {0: 1}}
    assert top_k_accuracy(y, X, leaf, k=3) == 0.0
