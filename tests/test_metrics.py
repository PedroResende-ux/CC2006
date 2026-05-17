"""Unit tests for the additive metric helpers in :mod:`ai.dt_pipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.dt_pipeline import confusion_matrix, macro_f1


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
