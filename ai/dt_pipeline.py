"""PopOut-specific ID3 pipeline: load → balance → split → bin → train → tune → evaluate → save.

This module wraps the pure ID3 primitives in :mod:`ai.id3` with the data
handling, balancing, binning, hyper-parameter tuning, visualisation,
persistence and CLI required by the assignment. It also contains the
Iris warm-up demo (which uses the same :func:`ai.id3.id3` algorithm
after information-gain binary discretisation).

Public surface (import this module to reach all of it):

* Dataset I/O:        :func:`inspect_dataset`, :func:`save_tree`, :func:`load_tree`
* Pre-processing:     :func:`balance_classes`, :func:`stratified_split`,
                      :func:`split_dataset`, :func:`bin_features`
* Training & tuning:  :func:`tune_pruning`, :func:`train_final_tree`
* Evaluation:         :func:`evaluate`, :func:`visualise_tree`
* Warm-up demo:       :func:`run_iris_demo`
* Convenience:        :func:`int_to_move`, :const:`POPOUT_BIN_DEFINITIONS`

CLI::

    # Full pipeline (with iris warm-up first):
    python -m ai.dt_pipeline [csv_path]

    # Predict on saved tree:
    python -m ai.dt_pipeline --predict <test_csv>
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from ai.id3 import (
    count_leaves,
    count_nodes,
    id3,
    information_gain,
    predict,
    tree_depth,
)


# ---------------------------------------------------------------------------
# Constants — PopOut-specific binning for continuous features
# ---------------------------------------------------------------------------


# Bin definitions for the three numeric, non-cell columns of the PopOut
# self-play dataset. Imported by :mod:`ai.dt_player` so train-time and
# inference-time binning stay in sync.
POPOUT_BIN_DEFINITIONS: dict = {
    'move_count': {
        'bins':   [-1, 10, 20, 30, 60],
        'labels': ['early', 'mid', 'late', 'endgame'],
    },
    'own_pieces_bottom_row': {
        'bins':   [-1, 1, 3, 6],
        'labels': ['low', 'mid', 'high'],
    },
    'opp_pieces_bottom_row': {
        'bins':   [-1, 1, 3, 6],
        'labels': ['low', 'mid', 'high'],
    },
}


# Default dataset path (relative to repository root).
_DEFAULT_DATASET_REL = os.path.join('data', 'popout_dataset_150k.csv')


# ---------------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------------


def int_to_move(n: int) -> str:
    """Convert an integer class label (0–13) back to a move string.

    Classes ``0..6`` are ``drop_0..drop_6``; ``7..13`` are ``pop_0..pop_6``.
    """
    if n < 7:
        return f"drop_{n}"
    return f"pop_{n - 7}"


def _move_to_int(x) -> int | None:
    """Inverse of :func:`int_to_move` with tolerance for raw int labels."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s.startswith('drop_'):
        return int(s.split('_')[1])
    if s.startswith('pop_'):
        return 7 + int(s.split('_')[1])
    try:
        return int(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Iris warm-up
# ---------------------------------------------------------------------------


def run_iris_demo(csv_path: str | None = None) -> tuple[dict, float]:
    """Run the ID3 warm-up on Iris (continuous → binary discretisation).

    Uses the same :func:`ai.id3.id3` algorithm as the PopOut pipeline, after
    discretising each feature via an information-gain-optimal binary split.

    Returns the trained tree and the test-set accuracy (%).
    """
    print("=" * 60)
    print("IRIS DATASET  WARM-UP")
    print("=" * 60)

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__),
                                '..', 'data', 'iris.csv')
    df = pd.read_csv(csv_path)

    col_map = {
        'sepallength': 'sepal_length', 'sepalwidth': 'sepal_width',
        'petallength': 'petal_length', 'petalwidth': 'petal_width',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    target_col = 'class'
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"\nLoaded Iris: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Classes: {df[target_col].unique()}")
    print(f"Features: {feature_cols}")

    print("\nDiscretising continuous features:")
    for col in feature_cols:
        unique_vals = np.sort(df[col].unique())
        best_ig = -1
        best_thresh = None

        for i in range(len(unique_vals) - 1):
            thresh = (unique_vals[i] + unique_vals[i + 1]) / 2.0
            temp = df[col].apply(lambda x: 'low' if x <= thresh else 'high')
            temp_df = pd.DataFrame({col: temp})
            ig = information_gain(temp_df, df[target_col], col)
            if ig > best_ig:
                best_ig = ig
                best_thresh = thresh

        df[col] = df[col].apply(
            lambda x: 'low' if x <= best_thresh else 'high'
        )
        print(f"  {col:<20} threshold={best_thresh:.3f}  IG={best_ig:.4f}")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split = int(0.8 * len(df_shuffled))
    train_df = df_shuffled.iloc[:split]
    test_df = df_shuffled.iloc[split:]

    X_train_iris = train_df[feature_cols]
    y_train_iris = train_df[target_col]
    X_test_iris = test_df[feature_cols]
    y_test_iris = test_df[target_col]

    print(f"\nTrain: {len(X_train_iris)} rows  |  Test: {len(X_test_iris)} rows")

    print("\nTraining ID3 on Iris...")
    iris_tree = id3(
        X_train_iris, y_train_iris,
        feature_cols,
        max_depth=10, min_samples=2,
    )

    y_pred_iris = predict(X_test_iris, iris_tree)
    acc = (y_pred_iris == y_test_iris.values).mean() * 100
    print(f"\nIris test accuracy: {acc:.2f}%")

    print("\nIris Decision Tree (full — small enough to display):")
    print("-" * 60)
    _print_node_iris(iris_tree, prefix="", is_last=True, depth=0)
    print("-" * 60)
    return iris_tree, acc


def _print_node_iris(tree: dict, prefix: str, is_last: bool, depth: int) -> None:
    """Box-drawing pretty-printer for the iris tree (string labels)."""
    connector = "└── " if is_last else "├── "
    child_pfx = prefix + ("    " if is_last else "│   ")

    if tree['is_leaf']:
        print(f"{prefix}{connector}🍃 {tree['label']}")
        return

    majority = tree['majority']
    print(f"{prefix}{connector}📦 [{tree['feature']}]  (majority: {majority})")

    children_items = list(tree['children'].items())
    for i, (val, child) in enumerate(children_items):
        last = (i == len(children_items) - 1)
        val_connector = "└── " if last else "├── "
        val_child_pfx = child_pfx + ("    " if last else "│   ")
        print(f"{child_pfx}{val_connector}= {val}")
        _print_node_iris(child, val_child_pfx, True, depth + 1)


# ---------------------------------------------------------------------------
# Load & inspect
# ---------------------------------------------------------------------------


def inspect_dataset(csv_path: str) -> tuple[pd.DataFrame, str, pd.Series, list]:
    """Load the PopOut dataset and report shape / class distribution / dtypes.

    Returns ``(df, move_col, counts, continuous_cols)``. ``df`` has the
    move column converted to integer class IDs in place.
    """
    print("=" * 60)
    print("LOAD and INSPECT DATASET")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"\nLoaded: {csv_path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    missing = df.isnull().sum().sum()
    if missing == 0:
        print(" No missing values")
    else:
        print(f" missing values found")

    move_col = None
    for col in df.columns:
        if col.lower() in ['class', 'move', 'label']:
            move_col = col
            break
    if move_col is None:
        move_col = df.columns[-1]
    print(f"Move column: '{move_col}'")

    y = df[move_col].apply(_move_to_int)
    valid = y.notna()
    print(f"Valid moves: {valid.sum():,} / {len(df):,} "
          f"({valid.sum() / len(df) * 100:.1f}%)")

    print("\nMove distribution:")
    print(f"{'Class':<6} {'Type':<12} {'Count':>10} {'%':>8}")
    print("-" * 40)

    counts = y[valid].value_counts().sort_index()
    all_classes = set(range(14))
    missing_classes = all_classes - set(counts.index)

    for move in range(14):
        count = counts.get(move, 0)
        pct = count / valid.sum() * 100 if valid.sum() > 0 else 0
        move_type = "DROP" if move < 7 else "POP"
        col = move if move < 7 else move - 7
        marker = "⚠️" if count == 0 else " "
        print(f"{marker} {move:<4}  {move_type} col {col:<5} {count:10,d} {pct:7.2f}%")

    if missing_classes:
        print(f"\n Missing classes: {sorted(missing_classes)}")
    else:
        print("\n All 14 move classes present (0-13)")

    if len(counts) > 0:
        max_c = counts.max()
        min_c = counts[counts > 0].min()
        ratio = max_c / min_c
        print(f"\nImbalance ratio (max/min): {ratio:.1f}:1")
        if ratio > 50:
            print(" Severe imbalance")

    continuous_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col == move_col:
            continue
        unique_vals = df[col].nunique()
        min_val = df[col].min()
        max_val = df[col].max()
        if unique_vals <= 3 or (min_val == 0 and max_val == 2 and unique_vals <= 3):
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    print(f"   Categorical features (no binning needed): {len(categorical_cols)}")
    print(f"   Continuous features (need binning): {len(continuous_cols)}")

    if continuous_cols:
        print("\n   Continuous features requiring binning:")
        for col in continuous_cols:
            print(f"     - {col}: range [{df[col].min()}, {df[col].max()}], "
                  f"{df[col].nunique()} unique values")

    df[move_col] = y
    return df, move_col, counts, continuous_cols


# ---------------------------------------------------------------------------
# Balance classes
# ---------------------------------------------------------------------------


def balance_classes(
    df: pd.DataFrame,
    move_col: str,
    cap: int = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Undersample over-represented classes to ``cap``.

    Under-represented classes are kept whole. Returns a shuffled,
    index-reset DataFrame.
    """
    print("=" * 60)
    print("BALANCE CLASSES")
    print("=" * 60)

    print(f"\nCap per class: {cap:,}")
    print(f"Before balancing: {len(df):,} rows")

    balanced_parts = []

    print(f"\n{'Class':<6} {'Type':<12} {'Before':>10} {'After':>10}")
    print("-" * 42)

    for move_int in range(14):
        subset = df[df[move_col] == move_int]
        before_count = len(subset)

        if before_count == 0:
            col_n = move_int if move_int < 7 else move_int - 7
            print(f"  {move_int:<4}  "
                  f"{'DROP' if move_int < 7 else 'POP'} col {col_n}"
                  f"   {0:>10,}  {0:>10,}  ⚠️ missing")
            continue

        sampled = subset.sample(n=min(before_count, cap), random_state=random_state)
        after_count = len(sampled)
        move_type = "DROP" if move_int < 7 else "POP"
        col_num = move_int if move_int < 7 else move_int - 7
        print(f"  {move_int:<4}  {move_type} col {col_num:<5} "
              f"{before_count:>10,} {after_count:>10,}")
        balanced_parts.append(sampled)

    df_balanced = (pd.concat(balanced_parts)
                   .sample(frac=1, random_state=random_state)
                   .reset_index(drop=True))

    print(f"\nAfter balancing: {len(df_balanced):,} rows")
    print(f"Reduction: {len(df):,} → {len(df_balanced):,} "
          f"({len(df_balanced) / len(df) * 100:.1f}% of original)")

    return df_balanced


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple:
    """Stratified train/test split using numpy/pandas only.

    Each class is permuted independently and split proportionally, then
    re-shuffled across classes.
    """
    rng = np.random.RandomState(random_state)

    train_indices: list = []
    test_indices: list = []

    for label in y.unique():
        class_idx = y[y == label].index.tolist()
        class_idx = [class_idx[i] for i in rng.permutation(len(class_idx))]

        n_test = max(1, int(len(class_idx) * test_size))
        n_train = len(class_idx) - n_test

        train_indices.extend(class_idx[:n_train])
        test_indices.extend(class_idx[n_train:])

    train_indices = [train_indices[i]
                     for i in rng.permutation(len(train_indices))]
    test_indices = [test_indices[i]
                    for i in rng.permutation(len(test_indices))]

    return (X.loc[train_indices].reset_index(drop=True),
            X.loc[test_indices].reset_index(drop=True),
            y.loc[train_indices].reset_index(drop=True),
            y.loc[test_indices].reset_index(drop=True))


def split_dataset(
    df_balanced: pd.DataFrame,
    move_col: str,
    random_state: int = 42,
) -> tuple:
    """Stratified split: 80% train+val, 20% test; then 90/10 of train+val."""
    X = df_balanced.drop(columns=[move_col])
    y = df_balanced[move_col]

    print(f"\nTotal samples: {len(df_balanced):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {y.nunique()}")

    X_trainval, X_test, y_trainval, y_test = stratified_split(
        X, y, test_size=0.20, random_state=random_state,
    )

    X_train, X_val, y_train, y_val = stratified_split(
        X_trainval, y_trainval, test_size=0.10, random_state=random_state,
    )

    total = len(df_balanced)
    print(f"\n{'Split':<12} {'Rows':>10} {'% of total':>12}")
    print("-" * 36)
    print(f"{'Train':<12} {len(X_train):>10,} {len(X_train) / total * 100:>11.1f}%")
    print(f"{'Validation':<12} {len(X_val):>10,} {len(X_val) / total * 100:>11.1f}%")
    print(f"{'Test':<12} {len(X_test):>10,} {len(X_test) / total * 100:>11.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Bin numeric features
# ---------------------------------------------------------------------------


def bin_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_cols: list,
) -> tuple:
    """Bin the PopOut continuous features using :const:`POPOUT_BIN_DEFINITIONS`.

    Returns copies of ``(X_train, X_val, X_test)`` with the configured
    continuous columns converted to string-typed categorical labels.
    """
    print("=" * 60)
    print("BIN NUMERIC FEATURES")
    print("=" * 60)

    if not continuous_cols:
        print("No continuous features to bin.")
        return X_train, X_val, X_test

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    print(f"\nBinning {len(continuous_cols)} continuous feature(s):\n")

    for col in continuous_cols:
        if col not in POPOUT_BIN_DEFINITIONS:
            print(f"  ⚠️  No bin definition for '{col}' — skipping")
            continue

        bins = POPOUT_BIN_DEFINITIONS[col]['bins']
        labels = POPOUT_BIN_DEFINITIONS[col]['labels']

        X_train[col] = pd.cut(X_train[col], bins=bins, labels=labels).astype(str)
        X_val[col] = pd.cut(X_val[col], bins=bins, labels=labels).astype(str)
        X_test[col] = pd.cut(X_test[col], bins=bins, labels=labels).astype(str)

        print(f"   '{col}'")
        print(f"     Bins:   {bins}")
        print(f"     Labels: {labels}")

    return X_train, X_val, X_test


# ---------------------------------------------------------------------------
# Hyper-parameter tuning (pre-pruning grid search)
# ---------------------------------------------------------------------------


def tune_pruning(
    X_train,
    y_train,
    X_val,
    y_val,
    skip: bool = False,
    preset_depth: int | None = None,
    preset_min_samples: int | None = None,
) -> tuple:
    """Grid search over ``max_depth`` × ``min_samples`` on the validation set.

    Args:
        skip: If True, skip the grid search and return ``preset_*`` values.
        preset_depth: ``max_depth`` to use when ``skip=True``.
        preset_min_samples: ``min_samples`` to use when ``skip=True``.

    Returns:
        ``(best_depth, best_min_samples, results)``. ``results`` is empty
        when ``skip=True``.
    """
    print("=" * 60)
    print("TUNE PRUNING PARAMETERS")
    print("=" * 60)

    if skip:
        if preset_depth is None or preset_min_samples is None:
            preset_depth = 15
            preset_min_samples = 20

        print(f"\n⏭️  SKIPPING grid search (using pre-computed values)")
        print(f"   max_depth   = {preset_depth}")
        print(f"   min_samples = {preset_min_samples}")
        print("\n   To re-run tuning, use: tune_pruning(skip=False)")

        print("\n" + "=" * 60)
        print("Step 6 complete (skipped). Ready for Step 7.")
        print("=" * 60)

        return preset_depth, preset_min_samples, []

    depths = [5, 8, 10, 15, 20]
    min_samples_list = [10, 20, 50, 100]
    features = X_train.columns.tolist()
    pop_classes = list(range(7, 14))
    results = []

    print(f"\nGrid: {len(depths)} depths × {len(min_samples_list)} "
          f"min_samples = {len(depths) * len(min_samples_list)} combinations\n")

    for max_depth in depths:
        for min_samples in min_samples_list:
            start = time.time()
            tree = id3(X_train, y_train, features,
                       max_depth=max_depth, min_samples=min_samples)
            train_time = time.time() - start

            y_pred = predict(X_val, tree)
            overall = (y_pred == y_val.values).mean() * 100
            pop_mask = y_val.isin(pop_classes)
            pop_acc = ((y_pred[pop_mask] == y_val[pop_mask].values).mean() * 100
                       if pop_mask.sum() > 0 else 0.0)
            nodes = count_nodes(tree)

            results.append({
                'max_depth': max_depth, 'min_samples': min_samples,
                'val_acc': overall, 'pop_acc': pop_acc,
                'nodes': nodes, 'time_s': train_time,
            })
            print(f"  depth={max_depth:<3} min_samples={min_samples:<4} │ "
                  f"val_acc={overall:.2f}%  pop_acc={pop_acc:.2f}%  "
                  f"nodes={nodes:,}  time={train_time:.1f}s")

    print("\n" + "-" * 75)
    print(f"{'depth':<7} {'min_smp':<9} {'val_acc':>8} {'pop_acc':>8} "
          f"{'nodes':>9} {'time':>7}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        print(f"  {r['max_depth']:<5} {r['min_samples']:<8} "
              f"{r['val_acc']:>7.2f}%  {r['pop_acc']:>7.2f}%  "
              f"{r['nodes']:>9,}  {r['time_s']:>6.1f}s")

    best = max(results, key=lambda x: (round(x['val_acc'], 1), -x['nodes']))

    print(f"\n Best combination:")
    print(f"   max_depth   = {best['max_depth']}")
    print(f"   min_samples = {best['min_samples']}")
    print(f"   val_acc     = {best['val_acc']:.2f}%")
    print(f"   pop_acc     = {best['pop_acc']:.2f}%")
    print(f"   nodes       = {best['nodes']:,}")

    return best['max_depth'], best['min_samples'], results


# ---------------------------------------------------------------------------
# Train final tree
# ---------------------------------------------------------------------------


def train_final_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_max_depth: int,
    best_min_samples: int,
) -> dict:
    """Train one final tree on the combined train+validation set."""
    print("=" * 60)
    print("FINAL TREE")
    print("=" * 60)

    features = X_train.columns.tolist()

    print(f"\nParameters:")
    print(f"  max_depth     = {best_max_depth}")
    print(f"  min_samples   = {best_min_samples}")
    print(f"  Features      = {len(features)}")
    print(f"  Training rows = {len(X_train):,}")
    print(f"\nTraining... (this may take several minutes)")

    start = time.time()
    tree = id3(X_train, y_train, features,
               max_depth=best_max_depth, min_samples=best_min_samples)
    elapsed = time.time() - start

    total_nodes = count_nodes(tree)
    total_leaves = count_leaves(tree)
    actual_depth = tree_depth(tree)
    root_feature = tree.get('feature', 'N/A — root is leaf')

    print(f"\n✅ Training complete in {elapsed:.1f}s")
    print(f"\nTree statistics:")
    print(f"  Total nodes    : {total_nodes:,}")
    print(f"  Leaf nodes     : {total_leaves:,}")
    print(f"  Internal nodes : {total_nodes - total_leaves:,}")
    print(f"  Actual depth   : {actual_depth}")
    print(f"  Max depth set  : {best_max_depth}")
    print(f"  Root splits on : '{root_feature}'")

    return tree


# ---------------------------------------------------------------------------
# Visualise tree (top N levels)
# ---------------------------------------------------------------------------


def _print_node(
    tree: dict,
    prefix: str,
    is_last: bool,
    depth: int,
    max_display: int,
) -> None:
    """Box-drawing pretty-printer for the PopOut tree (integer labels)."""
    connector = "└── " if is_last else "├── "
    child_pfx = prefix + ("    " if is_last else "│   ")

    if tree['is_leaf']:
        label = int_to_move(tree['label'])
        majority = int_to_move(tree['majority'])
        print(f"{prefix}{connector}"
              f"🍃 LEAF → {label:<12} (majority: {majority})")
        return

    if depth >= max_display:
        majority = int_to_move(tree['majority'])
        n_children = len(tree['children'])
        print(f"{prefix}{connector}"
              f"📦 [{tree['feature']}]  "
              f"({n_children} branches, majority: {majority})  ···")
        return

    majority = int_to_move(tree['majority'])
    print(f"{prefix}{connector}"
          f"📦 SPLIT [{tree['feature']}]  (majority: {majority})")

    children_items = list(tree['children'].items())
    for i, (val, child) in enumerate(children_items):
        last = (i == len(children_items) - 1)
        val_connector = "└── " if last else "├── "
        val_child_pfx = child_pfx + ("    " if last else "│   ")

        print(f"{child_pfx}{val_connector}= {val}")
        _print_node(child, val_child_pfx, True, depth + 1, max_display)


def visualise_tree(tree: dict, max_display: int = 5) -> None:
    """Print the top ``max_display`` levels of the tree to stdout."""
    print("=" * 60)
    print(f"TREE VISUALISATION (top {max_display} levels)")
    print("=" * 60)

    actual_depth = tree_depth(tree)
    total = count_nodes(tree)
    leaves = count_leaves(tree)

    print(f"\nFull tree stats:")
    print(f"  Actual depth : {actual_depth}")
    print(f"  Total nodes  : {total:,}")
    print(f"  Leaf nodes   : {leaves:,}")
    print(f"\nDisplaying top {max_display} of {actual_depth} levels.")
    print("Nodes beyond this depth shown as ···")
    print()
    print("-" * 60)

    if tree['is_leaf']:
        print(f" LEAF → {int_to_move(tree['label'])}")
    else:
        majority = int_to_move(tree['majority'])
        print(f" ROOT SPLIT [{tree['feature']}]  (majority: {majority})")

        children_items = list(tree['children'].items())
        for i, (val, child) in enumerate(children_items):
            is_last = (i == len(children_items) - 1)
            connector = "└── " if is_last else "├── "
            child_pfx = "    " if is_last else "│   "

            print(f"{connector}= {val}")
            _print_node(child, child_pfx, True, depth=1,
                        max_display=max_display)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(tree: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Full evaluation on the test set: overall acc + per-class acc."""
    print("=" * 60)
    print(" EVALUATE ON TEST SET")
    print("=" * 60)

    print(f"\nTest set size: {len(X_test):,} rows")
    print("Running predictions...")

    start = time.time()
    y_pred = predict(X_test, tree)
    elapsed = time.time() - start
    print(f"Predictions complete in {elapsed:.1f}s")

    y_true = y_test.values

    overall_acc = (y_pred == y_true).mean() * 100
    baseline_class = int(y_test.value_counts().idxmax())
    baseline_acc = (y_true == baseline_class).mean() * 100

    print(f"\nOverall accuracy: {overall_acc:.2f}%")
    print(f"\nPer-class accuracy:")
    print(f"{'Class':<5} {'Move':<12} {'Correct':>8} {'Total':>8} "
          f"{'Acc':>8} {'Type'}")
    print("-" * 52)

    class_results: dict = {}
    for cls in range(14):
        mask = y_true == cls
        total = mask.sum()
        if total == 0:
            continue
        correct = (y_pred[mask] == cls).sum()
        acc = correct / total * 100
        move_type = "DROP" if cls < 7 else "POP "
        marker = "⚠️" if acc < 20 else ""
        print(f"  {cls:<3}  {int_to_move(cls):<12} {correct:>8,} {total:>8,} "
              f"{acc:>7.1f}%  {move_type} {marker}")
        class_results[cls] = {'correct': correct, 'total': total, 'acc': acc}

    drop_classes = [c for c in class_results if c < 7]
    pop_classes = [c for c in class_results if c >= 7]

    drop_correct = sum(class_results[c]['correct'] for c in drop_classes)
    drop_total = sum(class_results[c]['total'] for c in drop_classes)
    pop_correct = sum(class_results[c]['correct'] for c in pop_classes)
    pop_total = sum(class_results[c]['total'] for c in pop_classes)

    print(f"\n  DROP moves overall: {drop_correct / drop_total * 100:.1f}% "
          f"({drop_correct:,}/{drop_total:,})")
    print(f"  POP  moves overall: {pop_correct / pop_total * 100:.1f}% "
          f"({pop_correct:,}/{pop_total:,})")

    return {
        'overall_acc': overall_acc,
        'baseline_acc': baseline_acc,
        'class_results': class_results,
        'y_pred': y_pred,
    }


# ---------------------------------------------------------------------------
# Metric helpers (additive — do not modify `evaluate` above)
# ---------------------------------------------------------------------------


# POP class encoding under the integer schema used throughout the pipeline:
# classes 7..13 = pop_0..pop_6.
_POP_CLASS_INTS: tuple[int, ...] = tuple(range(7, 14))


def predict_batch(X: pd.DataFrame, tree: dict) -> np.ndarray:
    """Faster batch prediction without building a ``pd.Series`` per row.

    Functionally equivalent to :func:`ai.id3.predict` (same fallback to
    ``majority`` when a feature value is unseen). The speedup comes from
    indexing positionally into the underlying numpy array instead of
    constructing a Series. Useful in sweeps where ``predict`` would
    otherwise dominate wall time.
    """
    col_idx = {c: i for i, c in enumerate(X.columns)}
    values = X.values
    n = len(values)
    out = np.empty(n, dtype=object)

    for i in range(n):
        row = values[i]
        node = tree
        while not node['is_leaf']:
            v = row[col_idx[node['feature']]]
            child = node['children'].get(v)
            if child is None:
                out[i] = node['majority']
                break
            node = child
        else:
            out[i] = node['label']
    return out


def confusion_matrix(
    y_true,
    y_pred,
    classes: list | None = None,
) -> pd.DataFrame:
    """Return a ``k × k`` confusion-matrix DataFrame.

    Rows are the true class, columns are the predicted class. If ``classes``
    is ``None``, the union of values appearing in ``y_true`` and ``y_pred``
    is used (sorted).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if classes is None:
        seen = set(np.unique(y_true_arr).tolist())
        seen.update(np.unique(y_pred_arr).tolist())
        classes = sorted(seen)
    else:
        classes = list(classes)

    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    return pd.DataFrame(cm, index=classes, columns=classes)


def per_class_metrics(
    y_true,
    y_pred,
    classes: list | None = None,
) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support.

    Returns one row per class in ``classes`` (or the union of values seen
    in ``y_true`` and ``y_pred``). Precision and recall are defined as 0.0
    when their respective denominator is zero (no predictions / no support).
    """
    cm = confusion_matrix(y_true, y_pred, classes=classes)
    cls_labels = list(cm.index)
    rows = []
    for c in cls_labels:
        tp = int(cm.loc[c, c])
        fp = int(cm[c].sum() - tp)
        fn = int(cm.loc[c].sum() - tp)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        rows.append({
            'class': c, 'precision': precision, 'recall': recall,
            'f1': f1, 'support': support,
        })
    return pd.DataFrame(rows).set_index('class')


def macro_f1(
    y_true,
    y_pred,
    classes: list | None = None,
) -> float:
    """Unweighted mean of per-class F1 scores.

    Macro-F1 weights every class equally regardless of support, which is
    the right metric for an imbalanced multi-class problem like PopOut
    (drop_3 dominates even after capping).
    """
    pcm = per_class_metrics(y_true, y_pred, classes=classes)
    if len(pcm) == 0:
        return 0.0
    return float(pcm['f1'].mean())


def evaluate_quiet(
    tree: dict,
    X: pd.DataFrame,
    y,
    classes: list | None = None,
) -> dict:
    """Programmatic counterpart of :func:`evaluate` — no stdout.

    Returns ``{accuracy, macro_f1, pop_recall, n_correct, n_total}``
    where ``pop_recall`` is the mean per-class recall over the 7 POP
    classes (integer encoding 7..13). Classes absent from ``y`` are
    skipped in that average so the metric is well-defined even on tiny
    or imbalanced subsets.
    """
    y_true = np.asarray(y.values if hasattr(y, 'values') else y)
    y_pred = predict_batch(X, tree)

    n_total = int(len(y_true))
    n_correct = int((y_pred == y_true).sum())
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    if classes is None:
        seen = set(np.unique(y_true).tolist())
        seen.update(np.unique(y_pred).tolist())
        classes_used = sorted(seen)
    else:
        classes_used = list(classes)

    pcm = per_class_metrics(y_true, y_pred, classes=classes_used)
    macro = float(pcm['f1'].mean()) if len(pcm) > 0 else 0.0

    # POP-class recall, averaged over those POP classes that appear in y_true.
    pop_present = [c for c in _POP_CLASS_INTS if c in pcm.index
                   and pcm.loc[c, 'support'] > 0]
    if pop_present:
        pop_recall = float(pcm.loc[pop_present, 'recall'].mean())
    else:
        pop_recall = 0.0

    return {
        'accuracy': float(accuracy),
        'macro_f1': macro,
        'pop_recall': pop_recall,
        'n_correct': n_correct,
        'n_total': n_total,
    }


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


def save_tree(tree: dict, path: str | None = None) -> str:
    """Pickle ``tree`` to disk and return the path."""
    print("=" * 60)
    print("SAVE TREE")
    print("=" * 60)

    if path is None:
        path = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'id3_tree.pkl')

    with open(path, 'wb') as f:
        pickle.dump(tree, f)

    size_kb = os.path.getsize(path) / 1024
    print(f"\nsaved to: {path}")
    print(f" {size_kb:.1f} KB")
    return path


def load_tree(path: str) -> dict:
    """Load a previously pickled tree."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_predict(test_csv: str) -> int:
    """Handle ``--predict <csv>``: load the saved tree and classify rows."""
    tree_path = os.path.join(os.path.dirname(__file__),
                             '..', 'data', 'id3_tree.pkl')
    print(f"Loading tree from {tree_path}...")
    tree = load_tree(tree_path)

    print(f"Loading test examples from {test_csv}...")
    df_test = pd.read_csv(test_csv)

    for col, defn in POPOUT_BIN_DEFINITIONS.items():
        if col in df_test.columns:
            df_test[col] = pd.cut(
                df_test[col], bins=defn['bins'], labels=defn['labels'],
            ).astype(str)

    if 'class' in df_test.columns:
        df_test = df_test.drop(columns=['class'])

    preds = predict(df_test, tree)
    print("\nPredictions:")
    for i, p in enumerate(preds):
        print(f"  Row {i + 1}: {int_to_move(p)}")
    return 0


def _cli_train(csv_path: str) -> int:
    """Handle the default mode: iris warm-up + full training pipeline."""
    run_iris_demo()

    print("\n" + "POPOUT ID3".center(60))
    print("=" * 60)

    df, move_col, _counts, continuous_cols = inspect_dataset(csv_path)
    df_balanced = balance_classes(df, move_col, cap=50000)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df_balanced, move_col,
    )
    X_train, X_val, X_test = bin_features(
        X_train, X_val, X_test, continuous_cols,
    )

    # Set SKIP_TUNING=False to re-run the full grid search.
    SKIP_TUNING = True
    BEST_DEPTH = 15
    BEST_MIN = 20

    best_depth, best_min, _grid_results = tune_pruning(
        X_train, y_train, X_val, y_val,
        skip=SKIP_TUNING,
        preset_depth=BEST_DEPTH,
        preset_min_samples=BEST_MIN,
    )

    X_full_train = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full_train = pd.concat([y_train, y_val]).reset_index(drop=True)
    tree = train_final_tree(X_full_train, y_full_train, best_depth, best_min)

    visualise_tree(tree, max_display=5)
    evaluate(tree, X_test, y_test)
    save_tree(tree)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Module entry point for ``python -m ai.dt_pipeline``."""
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 2 and argv[0] == '--predict':
        return _cli_predict(argv[1])

    csv_path = os.path.join(os.path.dirname(__file__),
                            '..', _DEFAULT_DATASET_REL)
    if argv:
        csv_path = argv[0]

    return _cli_train(csv_path)


if __name__ == "__main__":
    # The visualisation helpers print box-drawing + emoji glyphs. On Windows
    # consoles defaulting to cp1252 these raise UnicodeEncodeError; force
    # UTF-8 best-effort so the CLI is usable without env tweaks.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception):
        pass
    raise SystemExit(main())
