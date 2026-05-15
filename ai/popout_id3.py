"""
PopOut ID3 Decision Tree
Full plan: iris dataset warm-up, load → balance → split → bin → train → evaluate → save
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import pickle

# ==========================================
# IRIS (warm-up dataset)
# ==========================================

def run_iris_demo(csv_path: str = None):
    """
    warm-up: ID3 on Iris dataset.
    Uses the same ID3 functions as PopOut.
    """
    print("=" * 60)
    print("IRIS DATASET  WARM-UP")
    print("=" * 60)

    # Load
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__),
                                '..', 'data', 'iris.csv')
    df = pd.read_csv(csv_path)

    # Rename columns to standard names if needed
    col_map = {
        'sepallength': 'sepal_length', 'sepalwidth': 'sepal_width',
        'petallength': 'petal_length', 'petalwidth': 'petal_width'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    target_col = 'class'
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"\nLoaded Iris: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Classes: {df[target_col].unique()}")
    print(f"Features: {feature_cols}")

    # Discretise continuous features using optimal binary split
    # Iris has continuous values
    print("\nDiscretising continuous features:")
    for col in feature_cols:
        unique_vals = np.sort(df[col].unique())
        best_ig   = -1
        best_thresh = None

        for i in range(len(unique_vals) - 1):
            thresh = (unique_vals[i] + unique_vals[i+1]) / 2.0
            temp   = df[col].apply(lambda x: 'low' if x <= thresh else 'high')
            temp_df = pd.DataFrame({col: temp})
            ig = information_gain(temp_df, df[target_col], col)
            if ig > best_ig:
                best_ig     = ig
                best_thresh = thresh

        df[col] = df[col].apply(
            lambda x: 'low' if x <= best_thresh else 'high'
        )
        print(f"  {col:<20} threshold={best_thresh:.3f}  "
              f"IG={best_ig:.4f}")

    # Split 80/20
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split       = int(0.8 * len(df_shuffled))
    train_df    = df_shuffled.iloc[:split]
    test_df     = df_shuffled.iloc[split:]

    X_train_iris = train_df[feature_cols]
    y_train_iris = train_df[target_col]
    X_test_iris  = test_df[feature_cols]
    y_test_iris  = test_df[target_col]

    print(f"\nTrain: {len(X_train_iris)} rows  |  Test: {len(X_test_iris)} rows")

    # Train using the same id3() function as PopOut
    print("\nTraining ID3 on Iris...")
    iris_tree = id3(
        X_train_iris, y_train_iris,
        feature_cols,
        max_depth=10, min_samples=2
    )

    # Accuracy
    y_pred_iris = predict(X_test_iris, iris_tree)
    acc = (y_pred_iris == y_test_iris.values).mean() * 100
    print(f"\nIris test accuracy: {acc:.2f}%")

    # Visual tree
    print("\nIris Decision Tree (full — small enough to display):")
    print("-" * 60)
    _print_node_iris(iris_tree, prefix="", is_last=True, depth=0)
    print("-" * 60)
    return iris_tree, acc


def _print_node_iris(tree: dict, prefix: str, is_last: bool, depth: int):
    connector = "└── " if is_last else "├── "
    child_pfx  = prefix + ("    " if is_last else "│   ")

    if tree['is_leaf']:
        print(f"{prefix}{connector}🍃 {tree['label']}")
        return

    majority = tree['majority']
    print(f"{prefix}{connector}📦 [{tree['feature']}]  (majority: {majority})")

    children_items = list(tree['children'].items())
    for i, (val, child) in enumerate(children_items):
        last = (i == len(children_items) - 1)
        val_connector = "└── " if last else "├── "
        val_child_pfx  = child_pfx + ("    " if last else "│   ")
        print(f"{child_pfx}{val_connector}= {val}")
        _print_node_iris(child, val_child_pfx, True, depth + 1)


# ==========================================
# LOAD & INSPECT DATASET
# ==========================================

def inspect_dataset(csv_path: str):
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

    def move_to_int(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        if s.startswith('drop_'):
            return int(s.split('_')[1])
        if s.startswith('pop_'):
            return 7 + int(s.split('_')[1])
        try:
            return int(s)
        except:
            return None

    y = df[move_col].apply(move_to_int)
    valid = y.notna()
    print(f"Valid moves: {valid.sum():,} / {len(df):,} ({valid.sum()/len(df)*100:.1f}%)")

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
    continuous_cols = []
    categorical_cols = []

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


# ==========================================
# BALANCE CLASSES
# ==========================================

def balance_classes(df: pd.DataFrame, move_col: str,
                    cap: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """Undersample overrepresented classes to cap. Keep underrepresented classes whole."""

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
          f"({len(df_balanced)/len(df)*100:.1f}% of original)")

    return df_balanced

# ==========================================
# TRAIN/TEST SPLIT
# ==========================================

def stratified_split(X: pd.DataFrame, y: pd.Series,
                     test_size: float, random_state: int) -> tuple:
    """
    numpy/pandas stratified split.
    Splits each class proportionally, then concatenates.
    """
    rng = np.random.RandomState(random_state)

    train_indices = []
    test_indices  = []

    for label in y.unique():
        class_idx = y[y == label].index.tolist()
        class_idx = [class_idx[i] for i in rng.permutation(len(class_idx))]

        n_test  = max(1, int(len(class_idx) * test_size))
        n_train = len(class_idx) - n_test

        train_indices.extend(class_idx[:n_train])
        test_indices.extend(class_idx[n_train:])

    train_indices = [train_indices[i]
                     for i in rng.permutation(len(train_indices))]
    test_indices  = [test_indices[i]
                     for i in rng.permutation(len(test_indices))]

    return (X.loc[train_indices].reset_index(drop=True),
            X.loc[test_indices].reset_index(drop=True),
            y.loc[train_indices].reset_index(drop=True),
            y.loc[test_indices].reset_index(drop=True))


def split_dataset(df_balanced: pd.DataFrame, move_col: str,
                  random_state: int = 42):
    """80/20 stratified split, 10% validation."""

    X = df_balanced.drop(columns=[move_col])
    y = df_balanced[move_col]

    print(f"\nTotal samples: {len(df_balanced):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {y.nunique()}")

    X_trainval, X_test, y_trainval, y_test = stratified_split(
        X, y, test_size=0.20, random_state=random_state
    )

    X_train, X_val, y_train, y_val = stratified_split(
        X_trainval, y_trainval, test_size=0.10, random_state=random_state
    )

    total = len(df_balanced)
    print(f"\n{'Split':<12} {'Rows':>10} {'% of total':>12}")
    print("-" * 36)
    print(f"{'Train':<12} {len(X_train):>10,} {len(X_train)/total*100:>11.1f}%")
    print(f"{'Validation':<12} {len(X_val):>10,} {len(X_val)/total*100:>11.1f}%")
    print(f"{'Test':<12} {len(X_test):>10,} {len(X_test)/total*100:>11.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# BIN NUMERIC FEATURES
# ==========================================

def bin_features(X_train: pd.DataFrame, X_val: pd.DataFrame,
                 X_test: pd.DataFrame, continuous_cols: list) -> tuple:
    """Bin continuous features into labeled categories for ID3."""

    print("=" * 60)
    print("BIN NUMERIC FEATURES")
    print("=" * 60)

    if not continuous_cols:
        print("No continuous features to bin.")
        return X_train, X_val, X_test

    bin_definitions = {
        'move_count': {
            'bins':   [-1, 10, 20, 30, 60],
            'labels': ['early', 'mid', 'late', 'endgame']
        },
        'own_pieces_bottom_row': {
            'bins':   [-1, 1, 3, 6],
            'labels': ['low', 'mid', 'high']
        },
        'opp_pieces_bottom_row': {
            'bins':   [-1, 1, 3, 6],
            'labels': ['low', 'mid', 'high']
        }
    }

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    print(f"\nBinning {len(continuous_cols)} continuous feature(s):\n")

    for col in continuous_cols:
        if col not in bin_definitions:
            print(f"  ⚠️  No bin definition for '{col}' — skipping")
            continue

        bins   = bin_definitions[col]['bins']
        labels = bin_definitions[col]['labels']

        X_train[col] = pd.cut(X_train[col], bins=bins, labels=labels).astype(str)
        X_val[col]   = pd.cut(X_val[col],   bins=bins, labels=labels).astype(str)
        X_test[col]  = pd.cut(X_test[col],  bins=bins, labels=labels).astype(str)

        nan_count = (X_train[col] == 'nan').sum()
        print(f"   '{col}'")
        print(f"     Bins:   {bins}")
        print(f"     Labels: {labels}")

    return X_train, X_val, X_test


# ==========================================
#  ID3 FUNCTIONS
# ==========================================

def entropy(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0
    counts = y.value_counts().values
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(X: pd.DataFrame, y: pd.Series, feature: str) -> float:
    total_n = len(y)
    if total_n == 0:
        return 0.0
    weighted = sum(
        (len(subset) / total_n) * entropy(subset)
        for val in X[feature].unique()
        for subset in [y[X[feature] == val]]
    )
    return entropy(y) - weighted


def best_feature(X: pd.DataFrame, y: pd.Series, features: list) -> str:
    gains = {f: information_gain(X, y, f) for f in features}
    return max(gains, key=gains.get)


def id3(X: pd.DataFrame, y: pd.Series, features: list,
        max_depth: int = 10, min_samples: int = 20,
        depth: int = 0, parent_majority: int = None) -> dict:
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
    remaining     = [f for f in features if f != split_feature]

    node = {
        'is_leaf':  False,
        'feature':  split_feature,
        'majority': majority,
        'children': {}
    }

    for val in X[split_feature].unique():
        mask = X[split_feature] == val
        node['children'][val] = id3(
            X[mask], y[mask], remaining,
            max_depth, min_samples,
            depth + 1, majority
        )

    return node


def predict_one(row: pd.Series, tree: dict) -> int:
    if tree['is_leaf']:
        return tree['label']
    value = row[tree['feature']]
    if value not in tree['children']:
        return tree['majority']
    return predict_one(row, tree['children'][value])


def predict(X: pd.DataFrame, tree: dict) -> np.ndarray:
    cols = X.columns.tolist()
    return np.array([
        predict_one(pd.Series(row, index=cols), tree)
        for row in X.itertuples(index=False)
    ])


# ==========================================
# TUNE PRUNING PARAMETERS
# ==========================================

def count_nodes(tree: dict) -> int:
    """Recursively count all nodes."""
    if tree['is_leaf']:
        return 1
    return 1 + sum(count_nodes(c) for c in tree['children'].values())


def tune_pruning(X_train, y_train, X_val, y_val,
                 skip: bool = False,
                 preset_depth: int = None,
                 preset_min_samples: int = None) -> tuple:
    """
    Grid search over max_depth and min_samples.

    Args:
        skip: If True, skip tuning and use preset values
        preset_depth: max_depth to use when skip=True
        preset_min_samples: min_samples to use when skip=True

    Returns:
        best_depth, best_min_samples, results (empty list if skipped)
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

    depths           = [5, 8, 10, 15, 20]
    min_samples_list = [10, 20, 50, 100]
    features         = X_train.columns.tolist()
    pop_classes      = list(range(7, 14))
    results          = []

    print(f"\nGrid: {len(depths)} depths × {len(min_samples_list)} "
          f"min_samples = {len(depths)*len(min_samples_list)} combinations\n")

    for max_depth in depths:
        for min_samples in min_samples_list:
            start = time.time()
            tree  = id3(X_train, y_train, features,
                        max_depth=max_depth, min_samples=min_samples)
            train_time = time.time() - start

            y_pred   = predict(X_val, tree)
            overall  = (y_pred == y_val.values).mean() * 100
            pop_mask = y_val.isin(pop_classes)
            pop_acc  = ((y_pred[pop_mask] == y_val[pop_mask].values).mean() * 100
                        if pop_mask.sum() > 0 else 0.0)
            nodes    = count_nodes(tree)

            results.append({
                'max_depth': max_depth, 'min_samples': min_samples,
                'val_acc': overall, 'pop_acc': pop_acc,
                'nodes': nodes, 'time_s': train_time
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


# ==========================================
# TRAIN FINAL TREE
# ==========================================

def tree_depth(tree: dict, current: int = 0) -> int:
    """Maximum depth of tree."""
    if tree['is_leaf']:
        return current
    return max(tree_depth(c, current + 1) for c in tree['children'].values())


def count_leaves(tree: dict) -> int:
    """Count leaf nodes only."""
    if tree['is_leaf']:
        return 1
    return sum(count_leaves(c) for c in tree['children'].values())


def train_final_tree(X_train: pd.DataFrame, y_train: pd.Series,
                     best_max_depth: int, best_min_samples: int) -> dict:
    """
    Train one final tree on the full training set (train + validation)
    """
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

    start   = time.time()
    tree    = id3(X_train, y_train, features,
                  max_depth=best_max_depth, min_samples=best_min_samples)
    elapsed = time.time() - start

    total_nodes  = count_nodes(tree)
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


# ==========================================
# VISUALISE TREE (top 5 levels)
# ==========================================

def _print_node(tree: dict, prefix: str, is_last: bool,
                depth: int, max_display: int):
    """
    Recursive helper for visualise_tree.
    Draws tree using box-drawing characters for a clean hierarchy.

    prefix    : the indentation string built up by parent calls
    is_last   : whether this node is the last child of its parent
                (controls whether to use └── or ├──)
    depth     : current depth from root
    max_display: max depth to expand before truncating
    """
    connector  = "└── " if is_last else "├── "
    child_pfx  = prefix + ("    " if is_last else "│   ")

    if tree['is_leaf']:
        label    = int_to_move(tree['label'])
        majority = int_to_move(tree['majority'])
        print(f"{prefix}{connector}"
              f"🍃 LEAF → {label:<12} (majority: {majority})")
        return

    if depth >= max_display:
        majority  = int_to_move(tree['majority'])
        n_children = len(tree['children'])
        print(f"{prefix}{connector}"
              f"📦 [{tree['feature']}]  "
              f"({n_children} branches, majority: {majority})  ···")
        return

    # Internal node
    majority = int_to_move(tree['majority'])
    print(f"{prefix}{connector}"
          f"📦 SPLIT [{tree['feature']}]  (majority: {majority})")

    children_items = list(tree['children'].items())
    for i, (val, child) in enumerate(children_items):
        last = (i == len(children_items) - 1)
        val_connector = "└── " if last else "├── "
        val_child_pfx = child_pfx + ("    " if last else "│   ")

        # Print the branch value label
        print(f"{child_pfx}{val_connector}= {val}")

        # Recurse into child with increased depth
        _print_node(child, val_child_pfx, True, depth + 1, max_display)


def visualise_tree(tree: dict, max_display: int = 5):
    """
    Top 5 levels displayed
    """
    print("=" * 60)
    print(f"TREE VISUALISATION (top {max_display} levels)")
    print("=" * 60)

    actual_depth = tree_depth(tree)
    total        = count_nodes(tree)
    leaves       = count_leaves(tree)

    print(f"\nFull tree stats:")
    print(f"  Actual depth : {actual_depth}")
    print(f"  Total nodes  : {total:,}")
    print(f"  Leaf nodes   : {leaves:,}")
    print(f"\nDisplaying top {max_display} of {actual_depth} levels.")
    print("Nodes beyond this depth shown as ···")
    print()
    print("-" * 60)

    # Print root node manually (no connector, no prefix)
    if tree['is_leaf']:
        print(f" LEAF → {int_to_move(tree['label'])}")
    else:
        majority = int_to_move(tree['majority'])
        print(f" ROOT SPLIT [{tree['feature']}]  (majority: {majority})")

        children_items = list(tree['children'].items())
        for i, (val, child) in enumerate(children_items):
            is_last   = (i == len(children_items) - 1)
            connector = "└── " if is_last else "├── "
            child_pfx = "    " if is_last else "│   "

            # Branch value
            print(f"{connector}= {val}")

            # Child node
            _print_node(child, child_pfx, True, depth=1,
                        max_display=max_display)

# ==========================================
# EVALUATE ON TEST SET
# ==========================================

def int_to_move(n: int) -> str:
    """Convert integer class back to move string."""
    if n < 7:
        return f"drop_{n}"
    return f"pop_{n - 7}"


def evaluate(tree: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Full evaluation on the test set.
    Computes overall accuracy, per-class accuracy, and confusion matrix.
    """
    print("=" * 60)
    print(" EVALUATE ON TEST SET")
    print("=" * 60)

    print(f"\nTest set size: {len(X_test):,} rows")
    print("Running predictions...")

    start   = time.time()
    y_pred  = predict(X_test, tree)
    elapsed = time.time() - start
    print(f"Predictions complete in {elapsed:.1f}s")

    y_true = y_test.values

    overall_acc    = (y_pred == y_true).mean() * 100
    baseline_class = int(y_test.value_counts().idxmax())
    baseline_acc   = (y_true == baseline_class).mean() * 100
    improvement    = overall_acc - baseline_acc

    print(f"\nOverall accuracy: {overall_acc:.2f}%")
    print(f"\nPer-class accuracy:")
    print(f"{'Class':<5} {'Move':<12} {'Correct':>8} {'Total':>8} "
          f"{'Acc':>8} {'Type'}")
    print("-" * 52)

    class_results = {}
    for cls in range(14):
        mask  = y_true == cls
        total = mask.sum()
        if total == 0:
            continue
        correct   = (y_pred[mask] == cls).sum()
        acc       = correct / total * 100
        move_type = "DROP" if cls < 7 else "POP "
        marker    = "⚠️" if acc < 20 else ""
        print(f"  {cls:<3}  {int_to_move(cls):<12} {correct:>8,} {total:>8,} "
              f"{acc:>7.1f}%  {move_type} {marker}")
        class_results[cls] = {'correct': correct, 'total': total, 'acc': acc}

    drop_classes = [c for c in class_results if c < 7]
    pop_classes  = [c for c in class_results if c >= 7]

    drop_correct = sum(class_results[c]['correct'] for c in drop_classes)
    drop_total   = sum(class_results[c]['total']   for c in drop_classes)
    pop_correct  = sum(class_results[c]['correct'] for c in pop_classes)
    pop_total    = sum(class_results[c]['total']   for c in pop_classes)

    print(f"\n  DROP moves overall: {drop_correct/drop_total*100:.1f}% "
          f"({drop_correct:,}/{drop_total:,})")
    print(f"  POP  moves overall: {pop_correct/pop_total*100:.1f}% "
          f"({pop_correct:,}/{pop_total:,})")

    results = {
        'overall_acc':   overall_acc,
        'baseline_acc':  baseline_acc,
        'class_results': class_results,
        'y_pred':        y_pred
    }

    return results

# ==========================================
#  SAVE TREE
# ==========================================

def save_tree(tree: dict, path: str = None):
    """Save trained tree to disk with pickle."""

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
    """Load a previously saved tree from disk for test examples."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

     # Iris warm-up 
    iris_tree, iris_acc = run_iris_demo()

    if len(sys.argv) == 3 and sys.argv[1] == '--predict':
        tree_path = os.path.join(os.path.dirname(__file__),
                                 '..', 'data', 'id3_tree.pkl')
        print(f"Loading tree from {tree_path}...")
        tree = load_tree(tree_path)

        test_csv = sys.argv[2]
        print(f"Loading test examples from {test_csv}...")
        df_test = pd.read_csv(test_csv)

        bin_defs = {
            'move_count':            ([-1,10,20,30,60],
                                      ['early','mid','late','endgame']),
            'own_pieces_bottom_row': ([-1,1,3,6], ['low','mid','high']),
            'opp_pieces_bottom_row': ([-1,1,3,6], ['low','mid','high']),
        }
        for col, (bins, labels) in bin_defs.items():
            if col in df_test.columns:
                df_test[col] = pd.cut(
                    df_test[col], bins=bins, labels=labels
                ).astype(str)

        if 'class' in df_test.columns:
            df_test = df_test.drop(columns=['class'])

        preds = predict(df_test, tree)
        print("\nPredictions:")
        for i, p in enumerate(preds):
            print(f"  Row {i+1}: {int_to_move(p)}")
        sys.exit(0)

    # Normal training

    csv_path = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'popout_200k.csv')
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print("\n" + "POPOUT ID3".center(60))
    print("=" * 60)

    df, move_col, counts, continuous_cols = inspect_dataset(csv_path)

    df_balanced = balance_classes(df, move_col, cap=50000)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df_balanced, move_col
    )

    X_train, X_val, X_test = bin_features(
        X_train, X_val, X_test, continuous_cols
    )

    #  set SKIP_TUNING=True to use pre-computed best values
    SKIP_TUNING = True
    BEST_DEPTH  = 15
    BEST_MIN    = 20

    best_depth, best_min, grid_results = tune_pruning(
        X_train, y_train, X_val, y_val,
        skip=SKIP_TUNING,
        preset_depth=BEST_DEPTH,
        preset_min_samples=BEST_MIN
    )

    # train on 80% (train + validation combined)
    X_full_train = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_full_train = pd.concat([y_train, y_val]).reset_index(drop=True)
    tree = train_final_tree(X_full_train, y_full_train, best_depth, best_min)

    # visualise
    visualise_tree(tree, max_display=5)

    eval_results = evaluate(tree, X_test, y_test)
    save_tree(tree)
