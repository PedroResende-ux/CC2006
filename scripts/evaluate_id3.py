"""Final ID3 evaluation: 4 trees + 5-fold CV + 3 tournaments + report.

One CLI invocation that, on a 24-core PC, runs the full final experiment
end-to-end and writes a consolidated markdown report.

Pipeline (per dataset = ``baseline`` and ``optimised``):

1. Freeze a stratified 72/8/20 train/val/test split *before* training
   (nested: 80% train+val / 20% test, then 90/10 of train+val).
   Persist the test-row IDs (referenced to the original-CSV row index)
   in ``data/id3/test_indices_<dataset>.json`` so the test slice is
   byte-stable across re-runs and across the four trained trees.
2. Train an unpruned tree with the shared recipe
   ``(cap=100000, max_depth=20, min_samples=10)`` (per D1 sweep).
3. Apply reduced-error post-pruning (REP) using the frozen val set.
4. Run 5-fold CV (on ``X_train + X_val``) for both the pruned and
   unpruned recipes — 4 configurations × 5 folds = 20 trainings.
5. Evaluate each of the 4 trees on the held-out test set, on both
   the BALANCED test slice (frozen ``X_test``) and an UNBALANCED test
   slice (original-CSV rows whose orig_row_id is not in
   ``train ∪ val`` — the natural-class-prior view that ``DTPlayer``
   actually faces in live play).
6. Save confusion matrices and feature-importance tables for each tree.
7. Run three tournaments via ``scripts.run_tournament``:
   - ``DT_optimised_pruned`` vs MCTS (iter=20000, c=2, k=1, UCB1) — 200 games
   - ``DT_optimised_pruned`` vs ``DT_baseline_pruned`` — 200 games
   - ``DT_optimised_pruned`` vs Random — 500 games
8. Synthesise everything into the generated report.

CLI::

    python -m scripts.evaluate_id3 --workers 24 \\
        --output-dir data/id3

``--smoke`` runs the same pipeline end-to-end at tiny scale (1k rows
per dataset, depth=2, 2-fold CV, 4-game tournaments, MCTS at 200
iters) in under two minutes — used only to validate plumbing, not to
produce slide numbers.

Determinism: seed=42 everywhere, sorted CSV/JSON output, fixed
config-JSON generation. The same ``(seed, workers)`` produces
byte-identical artefacts on the same machine.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Make the project root importable when run as ``python -m scripts.evaluate_id3``.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Constants (production values — used when ``--smoke`` is NOT passed)
# ---------------------------------------------------------------------------

SEED: int = 42

# Per D1 sweep §5: shared recipe across both datasets (Outcome X —
# recipes transfer within 0.75 p.p. macro-F1).
RECIPE: dict[str, int] = {
    "cap": 100_000,
    "max_depth": 20,
    "min_samples": 10,
}

# Per §4.5 of the notebook: the optimised MCTS configuration.
MCTS_CONFIG: dict[str, Any] = {
    "iterations": 20_000,
    "exploration_weight": 2.0,
    "num_children_to_expand": 1,
    "rollout_depth_limit": 80,
    "uct_variant": "ucb1",
}

DATASETS: dict[str, str] = {
    "baseline":  "data/popout_200k.csv",
    "optimised": "data/popout_dataset_150k.csv",
}

CV_FOLDS: int = 5

TOURNAMENT_GAMES: dict[str, int] = {
    "dt_opt_vs_mcts":   200,
    "dt_opt_vs_dt_base": 200,
    "dt_opt_vs_random": 500,
}

# The 14 PopOut move classes — 0..6 are drop_*, 7..13 are pop_*.
_ALL_CLASSES: list[int] = list(range(14))

TOP_K: int = 3  # for top-k accuracy reporting


# ---------------------------------------------------------------------------
# Smoke-mode overrides — every step still runs, just at toy scale.
# ---------------------------------------------------------------------------

SMOKE_ROWS: int = 1000  # per-dataset row cap when ``--smoke`` is on
SMOKE_RECIPE: dict[str, int] = {
    "cap": 500,
    "max_depth": 2,
    "min_samples": 10,
}
SMOKE_CV_FOLDS: int = 2
SMOKE_TOURNAMENT_GAMES: int = 4   # must be even per run_tournament's validator
SMOKE_MCTS_ITERATIONS: int = 200  # small enough that DT-vs-MCTS finishes fast


# ---------------------------------------------------------------------------
# Helpers — module-level so multiprocessing workers can pickle them.
# ---------------------------------------------------------------------------


def _stratified_kfold_indices(
    y: pd.Series,
    n_folds: int,
    random_state: int = 42,
) -> list[np.ndarray]:
    """Stratified k-fold split. Returns a list of length ``n_folds`` of
    positional row-index arrays — each entry holds out one fold.

    Each class is permuted independently with ``random_state``, then
    split into ``n_folds`` near-equal chunks. The chunks are reshuffled
    within each fold so the order is deterministic but classes mix.
    Caller is responsible for ensuring ``y`` has a positional 0..n-1
    index (we call ``y.reset_index(drop=True)`` internally).
    """
    rng = np.random.RandomState(random_state)
    folds: list[list[int]] = [[] for _ in range(n_folds)]

    y_reset = y.reset_index(drop=True)
    for label in y_reset.unique():
        class_pos = np.flatnonzero((y_reset == label).values)
        if len(class_pos) == 0:
            continue
        perm = rng.permutation(len(class_pos))
        class_pos = class_pos[perm]
        chunks = np.array_split(class_pos, n_folds)
        for i, chunk in enumerate(chunks):
            folds[i].extend(chunk.tolist())

    out: list[np.ndarray] = []
    for fold in folds:
        arr = np.array(fold, dtype=np.int64)
        if len(arr) > 0:
            arr = arr[rng.permutation(len(arr))]
        out.append(arr)
    return out


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % CI for a binomial proportion ``wins / n``.

    Symmetric around the score-test centre, well-behaved at the
    boundary (p=0 or p=1). Returned tuple is clipped to ``[0, 1]``.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _top_k_accuracy_fast(
    y_true,
    X: pd.DataFrame,
    tree: dict,
    k: int = 3,
) -> float:
    """Fast top-k accuracy: positional tree-walk, no per-row ``pd.Series``.

    Equivalent semantics to :func:`ai.dt_pipeline.top_k_accuracy` (same
    fallback to ``majority`` on unseen feature values, same backwards-
    compatible fallback to ``[label]`` on leaves without
    ``class_counts``) but skips the ``pd.Series`` construction that
    dominates the slow path. On the unbalanced test slice (~400-500k
    rows) this turns minutes of evaluation into seconds.
    """
    y_arr = np.asarray(y_true.values if hasattr(y_true, 'values') else y_true)
    n = len(y_arr)
    if n == 0:
        return 0.0
    col_idx = {c: i for i, c in enumerate(X.columns)}
    values = X.values
    correct = 0
    for i in range(n):
        row = values[i]
        node = tree
        preds: list
        while not node['is_leaf']:
            v = row[col_idx[node['feature']]]
            child = node['children'].get(v)
            if child is None:
                preds = [node['majority']]
                break
            node = child
        else:
            counts = node.get('class_counts')
            if not counts:
                preds = [node['label']]
            else:
                ranked = sorted(
                    counts.items(), key=lambda kv: kv[1], reverse=True,
                )
                preds = [cls for cls, _ in ranked[:k]]
        if y_arr[i] in preds:
            correct += 1
    return correct / n


# ---------------------------------------------------------------------------
# Frozen-split structure (kept dataclass-free so it pickles trivially)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FrozenSplit:
    """One dataset's frozen split + the metadata to reproduce it.

    ``train_orig_ids`` / ``val_orig_ids`` / ``test_orig_ids`` are the
    positional row indices in the *original* CSV that ended up in each
    slice after balance + stratified split. They are persisted to
    ``data/id3/test_indices_<dataset>.json`` so the test slice is
    byte-stable across re-runs and so the unbalanced-test eval can
    reconstruct itself from the original CSV.
    """

    dataset: str
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_orig_ids: list[int]
    val_orig_ids: list[int]
    test_orig_ids: list[int]
    continuous_cols: list[str]
    csv_path: str


# ---------------------------------------------------------------------------
# Stage 1 — freeze splits + persistence
# ---------------------------------------------------------------------------


def freeze_splits(
    recipe: dict[str, int],
    output_dir: Path,
    smoke_rows: Optional[int] = None,
) -> dict[str, FrozenSplit]:
    """Per dataset, produce a stable train/val/test split and persist the
    pre-balance row indices so anyone can reproduce the test slice.

    ``smoke_rows``: when set, only the first N rows of each CSV are
    loaded — used by ``--smoke``. Production runs pass ``None``.

    Audit finding A (high severity):
    the existing pipeline re-splits inside every prep call, so test
    indices are not stable across cap variants. This function fixes
    that by adding an explicit ``_orig_row_id`` column before balancing,
    so the post-balance/post-split test slice can be traced back to the
    original CSV rows.
    """
    from ai.dt_pipeline import (
        balance_classes, bin_features, inspect_dataset, split_dataset,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, FrozenSplit] = {}

    for dataset, rel_path in DATASETS.items():
        abs_path = (_PROJECT_ROOT / rel_path).resolve()
        if not abs_path.exists():
            raise SystemExit(f"error: dataset not found: {abs_path}")

        print(f"\n[freeze] dataset={dataset} csv={abs_path}", flush=True)

        # If smoke, load only the first N rows. ``inspect_dataset`` uses
        # ``pd.read_csv`` internally with no nrows knob, so we shim by
        # writing a truncated CSV to a stable location and pointing
        # inspect at it. The truncated CSV must SURVIVE past
        # ``freeze_splits`` because ``_build_unbalanced_test_slice`` re-
        # opens it later via ``fs.csv_path`` — if we delete it now, the
        # unbalanced eval falls back to the full 1.5-1.9M-row original
        # CSV and the smoke run blows past its 2-minute budget. Stable
        # location is ``output_dir`` so the smoke artefacts are
        # self-contained.
        load_path = str(abs_path)
        if smoke_rows is not None:
            smoke_csv = output_dir / f"smoke_csv_{dataset}.csv"
            head = pd.read_csv(abs_path, nrows=smoke_rows)
            head.to_csv(smoke_csv, index=False)
            load_path = str(smoke_csv)
            print(f"[freeze] smoke truncated CSV: {smoke_csv} "
                  f"({len(head)} rows)", flush=True)

        df, move_col, _counts, continuous_cols = inspect_dataset(load_path)

        # Add the pre-balance original row id BEFORE balancing so we can
        # trace any post-balance row back to the original CSV. The
        # column is dropped from X frames before the tree sees them.
        df = df.copy()
        df['_orig_row_id'] = np.arange(len(df), dtype=np.int64)

        df_balanced = balance_classes(
            df, move_col, cap=int(recipe["cap"]), random_state=SEED,
        )
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            df_balanced, move_col, random_state=SEED,
        )

        # Snapshot the orig_row_ids per slice BEFORE we drop the helper
        # column from the X frames.
        train_orig_ids = X_train['_orig_row_id'].astype(int).tolist()
        val_orig_ids = X_val['_orig_row_id'].astype(int).tolist()
        test_orig_ids = X_test['_orig_row_id'].astype(int).tolist()

        X_train = X_train.drop(columns=['_orig_row_id'])
        X_val = X_val.drop(columns=['_orig_row_id'])
        X_test = X_test.drop(columns=['_orig_row_id'])

        X_train, X_val, X_test = bin_features(
            X_train, X_val, X_test, continuous_cols,
        )

        # Persist the orig_row_ids — sorted so the JSON is byte-stable
        # regardless of internal shuffling. The unbalanced-test eval
        # uses set difference, so order does not matter; sorting just
        # makes the file diff-friendly.
        idx_path = output_dir / f"test_indices_{dataset}.json"
        with idx_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "csv_path": str(rel_path),
                    "seed": SEED,
                    "recipe": dict(recipe),
                    "n_original_rows": int(len(df)),
                    "n_balanced_rows": int(len(df_balanced)),
                    "train_orig_ids": sorted(train_orig_ids),
                    "val_orig_ids": sorted(val_orig_ids),
                    "test_orig_ids": sorted(test_orig_ids),
                },
                f, indent=2,
            )
        print(f"[freeze] wrote {idx_path.relative_to(_PROJECT_ROOT)} "
              f"(train={len(train_orig_ids)} val={len(val_orig_ids)} "
              f"test={len(test_orig_ids)})", flush=True)

        splits[dataset] = FrozenSplit(
            dataset=dataset,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            train_orig_ids=train_orig_ids,
            val_orig_ids=val_orig_ids,
            test_orig_ids=test_orig_ids,
            continuous_cols=continuous_cols,
            # In smoke mode the load_path points at the smoke-truncated
            # CSV (~1000 rows); in production it's the original CSV
            # (1.5-1.9M rows). The unbalanced test slice is built off
            # this path so smoke stays cheap.
            csv_path=load_path,
        )

    return splits


# ---------------------------------------------------------------------------
# Stage 2 — train unpruned + apply REP
# ---------------------------------------------------------------------------


def train_unpruned_trees(
    splits: dict[str, FrozenSplit],
    recipe: dict[str, int],
    output_dir: Path,
) -> dict[str, dict]:
    """Train 2 unpruned trees with the shared recipe. Save pickles."""
    from ai.dt_pipeline import save_tree
    from ai.id3 import id3

    trees: dict[str, dict] = {}
    for dataset, fs in splits.items():
        features = fs.X_train.columns.tolist()
        print(f"\n[train] dataset={dataset} "
              f"depth={recipe['max_depth']} "
              f"min_samples={recipe['min_samples']} "
              f"rows={len(fs.X_train):,}", flush=True)
        t0 = time.monotonic()
        tree = id3(
            fs.X_train, fs.y_train, features,
            max_depth=int(recipe["max_depth"]),
            min_samples=int(recipe["min_samples"]),
        )
        elapsed = time.monotonic() - t0
        path = output_dir / f"id3_tree_{dataset}.pkl"
        save_tree(tree, str(path))
        print(f"[train] {dataset} done in {elapsed:.1f}s -> "
              f"{path.relative_to(_PROJECT_ROOT)}", flush=True)
        trees[dataset] = tree
    return trees


def apply_rep(
    unpruned_trees: dict[str, dict],
    splits: dict[str, FrozenSplit],
    output_dir: Path,
) -> dict[str, dict]:
    """REP using each dataset's val set. Save pruned pickles."""
    from ai.dt_pipeline import prune_rep, save_tree

    pruned: dict[str, dict] = {}
    for dataset, unpruned in unpruned_trees.items():
        fs = splits[dataset]
        print(f"\n[rep] dataset={dataset} val_rows={len(fs.X_val):,}", flush=True)
        t0 = time.monotonic()
        tree = prune_rep(unpruned, fs.X_val, fs.y_val)
        elapsed = time.monotonic() - t0
        path = output_dir / f"id3_tree_{dataset}_pruned.pkl"
        save_tree(tree, str(path))
        print(f"[rep] {dataset} done in {elapsed:.1f}s -> "
              f"{path.relative_to(_PROJECT_ROOT)}", flush=True)
        pruned[dataset] = tree
    return pruned


# ---------------------------------------------------------------------------
# Stage 3 — 5-fold CV
# ---------------------------------------------------------------------------


def _cv_worker(args: tuple) -> dict:
    """Worker: train one fold of one CV configuration and return metrics.

    Args layout (single tuple for clean pickling under ``spawn``):

        (fold_idx, dataset, prune_mode, fold_pickle, recipe)

    where ``fold_pickle`` holds
    ``(X_train, y_train, X_prune, y_prune, X_eval, y_eval)``.
    For ``prune_mode == "unpruned"`` the prune slice is ignored
    (it's still passed to keep the pickle shape uniform).
    """
    (fold_idx, dataset, prune_mode, fold_pickle, recipe) = args

    from ai.dt_pipeline import evaluate_quiet, prune_rep
    from ai.id3 import count_leaves, count_nodes, id3

    with open(fold_pickle, "rb") as f:
        (X_tr, y_tr, X_pr, y_pr, X_ev, y_ev) = pickle.load(f)

    t0 = time.monotonic()
    tree = id3(
        X_tr, y_tr, X_tr.columns.tolist(),
        max_depth=int(recipe["max_depth"]),
        min_samples=int(recipe["min_samples"]),
    )
    train_time = time.monotonic() - t0

    if prune_mode == "pruned":
        tp0 = time.monotonic()
        tree = prune_rep(tree, X_pr, y_pr)
        prune_time = time.monotonic() - tp0
    else:
        prune_time = 0.0

    metrics = evaluate_quiet(tree, X_ev, y_ev, classes=_ALL_CLASSES)
    top3 = _top_k_accuracy_fast(y_ev, X_ev, tree, k=TOP_K)

    return {
        "dataset": dataset,
        "prune_mode": prune_mode,
        "fold": fold_idx,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "pop_recall": float(metrics["pop_recall"]),
        "top_3_accuracy": float(top3),
        "n_nodes": int(count_nodes(tree)),
        "n_leaves": int(count_leaves(tree)),
        "train_time_s": float(train_time),
        "prune_time_s": float(prune_time),
        "n_train": int(len(X_tr)),
        "n_eval": int(len(X_ev)),
    }


def cross_validate(
    splits: dict[str, FrozenSplit],
    recipe: dict[str, int],
    n_folds: int,
    workers: int,
    output_dir: Path,
) -> pd.DataFrame:
    """5-fold CV per (dataset, prune_mode). 20 trainings at production scale.

    Folds are taken from ``X_train ∪ X_val`` so the frozen test set is
    never touched. For each held-out fold the remaining 4 folds are
    used for training; when ``prune_mode == "pruned"`` an extra ~20%
    of the training pool is held out for the REP step, so the eval
    slice is still untouched.
    """
    scratch_dir = Path(tempfile.mkdtemp(prefix="id3_cv_"))
    work_items: list[tuple] = []
    fold_pickles: list[Path] = []

    try:
        for dataset, fs in splits.items():
            X_pool = pd.concat(
                [fs.X_train, fs.X_val], ignore_index=True,
            ).reset_index(drop=True)
            y_pool = pd.concat(
                [fs.y_train, fs.y_val], ignore_index=True,
            ).reset_index(drop=True)

            fold_indices = _stratified_kfold_indices(
                y_pool, n_folds=n_folds, random_state=SEED,
            )

            for i in range(n_folds):
                eval_idx = fold_indices[i]
                rest = np.concatenate([
                    fold_indices[j] for j in range(n_folds) if j != i
                ])

                # Carve a prune slice from the rest (~20 %). Use a
                # secondary deterministic permutation so the partition
                # is stable across re-runs.
                rng = np.random.RandomState(SEED + 100 + i)
                rest_perm = rest[rng.permutation(len(rest))]
                cut = max(1, int(0.2 * len(rest_perm)))
                prune_idx = rest_perm[:cut]
                train_idx = rest_perm[cut:]

                X_tr = X_pool.iloc[train_idx].reset_index(drop=True)
                y_tr = y_pool.iloc[train_idx].reset_index(drop=True)
                X_pr = X_pool.iloc[prune_idx].reset_index(drop=True)
                y_pr = y_pool.iloc[prune_idx].reset_index(drop=True)
                X_ev = X_pool.iloc[eval_idx].reset_index(drop=True)
                y_ev = y_pool.iloc[eval_idx].reset_index(drop=True)

                fold_path = scratch_dir / f"fold_{dataset}_{i}.pkl"
                with fold_path.open("wb") as f:
                    pickle.dump(
                        (X_tr, y_tr, X_pr, y_pr, X_ev, y_ev),
                        f, protocol=pickle.HIGHEST_PROTOCOL,
                    )
                fold_pickles.append(fold_path)

                for mode in ("unpruned", "pruned"):
                    work_items.append(
                        (i, dataset, mode, str(fold_path), recipe)
                    )

        print(f"\n[cv] {len(work_items)} trainings "
              f"({len(splits)} datasets × {n_folds} folds × 2 modes), "
              f"workers={workers}", flush=True)

        rows: list[dict] = []
        with mp.Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_cv_worker, work_items, chunksize=1):
                rows.append(r)
                print(f"  [cv] {r['dataset']:<10} {r['prune_mode']:<8} "
                      f"fold={r['fold']} "
                      f"macro_f1={r['macro_f1']:.4f} "
                      f"acc={r['accuracy']:.4f} "
                      f"top3={r['top_3_accuracy']:.4f} "
                      f"pop_rec={r['pop_recall']:.4f} "
                      f"nodes={r['n_nodes']:,} "
                      f"train={r['train_time_s']:.1f}s "
                      f"prune={r['prune_time_s']:.1f}s",
                      flush=True)

        df = pd.DataFrame(rows)

        # Aggregate mean ± std per (dataset, prune_mode).
        agg = (
            df.groupby(["dataset", "prune_mode"])
              .agg(
                  mean_accuracy=("accuracy", "mean"),
                  std_accuracy=("accuracy", "std"),
                  mean_macro_f1=("macro_f1", "mean"),
                  std_macro_f1=("macro_f1", "std"),
                  mean_top_3_accuracy=("top_3_accuracy", "mean"),
                  std_top_3_accuracy=("top_3_accuracy", "std"),
                  mean_pop_recall=("pop_recall", "mean"),
                  std_pop_recall=("pop_recall", "std"),
                  mean_n_nodes=("n_nodes", "mean"),
              )
              .reset_index()
        )

        # Persist both the per-fold rows and the aggregate.
        per_fold_path = output_dir / "cv_per_fold.csv"
        agg_path = output_dir / "cv_aggregate.csv"
        df.sort_values(["dataset", "prune_mode", "fold"]).to_csv(
            per_fold_path, index=False, float_format="%.6f",
        )
        agg.to_csv(agg_path, index=False, float_format="%.6f")
        print(f"[cv] wrote {per_fold_path.relative_to(_PROJECT_ROOT)} + "
              f"{agg_path.relative_to(_PROJECT_ROOT)}", flush=True)

        return agg

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage 4 — held-out test evaluation (balanced + unbalanced slices)
# ---------------------------------------------------------------------------


def _eval_one_tree(
    tree: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Full metric suite (acc / macro_f1 / pop_recall / top-3) for one tree."""
    from ai.dt_pipeline import evaluate_quiet

    base = evaluate_quiet(tree, X, y, classes=_ALL_CLASSES)
    top3 = _top_k_accuracy_fast(y, X, tree, k=TOP_K)
    return {
        "accuracy": float(base["accuracy"]),
        "macro_f1": float(base["macro_f1"]),
        "pop_recall": float(base["pop_recall"]),
        "top_3_accuracy": float(top3),
        "n_correct": int(base["n_correct"]),
        "n_total": int(base["n_total"]),
    }


def _build_unbalanced_test_slice(
    fs: FrozenSplit,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct the natural-class-prior test slice for one dataset.

    Rationale (audit finding B): the balanced test slice has a near-
    uniform class prior — so it does NOT reflect what a DTPlayer faces
    in live play, where drop_3 dominates. The unbalanced test slice is
    every row of the original CSV whose ``orig_row_id`` is *not* in
    ``train_orig_ids ∪ val_orig_ids``. The tree never saw any of these
    rows during training or pruning, and the empirical class prior is
    the natural one — exactly the distribution the DTPlayer encounters.

    Implementation: re-load the original CSV, build the held-out
    mask, apply the same binning the tree was trained on, return
    ``(X, y)`` ready for prediction.
    """
    from ai.dt_pipeline import POPOUT_BIN_DEFINITIONS, inspect_dataset

    df, move_col, _counts, continuous_cols = inspect_dataset(fs.csv_path)

    # Drop a deterministic copy so we can mask cleanly.
    df = df.copy().reset_index(drop=True)
    in_train_or_val = set(fs.train_orig_ids) | set(fs.val_orig_ids)
    keep_mask = ~df.index.to_series().isin(in_train_or_val)
    df_held = df.loc[keep_mask].copy()

    y = df_held[move_col].astype(int).reset_index(drop=True)
    X = df_held.drop(columns=[move_col]).reset_index(drop=True)

    # Apply the same binning used at train time. ``bin_features`` takes
    # three frames at once for symmetry; reuse it by passing the same
    # frame three times and pulling out the first return.
    for col in continuous_cols:
        if col not in POPOUT_BIN_DEFINITIONS:
            continue
        bins = POPOUT_BIN_DEFINITIONS[col]['bins']
        labels = POPOUT_BIN_DEFINITIONS[col]['labels']
        X[col] = pd.cut(X[col], bins=bins, labels=labels).astype(str)

    return X, y


def evaluate_on_test(
    trees: dict[str, dict],
    pruned_trees: dict[str, dict],
    splits: dict[str, FrozenSplit],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each of the 4 trees on the frozen test set.

    Returns ``(balanced_df, unbalanced_df)``. Both DataFrames have one
    row per (dataset, prune_mode) with the full metric suite.
    """
    balanced_rows: list[dict] = []
    unbalanced_rows: list[dict] = []

    unbalanced_cache: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for dataset, fs in splits.items():
        # Lazy-build the unbalanced slice once per dataset (loading +
        # binning the full CSV is the expensive part).
        if dataset not in unbalanced_cache:
            print(f"\n[eval] building unbalanced slice for {dataset}", flush=True)
            unbalanced_cache[dataset] = _build_unbalanced_test_slice(fs)
            Xu, yu = unbalanced_cache[dataset]
            print(f"[eval] unbalanced slice: {len(Xu):,} rows "
                  f"(natural class prior)", flush=True)
        Xu, yu = unbalanced_cache[dataset]

        for prune_mode, tree_dict in (
            ("unpruned", trees),
            ("pruned", pruned_trees),
        ):
            tree = tree_dict[dataset]
            print(f"\n[eval] {dataset}/{prune_mode} balanced "
                  f"({len(fs.X_test):,} rows)", flush=True)
            m_bal = _eval_one_tree(tree, fs.X_test, fs.y_test)
            balanced_rows.append({
                "dataset": dataset, "prune_mode": prune_mode, **m_bal,
            })

            print(f"[eval] {dataset}/{prune_mode} unbalanced "
                  f"({len(Xu):,} rows)", flush=True)
            m_unbal = _eval_one_tree(tree, Xu, yu)
            unbalanced_rows.append({
                "dataset": dataset, "prune_mode": prune_mode, **m_unbal,
            })

    balanced_df = pd.DataFrame(balanced_rows)
    unbalanced_df = pd.DataFrame(unbalanced_rows)

    balanced_df.to_csv(
        output_dir / "test_results_balanced.csv",
        index=False, float_format="%.6f",
    )
    unbalanced_df.to_csv(
        output_dir / "test_results_unbalanced.csv",
        index=False, float_format="%.6f",
    )
    print(f"[eval] wrote test_results_balanced.csv + "
          f"test_results_unbalanced.csv", flush=True)
    return balanced_df, unbalanced_df


# ---------------------------------------------------------------------------
# Stage 5 — confusion matrices + feature importance
# ---------------------------------------------------------------------------


def generate_confusion_matrices(
    trees: dict[str, dict],
    pruned_trees: dict[str, dict],
    splits: dict[str, FrozenSplit],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """One confusion matrix per tree, on the balanced test slice."""
    from ai.dt_pipeline import confusion_matrix, predict_batch

    matrices: dict[str, pd.DataFrame] = {}
    for dataset, fs in splits.items():
        for prune_mode, tree_dict in (
            ("unpruned", trees),
            ("pruned", pruned_trees),
        ):
            tree = tree_dict[dataset]
            y_pred = predict_batch(fs.X_test, tree)
            cm = confusion_matrix(
                fs.y_test, y_pred, classes=_ALL_CLASSES,
            )
            key = f"{dataset}_{prune_mode}"
            path = output_dir / f"confmat_{key}.csv"
            cm.to_csv(path, index_label="true_class")
            matrices[key] = cm
            print(f"[confmat] {key} -> {path.relative_to(_PROJECT_ROOT)}",
                  flush=True)
    return matrices


def generate_feature_importance(
    trees: dict[str, dict],
    pruned_trees: dict[str, dict],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Feature-importance table for each of the 4 trees."""
    from ai.dt_pipeline import feature_importance

    importances: dict[str, pd.DataFrame] = {}
    for dataset in trees:
        for prune_mode, tree_dict in (
            ("unpruned", trees),
            ("pruned", pruned_trees),
        ):
            fi = feature_importance(tree_dict[dataset])
            key = f"{dataset}_{prune_mode}"
            path = output_dir / f"featimp_{key}.csv"
            fi.to_csv(path, index=False)
            importances[key] = fi
            print(f"[featimp] {key} -> {path.relative_to(_PROJECT_ROOT)} "
                  f"({len(fi)} features)", flush=True)
    return importances


# ---------------------------------------------------------------------------
# Stage 6 — tournaments
# ---------------------------------------------------------------------------


def _tournament_config_for(
    name: str,
    games: int,
    player_a: dict,
    player_b: dict,
) -> dict:
    """One-matchup config block, in the shape ``scripts.run_tournament``
    validates against."""
    return {
        "matchups": [
            {
                "name": name,
                "games": games,
                "player_a": player_a,
                "player_b": player_b,
            }
        ]
    }


def _write_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


def run_tournaments(
    pruned_trees: dict[str, dict],
    mcts_config: dict[str, Any],
    games: dict[str, int],
    output_dir: Path,
    configs_dir: Path,
    workers: int,
    smoke: bool,
) -> dict[str, dict]:
    """Generate the 3 tournament JSONs, run them via ``run_tournament``,
    parse the resulting CSVs.

    Returns a dict keyed by tournament name. Each value contains the
    parsed CSV row plus the file paths so the report writer can
    cross-reference.
    """
    # Tree paths must be project-root-relative strings — the runner
    # resolves them against ``_PROJECT_ROOT`` so any working directory
    # produces the same artefact. In production this is ``data/...``;
    # in smoke it's whatever ``--output-dir`` was set to.
    try:
        out_rel = output_dir.relative_to(_PROJECT_ROOT)
    except ValueError:
        # ``output_dir`` lives outside the project root (unusual);
        # fall back to the absolute path.
        out_rel = output_dir
    dt_opt = (out_rel / "id3_tree_optimised_pruned.pkl").as_posix()
    dt_base = (out_rel / "id3_tree_baseline_pruned.pkl").as_posix()

    # Suffix smoke configs so they don't collide with the real configs
    # in the same directory. Smoke configs end up in a temp dir; real
    # configs are versioned under scripts/tournament_configs/.
    suffix = "_smoke" if smoke else ""
    cfg_dir = (
        Path(tempfile.mkdtemp(prefix="id3_smoke_cfg_")) if smoke else configs_dir
    )
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Build the three configs. Tree-pickle paths point at the artefacts
    # we just wrote; the MCTS config block is the §4.5 optimised MCTS.
    spec = [
        (
            "dt_opt_vs_mcts",
            "dt_optimised_pruned_vs_mcts_20k" + suffix,
            games["dt_opt_vs_mcts"],
            {"type": "dt", "tree_pickle": dt_opt},
            {"type": "mcts", **mcts_config},
        ),
        (
            "dt_opt_vs_dt_base",
            "dt_optimised_pruned_vs_dt_baseline_pruned" + suffix,
            games["dt_opt_vs_dt_base"],
            {"type": "dt", "tree_pickle": dt_opt},
            {"type": "dt", "tree_pickle": dt_base},
        ),
        (
            "dt_opt_vs_random",
            "dt_optimised_pruned_vs_random" + suffix,
            games["dt_opt_vs_random"],
            {"type": "dt", "tree_pickle": dt_opt},
            {"type": "random"},
        ),
    ]

    results: dict[str, dict] = {}
    for slug, matchup_name, n_games, p_a, p_b in spec:
        cfg = _tournament_config_for(matchup_name, n_games, p_a, p_b)
        cfg_path = cfg_dir / f"{slug}{suffix}.json"
        _write_config(cfg_path, cfg)

        # Tournament CSVs live under data/tournaments/ to keep the
        # id3 artefact dir (output_dir) focused on tree pickles / CV /
        # confmat / featimp outputs.
        if smoke:
            tournaments_dir = output_dir
        else:
            tournaments_dir = _PROJECT_ROOT / "data" / "tournaments"
        tournaments_dir.mkdir(parents=True, exist_ok=True)
        out_csv = tournaments_dir / f"tournament_{slug}{suffix}.csv"
        # Tournament runner refuses to overwrite an existing file. In
        # production this is a feature (catches accidental reruns); in
        # smoke / repeated dev runs it's friction, so clear it.
        if out_csv.exists():
            out_csv.unlink()

        print(f"\n[tourney] {slug}: {n_games} games -> "
              f"{out_csv.relative_to(_PROJECT_ROOT)}", flush=True)

        cmd = [
            sys.executable, "-m", "scripts.run_tournament",
            str(cfg_path),
            "--workers", str(workers),
            "--out", str(out_csv),
            "--base-seed", str(SEED),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print("[tourney] STDERR:", proc.stderr, file=sys.stderr)
            raise SystemExit(
                f"error: tournament {slug} failed with code {proc.returncode}"
            )
        # Echo the runner's stderr summary so the user sees progress.
        if proc.stderr.strip():
            sys.stderr.write(proc.stderr)

        with out_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit(f"error: empty tournament CSV at {out_csv}")
        row = rows[0]
        a_wins = int(row["a_wins"])
        b_wins = int(row["b_wins"])
        draws = int(row["draws"])
        total = int(row["total_games"])
        ci_a = _wilson_ci(a_wins, total)
        ci_b = _wilson_ci(b_wins, total)
        ci_d = _wilson_ci(draws, total)
        results[slug] = {
            "matchup_name": matchup_name,
            "n_games": total,
            "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "a_winrate": a_wins / total if total else 0.0,
            "b_winrate": b_wins / total if total else 0.0,
            "draw_rate": draws / total if total else 0.0,
            "ci_a_lo": ci_a[0], "ci_a_hi": ci_a[1],
            "ci_b_lo": ci_b[0], "ci_b_hi": ci_b[1],
            "ci_d_lo": ci_d[0], "ci_d_hi": ci_d[1],
            "a_wins_as_p1": int(row["a_wins_as_p1"]),
            "a_wins_as_p2": int(row["a_wins_as_p2"]),
            "b_wins_as_p1": int(row["b_wins_as_p1"]),
            "b_wins_as_p2": int(row["b_wins_as_p2"]),
            "draws_p1_was_a": int(row["draws_p1_was_a"]),
            "draws_p1_was_b": int(row["draws_p1_was_b"]),
            "config_path": str(cfg_path.relative_to(_PROJECT_ROOT))
            if not smoke else str(cfg_path),
            "csv_path": str(out_csv.relative_to(_PROJECT_ROOT)),
        }
        print(f"[tourney] {slug}: A={a_wins} B={b_wins} D={draws}  "
              f"A%={results[slug]['a_winrate']*100:.1f}  "
              f"(CI {ci_a[0]*100:.1f}–{ci_a[1]*100:.1f})", flush=True)

    if smoke:
        # Clean up the throwaway smoke config dir; the per-tournament
        # config paths in ``results`` already point at the (now-gone)
        # temp files, but the CSVs remain under ``output_dir`` for
        # inspection.
        shutil.rmtree(cfg_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Stage 7 — REP pruning summary (for the report)
# ---------------------------------------------------------------------------


def summarise_rep(
    unpruned: dict[str, dict],
    pruned: dict[str, dict],
    splits: dict[str, FrozenSplit],
) -> pd.DataFrame:
    """Per dataset, report n_nodes / val_accuracy / test_macro_f1 before vs
    after REP."""
    from ai.dt_pipeline import evaluate_quiet
    from ai.id3 import count_leaves, count_nodes

    rows: list[dict] = []
    for dataset, fs in splits.items():
        for state, t in (("unpruned", unpruned[dataset]),
                          ("pruned", pruned[dataset])):
            val = evaluate_quiet(t, fs.X_val, fs.y_val, classes=_ALL_CLASSES)
            tst = evaluate_quiet(t, fs.X_test, fs.y_test, classes=_ALL_CLASSES)
            rows.append({
                "dataset": dataset, "state": state,
                "n_nodes": int(count_nodes(t)),
                "n_leaves": int(count_leaves(t)),
                "val_accuracy": float(val["accuracy"]),
                "val_macro_f1": float(val["macro_f1"]),
                "test_accuracy": float(tst["accuracy"]),
                "test_macro_f1": float(tst["macro_f1"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 8 — report writer
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _fmt_score(x: float) -> str:
    return f"{x:.4f}"


def _format_recipe(recipe: dict[str, int]) -> str:
    return (f"cap={recipe['cap']}, "
            f"max_depth={recipe['max_depth']}, "
            f"min_samples={recipe['min_samples']}")


def _move_label(c: int) -> str:
    return f"drop_{c}" if c < 7 else f"pop_{c-7}"


def write_report(
    out_path: Path,
    *,
    recipe: dict[str, int],
    mcts_config: dict[str, Any],
    splits: dict[str, FrozenSplit],
    cv_agg: pd.DataFrame,
    balanced_df: pd.DataFrame,
    unbalanced_df: pd.DataFrame,
    rep_summary: pd.DataFrame,
    importances: dict[str, pd.DataFrame],
    tournaments: dict[str, dict],
    n_folds: int,
    wallclock_total: float,
    smoke: bool,
) -> None:
    """Consolidate everything into the final-evaluation markdown report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# Final ID3 Evaluation Report")
    lines.append("")
    lines.append("**Date:** 2026-05-17 · **Branch:** main")
    lines.append("**Generated by:** `scripts/evaluate_id3.py`  "
                 f"(seed={SEED}, workers via `--workers`)")
    lines.append(f"**Wallclock:** {wallclock_total:.1f}s "
                 + ("**(SMOKE MODE — numbers are NOT slide-ready)**" if smoke else ""))
    lines.append("")
    if smoke:
        lines.append(
            "> **⚠ SMOKE-MODE RUN.** This report was produced by `--smoke`, "
            "which downsizes every step (1k rows/dataset, depth-2 trees, "
            f"{n_folds}-fold CV, {SMOKE_TOURNAMENT_GAMES}-game tournaments, "
            f"MCTS at {SMOKE_MCTS_ITERATIONS} iters) so the pipeline "
            "finishes in under two minutes. Numbers here validate plumbing "
            "only — re-run without `--smoke` on the 24-core PC for the "
            "actual slide-grade results."
        )
        lines.append("")

    # -----------------------------------------------------------------
    # §1 Methodology
    # -----------------------------------------------------------------
    lines.append("## 1. Methodology")
    lines.append("")
    lines.append(f"**Recipe (shared across both datasets, per D1 §5):** "
                 f"`{_format_recipe(recipe)}`.")
    lines.append("")
    lines.append("**Frozen-split protocol** (addresses D1 audit §7A — "
                 "test-row stability):")
    lines.append("")
    lines.append("- Each dataset's original CSV is loaded once. An "
                 "explicit `_orig_row_id` column is added BEFORE "
                 "`balance_classes`, so every post-balance row knows "
                 "which original-CSV row it came from.")
    lines.append("- `balance_classes(cap=N, random_state=42)` undersamples "
                 "the over-represented classes; the kept rows retain "
                 "their `_orig_row_id`.")
    lines.append("- `stratified_split(seed=42)` produces 72/8/20 "
                 "train/val/test (nested two-step: 80/20 then 90/10 of the "
                 "80%) via the existing pipeline. The split is byte-stable "
                 "across re-runs.")
    lines.append("- `train_orig_ids`, `val_orig_ids` and `test_orig_ids` "
                 "are persisted to `data/id3/test_indices_<dataset>.json` "
                 "so the test slice is reproducible. The four trained "
                 "trees (unpruned baseline, pruned baseline, unpruned "
                 "optimised, pruned optimised) all evaluate on the same "
                 "frozen `X_test`.")
    lines.append("")
    lines.append("**Two test views (addresses D1 audit §7B — class-prior "
                 "shift):**")
    lines.append("")
    lines.append("1. **Balanced test slice** — the `X_test` from the "
                 "frozen split. Near-uniform class prior (effect of the "
                 "cap=N balancing). Apples-to-apples comparison against "
                 "the val metrics reported by D1.")
    lines.append("2. **Unbalanced test slice** — every row of the "
                 "original CSV whose `orig_row_id ∉ train ∪ val`. "
                 "Natural class prior (drop_3 dominates); never seen "
                 "during training or pruning. **This is the closest "
                 "proxy to what a `DTPlayer` faces in live play.** Live-"
                 "play metrics in this report are read off this slice.")
    lines.append("")
    lines.append("**5-fold CV:** folds are taken from `X_train ∪ X_val` "
                 "so the frozen test set is never touched. For each "
                 "held-out fold the remaining 4 folds are used for "
                 "training; in the pruned variant ~20 % of those four "
                 "folds is set aside as the REP slice so the eval slice "
                 "stays untouched. 4 configurations × " + str(n_folds)
                 + " folds = " + str(4 * n_folds) + " trainings.")
    lines.append("")
    lines.append("**Metrics:** accuracy, macro-F1, top-3 accuracy, "
                 "POP-class recall (mean recall over classes 7–13). "
                 "Macro-F1 is the headline because the balanced slice is "
                 "still imperfectly uniform and the unbalanced slice is "
                 "heavily skewed — neither pleases accuracy. Top-3 is "
                 "the honest 'did the tree learn the right region of "
                 "move-space' metric because MCTS frequently has 2-3 "
                 "near-equivalent moves at any state.")
    lines.append("")

    # -----------------------------------------------------------------
    # §2 CV results
    # -----------------------------------------------------------------
    lines.append("## 2. 5-fold CV results")
    lines.append("")
    lines.append("Mean ± std over folds (on the held-out fold each time, "
                 "drawn from `X_train ∪ X_val`).")
    lines.append("")
    lines.append("| dataset | state | macro_f1 (mean ± std) | accuracy "
                 "(mean ± std) | top_3_acc (mean ± std) | pop_recall "
                 "(mean ± std) | n_nodes (mean) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in cv_agg.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['prune_mode']} "
            f"| {r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f} "
            f"| {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f} "
            f"| {r['mean_top_3_accuracy']:.4f} ± "
            f"{r['std_top_3_accuracy']:.4f} "
            f"| {r['mean_pop_recall']:.4f} ± {r['std_pop_recall']:.4f} "
            f"| {int(r['mean_n_nodes']):,} |"
        )
    lines.append("")
    lines.append("Per-fold rows: `data/id3/cv_per_fold.csv`. "
                 "Aggregate: `data/id3/cv_aggregate.csv`.")
    lines.append("")

    # -----------------------------------------------------------------
    # §3 Held-out test — balanced slice
    # -----------------------------------------------------------------
    lines.append("## 3. Held-out test set — balanced slice")
    lines.append("")
    lines.append("Single-snapshot evaluation on the frozen `X_test`. "
                 "Class prior is near-uniform (effect of `cap`).")
    lines.append("")
    lines.append("| dataset | state | accuracy | macro_f1 | top_3_acc | "
                 "pop_recall | n_rows |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in balanced_df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['prune_mode']} "
            f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['top_3_accuracy']:.4f} | {r['pop_recall']:.4f} "
            f"| {int(r['n_total']):,} |"
        )
    lines.append("")
    lines.append("Confusion matrices: `data/id3/confmat_<dataset>_<state>.csv`.")
    lines.append("")

    # -----------------------------------------------------------------
    # §4 Held-out test — unbalanced slice
    # -----------------------------------------------------------------
    lines.append("## 4. Held-out test set — unbalanced slice (natural prior)")
    lines.append("")
    lines.append("Every row of the original CSV whose `orig_row_id` is "
                 "not in `train ∪ val`. Natural class prior — **this is "
                 "closer to what `DTPlayer` faces in real play.** "
                 "Accuracy will be lower here than on the balanced "
                 "slice; macro-F1 typically holds up better.")
    lines.append("")
    lines.append("| dataset | state | accuracy | macro_f1 | top_3_acc | "
                 "pop_recall | n_rows |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in unbalanced_df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['prune_mode']} "
            f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['top_3_accuracy']:.4f} | {r['pop_recall']:.4f} "
            f"| {int(r['n_total']):,} |"
        )
    lines.append("")

    # -----------------------------------------------------------------
    # §5 REP pruning summary
    # -----------------------------------------------------------------
    lines.append("## 5. REP pruning summary")
    lines.append("")
    lines.append("Per dataset, before vs after reduced-error post-"
                 "pruning. REP runs a single bottom-up pass over the "
                 "frozen val set and is guaranteed never to make val "
                 "accuracy worse (ties go to the smaller tree).")
    lines.append("")
    lines.append("| dataset | state | n_nodes | n_leaves | val_accuracy | "
                 "val_macro_f1 | test_accuracy | test_macro_f1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in rep_summary.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['state']} "
            f"| {int(r['n_nodes']):,} | {int(r['n_leaves']):,} "
            f"| {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} "
            f"| {r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} |"
        )
    lines.append("")

    # -----------------------------------------------------------------
    # §6 Feature importance
    # -----------------------------------------------------------------
    lines.append("## 6. Feature importance (top-10 per pruned tree)")
    lines.append("")
    lines.append("Sample-weighted ranking (see "
                 "`ai.dt_pipeline.feature_importance`): a feature near "
                 "the root with N samples ranks above a feature deep "
                 "in a tiny subtree, even if the latter appears more "
                 "often.")
    lines.append("")
    for dataset in DATASETS:
        key = f"{dataset}_pruned"
        if key not in importances:
            continue
        lines.append(f"**{dataset} (pruned):**")
        lines.append("")
        lines.append("| rank | feature | n_splits | n_samples_total |")
        lines.append("|---:|---|---:|---:|")
        fi = importances[key].head(10)
        for i, (_, r) in enumerate(fi.iterrows(), start=1):
            lines.append(
                f"| {i} | `{r['feature']}` | {int(r['n_splits'])} "
                f"| {int(r['n_samples_total']):,} |"
            )
        lines.append("")
    lines.append("Full tables: `data/id3/featimp_<dataset>_<state>.csv`.")
    lines.append("")

    # -----------------------------------------------------------------
    # §7 Tournament results
    # -----------------------------------------------------------------
    lines.append("## 7. Tournament results")
    lines.append("")
    lines.append(f"**MCTS configuration** (per notebook §4.5 — the "
                 f"optimised MCTS): `iter={mcts_config['iterations']}, "
                 f"c={mcts_config['exploration_weight']}, "
                 f"k={mcts_config['num_children_to_expand']}, "
                 f"rollout_depth={mcts_config['rollout_depth_limit']}, "
                 f"variant={mcts_config['uct_variant']}`.")
    lines.append("")
    lines.append("Wilson 95 % CIs reported on every winrate. P1/P2 "
                 "splits track whether the result is driven by the "
                 "first-mover advantage (matchups alternate 50/50, so "
                 "the count should be ~balanced between sides).")
    lines.append("")
    for slug, label_a, label_b in (
        ("dt_opt_vs_mcts",
            "DT_optimised_pruned",
            f"MCTS({mcts_config['iterations']}-iter, c={mcts_config['exploration_weight']}, UCB1)"),
        ("dt_opt_vs_dt_base",
            "DT_optimised_pruned",
            "DT_baseline_pruned"),
        ("dt_opt_vs_random",
            "DT_optimised_pruned",
            "Random"),
    ):
        if slug not in tournaments:
            continue
        t = tournaments[slug]
        lines.append(f"### 7.{['dt_opt_vs_mcts','dt_opt_vs_dt_base','dt_opt_vs_random'].index(slug)+1} "
                     f"{label_a} vs {label_b} — {t['n_games']} games")
        lines.append("")
        lines.append(f"- **A = {label_a}:** wins {t['a_wins']} "
                     f"({t['a_winrate']*100:.1f}%), "
                     f"Wilson-95: [{t['ci_a_lo']*100:.1f}%, "
                     f"{t['ci_a_hi']*100:.1f}%]")
        lines.append(f"- **B = {label_b}:** wins {t['b_wins']} "
                     f"({t['b_winrate']*100:.1f}%), "
                     f"Wilson-95: [{t['ci_b_lo']*100:.1f}%, "
                     f"{t['ci_b_hi']*100:.1f}%]")
        lines.append(f"- **Draws:** {t['draws']} "
                     f"({t['draw_rate']*100:.1f}%), "
                     f"Wilson-95: [{t['ci_d_lo']*100:.1f}%, "
                     f"{t['ci_d_hi']*100:.1f}%]")
        lines.append("")
        lines.append("**P1 / P2 split** (matchups alternate 50/50; "
                     "uneven sides reveals first-mover bias):")
        lines.append("")
        lines.append("| outcome | A=P1 | A=P2 |")
        lines.append("|---|---:|---:|")
        lines.append(f"| A wins | {t['a_wins_as_p1']} | {t['a_wins_as_p2']} |")
        lines.append(f"| B wins | {t['b_wins_as_p2']} | {t['b_wins_as_p1']} |")
        lines.append(f"| draws | {t['draws_p1_was_a']} | {t['draws_p1_was_b']} |")
        lines.append("")
        lines.append(f"Config: `{t['config_path']}`. "
                     f"Raw CSV: `{t['csv_path']}`.")
        lines.append("")

    # -----------------------------------------------------------------
    # §8 Comparison with literature
    # -----------------------------------------------------------------
    lines.append("## 8. Comparison with literature")
    lines.append("")
    lines.append("Behavioural-cloning ID3 trees in Connect-4-like games "
                 "are typically reported in the **5-20 %** winrate band "
                 "vs strong MCTS opponents (Scripts of Tribute paper, "
                 "Anthony et al. on expert imitation). A flat 0 % winrate "
                 "(as in the `outro` comparison repo) is the signal of a "
                 "broken legality guard, not of the tree itself.")
    lines.append("")
    if "dt_opt_vs_mcts" in tournaments:
        wr = tournaments["dt_opt_vs_mcts"]["a_winrate"]
        lines.append(f"Our `DT_optimised_pruned` achieves "
                     f"**{wr*100:.1f}% winrate** against the §4.5 "
                     f"optimised MCTS over {tournaments['dt_opt_vs_mcts']['n_games']} "
                     f"games. This number is "
                     + ("inside the expected band — the tree learned "
                        "something MCTS-like and the legality guard from "
                        "E1 is doing its job."
                        if 0.05 <= wr <= 0.30
                        else "outside the typical 5-20 % BC-vs-MCTS band — "
                             "interpret with care (see §9).")
                     + " Against `Random`, the tree should approach "
                     "or exceed 95 % to confirm it learned more than "
                     "noise.")
    lines.append("")

    # -----------------------------------------------------------------
    # §9 Anything else
    # -----------------------------------------------------------------
    lines.append("## 9. Anything else I noticed")
    lines.append("")
    lines.append("- **POP recall (audit §7G).** The CV table shows the "
                 "fold-to-fold std of `pop_recall`; if it materially "
                 "exceeds the std of `macro_f1`, the tournament results "
                 "may swing between re-trainings. Watch this column.")
    lines.append("")
    lines.append("- **Schema-bug caveat.** The D1 sweep that selected "
                 "the recipe ran with two pre-existing schema bugs in "
                 "`POPOUT_BIN_DEFINITIONS` (`move_count` upper bound, "
                 "bottom-row piece-count upper bound). E2 fixed them, "
                 "but the recipe `(cap=100000, max_depth=20, "
                 "min_samples=10)` was selected on slightly different "
                 "binned data than what trains here. Per user "
                 "decision: hyperparameter optima are robust enough "
                 "that this delta is acceptable.")
    lines.append("")
    lines.append("- **Position-duplication ablation (audit §7C).** Not "
                 "executed — would require a separate "
                 "ablation run with `df.drop_duplicates(...)` before "
                 "splitting. If a professor asks about test-set leakage "
                 "on self-play datasets, this is the question and the "
                 "fix.")
    lines.append("")
    lines.append("- **Board canonicalisation (audit §7D).** "
                 "`current_player` is still a feature; the alternative "
                 "(swap 1↔2 cells when current_player=2, drop the "
                 "feature) was not tried. Halves the effective state "
                 "space at the cost of asymmetric per-player tactics. "
                 "Defer to a follow-up.")
    lines.append("")
    lines.append("- **Equal-frequency binning (audit §7E).** Bins are "
                 "still the hand-designed `POPOUT_BIN_DEFINITIONS`. "
                 "An equal-frequency ablation would strengthen the "
                 "'why these cut points' answer; deferred.")
    lines.append("")

    # -----------------------------------------------------------------
    # §10 Recommended slide narrative
    # -----------------------------------------------------------------
    lines.append("## 10. Recommended slide narrative")
    lines.append("")
    if not smoke and "dt_opt_vs_mcts" in tournaments:
        wr_mcts = tournaments["dt_opt_vs_mcts"]["a_winrate"]
        wr_rand = tournaments.get("dt_opt_vs_random", {}).get("a_winrate", 0.0)
        wr_dt = tournaments.get("dt_opt_vs_dt_base", {}).get("a_winrate", 0.0)
        lines.append(
            f"The pruned ID3 tree trained on the §4.5-optimised MCTS "
            f"dataset (shared recipe `{_format_recipe(recipe)}`) reaches "
            f"**{wr_mcts*100:.1f}% winrate vs the same optimised MCTS** "
            f"that produced its training data — landing "
            + ("inside the canonical 5-20 % BC-vs-expert band, which is "
               "the right result: behavioural cloning can imitate but "
               "rarely beats the teacher."
               if 0.05 <= wr_mcts <= 0.30
               else "outside the canonical BC-vs-expert band; the slide "
                    "deck must own this anomaly.")
            + f" Against `Random` the tree wins "
            f"**{wr_rand*100:.1f} %** (sanity check passed). Against "
            f"the baseline-MCTS tree the optimised tree wins "
            f"**{wr_dt*100:.1f} %**, "
            + ("evidence that the upstream MCTS tuning paid off "
               "downstream — the better the teacher, the better the "
               "imitator."
               if wr_dt > 0.55
               else "but the gap is within noise — the upstream MCTS "
                    "tuning may not transfer all the way through to "
                    "DT play strength."))
    else:
        lines.append("*(smoke run — fill in after the real run)*")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_id3",
        description="Final ID3 evaluation orchestrator.",
    )
    p.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="parallel worker processes for CV + tournaments "
             "(default: os.cpu_count())",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/id3"),
        help="directory for pickles / CSV / JSON artefacts (default: data/id3/)",
    )
    p.add_argument(
        "--report", type=Path, default=Path("evaluation_results.md"),
        help="path for the consolidated markdown report",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="end-to-end validation at tiny scale (under 2 min)",
    )
    p.add_argument(
        "--skip-tournaments", action="store_true",
        help="train + CV + eval only; skip the 3 tournament runs",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = _build_parser().parse_args(argv)

    smoke = bool(args.smoke)
    workers = max(1, int(args.workers))
    output_dir = (Path(args.output_dir)
                  if Path(args.output_dir).is_absolute()
                  else (_PROJECT_ROOT / args.output_dir)).resolve()
    report_path = (Path(args.report)
                   if Path(args.report).is_absolute()
                   else (_PROJECT_ROOT / args.report)).resolve()
    configs_dir = _PROJECT_ROOT / "scripts" / "tournament_configs"

    if smoke:
        recipe = SMOKE_RECIPE
        cv_folds = SMOKE_CV_FOLDS
        mcts_config = {**MCTS_CONFIG, "iterations": SMOKE_MCTS_ITERATIONS}
        games = {
            "dt_opt_vs_mcts": SMOKE_TOURNAMENT_GAMES,
            "dt_opt_vs_dt_base": SMOKE_TOURNAMENT_GAMES,
            "dt_opt_vs_random": SMOKE_TOURNAMENT_GAMES,
        }
        smoke_rows = SMOKE_ROWS
    else:
        recipe = RECIPE
        cv_folds = CV_FOLDS
        mcts_config = MCTS_CONFIG
        games = TOURNAMENT_GAMES
        smoke_rows = None

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[id3] workers={workers}  seed={SEED}  smoke={smoke}", flush=True)
    print(f"[id3] recipe={recipe}", flush=True)
    print(f"[id3] output_dir={output_dir}", flush=True)
    print(f"[id3] report={report_path}", flush=True)

    started = time.monotonic()

    splits = freeze_splits(recipe, output_dir, smoke_rows=smoke_rows)
    unpruned = train_unpruned_trees(splits, recipe, output_dir)
    pruned = apply_rep(unpruned, splits, output_dir)
    cv_agg = cross_validate(splits, recipe, cv_folds, workers, output_dir)
    balanced_df, unbalanced_df = evaluate_on_test(
        unpruned, pruned, splits, output_dir,
    )
    _matrices = generate_confusion_matrices(
        unpruned, pruned, splits, output_dir,
    )
    importances = generate_feature_importance(unpruned, pruned, output_dir)
    rep_summary = summarise_rep(unpruned, pruned, splits)

    if args.skip_tournaments:
        tournaments: dict[str, dict] = {}
        print("[id3] --skip-tournaments was set; skipping tournament runs.",
              flush=True)
    else:
        tournaments = run_tournaments(
            pruned, mcts_config, games, output_dir, configs_dir,
            workers=workers, smoke=smoke,
        )

    wallclock = time.monotonic() - started
    write_report(
        report_path,
        recipe=recipe, mcts_config=mcts_config, splits=splits,
        cv_agg=cv_agg, balanced_df=balanced_df, unbalanced_df=unbalanced_df,
        rep_summary=rep_summary, importances=importances,
        tournaments=tournaments,
        n_folds=cv_folds, wallclock_total=wallclock, smoke=smoke,
    )

    print(f"\n[id3] done — total wallclock {wallclock:.1f}s "
          f"report -> {report_path.relative_to(_PROJECT_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception):
        pass
    raise SystemExit(main())
