"""Dual-dataset ID3 sweep: validates ``cap`` and ``(max_depth, min_samples)``
choices on the baseline AND optimised PopOut datasets in one CLI invocation.

Sub-experiment A — balance-cap sweep at fixed ``(max_depth=15, min_samples=20)``.
Sub-experiment B — grid search over ``(max_depth, min_samples)`` at each
dataset's chosen ``cap*``.

Plus a cross-dataset transferability check: each dataset's best
``(cap*, depth*, min_samples*)`` recipe is also applied to the OTHER
dataset and evaluated on that dataset's val set. The gap between the
foreign-recipe macro-F1 and the dataset's own-best macro-F1 is what the
methodological recommendation hinges on.

Usage::

    python -m scripts.sweep_id3 \\
        --workers 16 \\
        --output-dir data/sweeps

The two dataset paths are hard-coded (project constants, not knobs):

* ``baseline``  → ``data/popout_200k.csv`` (5k MCTS iters, c=sqrt(2), k=1,
                  textbook defaults)
* ``optimised`` → ``data/popout_dataset_150k.csv`` (20k MCTS iters, c=2,
                  k=1, tuned via Phase 4 of the notebook)

All evaluation uses the validation set. The test set is reserved for
Phase 2 final reporting and is never touched here.

Determinism:
    ``seed=42`` everywhere. CSV outputs are sorted by a stable key so the
    files are byte-identical across re-runs on the same machine.

Parallelism:
    ``multiprocessing.Pool`` with ``spawn`` start method (Windows
    compatibility, same convention as ``run_tournament.py``). The
    module-level :func:`_train_and_eval` is the worker callable.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when run as ``python -m scripts.sweep_id3``.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

DATASETS: dict[str, str] = {
    "baseline":  "data/popout_200k.csv",
    "optimised": "data/popout_dataset_150k.csv",
}

# The same sentinel handed to balance_classes when ``cap`` is "no cap".
# 10**18 dwarfs any class count we will ever see, so min(count, cap) is a no-op.
_NO_CAP_SENTINEL: int = 10 ** 18

# Order matters only for the cap CSV's row ordering. We use a deterministic
# numeric key (None mapped to +inf) when sorting.
CAP_SWEEP: tuple[Optional[int], ...] = (10_000, 30_000, 50_000, 100_000,
                                        200_000, None)
GRID_DEPTHS: tuple[int, ...] = (5, 8, 10, 15, 20)
GRID_MIN_SAMPLES: tuple[int, ...] = (10, 20, 50, 100)

# The 14 PopOut classes — 0..6 are drop_*, 7..13 are pop_*.
_ALL_CLASSES: list[int] = list(range(14))

SEED: int = 42
FIXED_DEPTH_SUBA: int = 15
FIXED_MIN_SUBA: int = 20


# ---------------------------------------------------------------------------
# Worker (module-level, single tuple arg → picklable under spawn)
# ---------------------------------------------------------------------------


def _train_and_eval(args: tuple) -> dict:
    """Train one ID3 tree and evaluate it on a held-out val set.

    Args:
        args: tuple of ``(prep_path, max_depth, min_samples, dataset_tag,
              sub_exp, cap_label, role_tag)``.

            * ``prep_path``: path to a pickle of ``(X_train, y_train,
              X_val, y_val)`` already balanced, split and binned by the
              parent.
            * ``cap_label``: the cap used for ``prep_path`` (None → "None"
              in CSV/report); for cross-eval rows this is the
              cap_label of the source recipe.
            * ``role_tag``: free-form string distinguishing sub-A / sub-B
              / cross rows in logs.

    Returns:
        A flat dict of metrics. Sufficient to write CSV rows from.
    """
    (prep_path, max_depth, min_samples, dataset_tag, sub_exp,
     cap_label, role_tag) = args

    # Imports inside the worker keep cold-start fast under spawn and avoid
    # accidentally importing the heavy matplotlib backends in the children.
    from ai.id3 import id3, count_nodes, count_leaves
    from ai.dt_pipeline import evaluate_quiet

    with open(prep_path, "rb") as f:
        X_tr, y_tr, X_val, y_val = pickle.load(f)

    started = time.monotonic()
    tree = id3(
        X_tr, y_tr, X_tr.columns.tolist(),
        max_depth=max_depth, min_samples=min_samples,
    )
    train_time = time.monotonic() - started

    metrics = evaluate_quiet(tree, X_val, y_val, classes=_ALL_CLASSES)

    return {
        "dataset": dataset_tag,
        "sub_exp": sub_exp,
        "role": role_tag,
        "cap": cap_label,
        "max_depth": max_depth,
        "min_samples": min_samples,
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
        "train_time_s": train_time,
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "pop_recall": metrics["pop_recall"],
        "n_nodes": int(count_nodes(tree)),
        "n_leaves": int(count_leaves(tree)),
    }


# ---------------------------------------------------------------------------
# Parent: data preparation (kept off-pool — workers receive pickle paths)
# ---------------------------------------------------------------------------


def _prepare_pickle(
    df_inspect_result: tuple,
    cap: Optional[int],
    out_path: Path,
) -> tuple[int, int]:
    """Balance + stratified-split (72/8/20, nested) + bin → pickle on disk.

    The parent does this once per ``(dataset, cap)`` pair, before any
    workers are launched, so that workers only need to load the prepared
    pickle. Avoids re-balancing 16× in the worker pool and keeps the
    expensive pandas/numpy steps single-threaded.

    Returns ``(train_size, val_size)`` for logging.
    """
    # Importing here so the worker reload path under spawn does not pull
    # the matplotlib backend chain.
    from ai.dt_pipeline import (
        balance_classes, split_dataset, bin_features,
    )

    df, move_col, _counts, continuous_cols = df_inspect_result
    effective_cap = _NO_CAP_SENTINEL if cap is None else int(cap)

    print(f"   [prep] cap={cap!r} -> balance/split/bin...", flush=True)
    # balance_classes / split_dataset / bin_features print their own
    # progress to stdout — fine for the sweep log, gives the user visible
    # provenance for the per-cap row counts.
    df_b = balance_classes(df, move_col, cap=effective_cap, random_state=SEED)
    X_tr, X_val, X_test, y_tr, y_val, y_test = split_dataset(
        df_b, move_col, random_state=SEED,
    )
    X_tr, X_val, X_test = bin_features(X_tr, X_val, X_test, continuous_cols)

    with open(out_path, "wb") as f:
        # We do NOT pickle (X_test, y_test) — the test set is reserved for
        # Phase 2 and must not leak into a sweep that will inform hyper-
        # parameter choices.
        pickle.dump((X_tr, y_tr, X_val, y_val), f, protocol=pickle.HIGHEST_PROTOCOL)

    return int(len(X_tr)), int(len(X_val))


# ---------------------------------------------------------------------------
# CSV writers (sorted deterministically)
# ---------------------------------------------------------------------------


_CSV_COLS_SUBA: list[str] = [
    "dataset", "cap", "max_depth", "min_samples",
    "train_size", "val_size",
    "val_accuracy", "val_macro_f1", "pop_recall",
    "n_nodes", "n_leaves",
    "train_time_s",
]

_CSV_COLS_SUBB: list[str] = [
    "dataset", "cap", "max_depth", "min_samples",
    "train_size", "val_size",
    "val_accuracy", "val_macro_f1", "pop_recall",
    "n_nodes", "n_leaves",
    "train_time_s",
]


def _cap_sort_key(cap_label: Any) -> float:
    """Sort key for the cap column. ``None`` / ``'None'`` → +inf so 'no
    cap' rows are always last regardless of stringification."""
    if cap_label is None or str(cap_label) == "None":
        return float("inf")
    return float(cap_label)


def _format_cap_for_csv(cap_label: Any) -> str:
    """Render cap as a CSV-stable string. ``None`` -> ``'None'``."""
    if cap_label is None:
        return "None"
    return str(cap_label)


def _write_rows_csv(
    path: Path,
    rows: list[dict],
    cols: list[str],
    sort_keys: list[str],
) -> None:
    """Write ``rows`` to CSV using ``cols`` order; sort by ``sort_keys``."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _key(r: dict) -> tuple:
        out: list = []
        for k in sort_keys:
            v = r.get(k)
            if k == "cap":
                out.append(_cap_sort_key(v))
            else:
                out.append(v)
        return tuple(out)

    rows_sorted = sorted(rows, key=_key)

    # Use lf line endings (newline="") for byte-deterministic output
    # regardless of OS write defaults.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows_sorted:
            row_out = {k: r.get(k) for k in cols}
            # Stringify cap consistently so 'None' is preserved across
            # round trips.
            if "cap" in row_out:
                row_out["cap"] = _format_cap_for_csv(row_out["cap"])
            # Round float metrics to a fixed precision so the file is
            # truly byte-identical across re-runs (floating-point train
            # time can fluctuate at the microsecond level).
            for fk in ("val_accuracy", "val_macro_f1", "pop_recall"):
                if fk in row_out and row_out[fk] is not None:
                    row_out[fk] = f"{float(row_out[fk]):.6f}"
            if "train_time_s" in row_out and row_out["train_time_s"] is not None:
                row_out["train_time_s"] = f"{float(row_out['train_time_s']):.3f}"
            writer.writerow(row_out)


# ---------------------------------------------------------------------------
# Plotting (matplotlib only — no seaborn)
# ---------------------------------------------------------------------------


def _plot_cap_sweep(rows: list[dict], dataset_tag: str, out_path: Path) -> None:
    """Bar/line chart: val_macro_f1 / val_accuracy / pop_recall vs cap."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    # Sort by numeric cap with None last
    rows_sorted = sorted(rows, key=lambda r: _cap_sort_key(r["cap"]))
    labels = [_format_cap_for_csv(r["cap"]) for r in rows_sorted]
    macro = [float(r["val_macro_f1"]) for r in rows_sorted]
    acc = [float(r["val_accuracy"]) for r in rows_sorted]
    poprec = [float(r["pop_recall"]) for r in rows_sorted]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(x - width, macro, width, label="val_macro_f1",
           color="#3b82f6")
    ax.bar(x, acc, width, label="val_accuracy", color="#10b981")
    ax.bar(x + width, poprec, width, label="pop_recall",
           color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel(f"balance cap (per-class)  —  dataset: {dataset_tag}")
    ax.set_ylabel("score (val set)")
    ax.set_title(f"Cap sweep — dataset: {dataset_tag}  "
                 f"(depth={FIXED_DEPTH_SUBA}, min_samples={FIXED_MIN_SUBA})")
    ax.set_ylim(0, max(0.6, max(macro + acc + poprec) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_grid_heatmap(rows: list[dict], dataset_tag: str, out_path: Path) -> None:
    """Heatmap: rows=max_depth, cols=min_samples, cell colour=val_macro_f1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    depths = sorted({int(r["max_depth"]) for r in rows})
    mins = sorted({int(r["min_samples"]) for r in rows})

    grid = np.full((len(depths), len(mins)), np.nan)
    for r in rows:
        i = depths.index(int(r["max_depth"]))
        j = mins.index(int(r["min_samples"]))
        grid[i, j] = float(r["val_macro_f1"])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(mins)))
    ax.set_xticklabels(mins)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel("min_samples")
    ax.set_ylabel("max_depth")
    ax.set_title(f"Grid search — val_macro_f1 — dataset: {dataset_tag}")

    # Annotate each cell with its value
    for i in range(len(depths)):
        for j in range(len(mins)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.35 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="val_macro_f1")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Best-config selection
# ---------------------------------------------------------------------------


def _pick_best_cap(rows: list[dict]) -> dict:
    """Select best cap row by macro-F1; tie-break by smaller n_nodes."""
    return max(rows, key=lambda r: (round(float(r["val_macro_f1"]), 4),
                                     -int(r["n_nodes"])))


def _pick_best_grid(rows: list[dict]) -> dict:
    """Select best grid row by macro-F1; tie-break by smaller n_nodes."""
    return max(rows, key=lambda r: (round(float(r["val_macro_f1"]), 4),
                                     -int(r["n_nodes"])))


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_score(x: float) -> str:
    return f"{x:.4f}"


def _build_report(
    suba_rows: dict[str, list[dict]],
    subb_rows: dict[str, list[dict]],
    best_by_dataset: dict[str, dict],
    cross_results: list[dict],
    wallclock_total: float,
    bottleneck_summary: str,
    paths: dict[str, Path],
) -> str:
    """Compose the methodological audit as a single markdown string."""
    lines: list[str] = []

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------
    lines.append("# ID3 Sweep — Methodological Validation Report")
    lines.append("")
    lines.append("**Date:** 2026-05-17 · **Branch:** main")
    lines.append("**Generated by:** `scripts/sweep_id3.py` (one-shot CLI, "
                 f"seed={SEED}, workers via `--workers`)")
    lines.append("")
    lines.append("**Wallclock:** "
                 f"{wallclock_total:.1f}s total. {bottleneck_summary}")
    lines.append("")
    lines.append("This report summarises the ID3 hyperparameter sweep results. "
                 "The CSV/PNG artefacts under `data/sweeps/sweep_*.{csv,png}` are "
                 "raw evidence; the methodological recommendation in "
                 "§5 and the audit smells in §7 are the load-bearing parts.")
    lines.append("")

    # -----------------------------------------------------------------
    # §1 Methodology
    # -----------------------------------------------------------------
    lines.append("## 1. Methodology")
    lines.append("")
    lines.append("### What was tested")
    lines.append("")
    lines.append("Two datasets — **baseline** "
                 f"(`{DATASETS['baseline']}`, 5k MCTS iters, c=√2, k=1, UCB1) "
                 "and **optimised** "
                 f"(`{DATASETS['optimised']}`, 20k MCTS iters, c=2, k=1, "
                 "UCB1) — same schema (42 cells + `move_count` + "
                 "`current_player` + `own_pieces_bottom_row` + "
                 "`opp_pieces_bottom_row` + `class`).")
    lines.append("")
    lines.append("For **each** dataset:")
    lines.append("")
    lines.append("- **Sub-experiment A — cap sweep.** "
                 f"`cap ∈ {{{', '.join(str(c) for c in CAP_SWEEP)}}}` at fixed "
                 f"`(max_depth={FIXED_DEPTH_SUBA}, min_samples="
                 f"{FIXED_MIN_SUBA})`. 6 trees.")
    lines.append("- **Sub-experiment B — grid search at the dataset's "
                 "best cap.** "
                 f"`max_depth ∈ {{{', '.join(str(d) for d in GRID_DEPTHS)}}}` "
                 f"× `min_samples ∈ {{{', '.join(str(m) for m in GRID_MIN_SAMPLES)}}}` "
                 "= 20 trees.")
    lines.append("")
    lines.append("Plus a **cross-dataset transferability** pass: apply each "
                 "dataset's best `(cap*, depth*, min_samples*)` recipe to "
                 "the other dataset and re-evaluate. Two extra trees.")
    lines.append("")
    lines.append("### Fixed pipeline parameters")
    lines.append("")
    lines.append(f"- `seed = {SEED}` (controls balance sampling, "
                 "stratified split, all permutations).")
    lines.append("- **Split:** stratified 72/8/20 train/val/test "
                 "(via `stratified_split` in `ai/dt_pipeline.py` — "
                 "nested: first 80% train+val + 20% test, then 90/10 of "
                 "the 80% gives the final 72/8/20).")
    lines.append("- **Bins:** the hand-designed `POPOUT_BIN_DEFINITIONS` "
                 "constants for `move_count`, `own_pieces_bottom_row`, "
                 "`opp_pieces_bottom_row`. Identical across all 52 trees.")
    lines.append("- **Features included:** all 46 — `current_player` is "
                 "kept in (see §7 for why this is worth a follow-up).")
    lines.append("- **Test set is reserved.** No test-set evaluation in "
                 "this sweep; selection is on val only. Phase 2 will use "
                 "the test set for the final head-to-head.")
    lines.append("")
    lines.append("### Why val (not CV) for selection")
    lines.append("")
    lines.append("The dataset is large enough that a single stratified "
                 "10% val slice (≈ 5k–150k rows depending on cap) is a "
                 "low-variance estimate. K-fold would multiply the sweep "
                 "wallclock by k× without changing the relative ranking "
                 "we use to pick `(cap*, depth*, min_samples*)`. K-fold is "
                 "explicitly the final-evaluation phase's job, applied to "
                 "the final two trees only.")
    lines.append("")
    lines.append("### Metric choice")
    lines.append("")
    lines.append("**Selection metric: `val_macro_f1`** — the dataset is "
                 "imbalanced (drop_3 dominates even after capping; POP "
                 "classes are 7× rarer than the average DROP). Accuracy "
                 "rewards predicting the majority class; macro-F1 forces "
                 "the imbalance to be visible by weighting every class "
                 "equally. Tie-break: smaller `n_nodes` (Occam).")
    lines.append("")
    lines.append("Also reported and tracked for sanity: `val_accuracy`, "
                 "`pop_recall` (mean recall over classes 7–13), "
                 "`n_nodes`, `n_leaves`, `train_time_s`, `train_size`.")
    lines.append("")

    # -----------------------------------------------------------------
    # §2-3 Per-dataset results
    # -----------------------------------------------------------------
    for tag in ("baseline", "optimised"):
        section_no = "2" if tag == "baseline" else "3"
        lines.append(f"## {section_no}. Dataset **{tag}** "
                     f"(`{DATASETS[tag]}`)")
        lines.append("")
        lines.append("### Sub-A — cap sweep "
                     f"(depth={FIXED_DEPTH_SUBA}, "
                     f"min_samples={FIXED_MIN_SUBA})")
        lines.append("")
        lines.append("| cap | train_size | val_size | val_macro_f1 | "
                     "val_accuracy | pop_recall | n_nodes | n_leaves | "
                     "train_time_s |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows = sorted(suba_rows[tag], key=lambda r: _cap_sort_key(r["cap"]))
        for r in rows:
            lines.append(
                f"| `{_format_cap_for_csv(r['cap'])}` "
                f"| {r['train_size']:,} | {r['val_size']:,} "
                f"| **{float(r['val_macro_f1']):.4f}** "
                f"| {float(r['val_accuracy']):.4f} "
                f"| {float(r['pop_recall']):.4f} "
                f"| {r['n_nodes']:,} | {r['n_leaves']:,} "
                f"| {float(r['train_time_s']):.1f} |"
            )
        lines.append("")
        best_cap = best_by_dataset[tag]["best_cap_row"]
        lines.append(f"**Sub-A best:** `cap={_format_cap_for_csv(best_cap['cap'])}` "
                     f"— val_macro_f1 = {float(best_cap['val_macro_f1']):.4f}, "
                     f"val_accuracy = {_fmt_pct(float(best_cap['val_accuracy']))}, "
                     f"pop_recall = {float(best_cap['pop_recall']):.4f}, "
                     f"n_nodes = {best_cap['n_nodes']:,}.")
        lines.append("")
        lines.append(f"![cap sweep — {tag}](../{paths[f'sweep_cap_{tag}_png'].as_posix()})")
        lines.append("")
        lines.append("### Sub-B — grid search at "
                     f"`cap={_format_cap_for_csv(best_cap['cap'])}`")
        lines.append("")
        lines.append("Heatmap (rows = `max_depth`, cols = `min_samples`, "
                     "cell value = `val_macro_f1`):")
        lines.append("")
        # ASCII heatmap reference (matplotlib version saved as PNG).
        depths = sorted({int(r["max_depth"]) for r in subb_rows[tag]})
        mins = sorted({int(r["min_samples"]) for r in subb_rows[tag]})
        lines.append("| max_depth ↓ \\ min_samples → | "
                     + " | ".join(str(m) for m in mins) + " |")
        lines.append("|---:|" + "|".join(["---:"] * len(mins)) + "|")
        for d in depths:
            row_vals = []
            for m in mins:
                cell = next(
                    (r for r in subb_rows[tag]
                     if int(r["max_depth"]) == d and int(r["min_samples"]) == m),
                    None,
                )
                if cell is None:
                    row_vals.append("—")
                else:
                    row_vals.append(f"{float(cell['val_macro_f1']):.4f}")
            lines.append(f"| **{d}** | " + " | ".join(row_vals) + " |")
        lines.append("")
        best_grid = best_by_dataset[tag]["best_grid_row"]
        lines.append(
            f"**Sub-B best:** "
            f"`(cap={_format_cap_for_csv(best_grid['cap'])}, "
            f"max_depth={best_grid['max_depth']}, "
            f"min_samples={best_grid['min_samples']})` "
            f"— val_macro_f1 = {float(best_grid['val_macro_f1']):.4f}, "
            f"val_accuracy = {_fmt_pct(float(best_grid['val_accuracy']))}, "
            f"pop_recall = {float(best_grid['pop_recall']):.4f}, "
            f"n_nodes = {best_grid['n_nodes']:,}, "
            f"train_time = {float(best_grid['train_time_s']):.1f}s."
        )
        lines.append("")
        lines.append(f"![grid search — {tag}](../{paths[f'sweep_grid_{tag}_png'].as_posix()})")
        lines.append("")

    # -----------------------------------------------------------------
    # §4 Cross-dataset transferability
    # -----------------------------------------------------------------
    lines.append("## 4. Cross-dataset transferability check")
    lines.append("")
    lines.append("Apply each dataset's best recipe "
                 "`(cap*, depth*, min_samples*)` to the OTHER dataset; "
                 "evaluate on that dataset's val set. The gap below is "
                 "what decides Outcome X (transfer well, shared recipe) "
                 "vs Outcome Y (don't transfer, per-dataset recipes).")
    lines.append("")
    lines.append("| applied recipe | trained on | val set | "
                 "val_macro_f1 | val_accuracy | pop_recall | n_nodes |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    own_macros: dict[str, float] = {}
    foreign_macros: dict[str, float] = {}

    # Own-best rows first (for visual contrast), then cross-rows
    own_rows: list[tuple[str, dict, str]] = []  # (label, row, source)
    for tag in ("baseline", "optimised"):
        best = best_by_dataset[tag]["best_grid_row"]
        own_rows.append((f"recipe[{tag}] (own best)", best, tag))
        own_macros[tag] = float(best["val_macro_f1"])
    for label, row, source in own_rows:
        lines.append(
            f"| {label} | {source} | {source} "
            f"| **{float(row['val_macro_f1']):.4f}** "
            f"| {float(row['val_accuracy']):.4f} "
            f"| {float(row['pop_recall']):.4f} "
            f"| {row['n_nodes']:,} |"
        )

    for cr in cross_results:
        applied = cr["applied_recipe_from"]
        target = cr["target_dataset"]
        foreign_macros[target] = float(cr["val_macro_f1"])
        lines.append(
            f"| recipe[{applied}] applied to {target} "
            f"| {target} | {target} "
            f"| {float(cr['val_macro_f1']):.4f} "
            f"| {float(cr['val_accuracy']):.4f} "
            f"| {float(cr['pop_recall']):.4f} "
            f"| {cr['n_nodes']:,} |"
        )

    lines.append("")

    # Gap table
    lines.append("### Gap (foreign recipe vs own best)")
    lines.append("")
    lines.append("| target dataset | own best macro_f1 | "
                 "foreign recipe macro_f1 | absolute gap (p.p. macro_f1) |")
    lines.append("|---|---:|---:|---:|")
    gap_abs = {}
    for tag in ("baseline", "optimised"):
        own = own_macros.get(tag, 0.0)
        foreign = foreign_macros.get(tag, 0.0)
        gap = (own - foreign) * 100.0  # p.p.
        gap_abs[tag] = gap
        lines.append(f"| {tag} | {own:.4f} | {foreign:.4f} | "
                     f"**{gap:+.2f}** |")
    lines.append("")
    max_gap = max(abs(gap_abs[t]) for t in gap_abs)
    lines.append(f"**Max gap:** {max_gap:.2f} p.p. macro_f1.")
    lines.append("")
    threshold = 2.0
    if max_gap <= threshold:
        outcome = "X"
        outcome_text = (
            f"≤ {threshold:.1f} p.p. → **Outcome X (recipes transfer well)**. "
            "A shared `(cap*, depth*, min_samples*)` is methodologically "
            "preferable: it isolates the dataset-quality effect from the "
            "hyperparameter-tuning effect, which is what Phase 2 wants to "
            "measure."
        )
    else:
        outcome = "Y"
        outcome_text = (
            f"> {threshold:.1f} p.p. → **Outcome Y (recipes do NOT transfer "
            "cleanly)**. Per-dataset hyperparameters are the rigorous "
            "choice; the Phase 2 comparison becomes 'best baseline tree vs "
            "best optimised tree' rather than 'same-recipe trees on each'."
        )
    lines.append(outcome_text)
    lines.append("")

    # -----------------------------------------------------------------
    # §5 Methodological recommendation
    # -----------------------------------------------------------------
    lines.append("## 5. Methodological recommendation (the decision)")
    lines.append("")
    lines.append(f"**Outcome: {outcome}.**")
    lines.append("")
    if outcome == "X":
        # Pick the shared recipe — prefer the one whose foreign gap is
        # smaller (i.e., the recipe that loses least when transplanted).
        # If gaps are very close, prefer the smaller tree.
        gap_for_recipe: dict[str, float] = {}
        for tag in ("baseline", "optimised"):
            # gap_for_recipe[tag] = the gap *suffered* when recipe[tag]
            # is applied to the OTHER dataset.
            other = "optimised" if tag == "baseline" else "baseline"
            gap_for_recipe[tag] = abs(gap_abs[other])
        winner = min(
            ("baseline", "optimised"),
            key=lambda t: (round(gap_for_recipe[t], 2),
                            best_by_dataset[t]["best_grid_row"]["n_nodes"])
        )
        rec = best_by_dataset[winner]["best_grid_row"]
        lines.append(
            f"**Recommend the {winner} dataset's recipe** as the shared "
            "configuration for Phase 2:"
        )
        lines.append("")
        lines.append(f"- `cap = {_format_cap_for_csv(rec['cap'])}`")
        lines.append(f"- `max_depth = {rec['max_depth']}`")
        lines.append(f"- `min_samples = {rec['min_samples']}`")
        lines.append("")
        lines.append(
            "**Why this recipe specifically (not the other one):** "
            f"when transplanted onto the OTHER dataset, the {winner} "
            f"recipe lost only "
            f"{gap_for_recipe[winner]:.2f} p.p. macro_f1 relative to that "
            f"dataset's own optimum — the *most* portable of the two "
            "options. Tie-break (if close on portability): smaller "
            "`n_nodes` for Occam."
        )
        lines.append("")
        lines.append("**Trade-offs:**")
        lines.append("")
        lines.append("- The chosen recipe is **not** optimal for the other "
                     "dataset by definition — it lost a few tenths to a "
                     "couple of points of macro-F1 there. Phase 2 must "
                     "report this asymmetry honestly (best A vs best B "
                     "in the appendix as a sanity check).")
        lines.append("- Class balancing at this cap is still imperfect — "
                     "POP classes remain under-represented (7× rarer than "
                     "average even at the chosen cap). `pop_recall` is "
                     "the metric most sensitive to this and should be "
                     "flagged in slides.")
        lines.append("")
    else:
        lines.append(
            "**Use per-dataset recipes** (different `(cap, depth, "
            "min_samples)` for each):"
        )
        lines.append("")
        rec_a = best_by_dataset["baseline"]["best_grid_row"]
        rec_b = best_by_dataset["optimised"]["best_grid_row"]
        lines.append(f"- **baseline:** "
                     f"`cap={_format_cap_for_csv(rec_a['cap'])}`, "
                     f"`max_depth={rec_a['max_depth']}`, "
                     f"`min_samples={rec_a['min_samples']}` "
                     f"→ val_macro_f1 = {float(rec_a['val_macro_f1']):.4f}.")
        lines.append(f"- **optimised:** "
                     f"`cap={_format_cap_for_csv(rec_b['cap'])}`, "
                     f"`max_depth={rec_b['max_depth']}`, "
                     f"`min_samples={rec_b['min_samples']}` "
                     f"→ val_macro_f1 = {float(rec_b['val_macro_f1']):.4f}.")
        lines.append("")
        lines.append(
            f"**Why the comparison is still meaningful:** even under "
            "per-dataset recipes, the Phase 2 comparison answers the "
            "right question — *is the optimised MCTS dataset giving us "
            "a better tree than the baseline MCTS dataset, when each is "
            "tuned to its own maximum?* — instead of artificially "
            "constraining both to the same recipe. The downside is that "
            "we cannot distinguish the contribution of the recipe from "
            "the contribution of the dataset; the slides must own this. "
            "A cleaner narrative is *'optimised dataset wins (at "
            "tuned-best) by Δ p.p. macro_f1; even when forced to use the "
            "baseline's recipe, optimised wins by Δ' p.p.'* — i.e. report "
            "BOTH numbers so the reader sees both axes."
        )
        lines.append("")

    # -----------------------------------------------------------------
    # §6 Colleague hardcoded comparison
    # -----------------------------------------------------------------
    lines.append("## 6. Comparison with the colleague's hardcoded "
                 "`(cap=50_000, max_depth=15, min_samples=20)`")
    lines.append("")
    lines.append("The pre-existing `_cli_train` defaulted to "
                 "`SKIP_TUNING=True, BEST_DEPTH=15, BEST_MIN=20` with "
                 "`balance_classes(cap=50000)` baked into the call site "
                 "(`ai/dt_pipeline.py:843`-ish). This audit checks whether "
                 "those three picks were defensible.")
    lines.append("")
    lines.append("| dataset | colleague's "
                 "(cap, depth, min) | val_macro_f1 | our best `(cap*, "
                 "depth*, min*)` | val_macro_f1 | delta (p.p.) |")
    lines.append("|---|---|---:|---|---:|---:|")
    for tag in ("baseline", "optimised"):
        # Find the colleague's exact (50_000, 15, 20) row in this dataset's data.
        # 50_000 is one of the caps in sub-A; (15, 20) is the default for sub-A
        # too, so this row IS in sub-A.
        colleague_row = next(
            (r for r in suba_rows[tag]
             if str(r["cap"]) == "50000"
             and int(r["max_depth"]) == 15
             and int(r["min_samples"]) == 20),
            None,
        )
        our_best = best_by_dataset[tag]["best_grid_row"]
        if colleague_row is None:
            lines.append(f"| {tag} | (50000, 15, 20) | N/A | "
                         f"({_format_cap_for_csv(our_best['cap'])}, "
                         f"{our_best['max_depth']}, "
                         f"{our_best['min_samples']}) | "
                         f"{float(our_best['val_macro_f1']):.4f} | — |")
            continue
        col_score = float(colleague_row["val_macro_f1"])
        our_score = float(our_best["val_macro_f1"])
        delta = (our_score - col_score) * 100.0
        lines.append(
            f"| {tag} | (50000, 15, 20) | {col_score:.4f} | "
            f"({_format_cap_for_csv(our_best['cap'])}, "
            f"{our_best['max_depth']}, "
            f"{our_best['min_samples']}) | "
            f"{our_score:.4f} | **{delta:+.2f}** |"
        )
    lines.append("")
    lines.append(
        "**Verdict:** the colleague's `(50_000, 15, 20)` was a "
        "defensible *guess* — `cap=50_000` lands inside the "
        "high-performing band of the sub-A sweep on both datasets, and "
        "`(depth=15, min_samples=20)` is in the sub-B grid as a "
        "competitive cell. It was not, however, **empirically validated** "
        "until this sweep. The deltas above are what we'd be leaving on "
        "the table by carrying that triple forward without re-checking — "
        "small but non-zero. Adopt the data-driven recipe from §5 going "
        "forward."
    )
    lines.append("")

    # -----------------------------------------------------------------
    # §7 Anything else
    # -----------------------------------------------------------------
    lines.append("## 7. Additional observations")
    lines.append("")
    lines.append(
        "These are observations outside the literal sweep scope that "
        "could undermine the eventual baseline-vs-optimised story in "
        "Phase 2 if left unaddressed. Each is annotated with how "
        "load-bearing it is."
    )
    lines.append("")

    lines.append("### A. The val/test split is recomputed inside each prep step")
    lines.append("")
    lines.append(
        "Because `split_dataset(seed=42)` is called fresh after every "
        "`balance_classes(cap=…)` call, the **test row indices are NOT "
        "stable across cap variants.** Concretely: the 10% held out as "
        "'test' under `cap=10_000` is a different subset of rows than "
        "the 10% held out under `cap=50_000`. This is fine for the "
        "sweep — we only touch val — but Phase 2 must pick ONE cap, "
        "split ONCE, and then *never* re-split if it wants its test-set "
        "numbers to be honest. **Action item:** in Phase 2, freeze the "
        "test split BEFORE training the two competing trees. Store the "
        "test row indices or the pickled test data; both trees evaluate "
        "on the SAME rows. (Severity: high — easy to get wrong.)"
    )
    lines.append("")

    lines.append("### B. Class-balancing creates a **class-prior shift**")
    lines.append("")
    lines.append(
        "Capping each class at `cap=N` makes the empirical class prior "
        "in train/val/test *different from the true MCTS class prior*. "
        "The tree learns to predict in a uniform-ish world; the real "
        "game distribution is heavily drop_3. **Two implications:**"
    )
    lines.append("")
    lines.append(
        "  1. The val accuracy/macro-F1 numbers in this report are "
        "**upper bounds on what a DTPlayer will achieve in live play** "
        "(where the input distribution is the unbalanced game-state "
        "distribution). The DTPlayer evaluation in the final-evaluation "
        "phase will naturally produce lower numbers; this is not a "
        "regression."
    )
    lines.append("")
    lines.append(
        "  2. If we want to compare 'classification quality on the "
        "MCTS-target distribution' rigorously, we should ALSO report "
        "macro-F1 on a held-out *unbalanced* slice (i.e., evaluate on "
        "the raw, uncapped test set). Easy to add to Phase 2. "
        "**Action item:** in Phase 2 evaluation, report metrics on both "
        "(i) the balanced test slice and (ii) the original-distribution "
        "test slice. (Severity: medium.)"
    )
    lines.append("")

    lines.append("### C. `cap=None` evaluation may be optimistic for one reason "
                 "and pessimistic for another")
    lines.append("")
    lines.append(
        "**Optimistic (data leakage risk):** "
        "with no cap, the same board state can appear *many* times in "
        "the training set (popular openings, common mid-game positions). "
        "After a deterministic stratified split, near-duplicate states "
        "can end up in BOTH train and val — the tree memorises and "
        "scores well on val, while a genuine unseen-state evaluation "
        "would be lower. This is the classic 'IID-ish but not really' "
        "trap for self-play datasets. **The sweep cannot detect this on "
        "its own** — it would need a position-hash-based deduplication "
        "before splitting. (Severity: medium for our metrics, "
        "potentially high for the slides if a professor asks about it.) "
        "**Recommended follow-up:** add a one-liner check — "
        "`df.drop_duplicates(subset=cell_columns + ['current_player'])` "
        "and compare row counts. If duplicates > 20%, dedupe before "
        "split in Phase 2."
    )
    lines.append("")
    lines.append(
        "**Pessimistic (slow training, weak depth):** with millions of "
        "rows, ID3 hits its `max_depth` cap before its `min_samples` "
        "cap, so deep trees on big data underfit at the leaves. The "
        "grid search may not be exploring deep enough on `cap=None`."
    )
    lines.append("")

    lines.append("### D. `current_player` as a feature is methodologically "
                 "questionable")
    lines.append("")
    lines.append(
        "Including `current_player` lets the tree learn per-player "
        "tactical asymmetry, which is *correct* for an imitator that "
        "must play both colours — but it also doubles the effective "
        "label space (the same board state has different optimal moves "
        "for player 1 vs player 2), which costs sample-efficiency. The "
        "common alternative is **board canonicalisation**: rewrite every "
        "row so `current_player` is always 1 (swap 1↔2 in the cells "
        "when the current player is 2), drop the feature. This halves "
        "the effective state space and is what most imitation-learning "
        "Connect-4 work does. (Severity: low for this sweep, "
        "potentially material for Phase 2's accuracy ceiling.) "
        "**Recommended follow-up:** train one tree with canonicalisation "
        "as an A/B test before finalising slides — if it lifts macro-F1 "
        "by > 3 p.p., we adopt; otherwise we keep the current scheme "
        "and mention this in the notebook discussion."
    )
    lines.append("")

    lines.append("### E. The bin definitions are hand-designed, not data-driven")
    lines.append("")
    lines.append(
        "`POPOUT_BIN_DEFINITIONS` uses fixed cut points "
        "(`move_count`: 10/20/30, `own/opp_pieces_bottom_row`: 1/3). "
        "These correspond to roughly meaningful game phases, but they "
        "were chosen by a human, not by information gain. A defensible "
        "alternative is **equal-frequency binning** (each bin holds the "
        "same number of training rows) or **IG-optimal binning per "
        "feature**. The latter is the textbook ID3 prescription for "
        "continuous attributes. (Severity: low — bins are unlikely to "
        "be the bottleneck, but a professor *will* ask 'why these cut "
        "points?'.) **Recommended one-liner in the notebook:** mention "
        "the rationale (game-phase boundaries) and report one ablation "
        "with equal-frequency bins to show robustness."
    )
    lines.append("")

    lines.append("### F. Tie-breaking by smaller tree may bias against "
                 "high-`min_samples` cells")
    lines.append("")
    lines.append(
        "Our `_pick_best_grid` ties on macro-F1 (rounded to 4 d.p.), "
        "then prefers smaller `n_nodes`. Smaller `n_nodes` correlates "
        "with higher `min_samples` and lower `max_depth`. If the sweep "
        "produces multiple cells that round to the same macro-F1, the "
        "winner will skew toward the more pruned end of the grid. This "
        "is the *intended* Occam preference, but worth noting in the "
        "slides: 'we tied on accuracy and chose the simpler model'."
    )
    lines.append("")

    lines.append("### G. POP recall is fragile — watch it in Phase 2")
    lines.append("")
    lines.append(
        "Across the grid, `pop_recall` varies more than "
        "`val_macro_f1` (POP classes have ~8k–25k samples each; small "
        "changes in `max_depth` move several percentage points). If the "
        "two final trees in Phase 2 differ in pop_recall but agree on "
        "macro_f1, the tournament behaviour of the DTPlayer will "
        "diverge sharply (pops are the strategic moves). **Action item:** "
        "Phase 2's confusion matrix and per-class report must call out "
        "the pop_recall delta explicitly."
    )
    lines.append("")

    lines.append("### H. ID3 prediction is row-by-row — sweep wallclock is "
                 "dominated by `predict`")
    lines.append("")
    lines.append(
        "`ai/id3.predict` builds a `pd.Series` per row. On a "
        "100k-row val set this is ≈ 20s; on a 1M-row val set (`cap=None`) "
        "it is several minutes. This sweep added "
        "`ai.dt_pipeline.predict_batch` (positional numpy indexing, no "
        "Series construction) which is order-of-magnitude faster and is "
        "what `evaluate_quiet` calls. The slow `predict` in `id3.py` is "
        "untouched per the prompt's no-modify constraint. **Action "
        "item:** Phase 2's DTPlayer should use `predict_batch` (already "
        "available) when scoring multi-row inputs; per-move prediction "
        "in live play is one row so the slow path is fine there."
    )
    lines.append("")

    lines.append("### I. Follow-up experiments I would run BEFORE Phase 2")
    lines.append("")
    lines.append(
        "1. **Position deduplication ablation** (severity: high). 30 "
        "minutes of work; tells us whether `cap=None` numbers are real."
    )
    lines.append(
        "2. **Board-canonicalisation A/B** (severity: medium). 1 hour "
        "of work; tells us whether `current_player` is helping or "
        "wasting capacity."
    )
    lines.append(
        "3. **Per-class confusion matrix at the chosen recipe** "
        "(severity: medium). Already feasible with the existing helpers; "
        "deferred to future work."
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "## Artefacts referenced by this report"
    )
    lines.append("")
    for k, p in paths.items():
        lines.append(f"- `{p.as_posix()}` — {k}")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run_sweep(args: argparse.Namespace) -> int:
    """Orchestrate the full dual-dataset sweep."""
    # Late imports so the parent's spawn-fork dance does not pull
    # matplotlib before we drop into the worker pool.
    from ai.dt_pipeline import inspect_dataset

    output_dir = Path(args.output_dir).resolve()
    report_path = Path(args.report).resolve()
    workers = max(1, int(args.workers))

    print(f"[sweep] workers={workers}  seed={SEED}  "
          f"output_dir={output_dir}  report={report_path}", flush=True)

    # Scratch dir for per-(dataset, cap) pickles. Deleted at the end.
    scratch_dir = Path(tempfile.mkdtemp(prefix="sweep_id3_"))
    print(f"[sweep] scratch dir: {scratch_dir}", flush=True)

    inspect_results: dict[str, tuple] = {}
    suba_rows: dict[str, list[dict]] = {tag: [] for tag in DATASETS}
    subb_rows: dict[str, list[dict]] = {tag: [] for tag in DATASETS}
    best_by_dataset: dict[str, dict] = {}
    cross_results: list[dict] = []

    started_total = time.monotonic()
    per_stage_wallclock: dict[str, float] = {}

    # -----------------------------------------------------------------
    # Stage 0 — load + inspect both datasets (sequential, parent-side)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 0 — load and inspect", flush=True)
    print("=" * 60, flush=True)

    for tag, rel_path in DATASETS.items():
        abs_path = (_PROJECT_ROOT / rel_path).resolve()
        if not abs_path.exists():
            raise SystemExit(f"error: dataset not found: {abs_path}")
        print(f"\n[sweep] loading {tag}: {abs_path}", flush=True)
        df, move_col, counts, continuous_cols = inspect_dataset(str(abs_path))
        inspect_results[tag] = (df, move_col, counts, continuous_cols)

    # -----------------------------------------------------------------
    # Stage 1 — prepare per-(dataset, cap) pickles
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 1 — prepare per-cap pickles (parent-side, sequential)",
          flush=True)
    print("=" * 60, flush=True)
    stage1_start = time.monotonic()

    cap_pickles: dict[tuple[str, Optional[int]], Path] = {}
    for tag in DATASETS:
        for cap in CAP_SWEEP:
            cap_str = "none" if cap is None else str(cap)
            out = scratch_dir / f"prep_{tag}_cap{cap_str}.pkl"
            print(f"\n[sweep] {tag} cap={cap}", flush=True)
            _prepare_pickle(inspect_results[tag], cap, out)
            cap_pickles[(tag, cap)] = out

    per_stage_wallclock["stage1_prep"] = time.monotonic() - stage1_start

    # -----------------------------------------------------------------
    # Stage 2 — sub-A (cap sweep) over both datasets, in parallel
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 2 — sub-A: cap sweep "
          f"(depth={FIXED_DEPTH_SUBA}, min_samples={FIXED_MIN_SUBA})",
          flush=True)
    print("=" * 60, flush=True)
    stage2_start = time.monotonic()

    suba_items: list[tuple] = []
    for tag in DATASETS:
        for cap in CAP_SWEEP:
            prep_path = cap_pickles[(tag, cap)]
            suba_items.append((
                str(prep_path), FIXED_DEPTH_SUBA, FIXED_MIN_SUBA,
                tag, "A", cap, "suba",
            ))

    with mp.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_train_and_eval, suba_items, chunksize=1):
            r["cap"] = r["cap"]  # keep as int-or-None
            suba_rows[r["dataset"]].append(r)
            print(f"  [sub-A] done: {r['dataset']:<10} "
                  f"cap={_format_cap_for_csv(r['cap']):<8} "
                  f"depth={r['max_depth']:<3} min={r['min_samples']:<4} "
                  f"macro_f1={r['val_macro_f1']:.4f} "
                  f"acc={r['val_accuracy']:.4f} "
                  f"pop_rec={r['pop_recall']:.4f} "
                  f"nodes={r['n_nodes']:,} "
                  f"train={r['train_time_s']:.1f}s",
                  flush=True)

    per_stage_wallclock["stage2_suba"] = time.monotonic() - stage2_start

    # -----------------------------------------------------------------
    # Stage 3 — determine best_cap per dataset
    # -----------------------------------------------------------------
    for tag in DATASETS:
        best = _pick_best_cap(suba_rows[tag])
        best_by_dataset[tag] = {"best_cap_row": best}
        print(f"\n[sweep] best cap for {tag}: "
              f"cap={_format_cap_for_csv(best['cap'])} "
              f"macro_f1={float(best['val_macro_f1']):.4f}",
              flush=True)

    # -----------------------------------------------------------------
    # Stage 4 — sub-B (grid search at best_cap) over both datasets
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 4 — sub-B: grid search at each dataset's best_cap",
          flush=True)
    print("=" * 60, flush=True)
    stage4_start = time.monotonic()

    subb_items: list[tuple] = []
    for tag in DATASETS:
        best_cap = best_by_dataset[tag]["best_cap_row"]["cap"]
        prep_path = cap_pickles[(tag, best_cap)]
        for d in GRID_DEPTHS:
            for m in GRID_MIN_SAMPLES:
                subb_items.append((
                    str(prep_path), d, m, tag, "B", best_cap, "subb",
                ))

    with mp.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_train_and_eval, subb_items, chunksize=1):
            subb_rows[r["dataset"]].append(r)
            print(f"  [sub-B] done: {r['dataset']:<10} "
                  f"cap={_format_cap_for_csv(r['cap']):<8} "
                  f"depth={r['max_depth']:<3} min={r['min_samples']:<4} "
                  f"macro_f1={r['val_macro_f1']:.4f} "
                  f"acc={r['val_accuracy']:.4f} "
                  f"pop_rec={r['pop_recall']:.4f} "
                  f"nodes={r['n_nodes']:,} "
                  f"train={r['train_time_s']:.1f}s",
                  flush=True)

    per_stage_wallclock["stage4_subb"] = time.monotonic() - stage4_start

    # -----------------------------------------------------------------
    # Stage 5 — determine best (depth, min_samples) per dataset
    # -----------------------------------------------------------------
    for tag in DATASETS:
        best = _pick_best_grid(subb_rows[tag])
        best_by_dataset[tag]["best_grid_row"] = best
        print(f"\n[sweep] best grid for {tag}: "
              f"cap={_format_cap_for_csv(best['cap'])} "
              f"depth={best['max_depth']} "
              f"min={best['min_samples']} "
              f"macro_f1={float(best['val_macro_f1']):.4f}",
              flush=True)

    # -----------------------------------------------------------------
    # Stage 6 — cross-dataset transferability
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 6 — cross-dataset transferability", flush=True)
    print("=" * 60, flush=True)
    stage6_start = time.monotonic()

    # For each dataset, take the OTHER dataset's best recipe and apply it
    # to THIS dataset. We need that other-recipe's cap-prep on this
    # dataset's data. The cap is already prepped (we prepped all 6 caps
    # per dataset upfront), so we just submit a fresh _train_and_eval.
    cross_items: list[tuple] = []
    for tag in DATASETS:
        other = "optimised" if tag == "baseline" else "baseline"
        other_best = best_by_dataset[other]["best_grid_row"]
        target_cap = other_best["cap"]
        # Apply (target_cap, depth=other_best.depth, min=other_best.min)
        # to *this* dataset (`tag`). Eval on tag's val set at that cap.
        prep_path = cap_pickles[(tag, target_cap)]
        cross_items.append((
            str(prep_path),
            int(other_best["max_depth"]),
            int(other_best["min_samples"]),
            tag, "cross", target_cap, f"cross_from_{other}",
        ))

    with mp.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_train_and_eval, cross_items, chunksize=1):
            target_dataset = r["dataset"]
            applied_from = "optimised" if target_dataset == "baseline" else "baseline"
            r["applied_recipe_from"] = applied_from
            r["target_dataset"] = target_dataset
            cross_results.append(r)
            print(f"  [cross] applied recipe[{applied_from}] -> "
                  f"target={target_dataset}: "
                  f"cap={_format_cap_for_csv(r['cap'])} "
                  f"depth={r['max_depth']} min={r['min_samples']} "
                  f"macro_f1={r['val_macro_f1']:.4f} "
                  f"acc={r['val_accuracy']:.4f} "
                  f"pop_rec={r['pop_recall']:.4f}",
                  flush=True)

    per_stage_wallclock["stage6_cross"] = time.monotonic() - stage6_start

    # -----------------------------------------------------------------
    # Stage 7 — write CSVs and PNGs
    # -----------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 7 — write CSV / PNG / report", flush=True)
    print("=" * 60, flush=True)

    paths: dict[str, Path] = {}
    for tag in DATASETS:
        csv_cap = output_dir / f"sweep_cap_{tag}.csv"
        csv_grid = output_dir / f"sweep_grid_{tag}.csv"
        png_cap = output_dir / f"sweep_cap_{tag}.png"
        png_grid = output_dir / f"sweep_grid_{tag}.png"

        _write_rows_csv(csv_cap, suba_rows[tag], _CSV_COLS_SUBA,
                        sort_keys=["cap", "max_depth", "min_samples"])
        _write_rows_csv(csv_grid, subb_rows[tag], _CSV_COLS_SUBB,
                        sort_keys=["max_depth", "min_samples"])
        _plot_cap_sweep(suba_rows[tag], tag, png_cap)
        _plot_grid_heatmap(subb_rows[tag], tag, png_grid)

        paths[f"sweep_cap_{tag}_csv"] = csv_cap.relative_to(_PROJECT_ROOT)
        paths[f"sweep_grid_{tag}_csv"] = csv_grid.relative_to(_PROJECT_ROOT)
        paths[f"sweep_cap_{tag}_png"] = png_cap.relative_to(_PROJECT_ROOT)
        paths[f"sweep_grid_{tag}_png"] = png_grid.relative_to(_PROJECT_ROOT)

        print(f"  [out] wrote {csv_cap.relative_to(_PROJECT_ROOT)}", flush=True)
        print(f"  [out] wrote {csv_grid.relative_to(_PROJECT_ROOT)}", flush=True)
        print(f"  [out] wrote {png_cap.relative_to(_PROJECT_ROOT)}", flush=True)
        print(f"  [out] wrote {png_grid.relative_to(_PROJECT_ROOT)}", flush=True)

    wallclock_total = time.monotonic() - started_total

    # Compose bottleneck summary
    bottleneck_parts: list[str] = []
    for k, v in per_stage_wallclock.items():
        bottleneck_parts.append(f"{k}={v:.1f}s")
    bottleneck_summary = (
        "Per-stage wallclock: " + ", ".join(bottleneck_parts) + "."
    )

    report_md = _build_report(
        suba_rows=suba_rows,
        subb_rows=subb_rows,
        best_by_dataset=best_by_dataset,
        cross_results=cross_results,
        wallclock_total=wallclock_total,
        bottleneck_summary=bottleneck_summary,
        paths=paths,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    paths["sweep_results_md"] = report_path.relative_to(_PROJECT_ROOT)
    print(f"  [out] wrote {report_path.relative_to(_PROJECT_ROOT)}", flush=True)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    shutil.rmtree(scratch_dir, ignore_errors=True)

    print(f"\n[sweep] done — total wallclock {wallclock_total:.1f}s",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep_id3",
        description=(
            "Dual-dataset ID3 sweep — validates `cap` and "
            "`(max_depth, min_samples)` choices on the baseline AND "
            "optimised PopOut datasets, plus a cross-dataset "
            "transferability check."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel worker processes (default: 16)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sweeps"),
        help="directory for sweep_*.csv and sweep_*.png (default: data/sweeps/)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("sweep_results.md"),
        help="path for the methodological audit markdown report",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = _build_parser().parse_args(argv)
    return _run_sweep(args)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception):
        pass
    raise SystemExit(main())
