"""Self-play dataset generator for PopOut.

Runs many MCTS-vs-MCTS games in parallel processes. Each game is seeded
deterministically by (base_seed, game_index) so the entire batch is
reproducible. Only MCTS-chosen moves are written to the CSV; random
exploration moves are used to diversify the state distribution but never
labelled.

Usage examples:
    python -m scripts.generate_dataset --smoke
    python -m scripts.generate_dataset --games 10000 --iterations 5000 \\
        --workers 16 --out data/popout_50k.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Make the project root importable when this script is run as
# ``python -m scripts.generate_dataset`` from anywhere.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.mcts import MCTS  # noqa: E402
from game.board import COLS, ROWS, PopOutBoard  # noqa: E402


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

_CELL_COLS: list[str] = [f"cell_{r}{c}" for r in range(ROWS) for c in range(COLS)]
HEADER: list[str] = (
    _CELL_COLS
    + [
        "move_count",
        "current_player",
        "own_pieces_bottom_row",
        "opp_pieces_bottom_row",
        "class",
    ]
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenConfig:
    """All knobs that drive a generation run.

    Frozen so it pickles cleanly across the spawn boundary on Windows.
    """

    games: int
    iterations: int
    workers: int
    eps: float
    opening_max: int
    depth_limit: int
    base_seed: int
    out_path: Path
    smoke: bool
    progress_every_games: int


# ---------------------------------------------------------------------------
# Worker-side logic
# ---------------------------------------------------------------------------


def _derive_seed(base_seed: int, game_index: int) -> int:
    """Per-game master seed.

    Independent of worker count so the dataset is the same whether the
    pool has 1 or 32 workers.
    """
    return (base_seed * 1_000_003 + game_index) & 0x7FFF_FFFF


def _row_from_decision(board: PopOutBoard, move: tuple[str, int]) -> tuple:
    """Build the CSV row for a labelled MCTS decision at ``board``."""
    cells = [int(board.board[r, c]) for r in range(ROWS) for c in range(COLS)]
    bottom = board.board[ROWS - 1]
    current = int(board.current_player)
    opp = 2 if current == 1 else 1
    own_bot = int(np.count_nonzero(bottom == current))
    opp_bot = int(np.count_nonzero(bottom == opp))
    cls = f"{move[0]}_{move[1]}"
    return tuple(cells) + (int(board.move_count), current, own_bot, opp_bot, cls)


def _play_opening(seed: int, opening_plies: int) -> PopOutBoard:
    """Produce a non-terminal board after ``opening_plies`` random plies.

    Restart with a fresh salt if a candidate trajectory hits a terminal
    state. After a generous retry cap we fall back to the empty board —
    in practice this branch is never taken because a 12-ply random
    PopOut trajectory is essentially never terminal.
    """
    if opening_plies <= 0:
        return PopOutBoard()
    for attempt in range(128):
        salt = (seed ^ 0xCDCDCD ^ (attempt * 0x9E3779B9)) & 0x7FFF_FFFF
        rng = random.Random(salt)
        board = PopOutBoard()
        ok = True
        for _ in range(opening_plies):
            valid = board.get_valid_moves()
            if not valid:
                ok = False
                break
            board.make_move(*rng.choice(valid))
            if board.is_terminal():
                ok = False
                break
        if ok:
            return board
    return PopOutBoard()


def play_one_game(args: tuple[int, GenConfig]) -> tuple[int, list[tuple]]:
    """Play one MCTS-vs-MCTS game, return labelled rows for the parent.

    The function is module-level (required by ``multiprocessing`` with
    the spawn start method on Windows). It returns ``(game_index, rows)``
    so the parent can re-order results back into game order before
    writing.
    """
    game_index, cfg = args
    seed = _derive_seed(cfg.base_seed, game_index)
    # ``rng_main`` drives all stochastic choices outside the opening: the
    # opening length pick, the eps-greedy coin flip, and the random
    # exploration move when the coin says "explore".
    rng_main = random.Random(seed ^ 0xA5A5A5)

    opening_plies = rng_main.randint(0, cfg.opening_max)
    board = _play_opening(seed, opening_plies)

    mcts = MCTS(
        iterations=cfg.iterations,
        exploration_weight=math.sqrt(2),
        rollout_depth_limit=cfg.depth_limit,
        random_seed=seed,
    )

    rows: list[tuple] = []
    search_count = 0
    gc_every = max(cfg.depth_limit, 1)

    while not board.is_terminal():
        valid = board.get_valid_moves()
        if not valid:
            break  # ``is_terminal`` should have caught this; defensive.

        if rng_main.random() < cfg.eps:
            # Exploration move: unlabelled, just diversifies the state
            # distribution downstream of this position.
            board.make_move(*rng_main.choice(valid))
            continue

        move = mcts.search(board)
        rows.append(_row_from_decision(board, move))
        board.make_move(*move)
        search_count += 1
        if search_count % gc_every == 0:
            gc.collect(2)

    return game_index, rows


# ---------------------------------------------------------------------------
# Parent-side orchestration
# ---------------------------------------------------------------------------


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _verify_existing_header(path: Path) -> None:
    """Confirm ``path``'s first row matches HEADER; raise otherwise."""
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration as exc:
            raise SystemExit(
                f"--append target {path} exists but is empty (no header)."
            ) from exc
    if first != HEADER:
        raise SystemExit(
            f"--append target {path} has header that differs from the "
            f"expected schema; refusing to append.\n"
            f"  expected[:5]={HEADER[:5]}\n"
            f"  actual[:5]={first[:5]}"
        )


def _open_output(cfg: GenConfig, append: bool):
    """Open ``cfg.out_path`` for writing.

    Returns the file handle and a ``csv.writer`` already positioned to
    receive rows (header written if needed).
    """
    out = cfg.out_path
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        if append:
            _verify_existing_header(out)
            f = out.open("a", newline="")
            return f, csv.writer(f)
        raise SystemExit(
            f"Output file {out} already exists. "
            f"Pass --append to add rows, or delete the file. Aborting."
        )

    if append:
        # --append on a fresh path: write the header once and proceed.
        f = out.open("w", newline="")
        writer = csv.writer(f)
        writer.writerow(HEADER)
        return f, writer

    f = out.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(HEADER)
    return f, writer


def _run_pool(cfg: GenConfig, append: bool) -> tuple[int, float]:
    """Drive the worker pool. Returns (total_rows, elapsed_seconds)."""
    f, writer = _open_output(cfg, append)
    started = time.monotonic()
    total_rows = 0
    next_emit = 0
    pending: dict[int, list[tuple]] = {}

    work = [(i, cfg) for i in range(cfg.games)]

    try:
        with mp.Pool(processes=cfg.workers) as pool:
            done = 0
            for game_index, rows in pool.imap_unordered(
                play_one_game, work, chunksize=1
            ):
                pending[game_index] = rows
                # Drain any consecutive game indices ready for emission.
                while next_emit in pending:
                    out_rows = pending.pop(next_emit)
                    for row in out_rows:
                        writer.writerow(row)
                    total_rows += len(out_rows)
                    next_emit += 1
                done += 1
                if done % cfg.progress_every_games == 0 or done == cfg.games:
                    elapsed = time.monotonic() - started
                    rate = total_rows / elapsed if elapsed > 0 else 0.0
                    pct = 100.0 * done / cfg.games
                    remaining = cfg.games - done
                    avg_per_game = elapsed / max(done, 1)
                    eta = _fmt_eta(remaining * avg_per_game)
                    print(
                        f"[gen] {done}/{cfg.games} games ({pct:.1f}%) | "
                        f"{total_rows} rows | {rate:.1f} rows/s | ETA {eta}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        f.close()

    elapsed = time.monotonic() - started
    return total_rows, elapsed


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------


_CLASS_RE = re.compile(r"^(drop|pop)_[0-6]$")
_CELL_OK = {"0", "1", "2"}


def _validate_smoke(path: Path) -> tuple[bool, str, int]:
    """Validate a smoke-run CSV. Return (ok, message, n_rows)."""
    if not path.exists():
        return False, "smoke CSV was not produced", 0
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, "smoke CSV is empty (no header)", 0
        if header != HEADER:
            return False, "header does not match schema", 0
        rows = list(reader)
    n = len(rows)
    if n < 50:
        return False, f"only {n} rows (expected >= 50)", n
    expected_cols = len(HEADER)
    for i, row in enumerate(rows):
        if len(row) != expected_cols:
            return False, f"row {i}: {len(row)} cols (expected {expected_cols})", n
        for j in range(42):
            v = row[j]
            if v == "" or v.lower() == "nan":
                return False, f"row {i}, cell {j}: empty/NaN", n
            if v not in _CELL_OK:
                return False, f"row {i}, cell {j}: bad value {v!r}", n
        for j in range(42, expected_cols - 1):
            v = row[j]
            if v == "" or v.lower() == "nan":
                return False, f"row {i}, col {HEADER[j]}: empty/NaN", n
        cls = row[-1]
        if not _CLASS_RE.match(cls):
            return False, f"row {i}: bad class {cls!r}", n
    return True, "ok", n


def _run_smoke() -> int:
    """Run a sanity-check batch and report PASS/FAIL. Returns exit code."""
    cpu = os.cpu_count() or 1
    tmp_name = f"_smoke_{int(time.time() * 1000)}.csv"
    tmp_path = _PROJECT_ROOT / "data" / tmp_name
    cfg = GenConfig(
        games=100,
        iterations=500,
        workers=min(4, cpu),
        eps=0.25,
        opening_max=8,
        depth_limit=80,
        base_seed=0,
        out_path=tmp_path,
        smoke=True,
        progress_every_games=25,
    )
    started = time.monotonic()
    try:
        total_rows, _ = _run_pool(cfg, append=False)
    except SystemExit as exc:
        print(f"[smoke] FAIL — {exc}", flush=True)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface anything as smoke failure
        print(f"[smoke] FAIL — {type(exc).__name__}: {exc}", flush=True)
        return 1
    elapsed = time.monotonic() - started

    ok, msg, n = _validate_smoke(tmp_path)
    if not ok:
        print(f"[smoke] FAIL — {msg} (file kept at {tmp_path})", flush=True)
        return 1
    try:
        tmp_path.unlink()
    except OSError:
        pass
    print(
        f"[smoke] PASS — {cfg.games} games, {n} rows, {elapsed:.1f}s",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate a PopOut self-play dataset via MCTS.",
    )
    p.add_argument("--games", type=int, default=None,
                   help="total number of games to play (required unless --smoke)")
    p.add_argument("--iterations", type=int, default=5000,
                   help="MCTS iterations per labelled move")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel worker processes (default: os.cpu_count())")
    p.add_argument("--eps", type=float, default=0.25,
                   help="probability of a random exploration move")
    p.add_argument("--opening-max", type=int, default=12,
                   help="maximum random opening plies, inclusive")
    p.add_argument("--depth-limit", type=int, default=80,
                   help="MCTS rollout depth cap")
    p.add_argument("--base-seed", type=int, default=0,
                   help="master seed for the run")
    p.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "data" / "popout_dataset.csv",
        help="output CSV path",
    )
    p.add_argument("--append", action="store_true",
                   help="append to existing CSV (after header verification)")
    p.add_argument("--smoke", action="store_true",
                   help="run a 100-game / 500-iter sanity batch and exit")
    p.add_argument("--progress-every", type=int, default=50,
                   help="print progress every N completed games")
    return p


def _cfg_from_args(args: argparse.Namespace) -> GenConfig:
    if args.games is None:
        raise SystemExit("error: --games is required (or pass --smoke)")
    if args.games < 1:
        raise SystemExit("error: --games must be >= 1")
    if not (0.0 <= args.eps <= 1.0):
        raise SystemExit("error: --eps must be in [0, 1]")
    if args.iterations < 1:
        raise SystemExit("error: --iterations must be >= 1")
    if args.opening_max < 0:
        raise SystemExit("error: --opening-max must be >= 0")
    if args.depth_limit < 0:
        raise SystemExit("error: --depth-limit must be >= 0")
    if args.progress_every < 1:
        raise SystemExit("error: --progress-every must be >= 1")

    cpu = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else cpu
    if workers < 1:
        raise SystemExit("error: --workers must be >= 1")

    return GenConfig(
        games=args.games,
        iterations=args.iterations,
        workers=workers,
        eps=args.eps,
        opening_max=args.opening_max,
        depth_limit=args.depth_limit,
        base_seed=args.base_seed,
        out_path=Path(args.out),
        smoke=False,
        progress_every_games=args.progress_every,
    )


def main(argv: Optional[list[str]] = None) -> int:
    # ``force=True`` so this is safe whether or not another process in the
    # same interpreter has already initialised multiprocessing.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = _build_parser().parse_args(argv)

    if args.smoke:
        return _run_smoke()

    cfg = _cfg_from_args(args)
    total_rows, elapsed = _run_pool(cfg, append=args.append)
    rate = total_rows / elapsed if elapsed > 0 else 0.0
    print(
        f"[gen] done — {cfg.games} games, {total_rows} rows in "
        f"{elapsed:.1f}s ({rate:.1f} rows/s) -> {cfg.out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
