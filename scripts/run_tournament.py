"""Pairwise MCTS tournament runner.

Reads a JSON config that describes a list of matchups (pairs of MCTS
configurations). Plays each matchup in many independent games, alternating
who starts so each side is P1 half the time, and writes a CSV with the
aggregated results.

Usage:
    python -m scripts.run_tournament scripts/tournament_configs/iterations_sweep.json \\
        --workers 16 \\
        --out data/tournament_iterations.csv

The per-matchup ``wallclock_s`` column in the CSV is set to ``0.0`` so
that the output is byte-deterministic for the same ``(config, base-seed,
workers)`` tuple — real per-game timings vary across runs at the
millisecond level. The actual wall time per matchup (and for the whole
tournament) is still surfaced in the stdout summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Make the project root importable when this script is run as
# ``python -m scripts.run_tournament`` from anywhere.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.mcts import MCTS  # noqa: E402
from game.game import PopOutGame  # noqa: E402
from game.player import Player  # noqa: E402


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_PLAYER_FIELDS: tuple[str, ...] = (
    "iterations",
    "exploration_weight",
    "rollout_depth_limit",
    "num_children_to_expand",
    "uct_variant",
)

CSV_HEADER: list[str] = [
    "matchup_name",
    "total_games",
    "a_wins",
    "b_wins",
    "draws",
    "a_wins_as_p1",
    "a_wins_as_p2",
    "b_wins_as_p1",
    "b_wins_as_p2",
    "draws_p1_was_a",
    "draws_p1_was_b",
    "a_winrate",
    "b_winrate",
    "draw_rate",
    "wallclock_s",
    "config_a",
    "config_b",
]


# ---------------------------------------------------------------------------
# Worker-side player wrapper
# ---------------------------------------------------------------------------


class _MCTSConfiguredPlayer(Player):
    """:class:`Player` adapter that drives PopOutGame via a configured MCTS.

    Defined locally inside the tournament runner because
    :class:`ai.mcts_player.MCTSPlayer` does not yet forward the variant
    parameters (``num_children_to_expand``, ``uct_variant``) to its
    underlying engine; the tournament needs access to the full surface.
    """

    def __init__(self, player_id: int, config: dict[str, Any], seed: int) -> None:
        super().__init__(player_id, "MCTS")
        self._engine: MCTS = MCTS(
            iterations=int(config["iterations"]),
            exploration_weight=float(config["exploration_weight"]),
            rollout_depth_limit=int(config["rollout_depth_limit"]),
            num_children_to_expand=int(config["num_children_to_expand"]),
            uct_variant=str(config["uct_variant"]),
            random_seed=seed,
        )

    def choose_move(self, board):
        return self._engine.search(board)


def play_one_game(work_item: tuple) -> dict:
    """Worker: play one game and return a result record.

    Work item layout (tuple-based for clean pickling under the ``spawn``
    start method on Windows):

        (matchup_idx, game_idx, matchup_name,
         player_a_config, player_b_config, a_starts_p1, seed)

    The same ``seed`` is passed to both players in the pair, which makes
    each (matchup_idx, game_idx) deterministic given ``base_seed``. The
    two engines still advance independent RNG states because each one
    consumes random bits at its own pace.
    """
    (
        matchup_idx,
        game_idx,
        matchup_name,
        player_a_config,
        player_b_config,
        a_starts_p1,
        seed,
    ) = work_item

    started = time.monotonic()
    if a_starts_p1:
        p1 = _MCTSConfiguredPlayer(1, player_a_config, seed)
        p2 = _MCTSConfiguredPlayer(2, player_b_config, seed)
        a_was = "p1"
    else:
        p1 = _MCTSConfiguredPlayer(1, player_b_config, seed)
        p2 = _MCTSConfiguredPlayer(2, player_a_config, seed)
        a_was = "p2"

    winner = PopOutGame(p1, p2, verbose=False, max_moves=400).play()
    elapsed = time.monotonic() - started

    if winner is None:
        result = "draw"
    elif (a_was == "p1" and winner == 1) or (a_was == "p2" and winner == 2):
        result = "a_win"
    else:
        result = "b_win"

    return {
        "matchup_idx": matchup_idx,
        "game_idx": game_idx,
        "matchup_name": matchup_name,
        "a_was": a_was,
        "result": result,
        "wallclock_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Parent: config loading & validation
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"error: config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("error: config root must be a JSON object")
    if "matchups" not in data or not isinstance(data["matchups"], list):
        raise SystemExit("error: config must have a 'matchups' list")
    return data


def _validate_player_config(cfg: Any, where: str) -> dict:
    if not isinstance(cfg, dict):
        raise SystemExit(f"error: {where}: must be a JSON object")
    missing = [f for f in REQUIRED_PLAYER_FIELDS if f not in cfg]
    if missing:
        raise SystemExit(f"error: {where}: missing required field(s) {missing}")
    if cfg["uct_variant"] not in ("ucb1", "ucb1_tuned"):
        raise SystemExit(
            f"error: {where}: uct_variant must be 'ucb1' or 'ucb1_tuned', "
            f"got {cfg['uct_variant']!r}"
        )
    return cfg


def _seed_for(base_seed: int, matchup_idx: int, game_idx: int) -> int:
    """Per-game seed: deterministic in (base_seed, matchup_idx, game_idx)."""
    return (base_seed * 1_000_003 + matchup_idx * 1009 + game_idx) & 0x7FFF_FFFF


def _build_work_items(
    matchups: list[dict],
    default_games: Optional[int],
    base_seed: int,
) -> list[tuple]:
    items: list[tuple] = []
    for m_idx, m in enumerate(matchups):
        if not isinstance(m, dict):
            raise SystemExit(f"error: matchup {m_idx}: must be a JSON object")
        name = m.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(
                f"error: matchup {m_idx}: 'name' must be a non-empty string"
            )
        player_a = _validate_player_config(
            m.get("player_a"), f"matchup '{name}' player_a"
        )
        player_b = _validate_player_config(
            m.get("player_b"), f"matchup '{name}' player_b"
        )
        games = m.get("games", default_games)
        if games is None:
            raise SystemExit(
                f"error: matchup '{name}': 'games' missing and no top-level "
                f"'default_games' was provided"
            )
        if not isinstance(games, int) or games < 2:
            raise SystemExit(
                f"error: matchup '{name}': 'games' must be an int >= 2 (got {games!r})"
            )
        if games % 2 != 0:
            raise SystemExit(
                f"error: matchup '{name}': 'games' must be even (got {games}); "
                "matchups alternate P1/P2 50/50 so the count has to be divisible by 2"
            )
        for g in range(games):
            items.append(
                (
                    m_idx,
                    g,
                    name,
                    player_a,
                    player_b,
                    g % 2 == 0,                         # A is P1 on even games
                    _seed_for(base_seed, m_idx, g),
                )
            )
    return items


# ---------------------------------------------------------------------------
# Pretty helpers
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


# ---------------------------------------------------------------------------
# Tournament driver
# ---------------------------------------------------------------------------


def _run_tournament(
    work_items: list[tuple],
    workers: int,
    progress_every: int,
) -> tuple[list[dict], dict[str, float], float]:
    """Drive the worker pool. Returns (results, per_matchup_wallclock, total)."""
    started = time.monotonic()
    total = len(work_items)
    last_matchup_name = ""
    running_tally: dict[str, dict[str, int]] = {}
    results: list[dict] = []
    matchup_wallclock: dict[str, float] = {}

    with mp.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(play_one_game, work_items, chunksize=1):
            results.append(r)
            name = r["matchup_name"]
            tally = running_tally.setdefault(name, {"a": 0, "b": 0, "d": 0})
            res = r["result"]
            if res == "a_win":
                tally["a"] += 1
            elif res == "b_win":
                tally["b"] += 1
            else:
                tally["d"] += 1
            matchup_wallclock[name] = (
                matchup_wallclock.get(name, 0.0) + r["wallclock_s"]
            )
            last_matchup_name = name

            done = len(results)
            if done % progress_every == 0 or done == total:
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * done / total
                remaining = total - done
                eta = _fmt_eta(remaining / rate) if rate > 0 else "?"
                t = running_tally.get(
                    last_matchup_name, {"a": 0, "b": 0, "d": 0}
                )
                a, b, d = t["a"], t["b"], t["d"]
                if a > b:
                    side = "a"
                elif b > a:
                    side = "b"
                else:
                    side = "tie"
                print(
                    f"[tourney] {done}/{total} games done ({pct:.1f}%) | "
                    f"matchup {last_matchup_name} leading "
                    f"{a}-{b}-{d} ({side}) | "
                    f"{rate:.1f} games/s | ETA {eta}",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed = time.monotonic() - started
    # Sort by (matchup_idx, game_idx) so downstream aggregation is
    # deterministic regardless of which worker finished a given game first.
    results.sort(key=lambda r: (r["matchup_idx"], r["game_idx"]))
    return results, matchup_wallclock, elapsed


def _aggregate_per_matchup(
    matchups: list[dict],
    sorted_results: list[dict],
) -> list[dict]:
    """One row per matchup, in the matchup order from the config."""
    by_idx: dict[int, list[dict]] = {i: [] for i in range(len(matchups))}
    for r in sorted_results:
        by_idx[r["matchup_idx"]].append(r)

    rows: list[dict] = []
    for m_idx, m in enumerate(matchups):
        games = by_idx[m_idx]   # already sorted by game_idx via the outer sort
        n = len(games)

        a_wins = b_wins = draws = 0
        a_wins_as_p1 = a_wins_as_p2 = 0
        b_wins_as_p1 = b_wins_as_p2 = 0
        draws_p1_was_a = draws_p1_was_b = 0

        for r in games:
            res = r["result"]
            a_was = r["a_was"]
            if res == "a_win":
                a_wins += 1
                if a_was == "p1":
                    a_wins_as_p1 += 1
                else:
                    a_wins_as_p2 += 1
            elif res == "b_win":
                b_wins += 1
                # If A was P1 then B was P2 and vice-versa.
                if a_was == "p1":
                    b_wins_as_p2 += 1
                else:
                    b_wins_as_p1 += 1
            else:
                draws += 1
                if a_was == "p1":
                    draws_p1_was_a += 1
                else:
                    draws_p1_was_b += 1

        rows.append(
            {
                "matchup_name": m["name"],
                "total_games": n,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "draws": draws,
                "a_wins_as_p1": a_wins_as_p1,
                "a_wins_as_p2": a_wins_as_p2,
                "b_wins_as_p1": b_wins_as_p1,
                "b_wins_as_p2": b_wins_as_p2,
                "draws_p1_was_a": draws_p1_was_a,
                "draws_p1_was_b": draws_p1_was_b,
                "a_winrate": a_wins / n if n else 0.0,
                "b_winrate": b_wins / n if n else 0.0,
                "draw_rate": draws / n if n else 0.0,
                # Zero on purpose: real per-game timings vary at the ms
                # level, which would defeat byte-deterministic output.
                # Real wallclock is shown only in the stdout summary table.
                "wallclock_s": 0.0,
                "config_a": json.dumps(m["player_a"], sort_keys=True),
                "config_b": json.dumps(m["player_b"], sort_keys=True),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _print_summary(
    rows: list[dict],
    matchup_wallclock: dict[str, float],
    out_path: Path,
    total_wallclock: float,
) -> None:
    name_w = max(8, max((len(r["matchup_name"]) for r in rows), default=8))
    print(
        f"{'matchup':<{name_w}}  "
        f"{'A wins':>6}  {'B wins':>6}  {'draws':>5}  "
        f"{'A%':>6}  {'B%':>6}  {'draw%':>6}  {'time':>6}"
    )
    for r in rows:
        wall = matchup_wallclock.get(r["matchup_name"], 0.0)
        print(
            f"{r['matchup_name']:<{name_w}}  "
            f"{r['a_wins']:>6}  "
            f"{r['b_wins']:>6}  "
            f"{r['draws']:>5}  "
            f"{r['a_winrate'] * 100:>5.1f}%  "
            f"{r['b_winrate'] * 100:>5.1f}%  "
            f"{r['draw_rate'] * 100:>5.1f}%  "
            f"{wall:>5.0f}s"
        )
    total_games = sum(r["total_games"] for r in rows)
    print(
        f"[tourney] done — {len(rows)} matchups, "
        f"{total_games} games total, "
        f"{total_wallclock:.1f}s wallclock -> {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_tournament",
        description="Pairwise MCTS tournament runner.",
    )
    p.add_argument(
        "config",
        type=Path,
        help="JSON config file describing the matchups",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel worker processes (default: os.cpu_count())",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output CSV path "
            "(default: data/tournament_<config_basename>.csv)"
        ),
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="base seed for per-game seeding (default: 0)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print progress every N completed games (default: 25)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    # ``force=True`` so the call is safe whether or not another process
    # in the same interpreter has already initialised multiprocessing.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = _build_parser().parse_args(argv)

    cpu = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else cpu
    if workers < 1:
        raise SystemExit("error: --workers must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("error: --progress-every must be >= 1")

    cfg = _load_config(args.config)
    default_games = cfg.get("default_games")
    matchups = cfg["matchups"]
    if not matchups:
        raise SystemExit("error: matchups list is empty")

    work_items = _build_work_items(matchups, default_games, args.base_seed)

    out_path = args.out
    if out_path is None:
        out_path = _PROJECT_ROOT / "data" / f"tournament_{args.config.stem}.csv"
    out_path = Path(out_path)
    if out_path.exists():
        raise SystemExit(
            f"error: output file already exists: {out_path}. "
            f"Delete it or pass a different --out path."
        )

    print(
        f"[tourney] {len(work_items)} games across {len(matchups)} matchup(s), "
        f"{workers} workers, base_seed={args.base_seed}",
        file=sys.stderr,
        flush=True,
    )

    results, matchup_wallclock, total_wallclock = _run_tournament(
        work_items, workers, args.progress_every
    )
    rows = _aggregate_per_matchup(matchups, results)
    _write_csv(out_path, rows)
    _print_summary(rows, matchup_wallclock, out_path, total_wallclock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
