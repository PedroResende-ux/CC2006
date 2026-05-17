"""Tests for :class:`ai.dt_player.DTPlayer` and its tournament integration.

The three tests here cover the prompt's correctness contract for the
decision-tree player:

1. ``test_dtplayer_loads_and_predicts`` — end-to-end smoke: the tree
   pickle loads and ``choose_move`` returns a legal move from an empty
   board.

2. ``test_dtplayer_never_plays_illegal`` — the killer test for the
   regression that wrecked the comparison project's DT: across many
   randomly-played board states, ``choose_move`` must never return a
   move outside ``board.get_valid_moves()``.

3. ``test_dtplayer_beats_random`` — a small head-to-head sanity check
   that the DT actually plays *something*: 30 games vs RandomPlayer
   alternating starts, with a 60% win threshold.

These tests are integration smoke tests, not real evaluations. The
properly powered DT-vs-MCTS evaluation runs on a separate machine via
the real tournament configs.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import pandas as pd
import pytest

from ai.dt_pipeline import POPOUT_BIN_DEFINITIONS, _move_to_int
from ai.dt_player import DTPlayer
from ai.id3 import id3
from game.board import PopOutBoard
from game.game import PopOutGame
from game.player import RandomPlayer


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "data" / "popout_dataset_150k.csv"


# ---------------------------------------------------------------------------
# Module-scoped fixture: train a small DT once and share across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dt_tree_pickle(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a depth-2 ID3 tree on 5000 rows and return the pickle path.

    Module-scoped to amortise the (~few-second) training cost across the
    three tests in this file. We keep the tree intentionally weak — the
    point is to exercise the inference pipeline, not to benchmark
    accuracy. The depth-2 root + first split is more than enough to make
    ``test_dtplayer_beats_random`` clear the 60% bar because the dataset
    is dominated by the center-column heuristic.
    """
    if not DATASET_CSV.exists():
        pytest.skip(f"dataset missing: {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV, nrows=5000)

    # Convert the textual ``class`` column ("drop_3" / "pop_2" / ...) to
    # the integer encoding the production pipeline trains on.
    df['class'] = df['class'].apply(_move_to_int)
    df = df.dropna(subset=['class'])
    df['class'] = df['class'].astype(int)

    X = df.drop(columns=['class'])
    y = df['class']

    # Apply the same binning ``ai.dt_pipeline.bin_features`` does — kept
    # inline here to avoid the stdout noise of importing the pipeline's
    # CLI-flavoured helpers into the test process.
    for col, defn in POPOUT_BIN_DEFINITIONS.items():
        X[col] = pd.cut(X[col], bins=defn['bins'], labels=defn['labels']).astype(str)

    tree = id3(
        X, y,
        features=X.columns.tolist(),
        max_depth=2,
        min_samples=20,
    )

    out_path = tmp_path_factory.mktemp("dt_player") / "tree.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(tree, f)
    return str(out_path)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _random_play_until(board: PopOutBoard, plies: int, rng: random.Random) -> None:
    """Advance ``board`` by up to ``plies`` random plies, stopping at terminal."""
    for _ in range(plies):
        if board.is_terminal():
            return
        moves = board.get_valid_moves()
        if not moves:
            return
        kind, col = rng.choice(moves)
        board.make_move(kind, col)


# ---------------------------------------------------------------------------
# Test 1 — loading and a single prediction
# ---------------------------------------------------------------------------


def test_dtplayer_loads_and_predicts(dt_tree_pickle: str) -> None:
    """DTPlayer constructs from a pickle and returns a legal first move."""
    player = DTPlayer(player_id=1, tree_path=dt_tree_pickle, random_seed=0)
    board = PopOutBoard()

    move = player.choose_move(board)

    assert isinstance(move, tuple) and len(move) == 2
    kind, col = move
    assert kind in ("drop", "pop")
    assert 0 <= col < 7
    assert move in board.get_valid_moves()


# ---------------------------------------------------------------------------
# Test 2 — never plays an illegal move
# ---------------------------------------------------------------------------


def test_dtplayer_never_plays_illegal(dt_tree_pickle: str) -> None:
    """Across 50 random positions, choose_move always returns a legal move.

    This is the regression test for the bug that killed the comparison
    project's DT (``outro``: 0% win rate against MCTS because the tree
    emitted illegal moves and the game loop crashed). Each iteration
    advances the board by a different number of random plies (5-20) so
    we hit a mix of opening, midgame, and rare crowded positions where
    the tree's predicted ``drop_3`` is more likely to be illegal.
    """
    player = DTPlayer(player_id=1, tree_path=dt_tree_pickle, random_seed=0)
    rng = random.Random(42)

    n_checked = 0
    attempts = 0
    while n_checked < 50 and attempts < 200:
        attempts += 1
        board = PopOutBoard()
        n_plies = rng.randint(5, 20)
        _random_play_until(board, n_plies, rng)
        if board.is_terminal():
            continue
        # ``DTPlayer`` only cares about the position itself, not whose
        # turn it is, but :class:`Player`'s contract is bound to a
        # ``player_id`` — that doesn't affect legality, only book-keeping.
        legal = board.get_valid_moves()
        if not legal:
            continue

        move = player.choose_move(board)
        assert move in legal, (
            f"DT returned illegal move {move!r} on board with valid moves "
            f"{legal!r}; current_player={board.current_player}, "
            f"move_count={board.move_count}"
        )
        n_checked += 1

    assert n_checked == 50, f"only managed to check {n_checked}/50 positions"


# ---------------------------------------------------------------------------
# Test 3 — beats RandomPlayer by a wide margin
# ---------------------------------------------------------------------------


def test_dtplayer_beats_random(dt_tree_pickle: str) -> None:
    """30 games, alternating starts: DT must win more than 60% (>= 18/30).

    Threshold rationale: even a depth-2 tree, trained on 5000 rows
    drawn from the head of the dataset, learns the dominant
    "drop in the centre column" heuristic — that's the modal label for
    every early/mid position. Against a uniform-random opponent the
    centre column reliably builds a vertical 4-in-a-row before the
    random player can stop it, so a 60% win rate is conservative
    (empirically the depth-2 tree clears well over 80% in practice).
    Setting the bar at 60% (>= 18 wins) leaves slack for stochastic
    bad luck in the half of games where DT plays as P2.

    This is a sanity check on the end-to-end integration, not an
    evaluation — the real DT-vs-baseline tournament happens on a
    separate machine via the prompt E3 configs.
    """
    n_games = 30
    dt_wins = 0
    for game_idx in range(n_games):
        dt_is_p1 = (game_idx % 2 == 0)
        # Seed each player from game_idx so behaviour is fully
        # deterministic but varied across games.
        dt = DTPlayer(
            player_id=(1 if dt_is_p1 else 2),
            tree_path=dt_tree_pickle,
            random_seed=game_idx,
        )
        rnd = RandomPlayer(
            player_id=(2 if dt_is_p1 else 1),
            seed=1000 + game_idx,
        )
        if dt_is_p1:
            p1, p2 = dt, rnd
        else:
            p1, p2 = rnd, dt

        winner = PopOutGame(p1, p2, verbose=False, max_moves=200).play()
        if winner is not None:
            winning_player = p1 if winner == 1 else p2
            if winning_player is dt:
                dt_wins += 1

    assert dt_wins > 18, (
        f"DTPlayer only won {dt_wins}/{n_games} vs RandomPlayer "
        f"(expected > 60% as a sanity check on the integration)"
    )
