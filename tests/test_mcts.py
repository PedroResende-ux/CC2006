"""Acceptance tests for :mod:`ai.mcts` and :mod:`ai.mcts_player`."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ai.mcts import MCTS, TERMINAL_WIN
from ai.mcts_player import MCTSPlayer
from game.board import PLAYER_1, PLAYER_2, PopOutBoard
from game.game import PopOutGame
from game.player import RandomPlayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _board_from_array(state: list[list[int]], current_player: int) -> PopOutBoard:
    """Construct a PopOutBoard from a 6x7 nested list and turn assignment.

    The history is reset to a single entry corresponding to the manually
    set state, so 3-fold repetition checks are not triggered by the
    artificial initial-empty hash that the :class:`PopOutBoard`
    constructor would otherwise leave behind.
    """
    board = PopOutBoard()
    board.board = np.array(state, dtype=np.int8)
    board.current_player = current_player
    board.history = [board.get_state_hash()]
    return board


# ---------------------------------------------------------------------------
# Test 1 — basic construction and legal move output
# ---------------------------------------------------------------------------


def test_search_returns_legal_move() -> None:
    mcts = MCTS(iterations=20, random_seed=0)
    board = PopOutBoard()
    move = mcts.search(board)
    assert move in board.get_valid_moves()


# ---------------------------------------------------------------------------
# Test 2 — determinism under fixed seeding
# ---------------------------------------------------------------------------


def test_same_seed_same_move() -> None:
    board = PopOutBoard()
    a = MCTS(iterations=100, random_seed=42).search(board)
    b = MCTS(iterations=100, random_seed=42).search(board)
    assert a == b


# ---------------------------------------------------------------------------
# Test 3 — finds an immediate win
# ---------------------------------------------------------------------------


def test_finds_immediate_win() -> None:
    """P1 has three pieces in a row on the bottom; MCTS must close out the win."""
    board = _board_from_array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 2, 2, 2],
        ],
        current_player=PLAYER_1,
    )
    mcts = MCTS(iterations=200, random_seed=0)
    move = mcts.search(board)

    follow_up = board.clone()
    follow_up.make_move(*move)
    assert follow_up.check_winner() == PLAYER_1, (
        f"MCTS picked {move!r}, which does not win for P1.\n{follow_up.board}"
    )


# ---------------------------------------------------------------------------
# Test 4 — blocks an immediate threat
# ---------------------------------------------------------------------------


def test_blocks_immediate_threat() -> None:
    """P2 threatens row-5 four in a row at column 4; P1 must block (or win)."""
    board = _board_from_array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 2, 2, 2, 0, 1, 1],
        ],
        current_player=PLAYER_1,
    )
    mcts = MCTS(iterations=500, random_seed=0)
    move = mcts.search(board)

    after_block = board.clone()
    after_block.make_move(*move)

    if after_block.check_winner() == PLAYER_1:
        # P1 found a forced win — also acceptable per spec.
        return

    # Otherwise P2 must not have an immediate winning reply.
    for reply in after_block.get_valid_moves():
        twin = after_block.clone()
        twin.make_move(*reply)
        assert twin.check_winner() != PLAYER_2, (
            f"MCTS played {move!r}, but P2 wins instantly with {reply!r}."
        )


# ---------------------------------------------------------------------------
# Test 5 — win-rate vs RandomPlayer (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mcts_beats_random() -> None:
    wins = 0
    for seed in range(20):
        if seed % 2 == 0:
            p1 = MCTSPlayer(1, iterations=1000, random_seed=seed)
            p2 = RandomPlayer(2, seed=seed + 1000)
            result = PopOutGame(p1, p2, verbose=False, max_moves=200).play()
            if result == 1:
                wins += 1
        else:
            p1 = RandomPlayer(1, seed=seed + 1000)
            p2 = MCTSPlayer(2, iterations=1000, random_seed=seed)
            result = PopOutGame(p1, p2, verbose=False, max_moves=200).play()
            if result == 2:
                wins += 1
    assert wins >= 16, f"MCTS won only {wins}/20"


# ---------------------------------------------------------------------------
# Test 6 — terminal propagation reaches the root on a forced win
# ---------------------------------------------------------------------------


def test_terminal_propagation_marks_root_when_root_has_forced_win() -> None:
    """Root state has a forced one-move win; terminal=WIN must reach the root."""
    board = _board_from_array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 2, 2, 2],
        ],
        current_player=PLAYER_1,
    )
    mcts = MCTS(iterations=200, random_seed=0)
    mcts.search(board)
    assert mcts._last_root is not None
    assert mcts._last_root.terminal == TERMINAL_WIN


# ---------------------------------------------------------------------------
# Test 7 — only-move shortcut
# ---------------------------------------------------------------------------


def test_only_move_shortcut_runs_instantly() -> None:
    """When the position has a single legal move, search returns immediately.

    The board below leaves only ``("drop", 3)`` legal:

    * cols 0,1,2,4,5,6 are filled to the top, so they admit no drop;
    * col 3 is entirely empty, so its drop falls to row 5;
    * row 5 contains no P1 piece, so P1 cannot pop any column.
    """
    board = _board_from_array(
        [
            [2, 1, 2, 0, 2, 1, 2],
            [1, 2, 1, 0, 1, 2, 1],
            [2, 1, 2, 0, 2, 1, 2],
            [1, 2, 1, 0, 1, 2, 1],
            [2, 1, 2, 0, 2, 1, 2],
            [2, 2, 2, 0, 2, 2, 2],
        ],
        current_player=PLAYER_1,
    )

    # Sanity check the constructed state.
    assert board.get_valid_moves() == [("drop", 3)]
    assert board.check_winner() is None

    mcts = MCTS(iterations=999_999, random_seed=0)
    start = time.perf_counter()
    move = mcts.search(board)
    elapsed = time.perf_counter() - start

    assert move == ("drop", 3)
    assert elapsed < 0.05, f"only-move shortcut took {elapsed:.4f}s"
    # The shortcut must skip all iteration bookkeeping, including the root.
    assert mcts._last_root is None


# ---------------------------------------------------------------------------
# Bonus — MCTSPlayer integrates with the game loop without crashing
# ---------------------------------------------------------------------------


def test_mcts_player_returns_legal_move_on_empty_board() -> None:
    player = MCTSPlayer(1, iterations=10, random_seed=0)
    board = PopOutBoard()
    move = player.choose_move(board)
    assert move in board.get_valid_moves()


def test_uct_score_flips_exploitation_at_opponent_node() -> None:
    """The UCT exploitation term must invert when the opponent is to move
    at the parent node. Exploration term stays the same regardless.
    """
    import math
    from ai.mcts import MCTS, MCTSNode

    mcts = MCTS(iterations=1, exploration_weight=math.sqrt(2), random_seed=0)

    # Build a minimal parent/child pair without running a search.
    root_state = PopOutBoard()
    root_state.current_player = PLAYER_1
    parent = MCTSNode(state=root_state, parent=None, move=None, root_player=PLAYER_1)
    parent.visits = 100  # so ln(N_p) is well defined

    child_state = root_state.clone()
    child_state.make_move("drop", 3)
    # child_state.current_player is now PLAYER_2 — irrelevant; what matters is
    # parent.state.current_player vs parent.root_player.
    child = MCTSNode(state=child_state, parent=parent, move=("drop", 3), root_player=PLAYER_1)
    child.visits = 10
    child.value_sum = 7.0  # mean = 0.7 in root perspective

    # Case A — parent is root-player to move; exploitation must be 0.7.
    score_root = mcts._uct_score(child, parent)
    exploration = math.sqrt(2) * math.sqrt(math.log(100) / 10)
    assert math.isclose(score_root, 0.7 + exploration, rel_tol=1e-9)

    # Case B — flip parent's current player; exploitation must be 1 - 0.7 = 0.3.
    parent.state.current_player = PLAYER_2  # opponent to move at parent
    score_opp = mcts._uct_score(child, parent)
    assert math.isclose(score_opp, 0.3 + exploration, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Test A — Numba rollout matches pure-Python distribution
# ---------------------------------------------------------------------------


def test_numba_rollout_matches_python_distribution(capsys) -> None:
    """The Numba rollout must yield a statistically similar terminal mix.

    The two implementations use different RNGs (numpy's per-thread state
    vs ``random.Random``) so the trajectories diverge; only the
    aggregated win/loss/draw distribution is compared, with a generous
    tolerance.
    """
    # A mid-game state with both pops and drops legal — exercises both branches.
    board = _board_from_array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 1, 2, 1, 2],
        ],
        current_player=PLAYER_1,
    )

    def tally(use_numba: bool) -> tuple[int, int, int]:
        m = MCTS(iterations=1, random_seed=12345, use_numba_rollout=use_numba)
        m._root_player = PLAYER_1
        w1 = w2 = d = 0
        for _ in range(2000):
            r = m._simulate(board)
            if r == 1.0:
                w1 += 1
            elif r == 0.0:
                w2 += 1
            else:
                d += 1
        return w1, w2, d

    py = tally(False)
    nb = tally(True)
    # Print the tallies so they show up under ``pytest -s``.
    print(f"\npython: p1={py[0]} p2={py[1]} draws={py[2]}")
    print(f"numba:  p1={nb[0]} p2={nb[1]} draws={nb[2]}")

    # Loose tolerance: +/-10% relative or +/-50 absolute, whichever is larger.
    for a, b, name in zip(py, nb, ("p1_wins", "p2_wins", "draws")):
        tol = max(0.10 * a, 50)
        assert abs(a - b) <= tol, (
            f"{name}: python={a}, numba={b}, tol={tol:.0f}"
        )


# ---------------------------------------------------------------------------
# Test B — Numba rollout is at least 5x faster than pure Python (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_numba_rollout_is_at_least_5x_faster() -> None:
    board = PopOutBoard()                 # empty board

    # Warm up Numba JIT (first call compiles).
    warm = MCTS(iterations=1, random_seed=0, use_numba_rollout=True)
    warm._root_player = PLAYER_1
    warm._simulate(board)

    py = MCTS(iterations=1, random_seed=0, use_numba_rollout=False)
    nb = MCTS(iterations=1, random_seed=0, use_numba_rollout=True)
    py._root_player = PLAYER_1
    nb._root_player = PLAYER_1

    N = 500
    t = time.perf_counter()
    for _ in range(N):
        py._simulate(board)
    t_py = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(N):
        nb._simulate(board)
    t_nb = time.perf_counter() - t

    speedup = t_py / t_nb
    print(
        f"\npy={t_py * 1000:.0f}ms, nb={t_nb * 1000:.0f}ms, "
        f"speedup={speedup:.1f}x"
    )
    assert speedup >= 5.0, f"speedup {speedup:.2f}x < 5x"
