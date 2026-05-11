"""Numba-compiled rollout for the PopOut MCTS.

The functions in this module are pure number crunching on int8 arrays.
They do NOT import from ``game.board`` (Numba cannot see PopOutBoard).
The wrapper :meth:`ai.mcts.MCTS._simulate` is responsible for
translating the OO state into the primitive arguments.

All functions are decorated with ``@njit(cache=True)`` so subsequent
processes skip the JIT compile cost (the cache is written to
``__pycache__/`` next to this file).
"""

from __future__ import annotations

import numpy as np
from numba import njit


# Move-kind encoding inside numba: 0 = drop, 1 = pop.
DROP: int = 0
POP: int = 1

# Reward conventions (root-perspective), mirrored from ai.mcts.
REWARD_ROOT_WIN: float = 1.0
REWARD_ROOT_LOSS: float = 0.0
REWARD_DRAW: float = 0.5


@njit(cache=True)
def _has_four(board: np.ndarray, player: int) -> bool:
    """True iff ``player`` has any 4-in-a-row on ``board`` (6x7 int8).

    Uses explicit nested loops with early return because Numba compiles
    tight integer loops to ~C speed; numpy slicing on tiny arrays would
    allocate temporaries and lose to this form.
    """
    rows = 6
    cols = 7
    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if (
                board[r, c] == player
                and board[r, c + 1] == player
                and board[r, c + 2] == player
                and board[r, c + 3] == player
            ):
                return True
    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            if (
                board[r, c] == player
                and board[r + 1, c] == player
                and board[r + 2, c] == player
                and board[r + 3, c] == player
            ):
                return True
    # Diagonal "\"
    for r in range(rows - 3):
        for c in range(cols - 3):
            if (
                board[r, c] == player
                and board[r + 1, c + 1] == player
                and board[r + 2, c + 2] == player
                and board[r + 3, c + 3] == player
            ):
                return True
    # Diagonal "/"
    for r in range(3, rows):
        for c in range(cols - 3):
            if (
                board[r, c] == player
                and board[r - 1, c + 1] == player
                and board[r - 2, c + 2] == player
                and board[r - 3, c + 3] == player
            ):
                return True
    return False


@njit(cache=True)
def _check_winner(board: np.ndarray, last_move_player: int) -> int:
    """Return 1 if P1 has 4-in-a-row, 2 if P2 does, 0 otherwise.

    When both players have 4-in-a-row (only possible after a pop) the
    winner is ``last_move_player`` — PopOut rule 1 in ``CLAUDE.md``.
    """
    p1 = _has_four(board, 1)
    p2 = _has_four(board, 2)
    if p1 and p2:
        return last_move_player
    if p1:
        return 1
    if p2:
        return 2
    return 0


@njit(cache=True)
def _collect_valid_moves(
    board: np.ndarray,
    current_player: int,
    kinds_buf: np.ndarray,
    cols_buf: np.ndarray,
) -> int:
    """Fill ``kinds_buf`` / ``cols_buf`` with legal moves; return count.

    Same ordering as :meth:`PopOutBoard.get_valid_moves` (per-column,
    drop before pop) so distributions match the pure-Python rollout up
    to RNG differences.
    """
    n = 0
    for c in range(7):
        if board[0, c] == 0:                    # column not full -> drop legal
            kinds_buf[n] = DROP
            cols_buf[n] = c
            n += 1
        if board[5, c] == current_player:       # bottom is mine -> pop legal
            kinds_buf[n] = POP
            cols_buf[n] = c
            n += 1
    return n


@njit(cache=True)
def _apply_drop(board: np.ndarray, col: int, player: int) -> None:
    """Place ``player`` at the lowest empty cell of ``col``. In-place."""
    for r in range(5, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return


@njit(cache=True)
def _apply_pop(board: np.ndarray, col: int) -> None:
    """Remove the bottom piece of ``col``; shift the column down by one."""
    for r in range(5, 0, -1):
        board[r, col] = board[r - 1, col]
    board[0, col] = 0


@njit(cache=True)
def simulate_rollout(
    board: np.ndarray,
    current_player: int,
    last_move_player: int,
    root_player: int,
    depth_limit: int,
    seed: int,
) -> float:
    """Play random moves from ``board`` and return a root-perspective reward.

    ``board`` is mutated in place — pass a fresh copy from the caller.
    The function seeds numpy's per-thread RNG with ``seed`` once at the
    top so identical seeds reproduce identical trajectories.
    """
    np.random.seed(seed)
    kinds_buf = np.empty(14, dtype=np.int8)     # 7 cols x (drop + pop)
    cols_buf = np.empty(14, dtype=np.int8)

    plies = 0
    while plies < depth_limit:
        winner = _check_winner(board, last_move_player)
        if winner != 0:
            if winner == root_player:
                return REWARD_ROOT_WIN
            return REWARD_ROOT_LOSS

        n_moves = _collect_valid_moves(board, current_player, kinds_buf, cols_buf)
        if n_moves == 0:
            return REWARD_DRAW

        idx = np.random.randint(0, n_moves)
        kind = kinds_buf[idx]
        col = int(cols_buf[idx])

        if kind == DROP:
            _apply_drop(board, col, current_player)
        else:
            _apply_pop(board, col)
        last_move_player = current_player
        current_player = 3 - current_player     # 1 <-> 2
        plies += 1

    return REWARD_DRAW
