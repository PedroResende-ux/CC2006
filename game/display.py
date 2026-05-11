"""Text rendering of a PopOut board for terminal use."""

from __future__ import annotations

from typing import Union

import numpy as np

from game.board import COLS, EMPTY, PLAYER_1, PLAYER_2, ROWS, PopOutBoard


# Map internal player codes to the single-character symbols shown on screen.
_SYMBOLS: dict[int, str] = {EMPTY: "-", PLAYER_1: "X", PLAYER_2: "O"}


def render(board: Union[PopOutBoard, np.ndarray]) -> str:
    """Return a human-readable ASCII rendering of ``board``.

    The first line lists column numbers (1-indexed, more natural for end-users
    to type). The remaining lines are the six board rows, top first.
    """
    grid = board.board if isinstance(board, PopOutBoard) else board
    lines: list[str] = [" ".join(str(col + 1) for col in range(COLS))]
    for row in range(ROWS):
        cells = (_SYMBOLS[int(grid[row, col])] for col in range(COLS))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def display(board: Union[PopOutBoard, np.ndarray]) -> None:
    """Print ``board`` to standard output using :func:`render`."""
    print(render(board))
