"""PopOut board module.

Defines :class:`PopOutBoard`, which models the full state of a PopOut game
(Connect-4 variant with a "pop" move). See ``CLAUDE.md`` for the game rules.

The board is represented as a 6x7 NumPy array where:

* row 0 is the *top* of the board,
* row 5 is the *bottom* of the board (where pops occur),
* 0 = empty, 1 = player 1, 2 = player 2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ROWS: int = 6
COLS: int = 7
NUM_IN_A_ROW: int = 4

EMPTY: int = 0
PLAYER_1: int = 1
PLAYER_2: int = 2

MoveType = str            # "drop" or "pop"
Move = tuple[str, int]    # (move_type, column)


class PopOutBoard:
    """Full state of a PopOut game.

    Attributes
    ----------
    board:
        ``(ROWS, COLS)`` NumPy matrix of ``int8`` values. ``0`` means empty,
        ``1`` and ``2`` identify the two players. Row 0 is the topmost row;
        row ``ROWS - 1`` is the bottom (where pops take place).
    current_player:
        ``1`` or ``2`` — the player whose turn is to move.
    history:
        List of state hashes reached so far, including the initial state.
        Used to detect the three-fold repetition draw rule.
    move_count:
        Total number of moves (drops or pops) played so far.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player: int = PLAYER_1
        self.move_count: int = 0
        # The initial position counts as the first occurrence for the
        # 3-fold repetition rule.
        self.history: list[tuple[bytes, int]] = [self._state_hash()]
        # Metadata about the latest move; used by :meth:`check_winner` to
        # resolve the "both players win after a pop" rule.
        self._last_move_type: Optional[str] = None
        self._last_move_player: Optional[int] = None

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def get_valid_moves(self) -> list[Move]:
        """Return every legal move for the current player.

        Each move is a tuple ``(kind, column)`` with ``kind`` either
        ``"drop"`` or ``"pop"``.
        """
        moves: list[Move] = []
        for col in range(COLS):
            # DROP is legal when the topmost cell of the column is empty.
            if self.board[0, col] == EMPTY:
                moves.append(("drop", col))
            # POP is legal when the bottom cell belongs to the player on turn.
            if self.board[ROWS - 1, col] == self.current_player:
                moves.append(("pop", col))
        return moves

    def make_move(self, move_type: str, col: int) -> None:
        """Execute a move, update the history, and switch player.

        Parameters
        ----------
        move_type:
            ``"drop"`` or ``"pop"``.
        col:
            Zero-based column index (``0 <= col < COLS``).

        Raises
        ------
        ValueError
            If ``move_type`` is unknown, the column is out of range, or the
            move is illegal for the current state.
        """
        if move_type == "drop":
            self._apply_drop(col)
        elif move_type == "pop":
            self._apply_pop(col)
        else:
            raise ValueError(f"Unknown move type: {move_type!r}")

        # Remember the mover *before* switching player.
        self._last_move_type = move_type
        self._last_move_player = self.current_player

        self.move_count += 1
        self.current_player = PLAYER_2 if self.current_player == PLAYER_1 else PLAYER_1
        self.history.append(self._state_hash())

    def check_winner(self) -> Optional[int]:
        """Return the winning player (1 or 2) or ``None`` if nobody has won.

        When both players end up with a four-in-a-row simultaneously — which
        can only happen after a pop — the player who made the last move wins
        (see rule 1 in ``CLAUDE.md``).
        """
        p1_wins = self._has_four_in_a_row(PLAYER_1)
        p2_wins = self._has_four_in_a_row(PLAYER_2)

        if p1_wins and p2_wins:
            # Rule 1: the mover who caused the double four-in-a-row wins.
            # ``_last_move_player`` is always set for pops produced by the
            # engine; for pathological manually-built states we fall back to
            # ``None`` to signal the anomaly.
            return self._last_move_player
        if p1_wins:
            return PLAYER_1
        if p2_wins:
            return PLAYER_2
        return None

    def is_terminal(self) -> bool:
        """Return ``True`` if the game is over (someone won or it is a draw)."""
        if self.check_winner() is not None:
            return True
        return self.is_draw()

    def is_draw(self) -> bool:
        """Return ``True`` if the position is drawn.

        Two conditions trigger a draw:

        * no legal move is available (board full and no pops possible);
        * the current position has occurred at least three times (rule 3).
        """
        # A won position is never a draw.
        if self._has_four_in_a_row(PLAYER_1) or self._has_four_in_a_row(PLAYER_2):
            return False
        # 3-fold repetition — the current state hash already sits in
        # ``history`` (the engine appends it after each move).
        current_hash = self._state_hash()
        if self.history.count(current_hash) >= 3:
            return True
        # No legal moves for the player on turn.
        if not self.get_valid_moves():
            return True
        return False

    def clone(self) -> "PopOutBoard":
        """Return an independent deep copy suitable for MCTS simulations.

        The underlying NumPy array is copied; the history list is copied
        shallowly (its tuples are immutable, so that is sufficient).
        """
        twin = PopOutBoard.__new__(PopOutBoard)  # bypass ``__init__`` for speed
        twin.board = self.board.copy()
        twin.current_player = self.current_player
        twin.move_count = self.move_count
        twin.history = self.history.copy()
        twin._last_move_type = self._last_move_type
        twin._last_move_player = self._last_move_player
        return twin

    def get_state_hash(self) -> tuple[bytes, int]:
        """Return a hashable, comparable representation of the full state.

        The hash combines the raw bytes of the board with the player to move
        so that the same board position with different turn owners is treated
        as a different state.
        """
        return self._state_hash()

    @property
    def last_move_player(self) -> Optional[int]:
        """Player who made the most recent move, or ``None`` before any move.

        Public read-only view of the internal bookkeeping used by
        :meth:`check_winner` to resolve the "popper wins" tiebreak when a
        pop produces simultaneous four-in-a-rows. Exposed so external
        consumers (e.g. the MCTS rollout fast path) do not need to reach
        into the private ``_last_move_player`` attribute.
        """
        return self._last_move_player

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_hash(self) -> tuple[bytes, int]:
        return (self.board.tobytes(), self.current_player)

    def _apply_drop(self, col: int) -> None:
        """Place a piece of the current player into column ``col``."""
        if not (0 <= col < COLS):
            raise ValueError(f"Column out of range: {col}")
        if self.board[0, col] != EMPTY:
            raise ValueError(f"Cannot drop in column {col}: column is full.")
        # Find the lowest empty cell and fill it.
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                self.board[row, col] = self.current_player
                return

    def _apply_pop(self, col: int) -> None:
        """Pop the bottom piece of ``col`` for the current player."""
        if not (0 <= col < COLS):
            raise ValueError(f"Column out of range: {col}")
        if self.board[ROWS - 1, col] != self.current_player:
            raise ValueError(
                f"Cannot pop column {col}: bottom piece does not belong to "
                f"player {self.current_player}."
            )
        # Shift every cell in the column down by one, writing from bottom to
        # top so we always read the unmodified value from the row above.
        for row in range(ROWS - 1, 0, -1):
            self.board[row, col] = self.board[row - 1, col]
        self.board[0, col] = EMPTY

    def _has_four_in_a_row(self, player: int) -> bool:
        """Return ``True`` if ``player`` has four connected pieces anywhere."""
        b = self.board
        # Horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                if (
                    b[row, col] == player
                    and b[row, col + 1] == player
                    and b[row, col + 2] == player
                    and b[row, col + 3] == player
                ):
                    return True
        # Vertical
        for col in range(COLS):
            for row in range(ROWS - 3):
                if (
                    b[row, col] == player
                    and b[row + 1, col] == player
                    and b[row + 2, col] == player
                    and b[row + 3, col] == player
                ):
                    return True
        # Diagonal "\" (top-left to bottom-right)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if (
                    b[row, col] == player
                    and b[row + 1, col + 1] == player
                    and b[row + 2, col + 2] == player
                    and b[row + 3, col + 3] == player
                ):
                    return True
        # Diagonal "/" (bottom-left to top-right)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                if (
                    b[row, col] == player
                    and b[row - 1, col + 1] == player
                    and b[row - 2, col + 2] == player
                    and b[row - 3, col + 3] == player
                ):
                    return True
        return False

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PopOutBoard(player={self.current_player}, "
            f"moves={self.move_count})"
        )
