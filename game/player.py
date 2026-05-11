"""Player abstractions for PopOut.

This module defines the :class:`Player` protocol used by the game loop and
provides two concrete implementations:

* :class:`HumanPlayer` — driven by terminal input.
* :class:`RandomPlayer` — picks uniformly at random from the valid moves; a
  useful baseline and a handy opponent for testing.

Future AI players (MCTS, Decision Tree) are expected to subclass
:class:`Player` and override :meth:`Player.choose_move`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from game.board import PopOutBoard


class Player(ABC):
    """Abstract base class every PopOut player must implement.

    Subclasses only have to implement :meth:`choose_move`. The two optional
    draw-related hooks (:meth:`choose_pop_or_draw`, :meth:`choose_continue_or_draw`)
    default to the "keep playing" answer so that autonomous agents do not need
    to implement them explicitly.
    """

    def __init__(self, player_id: int, name: str) -> None:
        if player_id not in (1, 2):
            raise ValueError(
                f"player_id must be 1 or 2, got {player_id!r}"
            )
        self.player_id: int = player_id
        self.name: str = name

    @abstractmethod
    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        """Given the current board, return a legal move ``(move_type, col)``.

        ``move_type`` must be ``"drop"`` or ``"pop"``. The returned tuple must
        be an element of ``board.get_valid_moves()``.
        """

    def choose_pop_or_draw(self, board: PopOutBoard) -> str:
        """Decide between popping and declaring a draw (rule 2).

        The game loop calls this when the board is full (no drops possible)
        but at least one legal pop exists. Return ``"pop"`` to keep playing
        or ``"draw"`` to end the game. Default: ``"pop"``.
        """
        return "pop"

    def choose_continue_or_draw(self, board: PopOutBoard) -> str:
        """Decide between continuing and declaring a draw (rule 3).

        The game loop calls this when the current state has occurred three
        times. Return ``"continue"`` to keep playing or ``"draw"`` to end the
        game. Default: ``"continue"``.
        """
        return "continue"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.player_id}, name={self.name!r})"


class HumanPlayer(Player):
    """Player driven by textual input on the terminal."""

    def __init__(self, player_id: int, name: str = "Human") -> None:
        super().__init__(player_id, name)

    # ------------------------------------------------------------------
    # Required move selection
    # ------------------------------------------------------------------

    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        valid = board.get_valid_moves()
        drop_cols = sorted(col + 1 for kind, col in valid if kind == "drop")
        pop_cols = sorted(col + 1 for kind, col in valid if kind == "pop")

        if drop_cols:
            print(f"  Drop columns available: {drop_cols}")
        if pop_cols:
            print(f"  Pop columns available:  {pop_cols}")

        while True:
            kind_raw = input("Move type (drop/pop): ").strip().lower()
            if kind_raw in ("drop", "d"):
                move_type = "drop"
            elif kind_raw in ("pop", "p"):
                move_type = "pop"
            else:
                print("  Invalid move type. Enter 'drop' or 'pop'.")
                continue

            col_raw = input("Column (1-7): ").strip()
            try:
                col = int(col_raw) - 1
            except ValueError:
                print("  Invalid column. Enter an integer 1-7.")
                continue

            if (move_type, col) not in valid:
                print(
                    f"  '{move_type} {col + 1}' is not a legal move here. "
                    "Try again."
                )
                continue

            return (move_type, col)

    # ------------------------------------------------------------------
    # Optional draw hooks
    # ------------------------------------------------------------------

    def choose_pop_or_draw(self, board: PopOutBoard) -> str:
        while True:
            raw = input(
                "The board is full. Do you want to (p)op a piece or "
                "(d)eclare a draw? "
            ).strip().lower()
            if raw in ("p", "pop"):
                return "pop"
            if raw in ("d", "draw"):
                return "draw"
            print("  Invalid choice. Enter 'p' or 'd'.")

    def choose_continue_or_draw(self, board: PopOutBoard) -> str:
        while True:
            raw = input(
                "This state has appeared 3 times. Declare a (d)raw or "
                "(c)ontinue? "
            ).strip().lower()
            if raw in ("d", "draw"):
                return "draw"
            if raw in ("c", "continue"):
                return "continue"
            print("  Invalid choice. Enter 'd' or 'c'.")


class RandomPlayer(Player):
    """Baseline player that picks uniformly at random from legal moves."""

    def __init__(
        self,
        player_id: int,
        name: str = "Random",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id, name)
        self._rng = random.Random(seed)

    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        return self._rng.choice(board.get_valid_moves())
