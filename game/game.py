"""PopOut game loop.

Defines :class:`PopOutGame`, which drives a full game between two
:class:`~game.player.Player` instances from start to finish.

The loop has no knowledge of specific AI algorithms — it only talks to the
abstract :class:`Player` interface. That keeps the same loop reusable for
Human-vs-Human, Human-vs-AI, AI-vs-AI and automated self-play dataset
generation (Phase 4).
"""

from __future__ import annotations

from typing import Optional

from game.board import PopOutBoard
from game.display import display
from game.player import Player


class PopOutGame:
    """Run a single PopOut match between two players."""

    #: Safety cap on total moves. Random/self-play games with pops can
    #: theoretically loop forever when no side declares draws; this bounds
    #: the runtime. Tests and dataset generation can override it.
    DEFAULT_MAX_MOVES: int = 1000

    def __init__(
        self,
        player1: Player,
        player2: Player,
        verbose: bool = True,
        max_moves: int = DEFAULT_MAX_MOVES,
    ) -> None:
        if player1.player_id != 1:
            raise ValueError(
                f"player1 must have player_id 1, got {player1.player_id}"
            )
        if player2.player_id != 2:
            raise ValueError(
                f"player2 must have player_id 2, got {player2.player_id}"
            )
        if max_moves < 1:
            raise ValueError("max_moves must be positive")
        self.players: dict[int, Player] = {1: player1, 2: player2}
        self.verbose: bool = verbose
        self.max_moves: int = max_moves

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def play(self) -> Optional[int]:
        """Play a complete game.

        Returns the winning player id (1 or 2), or ``None`` if the game ends
        in a draw (forced, declared, or via the max-move safety cap).
        """
        board = PopOutBoard()

        while board.move_count < self.max_moves:
            if self.verbose:
                display(board)
                print()

            # --- 1. Check for a winner -----------------------------------
            winner = board.check_winner()
            if winner is not None:
                if self.verbose:
                    print(
                        f"Player {winner} ({self.players[winner].name}) "
                        f"wins in {board.move_count} moves!"
                    )
                return winner

            # --- 2. Forced draw (no legal move whatsoever) ---------------
            valid = board.get_valid_moves()
            if not valid:
                if self.verbose:
                    print("Draw: no valid moves available.")
                return None

            current = self.players[board.current_player]

            # --- 3. Rule 2 — board full, pops still available ----------
            has_drops = any(kind == "drop" for kind, _ in valid)
            if not has_drops:
                decision = current.choose_pop_or_draw(board)
                if decision == "draw":
                    if self.verbose:
                        print(
                            f"{current.name} declares a draw "
                            "(board full)."
                        )
                    return None

            # --- 4. Rule 3 — 3-fold repetition --------------------------
            if board.history.count(board.get_state_hash()) >= 3:
                decision = current.choose_continue_or_draw(board)
                if decision == "draw":
                    if self.verbose:
                        print(
                            f"{current.name} declares a draw by "
                            "repetition."
                        )
                    return None

            # --- 5. Ask the player for a move and execute it ------------
            if self.verbose:
                print(
                    f"Player {board.current_player}'s turn "
                    f"({current.name})."
                )
            move = current.choose_move(board)
            if self.verbose:
                kind, col = move
                print(f"{current.name} plays: {kind} in column {col + 1}")
                print()
            board.make_move(*move)

        # Exhausted the safety cap without a natural termination.
        if self.verbose:
            display(board)
            print()
            print(f"Draw: reached maximum of {self.max_moves} moves.")
        return None
