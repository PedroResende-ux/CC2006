"""MCTS-driven :class:`~game.player.Player` adapter.

Wraps :class:`ai.mcts.MCTS` so it can drop straight into the existing
:class:`game.game.PopOutGame` loop alongside human and random players.
"""

from __future__ import annotations

import math
from typing import Optional

from ai.mcts import MCTS
from game.board import PopOutBoard
from game.player import Player


class MCTSPlayer(Player):
    """A PopOut player whose moves are chosen by Monte Carlo Tree Search.

    Parameters
    ----------
    player_id:
        ``1`` or ``2`` — required by the :class:`Player` contract.
    name:
        Display name shown by the game loop.
    iterations:
        Number of MCTS iterations performed per move.
    exploration_weight:
        UCT exploration constant; defaults to the textbook ``sqrt(2)``.
    rollout_depth_limit:
        Maximum plies per random rollout. See :class:`ai.mcts.MCTS`.
    random_seed:
        Seed for the underlying MCTS RNG. ``None`` for non-reproducible play.
    """

    def __init__(
        self,
        player_id: int,
        name: str = "MCTS",
        iterations: int = 1000,
        exploration_weight: float = math.sqrt(2),
        rollout_depth_limit: int = 80,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id, name)
        self.iterations: int = iterations
        self.exploration_weight: float = exploration_weight
        self.rollout_depth_limit: int = rollout_depth_limit
        self.random_seed: Optional[int] = random_seed
        self._engine: MCTS = MCTS(
            iterations=iterations,
            exploration_weight=exploration_weight,
            rollout_depth_limit=rollout_depth_limit,
            random_seed=random_seed,
        )

    # ------------------------------------------------------------------
    # Player API
    # ------------------------------------------------------------------

    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        """Return the move recommended by MCTS for the given board."""
        return self._engine.search(board)
