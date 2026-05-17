"""Decision-tree-driven :class:`~game.player.Player` adapter — STUB.

This module establishes the file and class signature for the ID3
decision tree player so that :mod:`scripts.run_tournament` can be
extended in a future iteration to support DT-vs-MCTS matchups. The
actual move-selection logic is not implemented yet.

Planned implementation (deferred to a follow-up prompt)
-------------------------------------------------------

1. Constructor receives the path to a pickled tree produced by
   :func:`ai.dt_pipeline.save_tree` and (optionally) a fallback strategy
   for the case where the predicted move is illegal on the live board.

2. :meth:`choose_move` will:

   a. Extract the same 46 features used at training time from
      ``board``: 42 raw cell values (``cell_00`` … ``cell_56``,
      row-major), ``move_count``, ``current_player``,
      ``own_pieces_bottom_row``, ``opp_pieces_bottom_row``. The exact
      schema is defined by :func:`scripts.generate_dataset._row_from_decision`.
   b. Apply :const:`ai.dt_pipeline.POPOUT_BIN_DEFINITIONS` to the three
      continuous columns so the row matches the discretisation used at
      train time.
   c. Run :func:`ai.id3.predict_one` to obtain an integer class label.
   d. Decode it via :func:`ai.dt_pipeline.int_to_move` into
      ``(move_type, col)``.
   e. Validate against ``board.get_valid_moves()``. If illegal, fall back
      according to ``fallback_strategy`` — this is critical because
      the comparison project's DT could not play self-play games for
      exactly this reason (see ``outro/report.ipynb`` §"Decision Tree
      vs Decision tree: Not possible").

3. Tournament integration: :mod:`scripts.run_tournament` currently
   only constructs :class:`_MCTSConfiguredPlayer`. Adding DTPlayer
   support means extending the JSON matchup schema with a ``type``
   field (``"mcts"`` vs ``"dt"``) and a player-factory dispatch.
"""

from __future__ import annotations

from typing import Literal, Optional

from game.board import PopOutBoard
from game.player import Player

from ai.dt_pipeline import POPOUT_BIN_DEFINITIONS, int_to_move, load_tree
from ai.id3 import predict_one


# Fallback strategies considered for handling illegal predicted moves.
# - ``"raise"``:  fail fast (useful in tests).
# - ``"random"``: pick uniformly at random from ``board.get_valid_moves()``.
# - ``"second"``: walk the tree to the second-best leaf — not yet designed.
FallbackStrategy = Literal["raise", "random", "second"]


class DTPlayer(Player):
    """ID3 decision-tree player for PopOut — stubbed pending implementation.

    Parameters
    ----------
    player_id:
        ``1`` or ``2`` — required by the :class:`Player` contract.
    tree_path:
        Filesystem path to a pickled tree saved by
        :func:`ai.dt_pipeline.save_tree`.
    name:
        Display name shown by the game loop.
    fallback_strategy:
        How to handle a predicted move that is illegal on the current
        board. Defaults to ``"random"`` so the player never crashes
        mid-game, matching the safety profile of
        :class:`game.player.RandomPlayer`.
    random_seed:
        Seed for the random fallback's RNG. ``None`` for non-reproducible play.
    """

    def __init__(
        self,
        player_id: int,
        tree_path: str,
        name: str = "DT",
        fallback_strategy: FallbackStrategy = "random",
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id, name)
        self.tree_path: str = tree_path
        self.fallback_strategy: FallbackStrategy = fallback_strategy
        self.random_seed: Optional[int] = random_seed
        self._tree: dict = load_tree(tree_path)
        # Reserved for the actual implementation:
        self._bin_definitions = POPOUT_BIN_DEFINITIONS
        self._predict_one = predict_one
        self._int_to_move = int_to_move

    # ------------------------------------------------------------------
    # Player API
    # ------------------------------------------------------------------

    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        """Return the move recommended by the decision tree.

        Not implemented yet — see module docstring for the planned
        feature-extraction → binning → predict → decode → legality-check
        pipeline.
        """
        raise NotImplementedError(
            "DTPlayer.choose_move is a stub. The feature-extraction and "
            "legality-fallback logic will be added in a follow-up prompt. "
            "See ai/dt_player.py module docstring for the planned design."
        )
