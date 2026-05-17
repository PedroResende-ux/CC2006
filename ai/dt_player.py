"""Decision-tree-driven :class:`~game.player.Player` adapter.

This module wires an ID3 decision tree trained by :mod:`ai.dt_pipeline`
into the PopOut game loop and the tournament runner.

:meth:`DTPlayer.choose_move` extracts the 46 training features from the
live board, applies the same per-column binning, runs
:func:`ai.id3.predict_one`, decodes the integer leaf label back into a
``(kind, col)`` move tuple, and guards against the (rare) case where the
predicted move is illegal in the current position.

Why the legality guard matters
------------------------------

The comparison project (``outro``) reports a 0% win rate for its DT
against MCTS because the tree could happily emit an illegal move and the
game loop just crashed. Our spec is the opposite: ``choose_move`` must
return a legal move from ``board.get_valid_moves()`` 100% of the time.

Fallback strategy: ``"nearest"`` (default)
------------------------------------------

When the predicted move ``(kind, col)`` is not in the legal-move list,
:meth:`DTPlayer._fallback` returns the legal move closest to the
prediction on the ``(kind, col)`` lattice. Distance is

* ``|col_pred - col|``                       if the kinds match,
* ``|col_pred - col| + 7``                   if the kinds differ.

The ``+7`` kind-mismatch penalty is the board width — guarantees that any
same-kind alternative dominates any cross-kind alternative regardless of
column distance. Ties (e.g. ``drop_3`` predicted, both ``drop_2`` and
``drop_4`` legal) are broken in a deterministic order (DROP before POP,
then column ascending) so tournament replays with the same seed are
bit-identical. The fallback never consults the RNG.

This is "Option (c)" from the prompt brief — ad-hoc but tree-faithful:
the tree's column preference is honoured as long as any nearby column of
the predicted kind is playable, and only collapses to a different kind
when every column of the predicted kind is unavailable.

The historical ``"random"`` strategy is kept for ablations, plus a
``"raise"`` mode for unit tests that want to surface mismatches.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from game.board import PopOutBoard
from game.player import Player

from ai.dt_pipeline import (
    POPOUT_BIN_DEFINITIONS,
    int_to_move_tuple,
    load_tree,
)
from ai.id3 import predict_one


# How :class:`DTPlayer` should handle a predicted-but-illegal move:
#
# ``"raise"``    — raise ``RuntimeError``. Useful in unit tests where a
#                  mismatch indicates a real bug.
# ``"random"``   — pick uniformly at random from
#                  :meth:`PopOutBoard.get_valid_moves` using the seeded
#                  :class:`numpy.random.RandomState`. Deterministic given
#                  ``random_seed``.
# ``"nearest"``  — closest legal move on the ``(kind, col)`` lattice.
#                  Deterministic regardless of seed. Default.
FallbackStrategy = Literal["raise", "random", "nearest"]


# Pre-computed once: cell-feature column names in row-major order
# (``cell_00``, ``cell_01``, ..., ``cell_56``). Matches
# :data:`scripts.generate_dataset._CELL_COLS` exactly.
_CELL_COLUMN_NAMES: tuple[str, ...] = tuple(
    f"cell_{r}{c}" for r in range(6) for c in range(7)
)


class DTPlayer(Player):
    """ID3 decision-tree player for PopOut.

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
        board. See :data:`FallbackStrategy`. Defaults to ``"nearest"``,
        which is deterministic and preserves the tree's column intent.
    random_seed:
        Seed for the :class:`numpy.random.RandomState` consumed by the
        ``"random"`` fallback. ``None`` produces non-reproducible
        behaviour. Has no effect for ``"nearest"`` and ``"raise"`` —
        those strategies are intrinsically deterministic, so tournament
        replays remain bit-identical without consulting the RNG.
    """

    def __init__(
        self,
        player_id: int,
        tree_path: str,
        name: str = "DT",
        fallback_strategy: FallbackStrategy = "nearest",
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(player_id, name)
        self.tree_path: str = tree_path
        self.fallback_strategy: FallbackStrategy = fallback_strategy
        self.random_seed: Optional[int] = random_seed
        self._tree: dict = load_tree(tree_path)
        self._rng: np.random.RandomState = np.random.RandomState(random_seed)

    # ------------------------------------------------------------------
    # Player API
    # ------------------------------------------------------------------

    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        """Return a legal move recommended by the decision tree.

        Pipeline:

        1. Build a feature row matching the training schema (42 cell
           values + ``current_player`` as raw ints, plus three binned
           numeric features per
           :const:`ai.dt_pipeline.POPOUT_BIN_DEFINITIONS`).
        2. Predict the integer class label via :func:`ai.id3.predict_one`.
        3. Decode to ``(kind, col)`` via
           :func:`ai.dt_pipeline.int_to_move_tuple`.
        4. If the move is legal, return it. Otherwise delegate to
           :meth:`_fallback`.

        The result is guaranteed to be an element of
        ``board.get_valid_moves()``.
        """
        row = self._extract_features(board)
        predicted_label = predict_one(row, self._tree)
        predicted_move = int_to_move_tuple(int(predicted_label))

        legal = board.get_valid_moves()
        if predicted_move in legal:
            return predicted_move
        return self._fallback(legal, predicted_move)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, board: PopOutBoard) -> pd.Series:
        """Build the feature row in the schema the tree was trained on.

        Cell features (``cell_RC`` for ``R in 0..5, C in 0..6``) and
        ``current_player`` remain as ints — they are detected as
        categorical by :func:`ai.dt_pipeline.inspect_dataset`
        (``unique_vals <= 3``) so binning is never applied to them at
        train time, and applying it now would break the schema.

        The three numeric columns (``move_count``,
        ``own_pieces_bottom_row``, ``opp_pieces_bottom_row``) are
        bucketised with the same :func:`pandas.cut` call used in
        :func:`ai.dt_pipeline.bin_features`, then coerced to ``str``.
        Values outside the bin range therefore appear as the literal
        string ``"nan"`` — exactly what training did to any analogous
        rows in the dataset, so the predict-time schema matches the
        train-time schema even at the boundaries.

        Returns a :class:`pandas.Series` (object dtype) ready for
        :func:`ai.id3.predict_one`.
        """
        cells = board.board
        current = int(board.current_player)
        opp = 2 if current == 1 else 1
        bottom = cells[5]
        own_bot = int(np.count_nonzero(bottom == current))
        opp_bot = int(np.count_nonzero(bottom == opp))

        data: dict = {}
        for r in range(6):
            for c in range(7):
                data[f"cell_{r}{c}"] = int(cells[r, c])
        data['move_count'] = int(board.move_count)
        data['current_player'] = current
        data['own_pieces_bottom_row'] = own_bot
        data['opp_pieces_bottom_row'] = opp_bot

        for col, defn in POPOUT_BIN_DEFINITIONS.items():
            binned = pd.cut(
                [data[col]], bins=defn['bins'], labels=defn['labels'],
            ).astype(str)
            # ``binned`` is a length-1 ndarray of object strings; index 0
            # gives a Python ``str`` matching what ``bin_features`` produced
            # at training time.
            data[col] = binned[0]

        return pd.Series(data)

    # ------------------------------------------------------------------
    # Fallback strategies
    # ------------------------------------------------------------------

    def _fallback(
        self,
        legal_moves: list[tuple[str, int]],
        predicted_move: tuple[str, int],
    ) -> tuple[str, int]:
        """Return a legal move that approximates ``predicted_move``.

        See module docstring for the rationale behind each strategy. The
        default ``"nearest"`` strategy is fully deterministic; the
        ``"random"`` strategy is deterministic *given* ``random_seed``.
        """
        if not legal_moves:
            # The game loop only calls a player when a legal move exists,
            # so this branch is unreachable in practice — kept defensive.
            raise RuntimeError("DTPlayer asked to move with no legal moves.")

        if self.fallback_strategy == "raise":
            raise RuntimeError(
                f"DTPlayer predicted illegal move {predicted_move!r}; "
                f"legal moves were {legal_moves!r}."
            )

        if self.fallback_strategy == "random":
            idx = int(self._rng.randint(len(legal_moves)))
            return legal_moves[idx]

        # "nearest" — closest legal move on the (kind, col) lattice.
        pred_kind, pred_col = predicted_move

        def _distance_key(move: tuple[str, int]) -> tuple[int, int, int]:
            kind, col = move
            kind_penalty = 0 if kind == pred_kind else 7
            kind_order = 0 if kind == "drop" else 1
            return (abs(col - pred_col) + kind_penalty, kind_order, col)

        return min(legal_moves, key=_distance_key)
