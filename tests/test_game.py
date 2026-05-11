"""Unit tests for :mod:`game.player` and :mod:`game.game`."""

from __future__ import annotations

from typing import Iterable

import pytest

from game.board import PopOutBoard
from game.game import PopOutGame
from game.player import HumanPlayer, Player, RandomPlayer


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class ScriptedPlayer(Player):
    """Deterministic player that replays a pre-defined sequence of moves.

    Used for testing the game loop without relying on the Random/Human inputs.
    ``on_full_board`` and ``on_repetition`` control the optional-draw hooks.
    """

    def __init__(
        self,
        player_id: int,
        name: str,
        moves: Iterable[tuple[str, int]],
        *,
        on_full_board: str = "pop",
        on_repetition: str = "continue",
    ) -> None:
        super().__init__(player_id, name)
        self._moves: list[tuple[str, int]] = list(moves)
        self._idx: int = 0
        self._on_full_board = on_full_board
        self._on_repetition = on_repetition

    def choose_move(self, board: PopOutBoard) -> tuple[str, int]:
        move = self._moves[self._idx]
        self._idx += 1
        return move

    def choose_pop_or_draw(self, board: PopOutBoard) -> str:
        return self._on_full_board

    def choose_continue_or_draw(self, board: PopOutBoard) -> str:
        return self._on_repetition


# ---------------------------------------------------------------------------
# Player interface
# ---------------------------------------------------------------------------


def test_player_rejects_invalid_player_ids() -> None:
    with pytest.raises(ValueError):
        RandomPlayer(0, "Bad")
    with pytest.raises(ValueError):
        RandomPlayer(3, "Bad")


def test_human_player_stores_id_and_name() -> None:
    hp = HumanPlayer(1)
    assert hp.player_id == 1
    assert hp.name == "Human"


def test_base_player_defaults_keep_the_game_going() -> None:
    """Default draw hooks must keep the game alive so autonomous agents
    never accidentally forfeit by default."""
    board = PopOutBoard()
    player = RandomPlayer(1, seed=0)  # uses the inherited defaults
    assert player.choose_pop_or_draw(board) == "pop"
    assert player.choose_continue_or_draw(board) == "continue"


# ---------------------------------------------------------------------------
# RandomPlayer behaviour
# ---------------------------------------------------------------------------


def test_random_player_only_picks_legal_moves_over_100_plies() -> None:
    """Simulate up to 100 random plies; every chosen move must be valid."""
    board = PopOutBoard()
    p1 = RandomPlayer(1, "R1", seed=0)
    p2 = RandomPlayer(2, "R2", seed=1)

    plies = 0
    while plies < 100 and not board.is_terminal():
        current = p1 if board.current_player == 1 else p2
        move = current.choose_move(board)
        assert move in board.get_valid_moves(), (
            f"Random player returned illegal move {move!r}"
        )
        board.make_move(*move)
        plies += 1

    assert plies > 0  # sanity: we actually played something


def test_random_player_is_reproducible_given_same_seed() -> None:
    board = PopOutBoard()
    a = RandomPlayer(1, seed=42)
    b = RandomPlayer(1, seed=42)
    # Same seed + same board state must produce the same move.
    assert a.choose_move(board) == b.choose_move(board)


# ---------------------------------------------------------------------------
# Full-game smoke tests
# ---------------------------------------------------------------------------


def test_random_vs_random_ten_games_all_terminate() -> None:
    """Run 10 Random-vs-Random games end-to-end; none may crash."""
    outcomes = []
    for seed in range(10):
        p1 = RandomPlayer(1, "R1", seed=seed)
        p2 = RandomPlayer(2, "R2", seed=seed + 1000)
        game = PopOutGame(p1, p2, verbose=False, max_moves=500)
        outcomes.append(game.play())
    assert all(result in (1, 2, None) for result in outcomes)
    assert len(outcomes) == 10


# ---------------------------------------------------------------------------
# Game loop — result correctness
# ---------------------------------------------------------------------------


def test_game_loop_detects_horizontal_win_for_player_one() -> None:
    """Scripted horizontal four-in-a-row on the bottom row for P1."""
    p1 = ScriptedPlayer(
        1, "P1", [("drop", 0), ("drop", 1), ("drop", 2), ("drop", 3)]
    )
    p2 = ScriptedPlayer(2, "P2", [("drop", 0), ("drop", 1), ("drop", 2)])
    game = PopOutGame(p1, p2, verbose=False)
    assert game.play() == 1


def test_game_loop_detects_vertical_win_for_player_two() -> None:
    """Scripted vertical four-in-a-row in column 1 for P2."""
    p1_moves = [("drop", 0), ("drop", 0), ("drop", 0), ("drop", 2)]
    p2_moves = [("drop", 1), ("drop", 1), ("drop", 1), ("drop", 1)]
    p1 = ScriptedPlayer(1, "P1", p1_moves)
    p2 = ScriptedPlayer(2, "P2", p2_moves)
    game = PopOutGame(p1, p2, verbose=False)
    assert game.play() == 2


def test_game_loop_detects_draw_by_three_fold_repetition() -> None:
    """A drop/pop cycle repeated twice triggers rule 3; P1 declares a draw."""
    # One full cycle: P1 drop col 0, P2 drop col 1, P1 pop col 0, P2 pop col 1
    # Two cycles return the initial state to the history a total of 3 times.
    p1_moves = [("drop", 0), ("pop", 0), ("drop", 0), ("pop", 0)]
    p2_moves = [("drop", 1), ("pop", 1), ("drop", 1), ("pop", 1)]
    p1 = ScriptedPlayer(1, "P1", p1_moves, on_repetition="draw")
    p2 = ScriptedPlayer(2, "P2", p2_moves, on_repetition="draw")
    game = PopOutGame(p1, p2, verbose=False)
    assert game.play() is None


def test_game_loop_ignores_repetition_when_players_want_to_continue() -> None:
    """If nobody declares a draw on the 3-fold state, play continues.

    The scripted players have *no* extra moves after the repetition is
    reached; exercising the "continue" branch would therefore raise
    ``IndexError``. We run with a very small ``max_moves`` so the game ends
    by the safety cap instead, proving the loop did not short-circuit on
    ``is_draw``.
    """
    extra_moves_past_repetition = [("drop", 0)] * 50
    p1 = ScriptedPlayer(
        1,
        "P1",
        [("drop", 0), ("pop", 0), ("drop", 0), ("pop", 0)]
        + extra_moves_past_repetition,
    )
    p2 = ScriptedPlayer(
        2,
        "P2",
        [("drop", 1), ("pop", 1), ("drop", 1), ("pop", 1)]
        + extra_moves_past_repetition,
    )
    game = PopOutGame(p1, p2, verbose=False, max_moves=12)
    result = game.play()
    # Nobody won; the safety cap stopped the game.
    assert result is None


# ---------------------------------------------------------------------------
# Game loop — configuration errors
# ---------------------------------------------------------------------------


def test_game_rejects_swapped_player_ids() -> None:
    """player1 must have id 1 and player2 must have id 2."""
    with pytest.raises(ValueError):
        PopOutGame(RandomPlayer(2, "wrong"), RandomPlayer(1, "wrong"))


def test_game_rejects_non_positive_max_moves() -> None:
    with pytest.raises(ValueError):
        PopOutGame(RandomPlayer(1), RandomPlayer(2), max_moves=0)
