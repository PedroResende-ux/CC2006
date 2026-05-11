"""Unit tests for :mod:`game.board` — the PopOut engine."""

from __future__ import annotations

import numpy as np
import pytest

from game.board import COLS, EMPTY, PLAYER_1, PLAYER_2, ROWS, PopOutBoard


# ---------------------------------------------------------------------------
# Basic state
# ---------------------------------------------------------------------------


def test_empty_board_initial_state() -> None:
    b = PopOutBoard()
    assert b.board.shape == (ROWS, COLS)
    assert np.all(b.board == EMPTY)
    assert b.current_player == PLAYER_1
    assert b.move_count == 0
    # Initial state must be registered in history for 3-fold repetition.
    assert len(b.history) == 1
    assert b.check_winner() is None
    assert not b.is_terminal()


def test_initial_valid_moves_are_seven_drops() -> None:
    b = PopOutBoard()
    moves = b.get_valid_moves()
    assert len(moves) == COLS
    assert all(kind == "drop" for kind, _ in moves)
    # All pops are illegal on an empty board.
    assert not any(kind == "pop" for kind, _ in moves)


# ---------------------------------------------------------------------------
# Drop behaviour
# ---------------------------------------------------------------------------


def test_drop_on_empty_column_lands_on_bottom_row() -> None:
    b = PopOutBoard()
    b.make_move("drop", 3)
    assert b.board[ROWS - 1, 3] == PLAYER_1
    assert b.board[ROWS - 2, 3] == EMPTY
    assert b.current_player == PLAYER_2
    assert b.move_count == 1


def test_drop_stacks_on_existing_pieces() -> None:
    b = PopOutBoard()
    b.make_move("drop", 3)  # P1 at (5, 3)
    b.make_move("drop", 3)  # P2 at (4, 3)
    b.make_move("drop", 3)  # P1 at (3, 3)
    assert b.board[5, 3] == PLAYER_1
    assert b.board[4, 3] == PLAYER_2
    assert b.board[3, 3] == PLAYER_1
    assert b.board[2, 3] == EMPTY


def test_drop_on_full_column_is_rejected() -> None:
    b = PopOutBoard()
    # Fill column 3 with alternating P1/P2 drops.
    for _ in range(ROWS):
        b.make_move("drop", 3)
    moves = b.get_valid_moves()
    assert ("drop", 3) not in moves
    # The bottom piece is P1 (first drop), so the current player (P1 again
    # after 6 moves) can still pop the column.
    assert b.current_player == PLAYER_1
    assert ("pop", 3) in moves
    with pytest.raises(ValueError):
        b.make_move("drop", 3)


# ---------------------------------------------------------------------------
# Pop behaviour
# ---------------------------------------------------------------------------


def test_pop_own_piece_is_valid() -> None:
    b = PopOutBoard()
    b.make_move("drop", 3)  # P1 at (5, 3)
    b.make_move("drop", 0)  # P2 plays elsewhere; P1 is to move again.
    assert ("pop", 3) in b.get_valid_moves()


def test_pop_cannot_remove_opponent_piece() -> None:
    b = PopOutBoard()
    b.make_move("drop", 3)  # P1 drops at (5, 3); now P2's turn.
    assert b.current_player == PLAYER_2
    assert ("pop", 3) not in b.get_valid_moves()
    with pytest.raises(ValueError):
        b.make_move("pop", 3)


def test_pop_on_empty_column_is_rejected() -> None:
    b = PopOutBoard()
    assert not any(kind == "pop" for kind, _ in b.get_valid_moves())
    with pytest.raises(ValueError):
        b.make_move("pop", 0)


def test_pop_shifts_pieces_down_by_one() -> None:
    b = PopOutBoard()
    # Column 3 (bottom→top): P1, P2, P1
    b.make_move("drop", 3)  # P1 (5, 3)
    b.make_move("drop", 3)  # P2 (4, 3)
    b.make_move("drop", 3)  # P1 (3, 3)
    b.make_move("drop", 0)  # P2 plays elsewhere; P1 on turn.
    b.make_move("pop", 3)
    # After the pop, the column is shifted down by one row.
    assert b.board[5, 3] == PLAYER_2     # was at (4, 3)
    assert b.board[4, 3] == PLAYER_1     # was at (3, 3)
    assert b.board[3, 3] == EMPTY        # nothing above shifted here
    assert b.board[2, 3] == EMPTY


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------


def test_detect_horizontal_win() -> None:
    b = PopOutBoard()
    # P1 wins on the bottom row at columns 0–3.
    b.make_move("drop", 0)  # P1 (5, 0)
    b.make_move("drop", 0)  # P2 (4, 0)
    b.make_move("drop", 1)  # P1 (5, 1)
    b.make_move("drop", 1)  # P2 (4, 1)
    b.make_move("drop", 2)  # P1 (5, 2)
    b.make_move("drop", 2)  # P2 (4, 2)
    b.make_move("drop", 3)  # P1 (5, 3)  -> 4 in a row
    assert b.check_winner() == PLAYER_1
    assert b.is_terminal()


def test_detect_vertical_win() -> None:
    b = PopOutBoard()
    for _ in range(3):
        b.make_move("drop", 0)  # P1
        b.make_move("drop", 1)  # P2
    b.make_move("drop", 0)      # P1 — 4 stacked in column 0
    assert b.check_winner() == PLAYER_1
    assert b.is_terminal()


def test_detect_diagonal_up_right_win() -> None:
    # P1 occupies the diagonal (5,0) -> (2,3).
    b = PopOutBoard()
    b.board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 2, 0, 0, 0],
            [0, 1, 2, 2, 0, 0, 0],
            [1, 2, 2, 2, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    assert b.check_winner() == PLAYER_1


def test_detect_diagonal_down_right_win() -> None:
    # P1 occupies the diagonal (2,3) -> (5,6).
    b = PopOutBoard()
    b.board = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 2, 1, 0, 0],
            [0, 0, 0, 2, 2, 1, 0],
            [0, 0, 0, 2, 2, 2, 1],
        ],
        dtype=np.int8,
    )
    assert b.check_winner() == PLAYER_1


# ---------------------------------------------------------------------------
# Special pop rule (rule 1 from CLAUDE.md)
# ---------------------------------------------------------------------------


def test_pop_that_creates_four_for_both_players_grants_win_to_popper() -> None:
    """If popping creates four-in-a-row for both sides, the mover wins."""
    b = PopOutBoard()
    # Pre-pop layout (col 3 bottom = P1, to be popped):
    #
    #     Row 0: . . . 1 . . .
    #     Row 1: . . . . . . .
    #     Row 2: . . 1 . . . .
    #     Row 3: . 1 . . . . .
    #     Row 4: 1 . . 2 . . .
    #     Row 5: 2 2 2 1 . . .
    #
    # After P1 pops column 3 the whole column shifts down by one:
    # the P1 at (0, 3) slides to (1, 3), completing the P1 diagonal
    # (4,0)-(3,1)-(2,2)-(1,3); simultaneously the P2 at (4, 3) slides to
    # (5, 3), completing the P2 horizontal (5,0)-(5,1)-(5,2)-(5,3).
    b.board = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 2, 0, 0, 0],
            [2, 2, 2, 1, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    b.current_player = PLAYER_1
    # Sanity: nobody has won yet.
    assert b.check_winner() is None

    b.make_move("pop", 3)

    # Verify the new layout matches the expected double four-in-a-row.
    assert b.board[1, 3] == PLAYER_1   # diagonal for P1 closes here
    assert b.board[5, 3] == PLAYER_2   # horizontal for P2 closes here
    assert b.board[0, 3] == EMPTY      # topmost cell is cleared by the pop
    # Rule 1: both players have 4 in a row, the popper (P1) wins.
    assert b.check_winner() == PLAYER_1


# ---------------------------------------------------------------------------
# Clone & state hashing
# ---------------------------------------------------------------------------


def test_clone_is_independent_from_original() -> None:
    b = PopOutBoard()
    b.make_move("drop", 3)
    twin = b.clone()
    # Mutating the clone must not affect the original.
    twin.make_move("drop", 4)

    assert b.board[ROWS - 1, 4] == EMPTY
    assert twin.board[ROWS - 1, 4] == PLAYER_2
    assert b.current_player == PLAYER_2
    assert twin.current_player == PLAYER_1
    assert b.move_count == 1
    assert twin.move_count == 2
    # Histories are separate list objects.
    assert b.history is not twin.history
    assert len(b.history) != len(twin.history)


def test_state_hash_is_turn_sensitive() -> None:
    b1 = PopOutBoard()
    b2 = PopOutBoard()
    assert b1.get_state_hash() == b2.get_state_hash()
    # Same board but different player to move -> different hash.
    b2.current_player = PLAYER_2
    assert b1.get_state_hash() != b2.get_state_hash()


def test_state_hash_reflects_board_changes() -> None:
    b = PopOutBoard()
    h0 = b.get_state_hash()
    b.make_move("drop", 3)
    assert b.get_state_hash() != h0


# ---------------------------------------------------------------------------
# Three-fold repetition (rule 3)
# ---------------------------------------------------------------------------


def test_three_fold_repetition_declares_draw() -> None:
    """A repeated drop/pop cycle must trigger the draw rule on the 3rd pass."""
    b = PopOutBoard()
    initial = b.get_state_hash()

    def cycle() -> None:
        b.make_move("drop", 0)  # P1 -> (5, 0)
        b.make_move("drop", 1)  # P2 -> (5, 1)
        b.make_move("pop", 0)   # P1 pops -> column 0 empty
        b.make_move("pop", 1)   # P2 pops -> column 1 empty; back to initial

    # After the first cycle the initial state has been reached twice.
    cycle()
    assert b.get_state_hash() == initial
    assert b.history.count(initial) == 2
    assert not b.is_draw()

    # The second cycle produces the third occurrence, which is a draw.
    cycle()
    assert b.get_state_hash() == initial
    assert b.history.count(initial) == 3
    assert b.is_draw()
    assert b.is_terminal()


# ---------------------------------------------------------------------------
# Terminal flags
# ---------------------------------------------------------------------------


def test_is_draw_is_false_when_a_player_has_won() -> None:
    """A winning position is never classified as a draw."""
    b = PopOutBoard()
    # Bottom row filled with four P1 pieces: P1 wins horizontally.
    b.board[ROWS - 1, :4] = PLAYER_1
    assert b.check_winner() == PLAYER_1
    assert b.is_terminal()
    assert not b.is_draw()


def test_is_terminal_reflects_winner_or_draw() -> None:
    b = PopOutBoard()
    assert not b.is_terminal()
    # Drop four P1 pieces into column 0 — vertical win.
    b.make_move("drop", 0)
    b.make_move("drop", 1)
    b.make_move("drop", 0)
    b.make_move("drop", 1)
    b.make_move("drop", 0)
    b.make_move("drop", 1)
    b.make_move("drop", 0)
    assert b.is_terminal()
