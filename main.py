"""Command-line entry point for PopOut.

Shows a simple text menu and launches the selected game mode. Modes 2 and 3
currently use :class:`~game.player.RandomPlayer` as a placeholder for the
future MCTS / Decision Tree players (Phases 3 and 5 of the project).
"""

from __future__ import annotations

from game.game import PopOutGame
from game.player import HumanPlayer, Player, RandomPlayer


def main() -> None:
    """Run the interactive menu loop."""
    while True:
        choice = _read_menu_choice()
        if choice == "1":
            _launch(HumanPlayer(1, "Player 1"), HumanPlayer(2, "Player 2"))
        elif choice == "2":
            # ``num_sims`` is collected now so Phases 3/5 can simply replace
            # RandomPlayer with an MCTS-backed player that takes it.
            _num_sims = _ask_num_simulations()
            _launch(HumanPlayer(1, "Human"), RandomPlayer(2, "Computer"))
        elif choice == "3":
            _num_sims = _ask_num_simulations()
            _launch(RandomPlayer(1, "CPU-1"), RandomPlayer(2, "CPU-2"))
        elif choice == "4":
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Enter a number from 1 to 4.")


def _read_menu_choice() -> str:
    print()
    print("=== PopOut ===")
    print("Choose game mode:")
    print("1. Human vs Human")
    print("2. Human vs Computer")
    print("3. Computer vs Computer")
    print("4. Quit")
    print()
    return input("> ").strip()


def _ask_num_simulations() -> int:
    """Prompt for the number of MCTS simulations (used in future phases)."""
    while True:
        raw = input("Number of simulations for AI (default 1000): ").strip()
        if raw == "":
            return 1000
        try:
            value = int(raw)
        except ValueError:
            print("  Invalid number. Enter an integer.")
            continue
        if value < 1:
            print("  Must be at least 1.")
            continue
        return value


def _launch(player1: Player, player2: Player) -> None:
    PopOutGame(player1, player2, verbose=True).play()


if __name__ == "__main__":
    main()
