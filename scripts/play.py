"""scripts/play.py — three game modes per the project spec.

Examples:
    python -m scripts.play --p1 human --p2 human
    python -m scripts.play --p1 human --p2 dt
    python -m scripts.play --p1 mcts --p2 dt
    python -m scripts.play --p1 dt --p2 mcts --mcts-iter 1000   # fast MCTS demo

The two players are picked from {human, mcts, dt, random}. Default
configurations match the optimised values from the notebook §4.5
(MCTS: 20000 iterations, c=2.0, k=1, ucb1, depth-limit 80) and §10
(DT: data/id3/id3_tree_optimised_pruned.pkl).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai.dt_player import DTPlayer  # noqa: E402
from ai.mcts import MCTS  # noqa: E402
from game.game import PopOutGame  # noqa: E402
from game.player import HumanPlayer, Player, RandomPlayer  # noqa: E402


VALID_TYPES: tuple[str, ...] = ("human", "mcts", "dt", "random")

# Optimised MCTS configuration from notebook §4.5.
DEFAULT_MCTS_ITER: int = 20000
DEFAULT_MCTS_C: float = 2.0
DEFAULT_MCTS_K: int = 1
DEFAULT_MCTS_VARIANT: str = "ucb1"
DEFAULT_MCTS_ROLLOUT_DEPTH: int = 80

# Optimised + pruned DT from notebook §10.
DEFAULT_DT_PICKLE: str = "data/id3/id3_tree_optimised_pruned.pkl"

DEFAULT_SEED: int = 42


class _MCTSCliPlayer(Player):
    """Local adapter mirroring ``scripts.run_tournament._MCTSConfiguredPlayer``.

    :class:`ai.mcts_player.MCTSPlayer` does not forward the variant
    parameters (``num_children_to_expand``, ``uct_variant``) to the
    underlying engine, so we instantiate :class:`ai.mcts.MCTS` directly
    here — same pattern the tournament runner uses.
    """

    def __init__(
        self,
        player_id: int,
        iterations: int,
        seed: int,
    ) -> None:
        super().__init__(player_id, "MCTS")
        self._engine: MCTS = MCTS(
            iterations=iterations,
            exploration_weight=DEFAULT_MCTS_C,
            rollout_depth_limit=DEFAULT_MCTS_ROLLOUT_DEPTH,
            num_children_to_expand=DEFAULT_MCTS_K,
            uct_variant=DEFAULT_MCTS_VARIANT,
            random_seed=seed,
        )

    def choose_move(self, board):
        return self._engine.search(board)


def _resolve_pickle_path(rel_or_abs: str) -> str:
    """Resolve a DT pickle path relative to the project root when needed."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p)
    return str(_PROJECT_ROOT / p)


def make_player(type_str: str, player_id: int, args: argparse.Namespace) -> Player:
    """Construct one player matching the existing constructor signatures."""
    if type_str == "human":
        return HumanPlayer(player_id=player_id)
    if type_str == "random":
        return RandomPlayer(player_id=player_id, seed=args.seed)
    if type_str == "mcts":
        return _MCTSCliPlayer(
            player_id=player_id,
            iterations=args.mcts_iter,
            seed=args.seed,
        )
    if type_str == "dt":
        return DTPlayer(
            player_id=player_id,
            tree_path=_resolve_pickle_path(args.dt_pickle),
            random_seed=args.seed,
        )
    raise ValueError(f"Unknown player type: {type_str}")


def _winner_msg(winner: Optional[int], p1_type: str, p2_type: str) -> str:
    if winner is None:
        return "Draw."
    label = p1_type if winner == 1 else p2_type
    return f"Player {winner} ({label}) wins."


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="play",
        description=(
            "Run a PopOut game between two players. Supports the three "
            "modes required by the spec: human-vs-human, human-vs-computer, "
            "and computer-vs-computer (two different algorithms)."
        ),
    )
    p.add_argument("--p1", choices=VALID_TYPES, required=True,
                   help="Player 1 type.")
    p.add_argument("--p2", choices=VALID_TYPES, required=True,
                   help="Player 2 type.")
    p.add_argument("--mcts-iter", type=int, default=DEFAULT_MCTS_ITER,
                   help=f"MCTS iterations per move (default: {DEFAULT_MCTS_ITER}).")
    p.add_argument("--dt-pickle", type=str, default=DEFAULT_DT_PICKLE,
                   help=f"Path to the DT pickle (default: {DEFAULT_DT_PICKLE}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Seed for stochastic players (default: {DEFAULT_SEED}).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    p1 = make_player(args.p1, player_id=1, args=args)
    p2 = make_player(args.p2, player_id=2, args=args)

    print(f"Player 1 ({args.p1}) vs Player 2 ({args.p2})")
    print("=" * 40)

    game = PopOutGame(p1, p2, verbose=True)
    winner = game.play()
    print(f"\nResult: {_winner_msg(winner, args.p1, args.p2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
