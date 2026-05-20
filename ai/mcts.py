"""Monte Carlo Tree Search for PopOut.

This module implements UCT-based MCTS with:

* a tree of :class:`MCTSNode` instances (``__slots__`` for memory efficiency),
* root-perspective reward accounting (1.0 root-player win, 0.5 draw,
  0.0 root-player loss) accumulated unchanged through back-propagation,
* terminal proof propagation: nodes whose value is provably known
  (forced win/loss/draw) are marked and skipped in selection, with
  their value cascading up the tree via mini-max,
* an "only legal move" short-circuit at the very start of :meth:`MCTS.search`,
* a depth-capped uniform random rollout.

The search routine returns the **most-visited** child of the root, with
ties broken by mean value, and an override that always picks a proven
winning move when one is available.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from ai.rollout_numba import simulate_rollout
from game.board import PopOutBoard


# ---------------------------------------------------------------------------
# Constants for terminal-state encoding
# ---------------------------------------------------------------------------

# ``terminal`` values stored on MCTSNode:
#   -1   -> not (yet) proven; ordinary statistics apply.
#    0   -> proven loss for the root player.
#    0.5 -> proven draw.
#    1   -> proven win for the root player.
TERMINAL_UNDETERMINED: float = -1.0
TERMINAL_LOSS: float = 0.0
TERMINAL_DRAW: float = 0.5
TERMINAL_WIN: float = 1.0

Move = tuple[str, int]


# ---------------------------------------------------------------------------
# Fast (numpy-vectorised) helpers used inside the rollout hot loop.
# ---------------------------------------------------------------------------


def _has_four_pattern(mask: np.ndarray) -> bool:
    """Return ``True`` if a boolean board mask contains four-in-a-row.

    ``mask`` must be the ``(6, 7)`` boolean array obtained by comparing
    :attr:`PopOutBoard.board` against a player id. We check the four
    classic Connect-4 patterns with vectorised slicing — about 2x faster
    than the Python-loop implementation in :mod:`game.board`.
    """
    # Horizontal
    if (mask[:, :-3] & mask[:, 1:-2] & mask[:, 2:-1] & mask[:, 3:]).any():
        return True
    # Vertical
    if (mask[:-3, :] & mask[1:-2, :] & mask[2:-1, :] & mask[3:, :]).any():
        return True
    # Diagonal "\"
    if (mask[:-3, :-3] & mask[1:-2, 1:-2] & mask[2:-1, 2:-1] & mask[3:, 3:]).any():
        return True
    # Diagonal "/"
    if (mask[3:, :-3] & mask[2:-1, 1:-2] & mask[1:-2, 2:-1] & mask[:-3, 3:]).any():
        return True
    return False


def _fast_check_winner(state: PopOutBoard) -> Optional[int]:
    """Equivalent to :meth:`PopOutBoard.check_winner` but ~2x faster.

    Honours the "popper wins" tiebreaker (``rule 1`` in ``README.md``)
    via the engine's public :attr:`PopOutBoard.last_move_player` accessor.
    """
    board = state.board
    p1_wins = _has_four_pattern(board == 1)
    p2_wins = _has_four_pattern(board == 2)
    if p1_wins and p2_wins:
        return state.last_move_player
    if p1_wins:
        return 1
    if p2_wins:
        return 2
    return None


class MCTSNode:
    """A single node in the search tree.

    Attributes
    ----------
    state:
        Board state at this node. The constructor stores the board
        verbatim; callers must pass a freshly cloned :class:`PopOutBoard`.
    parent:
        Parent node, or ``None`` for the root.
    move:
        Move that produced ``state`` from the parent's state. ``None`` at
        the root.
    children:
        Children that have been expanded so far.
    untried_moves:
        Legal moves at ``state`` that have not yet produced a child node.
        Empty when ``state`` is terminal.
    visits:
        Number of times this node has been traversed by an iteration.
    value_sum:
        Sum of root-perspective rewards back-propagated through this node.
    terminal:
        ``-1`` while undetermined; otherwise the proven root-perspective
        value (``0``, ``0.5`` or ``1``). Once non-``-1`` the node is
        skipped in :meth:`MCTS._select`.
    undet_children:
        ``len(untried_moves) + |{c in children: c.terminal == -1}|``.
        Reaches zero exactly when every child has been resolved.
    root_player:
        The player whose perspective drives the reward sign. Stored on
        every node so :meth:`MCTS._propagate_terminal` does not need to
        thread it as an argument.
    """

    __slots__ = (
        "state",
        "parent",
        "move",
        "children",
        "untried_moves",
        "visits",
        "value_sum",
        "value_sum_sq",
        "terminal",
        "undet_children",
        "root_player",
    )

    def __init__(
        self,
        state: PopOutBoard,
        parent: Optional["MCTSNode"],
        move: Optional[Move],
        root_player: int,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.value_sum_sq: float = 0.0
        self.root_player = root_player

        # Determine terminal status and cache valid moves only once.
        # We avoid ``state.is_terminal()`` because it triggers
        # ``_has_four_in_a_row`` four times (twice via ``check_winner`` and
        # twice more via ``is_draw``); this hot path is the dominant cost in
        # MCTS construction, so we inline a leaner sequence here.
        winner = _fast_check_winner(state)
        if winner is not None:
            self.untried_moves: list[Move] = []
            self.terminal = TERMINAL_WIN if winner == root_player else TERMINAL_LOSS
        else:
            valid = state.get_valid_moves()
            if not valid:
                self.untried_moves = []
                self.terminal = TERMINAL_DRAW
            elif state.history.count(state.get_state_hash()) >= 3:
                self.untried_moves = []
                self.terminal = TERMINAL_DRAW
            else:
                self.untried_moves = valid
                self.terminal = TERMINAL_UNDETERMINED

        # No children yet, so undetermined-child count equals the number of
        # moves still to expand.
        self.undet_children: int = len(self.untried_moves)


class MCTS:
    """Vanilla Monte Carlo Tree Search with UCT and terminal proof propagation.

    Parameters
    ----------
    iterations:
        Number of selection-expansion-simulation-back-propagation cycles
        performed per call to :meth:`search`.
    exploration_weight:
        Constant ``c`` in the UCT formula.
    rollout_depth_limit:
        Hard cap on the number of plies played in a single rollout. When
        the limit is hit we return a draw value (``0.5``); without the
        cap PopOut games can loop forever via repeated drop/pop cycles.
    random_seed:
        Seed for the instance-local :class:`random.Random` used by both
        rollouts and untried-move expansion. ``None`` produces fresh
        non-reproducible behaviour.

    Notes
    -----
    The search uses a slightly enhanced terminal propagation scheme: a
    node is marked resolved as soon as a *forcing* child is found
    (the player to move can guarantee a win by playing that child),
    not only when every child has been resolved. This is the standard
    minimax shortcut and is strictly stronger than the all-children-resolved
    rule alone — the latter is still applied as the catch-all when no
    forcing child exists.

    UCT scoring uses ``child.value_sum / child.visits`` directly, as
    specified. Reward accounting is fixed to the root player's
    perspective and never flipped during back-propagation.

    The ``num_children_to_expand`` parameter controls how many children are
    created per visit to a fully-unexpanded node: ``1`` reproduces the
    standard MCTS behaviour, higher values create up to ``k`` children in
    one ``_expand`` call (capped by the remaining ``untried_moves``).

    The ``uct_variant`` parameter selects the selection formula:

    * ``"ucb1"`` (default) — classic UCT with a constant exploration weight.
    * ``"ucb1_tuned"`` — variance-aware bound of Auer, Cesa-Bianchi &
      Fischer (2002). The ``exploration_weight`` argument is ignored in
      this mode; the exploration term scales with the empirical reward
      variance instead.
    """

    def __init__(
        self,
        iterations: int = 1000,
        exploration_weight: float = math.sqrt(2),
        rollout_depth_limit: int = 80,
        random_seed: Optional[int] = None,
        use_numba_rollout: bool = True,
        num_children_to_expand: int = 1,
        uct_variant: str = "ucb1",
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if exploration_weight < 0:
            raise ValueError("exploration_weight must be non-negative")
        if rollout_depth_limit < 0:
            raise ValueError("rollout_depth_limit must be non-negative")
        if num_children_to_expand < 1:
            raise ValueError("num_children_to_expand must be >= 1")
        if uct_variant not in ("ucb1", "ucb1_tuned"):
            raise ValueError(
                f"uct_variant must be 'ucb1' or 'ucb1_tuned', got {uct_variant!r}"
            )

        self.iterations: int = iterations
        self.exploration_weight: float = exploration_weight
        self.rollout_depth_limit: int = rollout_depth_limit
        self._rng: random.Random = random.Random(random_seed)
        self._use_numba_rollout: bool = use_numba_rollout
        self.num_children_to_expand: int = num_children_to_expand
        self.uct_variant: str = uct_variant
        # Set per-search; exposed for tests that want to inspect the tree.
        self._last_root: Optional[MCTSNode] = None
        self._root_player: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, root_state: PopOutBoard) -> Move:
        """Run MCTS and return the recommended move for ``root_state``.

        Parameters
        ----------
        root_state:
            Board at which to plan. Not mutated.

        Returns
        -------
        Move
            ``(move_type, column)`` from ``root_state.get_valid_moves()``.

        Notes
        -----
        If the position has only one legal move we short-circuit and
        return it without running any iteration. ``self._last_root`` is
        set to ``None`` in that case.
        """
        moves = root_state.get_valid_moves()
        if len(moves) == 1:
            self._last_root = None
            return moves[0]

        self._root_player = root_state.current_player
        root = MCTSNode(
            state=root_state.clone(),
            parent=None,
            move=None,
            root_player=self._root_player,
        )
        self._last_root = root

        # Defensive: if the caller somehow passed a terminal state, just
        # return the first legal move.
        if root.terminal != TERMINAL_UNDETERMINED:
            return moves[0]

        for _ in range(self.iterations):
            if root.terminal != TERMINAL_UNDETERMINED:
                # Whole tree solved — further iterations cannot improve it.
                break

            leaf = self._select(root)

            # The leaf may either be terminal (skip), have untried moves
            # (expand), or be a fully expanded but non-resolved node where
            # selection terminated early because no undetermined children
            # remain. Only the middle case requires expansion.
            if leaf.terminal == TERMINAL_UNDETERMINED and leaf.untried_moves:
                leaf = self._expand(leaf)
                if leaf.terminal != TERMINAL_UNDETERMINED:
                    # The newly expanded child was terminal at birth; the
                    # statistics and propagation were handled inside
                    # :meth:`_expand` already. Nothing more to do.
                    continue

            if leaf.terminal != TERMINAL_UNDETERMINED:
                # Terminal leaf — counted via terminal propagation, not rollout.
                continue

            reward = self._simulate(leaf.state)
            self._backpropagate(leaf, reward)

        return self._best_move(root)

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend the tree until reaching a node to expand or simulate.

        We stop when:

        * the node is resolved (``terminal != -1``) — nothing to do;
        * the node still has untried moves — caller will expand;
        * the node has no undetermined children left — caller will skip
          (this branch is unreachable when terminal propagation works).
        """
        while True:
            if node.terminal != TERMINAL_UNDETERMINED:
                return node
            if node.untried_moves:
                return node

            undetermined = [
                c for c in node.children if c.terminal == TERMINAL_UNDETERMINED
            ]
            if not undetermined:
                # No children worth selecting; fall back to this node.
                return node

            node = max(
                undetermined,
                key=lambda child: self._uct_score(child, node),
            )

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Create up to ``self.num_children_to_expand`` new children of ``node``.

        With ``num_children_to_expand == 1`` exactly one child is created,
        reproducing the standard MCTS behaviour. With ``k > 1`` we create
        ``min(k, len(node.untried_moves))`` children in a single call. Each
        untried move is popped at a uniformly random index so UCT's
        unbiased exploration is preserved.

        For every child created we apply the same per-child logic as the
        single-expansion case: if the child's state is terminal at birth
        we back-propagate its proven value, decrement
        ``node.undet_children``, and attempt a terminal propagation
        upwards. If that propagation resolves ``node`` itself (because a
        forcing child was found) the loop stops — any further children
        would be wasted work in an already-decided subtree.

        Returns the first newly-created child that is not terminal-at-birth
        (so the caller can run a rollout from it). If every created child
        was terminal at birth, the last one is returned; its statistics
        have already been propagated by ``_backpropagate`` above and the
        caller's terminal-leaf branch will short-circuit the rollout.
        """
        if not node.untried_moves:
            # Defensive: nothing to expand. The search loop handles a
            # leaf with no rollout to do (terminal or no-children) without
            # crashing.
            return node

        k = self.num_children_to_expand
        created: list[MCTSNode] = []
        first_undetermined: Optional[MCTSNode] = None

        for _ in range(k):
            if not node.untried_moves:
                break

            # Random pop preserves UCT's unbiased exploration even when a
            # node has many untried moves.
            idx = self._rng.randrange(len(node.untried_moves))
            move = node.untried_moves.pop(idx)

            child_state = node.state.clone()
            child_state.make_move(*move)
            child = MCTSNode(
                state=child_state,
                parent=node,
                move=move,
                root_player=self._root_player,
            )
            node.children.append(child)
            created.append(child)

            if child.terminal != TERMINAL_UNDETERMINED:
                # Resolved at birth — propagate the proof through the stats
                # and update the parent's outstanding-children counter.
                self._backpropagate(child, child.terminal)
                node.undet_children -= 1
                self._try_resolve(node)
            else:
                # If undetermined: untried_moves dropped by 1 and
                # "undetermined children" grew by 1, so the parent's counter
                # is unchanged.
                if first_undetermined is None:
                    first_undetermined = child

            # If a forcing child resolved ``node`` itself, further expansion
            # would only waste work on a now-decided subtree.
            if node.terminal != TERMINAL_UNDETERMINED:
                break

        if first_undetermined is not None:
            return first_undetermined
        # All created children were terminal at birth; their stats were
        # already propagated, so the caller will skip the rollout.
        return created[-1]

    def _simulate(self, state: PopOutBoard) -> float:
        """Play random moves from ``state`` and return a root-perspective reward.

        Two implementations are available:

        * ``use_numba_rollout=True`` (default): delegate to the JIT-compiled
          :func:`ai.rollout_numba.simulate_rollout`. We translate the engine
          state into primitive arguments (a fresh copy of the board, the
          player ints, the depth limit and a seed drawn from ``self._rng``)
          and let the compiled function run the inner loop at native speed.
        * ``use_numba_rollout=False``: keep the original pure-Python loop
          for fall-back / cross-validation. Cap is :attr:`rollout_depth_limit`
          plies; if the limit is hit before a natural terminal, the
          position is treated as a draw (``0.5``). 3-fold-repetition draws
          are skipped — rare in 80-ply uniform-random rollouts and the most
          expensive check per ply.
        """
        if self._use_numba_rollout:
            last_mp = (
                state.last_move_player
                if state.last_move_player is not None
                else 0
            )
            seed = self._rng.randrange(2 ** 31 - 1)
            return float(
                simulate_rollout(
                    state.board.copy(),             # mutated in-place inside numba
                    int(state.current_player),
                    int(last_mp),
                    int(self._root_player),
                    int(self.rollout_depth_limit),
                    int(seed),
                )
            )

        # ---- Fallback: pure-Python rollout (original implementation) ----
        sim = state.clone()
        plies = 0
        root_player = self._root_player
        while plies < self.rollout_depth_limit:
            winner = _fast_check_winner(sim)
            if winner is not None:
                return TERMINAL_WIN if winner == root_player else TERMINAL_LOSS
            valid = sim.get_valid_moves()
            if not valid:
                return TERMINAL_DRAW
            move = self._rng.choice(valid)
            sim.make_move(*move)
            plies += 1

        # Depth cap reached without a natural terminal — score as a draw.
        return TERMINAL_DRAW

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Walk from ``node`` to the root adding ``reward`` to every ancestor.

        ``reward`` is always expressed in the root player's perspective
        (per :class:`MCTS` docs); we never flip its sign. The squared
        reward is accumulated alongside the running sum so UCB1-Tuned can
        recover the empirical variance in one O(1) step.
        """
        reward_sq = reward * reward
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current.value_sum_sq += reward_sq
            current = current.parent

    # ------------------------------------------------------------------
    # Terminal propagation
    # ------------------------------------------------------------------

    def _try_resolve(self, node: MCTSNode) -> None:
        """Attempt to mark ``node`` as resolved given its current children.

        Two triggers are checked:

        * **Forcing-child shortcut.** If the player to move at ``node``
          has a child whose proven value is the best they can hope for
          (``1`` for the root player, ``0`` for the opponent), the node
          is marked with that value immediately.
        * **All-children-resolved fallback.** Otherwise, when every child
          has been proven (``undet_children == 0``), the node's value is
          ``max`` (root player to move) or ``min`` (opponent to move) of
          the resolved children's terminal values.
        """
        if node.terminal != TERMINAL_UNDETERMINED:
            return

        is_root_turn = node.state.current_player == node.root_player
        forcing_value = TERMINAL_WIN if is_root_turn else TERMINAL_LOSS

        for child in node.children:
            if child.terminal == forcing_value:
                self._set_resolved(node, forcing_value)
                return

        if node.undet_children == 0:
            resolved_values = [c.terminal for c in node.children]
            value = max(resolved_values) if is_root_turn else min(resolved_values)
            self._set_resolved(node, value)

    def _set_resolved(self, node: MCTSNode, value: float) -> None:
        """Stamp ``node`` as resolved and recurse upwards."""
        node.terminal = value
        parent = node.parent
        if parent is not None:
            parent.undet_children -= 1
            self._try_resolve(parent)

    def _propagate_terminal(self, node: MCTSNode) -> None:
        """Public spelling of :meth:`_try_resolve`.

        The method name is part of the documented interface even though it
        is private; we keep the spec-mandated entry point and delegate to
        the internal implementation.
        """
        self._try_resolve(node)

    # ------------------------------------------------------------------
    # UCT scoring and final move selection
    # ------------------------------------------------------------------

    def _uct_score(self, child: MCTSNode, parent: MCTSNode) -> float:
        """UCT score from the perspective of the player to move at ``parent``.

        Rewards are stored in the root player's perspective. When the parent
        node is at the opponent's turn, the opponent maximises its own reward,
        which is ``1 - mean`` in the root-perspective accounting. The
        exploration term is symmetric and never flipped, regardless of
        variant.

        Dispatches on ``self.uct_variant``:

        * ``"ucb1"`` — classic UCT with the constant exploration weight
          ``self.exploration_weight``.
        * ``"ucb1_tuned"`` — variance-aware bound of Auer, Cesa-Bianchi &
          Fischer (2002). The exploration term scales with the empirical
          variance of the child's rewards, clamped above by ``1/4`` (the
          theoretical maximum variance of a reward in ``[0, 1]``). The
          ``exploration_weight`` argument is ignored in this mode.
        """
        if child.visits == 0:
            return math.inf

        mean = child.value_sum / child.visits
        if parent.state.current_player == parent.root_player:
            exploitation = mean
        else:
            exploitation = 1.0 - mean

        log_parent_visits = math.log(parent.visits)

        if self.uct_variant == "ucb1":
            exploration = self.exploration_weight * math.sqrt(
                log_parent_visits / child.visits
            )
        else:  # "ucb1_tuned"
            # Empirical variance of the child's rewards (clamped non-negative
            # to absorb floating-point noise from very small values).
            mean_sq = child.value_sum_sq / child.visits
            variance = max(0.0, mean_sq - mean * mean)
            # Variance bound term from the paper.
            v_bound = variance + math.sqrt(
                2.0 * log_parent_visits / child.visits
            )
            # The 1/4 cap reflects the maximum variance of a reward in [0, 1].
            exploration = math.sqrt(
                log_parent_visits / child.visits * min(0.25, v_bound)
            )

        return exploitation + exploration

    def _best_move(self, root: MCTSNode) -> Move:
        """Pick the recommended move from the root's expanded children.

        Selection order:

        1. A child with ``terminal == 1`` (proven win) wins outright.
        2. Otherwise, the child with the highest visit count, ties broken
           by the highest mean value (``value_sum / visits``).
        3. The "all children proven losses" rule from the spec is a
           special case of (2): every child has the same terminal proof
           in the statistics, so the most-visited one wins anyway.
        """
        children = root.children
        if not children:
            # Pathological case — no iterations completed. Fall back to
            # any legal move so we never raise on an unexpected input.
            valid = root.state.get_valid_moves()
            return valid[0]

        for child in children:
            if child.terminal == TERMINAL_WIN:
                return child.move  # type: ignore[return-value]

        best = max(
            children,
            key=lambda c: (c.visits, c.value_sum / c.visits if c.visits else 0.0),
        )
        return best.move  # type: ignore[return-value]
