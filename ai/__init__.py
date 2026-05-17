"""AI package: MCTS engine and ID3 decision-tree pipeline for PopOut.

Public modules:

* :mod:`ai.mcts`         — MCTS engine with UCT/UCB1-Tuned variants.
* :mod:`ai.mcts_player`  — :class:`game.player.Player` adapter for MCTS.
* :mod:`ai.rollout_numba`— Numba-accelerated rollout (used by MCTS).
* :mod:`ai.id3`          — Pure ID3 algorithm (entropy, info-gain, train, predict).
* :mod:`ai.dt_pipeline`  — PopOut-specific ID3 pipeline + Iris warm-up + CLI.
* :mod:`ai.dt_player`    — :class:`game.player.Player` adapter for the
                            trained decision tree (stub; pending follow-up).
"""
