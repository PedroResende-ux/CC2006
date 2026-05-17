"""Smoke test for ``scripts.play`` — the CLI for the three game modes.

Only Random-vs-Random is exercised here. Human cannot be tested in CI
(needs stdin), and MCTS/DT have dedicated test files of their own.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_play_random_vs_random_smoke():
    """``python -m scripts.play --p1 random --p2 random`` exits cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.play", "--p1", "random", "--p2", "random"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"crashed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Result" in result.stdout, (
        f"expected 'Result' in stdout, got:\n{result.stdout}"
    )
