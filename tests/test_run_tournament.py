"""Tests for the MCTS tournament runner."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_tournament", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def _tiny_config(tmp_path: Path) -> Path:
    cfg = {
        "default_games": 4,
        "matchups": [
            {
                "name": "smoke_a_vs_b",
                "player_a": {
                    "iterations": 20, "exploration_weight": 1.4142135624,
                    "rollout_depth_limit": 80, "num_children_to_expand": 1,
                    "uct_variant": "ucb1",
                },
                "player_b": {
                    "iterations": 20, "exploration_weight": 1.4142135624,
                    "rollout_depth_limit": 80, "num_children_to_expand": 1,
                    "uct_variant": "ucb1",
                },
            },
        ],
    }
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(cfg))
    return p


def test_runs_end_to_end_and_writes_csv(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    out = tmp_path / "out.csv"
    proc = _run([
        str(cfg), "--workers", "2",
        "--out", str(out), "--base-seed", "0",
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["matchup_name"] == "smoke_a_vs_b"
    assert int(row["total_games"]) == 4
    assert int(row["a_wins"]) + int(row["b_wins"]) + int(row["draws"]) == 4


def test_starts_alternate_50_50(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    out = tmp_path / "out.csv"
    _run([str(cfg), "--workers", "2", "--out", str(out)], timeout=240).check_returncode()
    row = next(csv.DictReader(out.open()))
    a_as_p1 = int(row["a_wins_as_p1"]) + int(row["b_wins_as_p2"]) + int(row["draws_p1_was_a"])
    a_as_p2 = int(row["a_wins_as_p2"]) + int(row["b_wins_as_p1"]) + int(row["draws_p1_was_b"])
    # 4 games total, exactly 2 each side.
    assert a_as_p1 == 2, f"A played as P1 in {a_as_p1}/4 games"
    assert a_as_p2 == 2, f"A played as P2 in {a_as_p2}/4 games"


def test_rejects_odd_games(tmp_path: Path) -> None:
    cfg = {
        "matchups": [{
            "name": "odd",
            "player_a": {"iterations": 10, "exploration_weight": 1.41,
                         "rollout_depth_limit": 80, "num_children_to_expand": 1,
                         "uct_variant": "ucb1"},
            "player_b": {"iterations": 10, "exploration_weight": 1.41,
                         "rollout_depth_limit": 80, "num_children_to_expand": 1,
                         "uct_variant": "ucb1"},
            "games": 5,
        }],
    }
    cfg_path = tmp_path / "odd.json"
    cfg_path.write_text(json.dumps(cfg))
    proc = _run([str(cfg_path), "--workers", "2", "--out", str(tmp_path / "x.csv")])
    assert proc.returncode != 0
    assert "even" in (proc.stdout + proc.stderr).lower()


def test_determinism_same_base_seed(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _run([str(cfg), "--workers", "2", "--out", str(a), "--base-seed", "42"]).check_returncode()
    _run([str(cfg), "--workers", "2", "--out", str(b), "--base-seed", "42"]).check_returncode()
    assert a.read_bytes() == b.read_bytes(), "tournament CSV must be identical for same base seed"
