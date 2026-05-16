"""Tests for the dataset generator."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    """Invoke ``python -m scripts.generate_dataset ...`` and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.generate_dataset", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_schema_matches_spec(tmp_path: Path) -> None:
    """A small real run produces a CSV whose header matches the spec verbatim."""
    out = tmp_path / "test.csv"
    proc = _run(
        [
            "--games", "5", "--iterations", "100",
            "--workers", "2", "--opening-max", "4",
            "--out", str(out), "--base-seed", "1",
        ],
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    with out.open() as f:
        header = next(csv.reader(f))
    cell_cols = [f"cell_{r}{c}" for r in range(6) for c in range(7)]
    expected = (
        cell_cols
        + ["move_count", "current_player",
           "own_pieces_bottom_row", "opp_pieces_bottom_row",
           "class"]
    )
    assert header == expected, f"header diff:\n{header}\nvs\n{expected}"


def test_class_values_are_valid(tmp_path: Path) -> None:
    """All labelled class values must match drop_0..drop_6 or pop_0..pop_6."""
    out = tmp_path / "test.csv"
    proc = _run(
        [
            "--games", "5", "--iterations", "100",
            "--workers", "2", "--opening-max", "4",
            "--out", str(out), "--base-seed", "2",
        ],
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    pattern = re.compile(r"^(drop|pop)_[0-6]$")
    with out.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            assert pattern.match(row["class"]), f"row {i}: bad class={row['class']!r}"


def test_determinism_with_same_seed(tmp_path: Path) -> None:
    """Two runs with the same flags must produce identical CSVs."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    common = [
        "--games", "5", "--iterations", "100",
        "--workers", "2", "--opening-max", "4",
        "--base-seed", "42",
    ]
    pa = _run(common + ["--out", str(a)], timeout=240)
    pb = _run(common + ["--out", str(b)], timeout=240)
    assert pa.returncode == 0, pa.stderr
    assert pb.returncode == 0, pb.stderr
    assert a.read_bytes() == b.read_bytes(), "CSVs differ for same seed"


def test_rejects_overwrite_without_append_flag(tmp_path: Path) -> None:
    """Running twice into the same path without --append must fail."""
    out = tmp_path / "test.csv"
    common = [
        "--games", "2", "--iterations", "50",
        "--workers", "2", "--opening-max", "2",
        "--out", str(out),
    ]
    first = _run(common, timeout=240)
    assert first.returncode == 0, first.stderr
    second = _run(common, timeout=240)
    assert second.returncode != 0
    combined = second.stdout + second.stderr
    assert "exists" in combined.lower() or "abort" in combined.lower()


# ---------------------------------------------------------------------------
# Prompt F: hyperparameter propagation + chunk-range support
# ---------------------------------------------------------------------------


def test_backward_compat(tmp_path: Path) -> None:
    """Legacy CLI (only --games + --iterations) still produces a valid CSV."""
    out = tmp_path / "bc.csv"
    proc = _run(
        [
            "--games", "3", "--iterations", "100",
            "--workers", "1", "--base-seed", "0",
            "--out", str(out),
        ],
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    with out.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows, "no rows produced"
    # We cannot recover game_index from rows (not in schema), but every
    # game contributes >=1 labelled row in practice. Check enough rows.
    assert len(rows) >= 3, f"expected at least 3 rows, got {len(rows)}"


def test_chunk_bit_identical(tmp_path: Path) -> None:
    """Single run vs. two chunked appends must produce byte-equal CSVs."""
    single = tmp_path / "single.csv"
    chunk = tmp_path / "chunk.csv"
    common = [
        "--base-seed", "42", "--iterations", "200",
        "--workers", "1", "--opening-max", "4",
    ]

    # Single run
    p_single = _run(
        common + ["--games", "6", "--out", str(single)],
        timeout=240,
    )
    assert p_single.returncode == 0, p_single.stderr

    # Chunked: first half
    p_a = _run(
        common + [
            "--start-game", "0", "--end-game", "3",
            "--out", str(chunk),
        ],
        timeout=240,
    )
    assert p_a.returncode == 0, p_a.stderr

    # Chunked: second half (appended)
    p_b = _run(
        common + [
            "--start-game", "3", "--end-game", "6",
            "--out", str(chunk), "--append",
        ],
        timeout=240,
    )
    assert p_b.returncode == 0, p_b.stderr

    assert single.read_bytes() == chunk.read_bytes(), (
        "Chunked output is not byte-identical to a single run"
    )


def test_exploration_weight_propagates(tmp_path: Path) -> None:
    """Changing --exploration-weight must yield a different CSV."""
    a = tmp_path / "ew_low.csv"
    b = tmp_path / "ew_high.csv"
    common = [
        "--games", "2", "--iterations", "200",
        "--workers", "1", "--opening-max", "4",
        "--base-seed", "7",
    ]
    pa = _run(common + ["--exploration-weight", "0.5", "--out", str(a)], timeout=240)
    pb = _run(common + ["--exploration-weight", "5.0", "--out", str(b)], timeout=240)
    assert pa.returncode == 0, pa.stderr
    assert pb.returncode == 0, pb.stderr
    assert a.read_bytes() != b.read_bytes(), (
        "exploration_weight does not seem to affect MCTS choices"
    )


def test_uct_variant_propagates(tmp_path: Path) -> None:
    """Changing --uct-variant must yield a different CSV."""
    a = tmp_path / "ucb1.csv"
    b = tmp_path / "tuned.csv"
    common = [
        "--games", "2", "--iterations", "200",
        "--workers", "1", "--opening-max", "4",
        "--base-seed", "11",
    ]
    pa = _run(common + ["--uct-variant", "ucb1", "--out", str(a)], timeout=240)
    pb = _run(common + ["--uct-variant", "ucb1_tuned", "--out", str(b)], timeout=240)
    assert pa.returncode == 0, pa.stderr
    assert pb.returncode == 0, pb.stderr
    assert a.read_bytes() != b.read_bytes(), (
        "uct_variant does not seem to affect MCTS choices"
    )


def test_num_children_propagates(tmp_path: Path) -> None:
    """Changing --num-children-to-expand must yield a different CSV."""
    a = tmp_path / "k1.csv"
    b = tmp_path / "k7.csv"
    common = [
        "--games", "2", "--iterations", "200",
        "--workers", "1", "--opening-max", "4",
        "--base-seed", "13",
    ]
    pa = _run(common + ["--num-children-to-expand", "1", "--out", str(a)], timeout=240)
    pb = _run(common + ["--num-children-to-expand", "7", "--out", str(b)], timeout=240)
    assert pa.returncode == 0, pa.stderr
    assert pb.returncode == 0, pb.stderr
    assert a.read_bytes() != b.read_bytes(), (
        "num_children_to_expand does not seem to affect MCTS choices"
    )


def test_invalid_range_rejected(tmp_path: Path) -> None:
    """--start-game >= --end-game must be rejected with SystemExit."""
    out = tmp_path / "bad.csv"
    proc = _run(
        [
            "--start-game", "100", "--end-game", "50",
            "--iterations", "50", "--workers", "1",
            "--out", str(out),
        ],
        timeout=60,
    )
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "end-game" in combined or "start-game" in combined


def test_invalid_uct_variant_rejected(tmp_path: Path) -> None:
    """argparse must reject an unknown --uct-variant value."""
    out = tmp_path / "bad.csv"
    proc = _run(
        [
            "--games", "1", "--iterations", "50", "--workers", "1",
            "--uct-variant", "foo",
            "--out", str(out),
        ],
        timeout=60,
    )
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "uct-variant" in combined or "invalid choice" in combined


def test_games_endgame_consistency_check(tmp_path: Path) -> None:
    """--games inconsistent with start/end-game must be rejected."""
    out = tmp_path / "bad.csv"
    proc = _run(
        [
            "--games", "5", "--start-game", "0", "--end-game", "10",
            "--iterations", "50", "--workers", "1",
            "--out", str(out),
        ],
        timeout=60,
    )
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "inconsistent" in combined or "implied" in combined


def test_help_shows_only_expected_flags() -> None:
    """The CLI surface must stay minimal: only the agreed flags appear in --help.

    Guards against re-introducing --depth-limit, --smoke, or --progress-every
    after their removal, and confirms every kept flag is still wired up.
    """
    # Import lazily so pytest collection doesn't fail if the module is broken.
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.generate_dataset import _build_parser

    help_text = _build_parser().format_help()

    for removed in ("--depth-limit", "--smoke", "--progress-every"):
        assert removed not in help_text, (
            f"flag {removed!r} reappeared in --help — it was supposed to be hardcoded/removed"
        )

    expected_flags = [
        "--games",
        "--iterations",
        "--exploration-weight",
        "--num-children-to-expand",
        "--uct-variant",
        "--start-game",
        "--end-game",
        "--base-seed",
        "--workers",
        "--eps",
        "--opening-max",
        "--out",
        "--append",
    ]
    for flag in expected_flags:
        assert flag in help_text, f"expected flag {flag!r} missing from --help"
