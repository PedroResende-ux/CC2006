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


def test_smoke_passes() -> None:
    """End-to-end smoke run must exit 0 and print PASS."""
    proc = _run(["--smoke"], timeout=240)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    combined = proc.stdout + proc.stderr
    assert "[smoke] PASS" in combined


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
