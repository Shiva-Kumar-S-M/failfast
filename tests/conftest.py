"""Shared pytest fixtures for the FailFast test suite."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def mini_repo(tmp_path: Path) -> Path:
    """Create a minimal git repository with a tiny source file and test.

    Returns:
        Path to the repository root.
    """
    # Initialise git repo
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_path, check=True, capture_output=True)

    # Source file
    src = tmp_path / "mylib.py"
    src.write_text(
        textwrap.dedent("""\
            def add(a, b):
                return a + b

            def subtract(a, b):
                return a - b
        """),
        encoding="utf-8",
    )

    # Test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_mylib.py").write_text(
        textwrap.dedent("""\
            from mylib import add, subtract

            def test_add():
                assert add(1, 2) == 3

            def test_subtract():
                assert subtract(5, 3) == 2
        """),
        encoding="utf-8",
    )

    # Initial commit
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path
