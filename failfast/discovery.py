"""Discover pytest test node IDs in a target repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def discover_tests(
    repo: Path,
    *,
    paths: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Return all pytest node IDs found in *repo* without running them.

    Uses ``pytest --collect-only -q`` under the hood. The process is spawned
    with the repository root as the working directory so that node IDs are
    relative to that root (e.g. ``tests/test_foo.py::test_bar``).

    Args:
        repo: Absolute path to the repository root.
        paths: Optional list of sub-paths (relative to *repo*) to restrict
            collection. Defaults to letting pytest discover everything.
        extra_args: Any additional arguments forwarded to pytest verbatim
            (e.g. ``["-m", "not slow"]``).

    Returns:
        A deduplicated, ordered list of pytest node ID strings.

    Raises:
        subprocess.CalledProcessError: If pytest exits with a status code
            other than 0 (tests found) or 5 (no tests collected).
        FileNotFoundError: If ``pytest`` is not on PATH.
    """
    cmd: list[str] = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-o",
        "addopts=",
        "-q",
        "--no-header",
    ]
    if paths:
        cmd.extend(paths)
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
    )

    # Exit code 5 means "no tests collected" – treat as empty list.
    if result.returncode not in (0, 5):
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    node_ids: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # pytest -q outputs node IDs like "tests/test_x.py::test_y"
        # Skip summary lines (they contain spaces but no "::")
        if "::" in stripped and not stripped.startswith("no tests ran"):
            node_ids.append(stripped)

    # Preserve order but remove any accidental duplicates
    seen: set[str] = set()
    unique: list[str] = []
    for nid in node_ids:
        if nid not in seen:
            seen.add(nid)
            unique.append(nid)

    return unique


def node_id_to_file(node_id: str) -> str:
    """Extract the file path component from a pytest node ID.

    Args:
        node_id: A pytest node ID such as ``tests/test_foo.py::TestClass::test_bar``.

    Returns:
        The file path part (``tests/test_foo.py``).
    """
    return node_id.split("::")[0]
