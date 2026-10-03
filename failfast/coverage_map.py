"""Build a mapping from each test to the source files and lines it executes.

Uses ``coverage.py`` with *dynamic context* support to collect per-test
coverage in a single pytest run, avoiding the overhead of running the suite
N times.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CoverageEntry:
    """Coverage information for one test.

    Attributes:
        node_id: The pytest node ID.
        executed_lines: Mapping from file path (relative to repo root) to the
            set of line numbers executed by this test.
    """

    node_id: str
    executed_lines: dict[str, set[int]] = field(default_factory=dict)


def build_coverage_map(
    repo: Path,
    *,
    source: str | None = None,
    extra_args: list[str] | None = None,
) -> list[CoverageEntry]:
    """Run the test suite with coverage and return a per-test coverage map.

    Internally this calls ``python -m pytest --cov=<source> --cov-context=test``
    to collect coverage data, then reads the ``.coverage`` database using
    ``coverage json`` to extract per-context (per-test) execution data.

    Args:
        repo: Absolute path to the repository root.
        source: Python package or directory to measure. Defaults to the repo
            root, which measures all source files.
        extra_args: Additional arguments forwarded to pytest verbatim
            (e.g. ``["-m", "unit"]``).

    Returns:
        A list of :class:`CoverageEntry` objects, one per test that actually
        ran. Tests that were collected but skipped may be absent.

    Raises:
        subprocess.CalledProcessError: If pytest or ``coverage json`` fails.
        RuntimeError: If ``pytest-cov`` is not installed.
    """
    src = source or str(repo)

    with tempfile.TemporaryDirectory() as tmp:
        cov_file = Path(tmp) / ".coverage"
        json_file = Path(tmp) / "coverage.json"

        # Step 1: Run pytest with coverage contexts enabled
        cmd: list[str] = [
            "python",
            "-m",
            "pytest",
            f"--cov={src}",
            "--cov-context=test",
            f"--cov-config={_inline_cov_config(cov_file)}",
            "--no-header",
            "-q",
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd,
            cwd=repo,
            capture_output=True,
            text=True,
        )
        # Exit codes 0 (all pass) and 1 (some fail) are both fine
        if result.returncode not in (0, 1, 5):
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

        if not cov_file.exists():
            raise RuntimeError(
                "Coverage data file was not created. "
                "Is pytest-cov installed in the target repo's environment?"
            )

        # Step 2: Export to JSON so we can parse context-aware data
        export_result = subprocess.run(
            [
                "python",
                "-m",
                "coverage",
                "json",
                "--data-file",
                str(cov_file),
                "-o",
                str(json_file),
                "--pretty-print",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if export_result.returncode != 0:
            raise subprocess.CalledProcessError(
                export_result.returncode,
                ["coverage", "json"],
                output=export_result.stdout,
                stderr=export_result.stderr,
            )

        return _parse_json_report(json_file, repo)


def _inline_cov_config(data_file: Path) -> str:
    """Write a minimal .coveragerc and return its path as a string.

    We need to point coverage at a temp file so it doesn't clobber any
    existing ``.coverage`` in the repo root.

    Args:
        data_file: The path where coverage should write its data.

    Returns:
        Path string of the written config file.
    """
    import os
    import tempfile as tf

    fd, path = tf.mkstemp(suffix=".ini", prefix="failfast_cov_")
    os.close(fd)
    with open(path, "w") as fh:
        fh.write(f"[run]\ndata_file = {data_file}\ncontext = test\n")
    return path


def _parse_json_report(json_file: Path, repo: Path) -> list[CoverageEntry]:
    """Extract per-test coverage entries from a ``coverage json`` output file.

    The JSON structure produced by coverage.py (with contexts) looks like::

        {
          "files": {
            "path/to/module.py": {
              "contexts": {
                "tests/test_foo.py::test_bar|run": [10, 11, 12],
                ...
              }
            }
          }
        }

    Args:
        json_file: Path to the JSON report.
        repo: Repository root (used to make paths relative).

    Returns:
        One :class:`CoverageEntry` per test node ID found in the data.
    """
    with json_file.open(encoding="utf-8") as fh:
        data = json.load(fh)

    entries: dict[str, CoverageEntry] = {}

    for file_path, file_data in data.get("files", {}).items():
        contexts: dict[str, list[int]] = file_data.get("contexts", {})

        # Make the file path relative to the repo root when possible
        try:
            rel_path = str(Path(file_path).relative_to(repo))
        except ValueError:
            rel_path = file_path

        for context_key, lines in contexts.items():
            # coverage.py appends "|run" or "|<phase>" to context names
            node_id = context_key.split("|")[0]
            if not node_id or node_id == "":
                continue

            if node_id not in entries:
                entries[node_id] = CoverageEntry(node_id=node_id)

            if lines:
                entries[node_id].executed_lines.setdefault(rel_path, set()).update(
                    lines
                )

    return list(entries.values())


def filter_by_changed_files(
    coverage_map: list[CoverageEntry],
    changed_paths: set[str],
) -> list[str]:
    """Return node IDs whose coverage overlaps with *changed_paths*.

    Args:
        coverage_map: The output of :func:`build_coverage_map`.
        changed_paths: Set of repository-relative file paths that changed.

    Returns:
        Ordered list of pytest node IDs that touch at least one changed file.
    """
    result: list[str] = []
    for entry in coverage_map:
        covered = set(entry.executed_lines.keys())
        if covered & changed_paths:
            result.append(entry.node_id)
    return result
