"""Synthetic training data generator for FailFast.

**This module produces synthetic (fabricated) data only.**

It creates a small, self-contained Python project in a temporary directory,
injects faults by mutating source lines, runs the test suite, and records the
outcomes to a Store.  The resulting dataset is useful for verifying the feature
pipeline end-to-end and for demos when real CI history is unavailable.

Synthetic data limitations
--------------------------
- Faults are trivial source mutations (line replacements) that do not
  reflect real-world failure distributions.
- Coverage maps from these micro-projects are not representative of large
  codebases.
- **Never** use this data to make claims about real-world FailFast performance.
  All output files are tagged ``synthetic=True`` in their manifests and carry
  an ``is_synthetic=1`` column.

Usage (command line)::

    failfast simulate --commits 20 --out data/synthetic

Usage (Python)::

    from failfast.simulate import run_simulation
    run_simulation(output_dir=Path("data/synthetic"), n_commits=20)
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from failfast.coverage_map import build_coverage_map
from failfast.dataset import build_labeled_dataset, save_dataset
from failfast.diff import parse_diff_text
from failfast.features.builder import build_features
from failfast.store import Store

# ---------------------------------------------------------------------------
# Fixture project template
# ---------------------------------------------------------------------------

_MODULE_SRC = '''\
"""Sample module for FailFast synthetic data generation."""


def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Return a - b."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Return a * b."""
    return a * b


def is_even(n: int) -> bool:
    """Return True if n is even."""
    return n % 2 == 0
'''

_TEST_SRC = '''\
"""Tests for the sample module."""
from sample import add, subtract, multiply, is_even


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(3, 4) == 12


def test_is_even_true():
    assert is_even(4) is True


def test_is_even_false():
    assert is_even(3) is False
'''

_MUTATIONS: list[tuple[str, str]] = [
    ("return a + b", "return a - b"),
    ("return a - b", "return a + b"),
    ("return a * b", "return a + b"),
    ("return n % 2 == 0", "return n % 2 == 1"),
]


def _write_fixture(project: Path) -> None:
    """Write the sample module and its tests into *project*."""
    (project / "sample.py").write_text(_MODULE_SRC, encoding="utf-8")
    tests_dir = project / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_sample.py").write_text(_TEST_SRC, encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n",
        encoding="utf-8",
    )


def _apply_mutation(project: Path, mutation: tuple[str, str]) -> str:
    """Replace *mutation[0]* with *mutation[1]* in sample.py and return a fake diff."""
    src_file = project / "sample.py"
    original = src_file.read_text(encoding="utf-8")
    mutated = original.replace(mutation[0], mutation[1], 1)
    src_file.write_text(mutated, encoding="utf-8")

    # Build a minimal unified diff string so parse_diff_text can process it
    old_lines = original.splitlines(keepends=True)
    new_lines = mutated.splitlines(keepends=True)

    changed_lineno = next(
        (i + 1 for i, (o, n) in enumerate(zip(old_lines, new_lines)) if o != n),
        1,
    )

    diff_text = (
        "--- a/sample.py\n"
        "+++ b/sample.py\n"
        f"@@ -{changed_lineno},1 +{changed_lineno},1 @@\n"
        f"-{old_lines[changed_lineno - 1]}"
        f"+{new_lines[changed_lineno - 1]}"
    )
    return diff_text


def _reset_module(project: Path) -> None:
    """Restore sample.py to its original content."""
    (project / "sample.py").write_text(_MODULE_SRC, encoding="utf-8")


def _run_tests(project: Path, store: Store, run_id: int) -> dict[str, str]:
    """Run pytest in *project* and record outcomes to *store*.

    Returns:
        Mapping from node ID to outcome string.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--tb=no",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=project,
        capture_output=True,
        text=True,
    )
    outcomes: dict[str, str] = {}

    for line in result.stdout.splitlines():
        line = line.strip()
        if " PASSED" in line or " FAILED" in line or " ERROR" in line:
            parts = line.split()
            if parts:
                node = parts[0]
                if "PASSED" in line:
                    outcome = "passed"
                elif "FAILED" in line:
                    outcome = "failed"
                else:
                    outcome = "error"
                outcomes[node] = outcome
                store.upsert_result(run_id, node, outcome, duration_s=0.1)

    # If verbose output didn't parse, fall back to exit code
    if not outcomes:
        if result.returncode == 0:
            default = "passed"
        else:
            default = "failed"
        # Read node ids by discovery
        disc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header"],
            cwd=project,
            capture_output=True,
            text=True,
        )
        for line in disc.stdout.splitlines():
            line = line.strip()
            if "::" in line and not line.startswith("="):
                outcomes[line] = default
                store.upsert_result(run_id, line, default, duration_s=0.1)

    return outcomes


def run_simulation(
    output_dir: Path,
    *,
    n_commits: int = 20,
    seed: int = 42,
    store_path: Path | None = None,
) -> Path:
    """Generate a synthetic labeled dataset.

    Creates a temporary fixture project, applies random mutations across
    *n_commits* rounds, runs the test suite after each mutation, collects
    features, and assembles a labeled DataFrame.

    Args:
        output_dir: Directory where the dataset will be written.
        n_commits: Number of synthetic commits (mutation rounds) to simulate.
        seed: Random seed for reproducibility.
        store_path: Path to the SQLite database.  Uses an in-memory store when
            ``None``.

    Returns:
        Path to the saved dataset file.

    Note:
        All output is tagged synthetic.  See module docstring.
    """
    import pandas as pd

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path: Path | str = store_path or ":memory:"
    store = Store(db_path)

    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "fixture"
        project.mkdir()
        _write_fixture(project)

        # Build the coverage map once on the clean project
        try:
            cov_entries = build_coverage_map(project, source="sample")
        except Exception:
            cov_entries = []

        import_graph_cache = None  # built lazily on first use

        all_rows: list[pd.DataFrame] = []
        base_ts = time.time() - (n_commits * 60)

        for i in range(n_commits):
            commit_sha = f"synthetic_{i:04d}"
            commit_ts = base_ts + i * 60
            mutation = rng.choice(_MUTATIONS)

            diff_text = _apply_mutation(project, mutation)
            changed_files = parse_diff_text(diff_text)

            run_id = store._conn.execute(
                "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
                (commit_sha, "HEAD~1", commit_ts),
            ).lastrowid
            store._conn.commit()

            outcomes = _run_tests(project, store, run_id)

            node_ids = list(outcomes.keys())
            if not node_ids:
                _reset_module(project)
                continue

            # Build features using history strictly before this commit
            try:
                feat_df = build_features(
                    project,
                    node_ids,
                    cov_entries,
                    changed_files,
                    store,
                    before_timestamp=commit_ts,
                    import_graph=import_graph_cache,
                )

                labeled = build_labeled_dataset(
                    feat_df, outcomes, is_synthetic=True
                )
                labeled["commit_sha"] = commit_sha
                labeled["commit_timestamp"] = commit_ts
                all_rows.append(labeled.reset_index())
            except Exception:
                pass

            _reset_module(project)

    store.close()

    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    saved_path = save_dataset(
        full_df.set_index("node_id") if not full_df.empty and "node_id" in full_df.columns else full_df,
        output_dir,
        "synthetic",
        is_synthetic=True,
    )
    return saved_path
