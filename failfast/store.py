"""SQLite store for FailFast run history."""

from __future__ import annotations

import importlib.resources
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Public data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """A single FailFast run stored in the database.

    Attributes:
        id: Auto-assigned primary key.
        commit_sha: The HEAD commit of the repo when the run was executed.
        base_sha: The base ref used for the diff comparison.
        created_at: Unix timestamp (float) when the run was recorded.
    """

    id: int
    commit_sha: str
    base_sha: str
    created_at: float


@dataclass
class TestRecord:
    """A unique test identifier stored in the database.

    Attributes:
        id: Auto-assigned primary key.
        node_id: The pytest node ID (e.g. ``tests/test_foo.py::test_bar``).
    """

    __test__ = False

    id: int
    node_id: str


@dataclass
class ResultRecord:
    """The outcome of one test in one run.

    Attributes:
        id: Auto-assigned primary key.
        run_id: Foreign key to :class:`RunRecord`.
        test_id: Foreign key to :class:`TestRecord`.
        outcome: One of ``passed``, ``failed``, ``error``, ``skipped``,
            ``xfailed``, or ``xpassed``.
        duration_s: Wall-clock seconds the test took to complete.
    """

    id: int
    run_id: int
    test_id: int
    outcome: str
    duration_s: float


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA_PACKAGE = "failfast"
_SCHEMA_FILE = "schema.sql"


def _load_schema() -> str:
    """Return the SQL DDL from the bundled schema.sql."""
    ref = importlib.resources.files(_SCHEMA_PACKAGE).joinpath(_SCHEMA_FILE)
    return ref.read_text(encoding="utf-8")


class Store:
    """Thin wrapper around an SQLite database for FailFast history.

    The store is intentionally minimal: it persists run metadata and
    per-test outcomes and exposes just enough queries for feature extraction
    in Day 2.

    Args:
        db_path: Path to the SQLite file. Use ``":memory:"`` for an in-memory
            database (useful in tests).

    Example::

        store = Store(Path("failfast.db"))
        run_id = store.create_run("abc123", "main")
        store.upsert_result(run_id, "tests/test_foo.py::test_bar", "passed", 0.42)
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_schema(self) -> None:
        """Create tables if they do not exist."""
        ddl = _load_schema()
        with self._conn:
            self._conn.executescript(ddl)

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Cursor, None, None]:
        """Yield a cursor inside an explicit transaction."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(self, commit_sha: str, base_sha: str) -> int:
        """Insert a new run and return its primary key.

        Args:
            commit_sha: The HEAD commit SHA at the time of the run.
            base_sha: The base ref used for the diff.

        Returns:
            The auto-assigned ``run.id``.
        """
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
                (commit_sha, base_sha, now),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_run(self, run_id: int) -> RunRecord | None:
        """Fetch a run by primary key.

        Args:
            run_id: The primary key to look up.

        Returns:
            A :class:`RunRecord` or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT id, commit_sha, base_sha, created_at FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            id=row["id"],
            commit_sha=row["commit_sha"],
            base_sha=row["base_sha"],
            created_at=row["created_at"],
        )

    def list_runs(self, *, limit: int = 100) -> list[RunRecord]:
        """Return the most recent runs, newest first.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of :class:`RunRecord`, ordered by descending ``created_at``.
        """
        rows = self._conn.execute(
            "SELECT id, commit_sha, base_sha, created_at FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            RunRecord(
                id=r["id"],
                commit_sha=r["commit_sha"],
                base_sha=r["base_sha"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def _get_or_create_test(self, node_id: str) -> int:
        """Return the ``test.id`` for *node_id*, inserting it if needed.

        Args:
            node_id: Pytest node ID string.

        Returns:
            The integer primary key.
        """
        with self._tx() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO tests (node_id) VALUES (?)",
                (node_id,),
            )
        row = self._conn.execute(
            "SELECT id FROM tests WHERE node_id = ?", (node_id,)
        ).fetchone()
        return row["id"]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def upsert_result(
        self,
        run_id: int,
        node_id: str,
        outcome: str,
        duration_s: float = 0.0,
    ) -> int:
        """Record (or update) the outcome of *node_id* in *run_id*.

        If a result for this ``(run_id, test_id)`` pair already exists it is
        replaced, which makes the method idempotent for re-runs.

        Args:
            run_id: The run's primary key.
            node_id: Pytest node ID of the test.
            outcome: One of the allowed outcome strings.
            duration_s: Wall-clock seconds the test took.

        Returns:
            The primary key of the inserted or replaced result row.
        """
        test_id = self._get_or_create_test(node_id)
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO results (run_id, test_id, outcome, duration_s)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, test_id) DO UPDATE SET
                    outcome = excluded.outcome,
                    duration_s = excluded.duration_s
                """,
                (run_id, test_id, outcome, duration_s),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_results_for_run(self, run_id: int) -> list[ResultRecord]:
        """Return all test results for a given run.

        Args:
            run_id: The run's primary key.

        Returns:
            List of :class:`ResultRecord` objects.
        """
        rows = self._conn.execute(
            """
            SELECT r.id, r.run_id, r.test_id, r.outcome, r.duration_s
            FROM results r
            WHERE r.run_id = ?
            """,
            (run_id,),
        ).fetchall()
        return [
            ResultRecord(
                id=row["id"],
                run_id=row["run_id"],
                test_id=row["test_id"],
                outcome=row["outcome"],
                duration_s=row["duration_s"],
            )
            for row in rows
        ]

    def get_test_history(self, node_id: str, *, limit: int = 50) -> list[ResultRecord]:
        """Return the most recent results for a test, newest run first.

        Args:
            node_id: Pytest node ID.
            limit: Maximum number of records to return.

        Returns:
            List of :class:`ResultRecord`, ordered by descending run ``created_at``.
        """
        rows = self._conn.execute(
            """
            SELECT res.id, res.run_id, res.test_id, res.outcome, res.duration_s
            FROM results res
            JOIN tests t ON t.id = res.test_id
            JOIN runs run ON run.id = res.run_id
            WHERE t.node_id = ?
            ORDER BY run.created_at DESC
            LIMIT ?
            """,
            (node_id, limit),
        ).fetchall()
        return [
            ResultRecord(
                id=r["id"],
                run_id=r["run_id"],
                test_id=r["test_id"],
                outcome=r["outcome"],
                duration_s=r["duration_s"],
            )
            for r in rows
        ]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
