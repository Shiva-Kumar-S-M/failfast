"""Historical test-execution features for test prioritization.

Features are derived exclusively from run history that predates the commit
being scored.  This strict cutoff prevents label leakage: the model must never
see the outcome of the commit it is asked to rank.

Feature definitions
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Feature
     - Description
   * - ``failure_rate``
     - Fraction of historical runs in which the test failed or errored.
       ``failed_count / total_runs`` (0.0 when no history).
   * - ``recent_failure_count``
     - Number of failures in the last *N* runs (default N = 10).
   * - ``runs_since_last_failure``
     - How many consecutive passing runs have occurred since the most recent
       failure.  Set to ``N_RUNS_SENTINEL = 999`` when the test has never
       failed in the available history.
   * - ``flakiness``
     - Fraction of consecutive run pairs where the outcome flipped between
       passing and failing.  0.0 = perfectly consistent, 1.0 = alternates
       every run.
   * - ``avg_duration``
     - Mean wall-clock duration (seconds) across all historical runs.
       0.0 when no history exists.
   * - ``last_duration``
     - Duration of the most recent historical run.  0.0 when no history.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from failfast.store import Store

# Sentinel assigned when a test has never failed in its recorded history.
N_RUNS_SENTINEL: int = 999

# Outcomes considered a "failure" for feature computation.
_FAILURE_OUTCOMES: frozenset[str] = frozenset({"failed", "error"})


@dataclass(frozen=True)
class HistoryFeatures:
    """Historical performance metrics for one test.

    Attributes:
        node_id: Pytest node ID of the test.
        failure_rate: Fraction of past runs that were a failure (0.0–1.0).
        recent_failure_count: Failures in the last *N* runs.
        runs_since_last_failure: Consecutive green runs since last failure.
            ``N_RUNS_SENTINEL`` when the test has never failed.
        flakiness: Flip-rate between consecutive runs (0.0–1.0).
        avg_duration: Mean execution time in seconds across all history.
        last_duration: Duration of the most recent run (seconds).
    """

    node_id: str
    failure_rate: float
    recent_failure_count: int
    runs_since_last_failure: int
    flakiness: float
    avg_duration: float
    last_duration: float


def compute_history_features(
    store: Store,
    node_ids: list[str],
    before_timestamp: float,
    *,
    recent_n: int = 10,
) -> list[HistoryFeatures]:
    """Compute history features for every node ID in *node_ids*.

    Only runs whose ``created_at`` timestamp is **strictly less than**
    *before_timestamp* are considered.  This matches the scenario where we are
    scoring a commit that was created at *before_timestamp* – we can only use
    data we would have had at that moment.

    Args:
        store: Open :class:`failfast.store.Store` instance.
        node_ids: Pytest node IDs to compute features for.
        before_timestamp: Unix timestamp upper bound (exclusive).  Only runs
            recorded before this point are used.
        recent_n: Window size for ``recent_failure_count``.

    Returns:
        One :class:`HistoryFeatures` per entry in *node_ids*, in order.
    """
    features: list[HistoryFeatures] = []
    for node_id in node_ids:
        rows = _fetch_history(store, node_id, before_timestamp)
        features.append(_compute_single(node_id, rows, recent_n))
    return features


def _fetch_history(
    store: Store,
    node_id: str,
    before_timestamp: float,
) -> list[tuple[str, float]]:
    """Return ``(outcome, duration_s)`` rows for *node_id* before the cutoff.

    Rows are ordered newest-first so index 0 is the most recent run.

    Args:
        store: Open store instance.
        node_id: Pytest node ID.
        before_timestamp: Exclusive upper bound on ``runs.created_at``.

    Returns:
        List of ``(outcome, duration_s)`` tuples, newest first.
    """
    rows = store._conn.execute(
        """
        SELECT res.outcome, res.duration_s
        FROM results res
        JOIN tests  t   ON t.id   = res.test_id
        JOIN runs   run ON run.id = res.run_id
        WHERE t.node_id    = ?
          AND run.created_at < ?
        ORDER BY run.created_at DESC
        """,
        (node_id, before_timestamp),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _compute_single(
    node_id: str,
    rows: list[tuple[str, float]],
    recent_n: int,
) -> HistoryFeatures:
    """Compute features from a pre-fetched history list."""
    if not rows:
        return HistoryFeatures(
            node_id=node_id,
            failure_rate=0.0,
            recent_failure_count=0,
            runs_since_last_failure=N_RUNS_SENTINEL,
            flakiness=0.0,
            avg_duration=0.0,
            last_duration=0.0,
        )

    outcomes = [r[0] for r in rows]
    durations = [r[1] for r in rows]

    is_failure = [o in _FAILURE_OUTCOMES for o in outcomes]
    total = len(rows)
    failed_total = sum(is_failure)

    failure_rate = failed_total / total

    recent = is_failure[:recent_n]
    recent_failure_count = sum(recent)

    # runs_since_last_failure: count leading non-failures (rows are newest-first)
    runs_since = 0
    for f in is_failure:
        if f:
            break
        runs_since += 1
    runs_since_last_failure = runs_since if failed_total > 0 else N_RUNS_SENTINEL

    # flakiness: fraction of consecutive pairs that flip
    flips = 0
    pairs = total - 1
    for i in range(pairs):
        if is_failure[i] != is_failure[i + 1]:
            flips += 1
    flakiness = flips / pairs if pairs > 0 else 0.0

    avg_duration = sum(durations) / total
    last_duration = durations[0]  # newest first

    return HistoryFeatures(
        node_id=node_id,
        failure_rate=failure_rate,
        recent_failure_count=recent_failure_count,
        runs_since_last_failure=runs_since_last_failure,
        flakiness=flakiness,
        avg_duration=avg_duration,
        last_duration=last_duration,
    )
