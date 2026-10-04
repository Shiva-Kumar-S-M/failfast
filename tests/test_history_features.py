"""Tests for failfast.features.history.

Includes a dedicated test proving that compute_history_features never uses
data from after the before_timestamp cutoff (leakage prevention).
"""

from __future__ import annotations

import time

import pytest

from failfast.features.history import (
    N_RUNS_SENTINEL,
    HistoryFeatures,
    compute_history_features,
)
from failfast.store import Store


def make_store() -> Store:
    return Store(":memory:")


def seed_history(
    store: Store,
    node_id: str,
    outcomes: list[str],
    durations: list[float] | None = None,
    base_ts: float | None = None,
) -> list[float]:
    """Insert runs oldest-first; return list of created_at timestamps."""
    if durations is None:
        durations = [0.1] * len(outcomes)
    if base_ts is None:
        base_ts = time.time() - len(outcomes) * 10

    timestamps: list[float] = []
    for i, (outcome, dur) in enumerate(zip(outcomes, durations, strict=True)):
        ts = base_ts + i * 10
        run_id = store._conn.execute(
            "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
            (f"sha_{i}", "base", ts),
        ).lastrowid
        store._conn.commit()
        store.upsert_result(run_id, node_id, outcome, duration_s=dur)
        timestamps.append(ts)
    return timestamps


class TestComputeHistoryFeatures:
    def test_no_history_returns_defaults(self) -> None:
        store = make_store()
        result = compute_history_features(store, ["tests/test_a.py::test_x"], 9e18)
        f = result[0]
        assert f.failure_rate == 0.0
        assert f.recent_failure_count == 0
        assert f.runs_since_last_failure == N_RUNS_SENTINEL
        assert f.flakiness == 0.0
        assert f.avg_duration == 0.0
        assert f.last_duration == 0.0

    def test_all_passing_history(self) -> None:
        store = make_store()
        seed_history(store, "tests/t.py::test_x", ["passed"] * 5)
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        f = result[0]
        assert f.failure_rate == 0.0
        assert f.recent_failure_count == 0
        assert f.runs_since_last_failure == N_RUNS_SENTINEL
        assert f.flakiness == 0.0

    def test_all_failing_history(self) -> None:
        store = make_store()
        seed_history(store, "tests/t.py::test_x", ["failed"] * 4)
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        f = result[0]
        assert f.failure_rate == pytest.approx(1.0)
        assert f.recent_failure_count == 4
        assert f.runs_since_last_failure == 0

    def test_failure_rate_partial(self) -> None:
        store = make_store()
        outcomes = ["passed", "failed", "passed", "failed"]
        seed_history(store, "tests/t.py::test_x", outcomes)
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        f = result[0]
        assert f.failure_rate == pytest.approx(0.5)

    def test_recent_failure_count_window(self) -> None:
        store = make_store()
        # 15 runs: first 10 passed, last 5 failed (newest first = [failed]*5 + [passed]*10)
        outcomes = ["passed"] * 10 + ["failed"] * 5  # oldest-first insertion
        seed_history(store, "tests/t.py::test_x", outcomes)
        result = compute_history_features(
            store, ["tests/t.py::test_x"], 9e18, recent_n=10
        )
        f = result[0]
        # Rows are newest-first; newest 10 are the 5 failed + 5 of the passed ones
        assert f.recent_failure_count == 5

    def test_runs_since_last_failure(self) -> None:
        store = make_store()
        outcomes = ["passed", "failed", "passed", "passed", "passed"]
        seed_history(store, "tests/t.py::test_x", outcomes)
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        f = result[0]
        # Newest-first: passed, passed, passed, failed, passed → 3 consecutive passes
        assert f.runs_since_last_failure == 3

    def test_flakiness_alternating(self) -> None:
        store = make_store()
        outcomes = ["passed", "failed", "passed", "failed", "passed"]
        seed_history(store, "tests/t.py::test_x", outcomes)
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        f = result[0]
        # 4 pairs, all flip → flakiness = 1.0
        assert f.flakiness == pytest.approx(1.0)

    def test_flakiness_stable(self) -> None:
        store = make_store()
        outcomes = ["passed"] * 5
        seed_history(store, "tests/t.py::test_x", outcomes)
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        assert result[0].flakiness == pytest.approx(0.0)

    def test_avg_duration(self) -> None:
        store = make_store()
        seed_history(
            store,
            "tests/t.py::test_x",
            ["passed", "passed", "passed"],
            durations=[0.1, 0.3, 0.5],
        )
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        assert result[0].avg_duration == pytest.approx(0.3)

    def test_last_duration_is_most_recent(self) -> None:
        store = make_store()
        # Insert oldest to newest with increasing durations
        seed_history(
            store,
            "tests/t.py::test_x",
            ["passed", "passed", "passed"],
            durations=[0.1, 0.5, 2.0],
        )
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        # Newest run had duration 2.0
        assert result[0].last_duration == pytest.approx(2.0)

    def test_multiple_node_ids(self) -> None:
        store = make_store()
        seed_history(store, "tests/a.py::test_a", ["failed", "failed"])
        seed_history(store, "tests/b.py::test_b", ["passed", "passed"])
        result = compute_history_features(
            store, ["tests/a.py::test_a", "tests/b.py::test_b"], 9e18
        )
        assert result[0].failure_rate == pytest.approx(1.0)
        assert result[1].failure_rate == pytest.approx(0.0)

    # -----------------------------------------------------------------------
    # Leakage prevention: no future data must ever be used
    # -----------------------------------------------------------------------

    def test_history_leakage_before_timestamp_excludes_future_runs(self) -> None:
        """Runs after before_timestamp must NEVER be visible to feature extraction.

        This test inserts a run that would change the failure_rate from 0 to 1,
        then verifies that the features computed with a cutoff BEFORE that run
        still report failure_rate == 0.  This proves no label leakage occurs.
        """
        store = make_store()
        base_ts = 1_000_000.0  # fixed past timestamp

        # Early passing run (will be visible)
        run_id_early = store._conn.execute(
            "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
            ("sha_early", "base", base_ts),
        ).lastrowid
        store._conn.commit()
        store.upsert_result(run_id_early, "tests/t.py::test_x", "passed", 0.1)

        cutoff = base_ts + 50  # our scoring point

        # FUTURE failing run (must NOT be visible)
        future_ts = base_ts + 100  # after cutoff
        run_id_future = store._conn.execute(
            "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
            ("sha_future", "base", future_ts),
        ).lastrowid
        store._conn.commit()
        store.upsert_result(run_id_future, "tests/t.py::test_x", "failed", 0.2)

        result = compute_history_features(
            store, ["tests/t.py::test_x"], before_timestamp=cutoff
        )
        f = result[0]

        # If leakage occurred, failure_rate would be 0.5 (1 failed / 2 total).
        # The correct answer is 0.0 (only the early passing run is visible).
        assert f.failure_rate == pytest.approx(0.0), (
            "Leakage detected: the future failing run was included in feature "
            f"computation. failure_rate={f.failure_rate} but should be 0.0."
        )
        assert f.recent_failure_count == 0, (
            "Leakage detected: future run counted in recent failures."
        )
        assert f.runs_since_last_failure == N_RUNS_SENTINEL, (
            "Leakage detected: future failure affected runs_since_last_failure."
        )

    def test_cutoff_is_strictly_less_than(self) -> None:
        """A run at exactly the cutoff timestamp must NOT be included."""
        store = make_store()
        exact_ts = 5_000_000.0

        run_id = store._conn.execute(
            "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
            ("sha_exact", "base", exact_ts),
        ).lastrowid
        store._conn.commit()
        store.upsert_result(run_id, "tests/t.py::test_x", "failed", 0.1)

        # Cutoff IS the timestamp of the run; the run must be excluded
        result = compute_history_features(
            store, ["tests/t.py::test_x"], before_timestamp=exact_ts
        )
        assert result[0].failure_rate == 0.0

    def test_history_features_are_frozen(self) -> None:
        feat = HistoryFeatures(
            node_id="t::t",
            failure_rate=0.0,
            recent_failure_count=0,
            runs_since_last_failure=N_RUNS_SENTINEL,
            flakiness=0.0,
            avg_duration=0.0,
            last_duration=0.0,
        )
        with pytest.raises(Exception):
            feat.failure_rate = 1.0  # type: ignore[misc]

    def test_error_outcome_counts_as_failure(self) -> None:
        store = make_store()
        seed_history(store, "tests/t.py::test_x", ["error", "passed"])
        result = compute_history_features(store, ["tests/t.py::test_x"], 9e18)
        f = result[0]
        assert f.failure_rate == pytest.approx(0.5)
