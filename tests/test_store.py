"""Tests for failfast.store – SQLite store layer."""

from __future__ import annotations

import time

import pytest

from failfast.store import ResultRecord, RunRecord, Store, TestRecord


@pytest.fixture()
def store() -> Store:
    """In-memory store, fresh for each test."""
    return Store(":memory:")


class TestCreateRun:
    def test_returns_integer_id(self, store: Store) -> None:
        run_id = store.create_run("abc123", "main")
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_two_runs_have_distinct_ids(self, store: Store) -> None:
        id1 = store.create_run("sha1", "main")
        id2 = store.create_run("sha2", "main")
        assert id1 != id2

    def test_created_at_is_recent(self, store: Store) -> None:
        before = time.time()
        run_id = store.create_run("sha1", "base")
        after = time.time()
        run = store.get_run(run_id)
        assert run is not None
        assert before <= run.created_at <= after


class TestGetRun:
    def test_get_existing_run(self, store: Store) -> None:
        run_id = store.create_run("deadbeef", "origin/main")
        run = store.get_run(run_id)
        assert run is not None
        assert isinstance(run, RunRecord)
        assert run.commit_sha == "deadbeef"
        assert run.base_sha == "origin/main"
        assert run.id == run_id

    def test_get_missing_run_returns_none(self, store: Store) -> None:
        assert store.get_run(9999) is None


class TestListRuns:
    def test_empty_store_returns_empty_list(self, store: Store) -> None:
        assert store.list_runs() == []

    def test_runs_ordered_newest_first(self, store: Store) -> None:
        id1 = store.create_run("sha1", "main")
        time.sleep(0.01)  # ensure different timestamps
        id2 = store.create_run("sha2", "main")
        runs = store.list_runs()
        assert len(runs) == 2
        assert runs[0].id == id2  # newest first
        assert runs[1].id == id1

    def test_limit_parameter(self, store: Store) -> None:
        for i in range(5):
            store.create_run(f"sha{i}", "main")
        runs = store.list_runs(limit=3)
        assert len(runs) == 3


class TestUpsertResult:
    def test_basic_upsert(self, store: Store) -> None:
        run_id = store.create_run("sha1", "main")
        result_id = store.upsert_result(run_id, "tests/test_foo.py::test_bar", "passed", 0.5)
        assert isinstance(result_id, int)

    def test_idempotent_update(self, store: Store) -> None:
        run_id = store.create_run("sha1", "main")
        store.upsert_result(run_id, "tests/test_foo.py::test_bar", "passed", 0.5)
        # Calling again should update, not raise a UNIQUE constraint error
        store.upsert_result(run_id, "tests/test_foo.py::test_bar", "failed", 0.6)

        results = store.get_results_for_run(run_id)
        assert len(results) == 1
        assert results[0].outcome == "failed"
        assert results[0].duration_s == pytest.approx(0.6)

    def test_all_valid_outcomes(self, store: Store) -> None:
        run_id = store.create_run("sha1", "main")
        for i, outcome in enumerate(["passed", "failed", "error", "skipped", "xfailed", "xpassed"]):
            store.upsert_result(run_id, f"tests/test_{i}.py::test_{i}", outcome, 0.1)
        results = store.get_results_for_run(run_id)
        assert len(results) == 6

    def test_invalid_outcome_raises(self, store: Store) -> None:
        run_id = store.create_run("sha1", "main")
        with pytest.raises(Exception):
            store.upsert_result(run_id, "tests/test_foo.py::test_bar", "INVALID", 0.0)


class TestGetResultsForRun:
    def test_returns_list_of_result_records(self, store: Store) -> None:
        run_id = store.create_run("sha1", "main")
        store.upsert_result(run_id, "tests/test_a.py::test_x", "passed", 0.1)
        store.upsert_result(run_id, "tests/test_b.py::test_y", "failed", 0.2)
        results = store.get_results_for_run(run_id)
        assert len(results) == 2
        assert all(isinstance(r, ResultRecord) for r in results)

    def test_empty_run(self, store: Store) -> None:
        run_id = store.create_run("sha1", "main")
        assert store.get_results_for_run(run_id) == []


class TestGetTestHistory:
    def test_returns_ordered_history(self, store: Store) -> None:
        node_id = "tests/test_foo.py::test_bar"
        for outcome in ["passed", "failed", "passed"]:
            run_id = store.create_run("sha", "main")
            store.upsert_result(run_id, node_id, outcome, 0.1)

        history = store.get_test_history(node_id)
        assert len(history) == 3
        # Newest first: last inserted run should be first
        assert history[0].outcome == "passed"

    def test_history_for_unknown_test_is_empty(self, store: Store) -> None:
        assert store.get_test_history("tests/does_not_exist.py::test_x") == []

    def test_limit_respected(self, store: Store) -> None:
        node_id = "tests/test_foo.py::test_bar"
        for _ in range(10):
            run_id = store.create_run("sha", "main")
            store.upsert_result(run_id, node_id, "passed", 0.1)
        assert len(store.get_test_history(node_id, limit=5)) == 5


class TestStoreContextManager:
    def test_context_manager_closes(self) -> None:
        with Store(":memory:") as s:
            run_id = s.create_run("sha", "main")
            assert s.get_run(run_id) is not None
        # After close, further operations should raise
        with pytest.raises(Exception):
            s.get_run(run_id)
