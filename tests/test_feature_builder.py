"""Tests for failfast.features.builder."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from failfast.coverage_map import CoverageEntry
from failfast.features.builder import FEATURE_COLUMNS, build_feature_row, build_features
from failfast.models import ChangedFile, Hunk
from failfast.store import Store


def make_changed_file(path: str, added: set[int]) -> ChangedFile:
    hunk = Hunk(
        old_start=1,
        old_count=0,
        new_start=1,
        new_count=len(added),
        added_lines=added,
        removed_lines=set(),
    )
    return ChangedFile(path=path, old_path=path, change_type="modified", hunks=[hunk])


def make_entry(node_id: str, executed: dict[str, set[int]]) -> CoverageEntry:
    return CoverageEntry(node_id=node_id, executed_lines=executed)


def make_store_with_history(node_id: str, outcomes: list[str]) -> Store:
    store = Store(":memory:")
    base_ts = time.time() - len(outcomes) * 10
    for i, outcome in enumerate(outcomes):
        ts = base_ts + i * 10
        run_id = store._conn.execute(
            "INSERT INTO runs (commit_sha, base_sha, created_at) VALUES (?, ?, ?)",
            (f"sha_{i}", "base", ts),
        ).lastrowid
        store._conn.commit()
        store.upsert_result(run_id, node_id, outcome, duration_s=0.2)
    return store


class TestBuildFeatureRow:
    def test_all_defaults_when_no_overlap_no_history(self) -> None:
        row = build_feature_row("t::x", {}, {}, {})
        assert row["overlap_lines"] == 0
        assert row["overlap_files"] == 0
        assert row["overlap_ratio"] == 0.0
        assert row["import_distance"] == 999  # UNREACHABLE
        assert row["failure_rate"] == 0.0
        assert row["runs_since_last_failure"] == 999

    def test_overlap_values_populated(self) -> None:
        overlap = {
            "t::x": {"overlap_lines": 5, "overlap_files": 2, "overlap_ratio": 0.5}
        }
        row = build_feature_row("t::x", overlap, {}, {})
        assert row["overlap_lines"] == 5
        assert row["overlap_files"] == 2
        assert row["overlap_ratio"] == pytest.approx(0.5)


class TestBuildFeatures:
    def test_output_has_correct_columns(self, tmp_path: Path) -> None:
        changed = [make_changed_file("src/foo.py", {1, 2})]
        entries = [make_entry("tests/test_foo.py::test_a", {"src/foo.py": {1}})]
        store = Store(":memory:")

        df = build_features(
            tmp_path, ["tests/test_foo.py::test_a"], entries, changed, store, 9e18
        )
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_index_is_node_id(self, tmp_path: Path) -> None:
        entries = [make_entry("tests/t.py::test_x", {})]
        store = Store(":memory:")

        df = build_features(tmp_path, ["tests/t.py::test_x"], entries, [], store, 9e18)
        assert df.index.name == "node_id"
        assert "tests/t.py::test_x" in df.index

    def test_overlap_features_in_output(self, tmp_path: Path) -> None:
        changed = [make_changed_file("src/foo.py", {10, 11})]
        entries = [make_entry("tests/t.py::test_x", {"src/foo.py": {10, 11}})]
        store = Store(":memory:")

        df = build_features(tmp_path, ["tests/t.py::test_x"], entries, changed, store, 9e18)
        row = df.loc["tests/t.py::test_x"]
        assert row["overlap_lines"] == 2
        assert row["overlap_ratio"] == pytest.approx(1.0)

    def test_history_features_in_output(self, tmp_path: Path) -> None:
        node_id = "tests/t.py::test_x"
        store = make_store_with_history(node_id, ["passed", "failed", "passed"])
        entries = [make_entry(node_id, {})]

        df = build_features(tmp_path, [node_id], entries, [], store, 9e18)
        row = df.loc[node_id]
        assert row["failure_rate"] == pytest.approx(1 / 3)

    def test_empty_node_ids_returns_empty_df(self, tmp_path: Path) -> None:
        store = Store(":memory:")
        df = build_features(tmp_path, [], [], [], store, 9e18)
        assert df.empty

    def test_column_order_matches_feature_columns(self, tmp_path: Path) -> None:
        node_id = "tests/t.py::test_x"
        store = Store(":memory:")
        entries = [make_entry(node_id, {})]

        df = build_features(tmp_path, [node_id], entries, [], store, 9e18)
        assert list(df.columns) == FEATURE_COLUMNS

    def test_multiple_tests_multiple_rows(self, tmp_path: Path) -> None:
        node_ids = ["tests/a.py::test_a", "tests/b.py::test_b"]
        entries = [make_entry(nid, {}) for nid in node_ids]
        store = Store(":memory:")

        df = build_features(tmp_path, node_ids, entries, [], store, 9e18)
        assert len(df) == 2
        assert set(df.index) == set(node_ids)
