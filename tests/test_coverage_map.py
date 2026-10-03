"""Tests for failfast.coverage_map – per-test coverage collection."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from failfast.coverage_map import CoverageEntry, filter_by_changed_files, _parse_json_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_coverage_json(tmp_path: Path) -> Path:
    """Write a minimal coverage.py JSON report with dynamic contexts."""
    data = {
        "meta": {"version": "7.0.0", "branch_coverage": False},
        "files": {
            "/repo/mylib.py": {
                "executed_lines": [1, 2, 3, 4],
                "contexts": {
                    "tests/test_mylib.py::test_add|run": [1, 2],
                    "tests/test_mylib.py::test_subtract|run": [3, 4],
                    "": [1],  # empty context – should be ignored
                },
            },
            "/repo/utils.py": {
                "executed_lines": [10],
                "contexts": {
                    "tests/test_mylib.py::test_add|run": [10],
                },
            },
        },
    }
    json_path = tmp_path / "coverage.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


# ---------------------------------------------------------------------------
# Tests: _parse_json_report
# ---------------------------------------------------------------------------


class TestParseJsonReport:
    def test_returns_list_of_entries(self, sample_coverage_json: Path, tmp_path: Path) -> None:
        entries = _parse_json_report(sample_coverage_json, Path("/repo"))
        assert isinstance(entries, list)
        assert all(isinstance(e, CoverageEntry) for e in entries)

    def test_correct_number_of_entries(self, sample_coverage_json: Path) -> None:
        # test_add and test_subtract are the two non-empty contexts
        entries = _parse_json_report(sample_coverage_json, Path("/repo"))
        node_ids = {e.node_id for e in entries}
        assert "tests/test_mylib.py::test_add" in node_ids
        assert "tests/test_mylib.py::test_subtract" in node_ids
        # empty context key should be ignored
        assert "" not in node_ids

    def test_test_add_covers_both_files(self, sample_coverage_json: Path) -> None:
        entries = _parse_json_report(sample_coverage_json, Path("/repo"))
        add_entry = next(e for e in entries if "test_add" in e.node_id)
        # mylib.py and utils.py should both appear
        assert len(add_entry.executed_lines) == 2

    def test_lines_are_sets(self, sample_coverage_json: Path) -> None:
        entries = _parse_json_report(sample_coverage_json, Path("/repo"))
        add_entry = next(e for e in entries if "test_add" in e.node_id)
        for lines in add_entry.executed_lines.values():
            assert isinstance(lines, set)

    def test_test_subtract_covers_only_mylib(self, sample_coverage_json: Path) -> None:
        entries = _parse_json_report(sample_coverage_json, Path("/repo"))
        sub_entry = next(e for e in entries if "test_subtract" in e.node_id)
        assert len(sub_entry.executed_lines) == 1

    def test_empty_json_returns_empty(self, tmp_path: Path) -> None:
        json_path = tmp_path / "empty.json"
        json_path.write_text(json.dumps({"files": {}}), encoding="utf-8")
        assert _parse_json_report(json_path, tmp_path) == []


# ---------------------------------------------------------------------------
# Tests: filter_by_changed_files
# ---------------------------------------------------------------------------


class TestFilterByChangedFiles:
    def _make_entries(self) -> list[CoverageEntry]:
        return [
            CoverageEntry(
                node_id="tests/test_a.py::test_foo",
                executed_lines={"src/a.py": {1, 2}, "src/shared.py": {10}},
            ),
            CoverageEntry(
                node_id="tests/test_b.py::test_bar",
                executed_lines={"src/b.py": {5, 6}},
            ),
            CoverageEntry(
                node_id="tests/test_c.py::test_baz",
                executed_lines={"src/shared.py": {10, 11}},
            ),
        ]

    def test_returns_tests_touching_changed_files(self) -> None:
        entries = self._make_entries()
        result = filter_by_changed_files(entries, {"src/a.py"})
        assert "tests/test_a.py::test_foo" in result
        assert "tests/test_b.py::test_bar" not in result

    def test_shared_file_matches_multiple_tests(self) -> None:
        entries = self._make_entries()
        result = filter_by_changed_files(entries, {"src/shared.py"})
        assert "tests/test_a.py::test_foo" in result
        assert "tests/test_c.py::test_baz" in result

    def test_no_overlap_returns_empty(self) -> None:
        entries = self._make_entries()
        result = filter_by_changed_files(entries, {"src/nonexistent.py"})
        assert result == []

    def test_empty_coverage_map(self) -> None:
        result = filter_by_changed_files([], {"src/a.py"})
        assert result == []

    def test_empty_changed_paths(self) -> None:
        entries = self._make_entries()
        result = filter_by_changed_files(entries, set())
        assert result == []
