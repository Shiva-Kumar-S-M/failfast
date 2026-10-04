"""Tests for failfast.features.coverage_overlap."""

from __future__ import annotations

import dataclasses

import pytest

from failfast.coverage_map import CoverageEntry
from failfast.features.coverage_overlap import OverlapFeatures, compute_coverage_overlap
from failfast.models import ChangedFile, Hunk


def make_hunk(added: set[int], removed: set[int]) -> Hunk:
    return Hunk(
        old_start=1,
        old_count=len(removed),
        new_start=1,
        new_count=len(added),
        added_lines=added,
        removed_lines=removed,
    )


def make_changed_file(path: str, added: set[int], removed: set[int]) -> ChangedFile:
    return ChangedFile(
        path=path,
        old_path=path,
        change_type="modified",
        hunks=[make_hunk(added, removed)],
    )


def make_entry(node_id: str, executed: dict[str, set[int]]) -> CoverageEntry:
    return CoverageEntry(node_id=node_id, executed_lines=executed)


class TestComputeCoverageOverlap:
    def test_no_changed_files_gives_zero_ratio(self) -> None:
        entries = [make_entry("tests/test_foo.py::test_a", {"src/foo.py": {1, 2, 3}})]
        result = compute_coverage_overlap(entries, [])
        assert result[0].overlap_lines == 0
        assert result[0].overlap_files == 0
        assert result[0].overlap_ratio == 0.0

    def test_full_overlap_ratio_is_one(self) -> None:
        """When the test covers every changed line, ratio == 1.0."""
        changed = [make_changed_file("src/foo.py", {10, 11, 12}, set())]
        entries = [
            make_entry("tests/test_foo.py::test_a", {"src/foo.py": {10, 11, 12}})
        ]
        result = compute_coverage_overlap(entries, changed)
        feat = result[0]
        assert feat.overlap_lines == 3
        assert feat.overlap_files == 1
        assert feat.overlap_ratio == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        changed = [make_changed_file("src/foo.py", {10, 11, 12, 13}, set())]
        entries = [make_entry("tests/test_foo.py::test_a", {"src/foo.py": {10, 11}})]
        result = compute_coverage_overlap(entries, changed)
        feat = result[0]
        assert feat.overlap_lines == 2
        assert feat.overlap_files == 1
        assert feat.overlap_ratio == pytest.approx(2 / 4)

    def test_no_coverage_match(self) -> None:
        changed = [make_changed_file("src/foo.py", {10, 11}, set())]
        entries = [make_entry("tests/test_bar.py::test_b", {"src/bar.py": {5, 6}})]
        result = compute_coverage_overlap(entries, changed)
        feat = result[0]
        assert feat.overlap_lines == 0
        assert feat.overlap_files == 0
        assert feat.overlap_ratio == 0.0

    def test_multiple_changed_files(self) -> None:
        changed = [
            make_changed_file("src/a.py", {1, 2}, set()),
            make_changed_file("src/b.py", {5, 6}, set()),
        ]
        entries = [
            make_entry(
                "tests/test_ab.py::test_both",
                {"src/a.py": {1}, "src/b.py": {5}},
            )
        ]
        result = compute_coverage_overlap(entries, changed)
        feat = result[0]
        assert feat.overlap_lines == 2  # 1 from a + 1 from b
        assert feat.overlap_files == 2
        assert feat.overlap_ratio == pytest.approx(2 / 4)

    def test_removed_lines_count_as_changed(self) -> None:
        """Removed lines are also changed and should count toward total."""
        changed = [make_changed_file("src/foo.py", set(), {7, 8})]
        entries = [make_entry("tests/test_foo.py::test_a", {"src/foo.py": {7}})]
        result = compute_coverage_overlap(entries, changed)
        feat = result[0]
        assert feat.overlap_lines == 1
        assert feat.overlap_ratio == pytest.approx(1 / 2)

    def test_multiple_tests_returned_in_order(self) -> None:
        changed = [make_changed_file("src/foo.py", {1}, set())]
        entries = [
            make_entry("tests/test_a.py::test_a", {"src/foo.py": {1}}),
            make_entry("tests/test_b.py::test_b", {"src/bar.py": {1}}),
        ]
        result = compute_coverage_overlap(entries, changed)
        assert [r.node_id for r in result] == [
            "tests/test_a.py::test_a",
            "tests/test_b.py::test_b",
        ]
        assert result[0].overlap_lines == 1
        assert result[1].overlap_lines == 0

    def test_ratio_uses_total_changed_lines_across_files(self) -> None:
        changed = [
            make_changed_file("src/a.py", {1}, set()),
            make_changed_file("src/b.py", {2, 3}, set()),
        ]
        entries = [make_entry("tests/t.py::test_x", {"src/a.py": {1}})]
        result = compute_coverage_overlap(entries, changed)
        # total_changed = 3 (lines 1, 2, 3)
        assert result[0].overlap_ratio == pytest.approx(1 / 3)

    def test_overlap_features_are_frozen(self) -> None:
        feat = OverlapFeatures(
            node_id="t::t", overlap_lines=0, overlap_files=0, overlap_ratio=0.0
        )
        with pytest.raises(dataclasses.FrozenInstanceError):  # frozen dataclass
            feat.overlap_lines = 1  # type: ignore[misc]
