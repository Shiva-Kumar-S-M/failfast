"""Coverage-overlap features for test prioritization.

For each (test, diff) pair this module computes three numeric features:

- ``overlap_lines``: the number of changed lines (added + removed) that the
  test actually executes according to the coverage map.
- ``overlap_files``: the number of changed files touched by the test.
- ``overlap_ratio``: ``overlap_lines / total_changed_lines``, or 0.0 when
  there are no changed lines (e.g. a rename-only diff).

A test with high overlap is strongly associated with the changed code and
therefore more likely to fail on that commit.
"""

from __future__ import annotations

from dataclasses import dataclass

from failfast.coverage_map import CoverageEntry
from failfast.models import ChangedFile


@dataclass(frozen=True)
class OverlapFeatures:
    """Overlap measurements for one (test, diff) pair.

    Attributes:
        node_id: The pytest node ID of the test.
        overlap_lines: Count of changed lines the test executes.
        overlap_files: Count of changed files the test covers.
        overlap_ratio: ``overlap_lines / total_changed_lines`` (0.0 when
            the diff has no changed lines).
    """

    node_id: str
    overlap_lines: int
    overlap_files: int
    overlap_ratio: float


def compute_coverage_overlap(
    coverage_entries: list[CoverageEntry],
    changed_files: list[ChangedFile],
) -> list[OverlapFeatures]:
    """Compute coverage-overlap features for every test in *coverage_entries*.

    Changed lines are the union of added *and* removed lines across all hunks.
    The ratio is computed relative to ``total_changed_lines`` so that tests
    are comparable across diffs of different sizes.

    Args:
        coverage_entries: Per-test coverage data from
            :func:`failfast.coverage_map.build_coverage_map`.
        changed_files: Parsed diff output from
            :func:`failfast.diff.diff_refs` or
            :func:`failfast.diff.parse_diff_text`.

    Returns:
        One :class:`OverlapFeatures` per entry in *coverage_entries*, in the
        same order.
    """
    # Build a lookup: file path -> set of changed line numbers
    changed_lines_by_file: dict[str, set[int]] = {}
    total_changed: int = 0

    for cf in changed_files:
        lines = cf.added_lines | cf.removed_lines
        if lines:
            changed_lines_by_file[cf.path] = lines
            total_changed += len(lines)

    results: list[OverlapFeatures] = []

    for entry in coverage_entries:
        hit_lines = 0
        hit_files = 0

        for file_path, changed_line_set in changed_lines_by_file.items():
            executed = entry.executed_lines.get(file_path, set())
            intersection = executed & changed_line_set
            if intersection:
                hit_lines += len(intersection)
                hit_files += 1

        ratio = hit_lines / total_changed if total_changed > 0 else 0.0

        results.append(
            OverlapFeatures(
                node_id=entry.node_id,
                overlap_lines=hit_lines,
                overlap_files=hit_files,
                overlap_ratio=ratio,
            )
        )

    return results
