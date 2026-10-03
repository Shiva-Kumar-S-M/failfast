"""Data models for representing git diff output."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hunk:
    """A single contiguous block of changes within a file.

    Attributes:
        old_start: First line number in the old file (1-based).
        old_count: Number of lines the hunk covers in the old file.
        new_start: First line number in the new file (1-based).
        new_count: Number of lines the hunk covers in the new file.
        added_lines: Set of new-file line numbers that were added.
        removed_lines: Set of old-file line numbers that were removed.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: set[int] = field(default_factory=set)
    removed_lines: set[int] = field(default_factory=set)


@dataclass
class ChangedFile:
    """A file that appears in a git diff.

    Attributes:
        path: Repository-relative path of the file after the diff.
        old_path: Path before rename/copy; equals ``path`` for ordinary edits.
        change_type: One of ``'added'``, ``'modified'``, ``'deleted'``,
            ``'renamed'``, or ``'copied'``.
        hunks: Ordered list of :class:`Hunk` objects within this file.
    """

    path: str
    old_path: str
    change_type: str  # 'added' | 'modified' | 'deleted' | 'renamed' | 'copied'
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def added_lines(self) -> set[int]:
        """Union of all added line numbers across every hunk."""
        result: set[int] = set()
        for hunk in self.hunks:
            result |= hunk.added_lines
        return result

    @property
    def removed_lines(self) -> set[int]:
        """Union of all removed line numbers across every hunk."""
        result: set[int] = set()
        for hunk in self.hunks:
            result |= hunk.removed_lines
        return result
