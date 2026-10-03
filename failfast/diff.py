"""Parse ``git diff`` output into structured :class:`~failfast.models.ChangedFile` objects."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from failfast.models import ChangedFile, Hunk

# Matches "diff --git a/<path> b/<path>"
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")

# Matches rename / copy source lines: "rename from <path>" or "copy from <path>"
_FROM_LINE = re.compile(r"^(?:rename|copy) from (.+)$")

# Matches rename / copy destination lines
_TO_LINE = re.compile(r"^(?:rename|copy) to (.+)$")

# Matches hunk headers: "@@ -old_start[,old_count] +new_start[,new_count] @@"
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _classify(lines: list[str]) -> str:
    """Return the change type from the lines between two diff headers.

    Args:
        lines: Lines of the diff section for one file (without the
            ``diff --git`` header line itself).

    Returns:
        One of ``'added'``, ``'deleted'``, ``'renamed'``, ``'copied'``,
        or ``'modified'``.
    """
    for line in lines:
        if line.startswith("new file mode"):
            return "added"
        if line.startswith("deleted file mode"):
            return "deleted"
        if line.startswith("rename from"):
            return "renamed"
        if line.startswith("copy from"):
            return "copied"
    return "modified"


def _parse_hunk(hunk_header: str, body_lines: list[str]) -> Hunk:
    """Build a :class:`Hunk` from a hunk header and its body lines.

    Args:
        hunk_header: The raw ``@@ ... @@`` header line.
        body_lines: Lines belonging to this hunk (context and diff lines).

    Returns:
        A populated :class:`Hunk` instance.
    """
    m = _HUNK_HEADER.match(hunk_header)
    assert m, f"Unexpected hunk header: {hunk_header!r}"

    old_start = int(m.group(1))
    old_count = int(m.group(2)) if m.group(2) is not None else 1
    new_start = int(m.group(3))
    new_count = int(m.group(4)) if m.group(4) is not None else 1

    hunk = Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
    )

    old_line = old_start
    new_line = new_start

    for line in body_lines:
        if not line:
            # Blank line at the end of file – treat as context
            old_line += 1
            new_line += 1
            continue
        prefix = line[0]
        if prefix == "+":
            hunk.added_lines.add(new_line)
            new_line += 1
        elif prefix == "-":
            hunk.removed_lines.add(old_line)
            old_line += 1
        else:
            # context line
            old_line += 1
            new_line += 1

    return hunk


def parse_diff_text(diff_text: str) -> list[ChangedFile]:
    """Parse the text produced by ``git diff`` into a list of :class:`ChangedFile`.

    Args:
        diff_text: The full output of a ``git diff`` command.

    Returns:
        One :class:`ChangedFile` per file mentioned in the diff, ordered by
        first appearance.
    """
    lines = diff_text.splitlines()

    # Split into per-file sections at "diff --git" boundaries
    sections: list[tuple[str, str, list[str]]] = []
    current_a: str | None = None
    current_b: str | None = None
    current_lines: list[str] = []

    for line in lines:
        m = _DIFF_HEADER.match(line)
        if m:
            if current_a is not None:
                sections.append((current_a, current_b, current_lines))  # type: ignore[arg-type]
            current_a = m.group(1)
            current_b = m.group(2)
            current_lines = []
        elif current_a is not None:
            current_lines.append(line)

    if current_a is not None:
        sections.append((current_a, current_b, current_lines))  # type: ignore[arg-type]

    changed_files: list[ChangedFile] = []

    for path_a, path_b, section_lines in sections:
        change_type = _classify(section_lines)

        # Determine canonical paths
        old_path = path_a
        new_path = path_b

        # For renames/copies the "b/<path>" in the header is the destination
        for line in section_lines:
            m_from = _FROM_LINE.match(line)
            if m_from:
                old_path = m_from.group(1)
            m_to = _TO_LINE.match(line)
            if m_to:
                new_path = m_to.group(1)

        # Parse hunks
        hunks: list[Hunk] = []
        hunk_header: str | None = None
        hunk_body: list[str] = []

        for line in section_lines:
            if _HUNK_HEADER.match(line):
                if hunk_header is not None:
                    hunks.append(_parse_hunk(hunk_header, hunk_body))
                hunk_header = line
                hunk_body = []
            elif hunk_header is not None:
                hunk_body.append(line)

        if hunk_header is not None:
            hunks.append(_parse_hunk(hunk_header, hunk_body))

        changed_files.append(
            ChangedFile(
                path=new_path,
                old_path=old_path,
                change_type=change_type,
                hunks=hunks,
            )
        )

    return changed_files


def diff_refs(
    repo: Path,
    base: str,
    head: str = "HEAD",
    *,
    unified: int = 0,
) -> list[ChangedFile]:
    """Return changed files between two git refs.

    Args:
        repo: Path to the root of a git repository.
        base: The base ref (e.g. ``"main"`` or a commit SHA).
        head: The head ref to compare against. Defaults to ``"HEAD"``.
        unified: Number of context lines to include (default 0 gives minimal
            output and speeds up parsing).

    Returns:
        List of :class:`ChangedFile` objects describing the diff.

    Raises:
        subprocess.CalledProcessError: If git exits with a non-zero status.
        FileNotFoundError: If ``git`` is not on PATH.
    """
    result = subprocess.run(
        ["git", "diff", f"-U{unified}", base, head],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_diff_text(result.stdout)
