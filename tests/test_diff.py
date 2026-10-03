"""Tests for failfast.diff – git diff parser."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from failfast.diff import diff_refs, parse_diff_text
from failfast.models import ChangedFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_DIFF = textwrap.dedent("""\
    diff --git a/foo.py b/foo.py
    index 1234567..abcdefg 100644
    --- a/foo.py
    +++ b/foo.py
    @@ -1,4 +1,5 @@
     def greet():
    -    return "hello"
    +    return "hi"
    +    # changed
     
     def bye():
""")

ADDED_FILE_DIFF = textwrap.dedent("""\
    diff --git a/new.py b/new.py
    new file mode 100644
    index 0000000..aaaaaaa
    --- /dev/null
    +++ b/new.py
    @@ -0,0 +1,3 @@
    +def new_func():
    +    pass
    +
""")

DELETED_FILE_DIFF = textwrap.dedent("""\
    diff --git a/old.py b/old.py
    deleted file mode 100644
    index bbbbbbb..0000000
    --- a/old.py
    +++ /dev/null
    @@ -1,2 +0,0 @@
    -def old():
    -    pass
""")

RENAMED_FILE_DIFF = textwrap.dedent("""\
    diff --git a/before.py b/after.py
    rename from before.py
    rename to after.py
    similarity index 80%
    index ccccccc..ddddddd 100644
    --- a/before.py
    +++ b/after.py
    @@ -1,3 +1,3 @@
     def fn():
    -    return 1
    +    return 2
""")


# ---------------------------------------------------------------------------
# Tests: parse_diff_text
# ---------------------------------------------------------------------------


class TestParseDiffText:
    def test_modified_file_detected(self) -> None:
        files = parse_diff_text(SIMPLE_DIFF)
        assert len(files) == 1
        f = files[0]
        assert f.path == "foo.py"
        assert f.change_type == "modified"

    def test_modified_line_numbers(self) -> None:
        files = parse_diff_text(SIMPLE_DIFF)
        f = files[0]
        # Line 2 was removed, lines 2 and 3 were added (new numbering)
        assert 2 in f.removed_lines
        assert 2 in f.added_lines
        assert 3 in f.added_lines

    def test_added_file(self) -> None:
        files = parse_diff_text(ADDED_FILE_DIFF)
        assert len(files) == 1
        assert files[0].change_type == "added"
        assert files[0].path == "new.py"

    def test_added_file_line_numbers(self) -> None:
        files = parse_diff_text(ADDED_FILE_DIFF)
        f = files[0]
        assert f.added_lines == {1, 2, 3}
        assert f.removed_lines == set()

    def test_deleted_file(self) -> None:
        files = parse_diff_text(DELETED_FILE_DIFF)
        assert len(files) == 1
        assert files[0].change_type == "deleted"
        assert files[0].removed_lines == {1, 2}

    def test_renamed_file(self) -> None:
        files = parse_diff_text(RENAMED_FILE_DIFF)
        assert len(files) == 1
        f = files[0]
        assert f.change_type == "renamed"
        assert f.path == "after.py"
        assert f.old_path == "before.py"

    def test_empty_diff(self) -> None:
        assert parse_diff_text("") == []

    def test_multiple_files(self) -> None:
        combined = SIMPLE_DIFF + ADDED_FILE_DIFF
        files = parse_diff_text(combined)
        assert len(files) == 2
        paths = {f.path for f in files}
        assert "foo.py" in paths
        assert "new.py" in paths

    def test_changed_file_is_dataclass(self) -> None:
        files = parse_diff_text(SIMPLE_DIFF)
        assert isinstance(files[0], ChangedFile)

    def test_no_hunk_file_has_no_lines(self) -> None:
        """Binary file or similarity-only diff has no hunks."""
        binary_diff = textwrap.dedent("""\
            diff --git a/img.png b/img.png
            index aaaaaaa..bbbbbbb 100644
            Binary files a/img.png and b/img.png differ
        """)
        files = parse_diff_text(binary_diff)
        assert len(files) == 1
        assert files[0].added_lines == set()
        assert files[0].removed_lines == set()


# ---------------------------------------------------------------------------
# Tests: diff_refs (integration – needs a real git repo)
# ---------------------------------------------------------------------------


class TestDiffRefs:
    def test_detects_modified_file(self, mini_repo: Path) -> None:
        """Editing a file between two commits shows up in diff_refs."""
        # Modify the source file and commit
        src = mini_repo / "mylib.py"
        src.write_text("def add(a, b):\n    return a + b + 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=mini_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Modify add"],
            cwd=mini_repo,
            check=True,
            capture_output=True,
        )

        changed = diff_refs(mini_repo, base="HEAD~1", head="HEAD")
        paths = {f.path for f in changed}
        assert "mylib.py" in paths

    def test_added_file_appears(self, mini_repo: Path) -> None:
        new_file = mini_repo / "extra.py"
        new_file.write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=mini_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add extra"],
            cwd=mini_repo,
            check=True,
            capture_output=True,
        )

        changed = diff_refs(mini_repo, base="HEAD~1", head="HEAD")
        paths = {f.path for f in changed}
        assert "extra.py" in paths
        assert any(f.change_type == "added" for f in changed if f.path == "extra.py")

    def test_no_changes_returns_empty(self, mini_repo: Path) -> None:
        changed = diff_refs(mini_repo, base="HEAD", head="HEAD")
        assert changed == []

    def test_invalid_ref_raises(self, mini_repo: Path) -> None:
        with pytest.raises(subprocess.CalledProcessError):
            diff_refs(mini_repo, base="nonexistent-branch")
