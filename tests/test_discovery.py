"""Tests for failfast.discovery – pytest test discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from failfast.discovery import discover_tests, node_id_to_file


class TestDiscoverTests:
    def test_finds_tests_in_mini_repo(self, mini_repo: Path) -> None:
        node_ids = discover_tests(mini_repo)
        assert len(node_ids) == 2
        assert any("test_add" in nid for nid in node_ids)
        assert any("test_subtract" in nid for nid in node_ids)

    def test_node_ids_contain_double_colon(self, mini_repo: Path) -> None:
        node_ids = discover_tests(mini_repo)
        assert all("::" in nid for nid in node_ids)

    def test_returns_list(self, mini_repo: Path) -> None:
        result = discover_tests(mini_repo)
        assert isinstance(result, list)

    def test_no_duplicates(self, mini_repo: Path) -> None:
        node_ids = discover_tests(mini_repo)
        assert len(node_ids) == len(set(node_ids))

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        result = discover_tests(tmp_path)
        assert result == []

    def test_restricted_path(self, mini_repo: Path) -> None:
        """Passing a specific path restricts discovery."""
        node_ids = discover_tests(mini_repo, paths=["tests/"])
        assert all("tests/" in nid for nid in node_ids)

    def test_extra_args_filter(self, mini_repo: Path) -> None:
        """Using -k flag filters results."""
        node_ids = discover_tests(mini_repo, extra_args=["-k", "add"])
        assert all("test_add" in nid for nid in node_ids)
        # test_subtract should be excluded
        assert not any("test_subtract" in nid for nid in node_ids)

    def test_additional_test_file(self, mini_repo: Path) -> None:
        """Adding a new test file increases the count."""
        tests_dir = mini_repo / "tests"
        (tests_dir / "test_extra.py").write_text(
            textwrap.dedent("""\
                def test_one():
                    assert 1 == 1
            """),
            encoding="utf-8",
        )
        node_ids = discover_tests(mini_repo)
        assert len(node_ids) == 3
        assert any("test_one" in nid for nid in node_ids)


class TestNodeIdToFile:
    def test_simple_node_id(self) -> None:
        assert node_id_to_file("tests/test_foo.py::test_bar") == "tests/test_foo.py"

    def test_class_method_node_id(self) -> None:
        assert (
            node_id_to_file("tests/test_foo.py::TestClass::test_method")
            == "tests/test_foo.py"
        )

    def test_file_only(self) -> None:
        # Edge case: if somehow only file is given
        assert node_id_to_file("tests/test_foo.py") == "tests/test_foo.py"
