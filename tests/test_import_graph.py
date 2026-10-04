"""Tests for failfast.features.import_graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from failfast.features.import_graph import (
    UNREACHABLE,
    ImportGraphFeatures,
    build_import_graph,
    compute_import_distances,
)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestBuildImportGraph:
    def test_empty_repo(self, tmp_path: Path) -> None:
        graph = build_import_graph(tmp_path)
        assert graph == {}

    def test_absolute_import(self, tmp_path: Path) -> None:
        write_file(tmp_path / "mymod.py", "x = 1\n")
        write_file(tmp_path / "main.py", "import mymod\n")
        graph = build_import_graph(tmp_path)
        assert "mymod.py" in graph.get("main.py", set())

    def test_from_import(self, tmp_path: Path) -> None:
        write_file(tmp_path / "util.py", "def helper(): pass\n")
        write_file(tmp_path / "app.py", "from util import helper\n")
        graph = build_import_graph(tmp_path)
        assert "util.py" in graph.get("app.py", set())

    def test_relative_import(self, tmp_path: Path) -> None:
        write_file(tmp_path / "pkg" / "__init__.py", "")
        write_file(tmp_path / "pkg" / "core.py", "x = 1\n")
        write_file(tmp_path / "pkg" / "utils.py", "from . import core\n")
        graph = build_import_graph(tmp_path)
        assert "pkg/core.py" in graph.get("pkg/utils.py", set())

    def test_relative_import_with_module(self, tmp_path: Path) -> None:
        write_file(tmp_path / "pkg" / "__init__.py", "")
        write_file(tmp_path / "pkg" / "models.py", "class M: pass\n")
        write_file(tmp_path / "pkg" / "views.py", "from .models import M\n")
        graph = build_import_graph(tmp_path)
        assert "pkg/models.py" in graph.get("pkg/views.py", set())

    def test_syntax_error_file_skipped(self, tmp_path: Path) -> None:
        write_file(tmp_path / "bad.py", "def (\n")  # invalid syntax
        write_file(tmp_path / "good.py", "import bad\n")
        # Should not raise; bad.py itself has no edges, good.py may try to import it
        graph = build_import_graph(tmp_path)
        assert "good.py" in graph  # good.py is present even if target can't be parsed

    def test_no_self_loop_for_init(self, tmp_path: Path) -> None:
        """__init__.py imports should not create self-referential edges."""
        write_file(tmp_path / "pkg" / "__init__.py", "from . import utils\n")
        write_file(tmp_path / "pkg" / "utils.py", "x = 1\n")
        graph = build_import_graph(tmp_path)
        # pkg/__init__.py should not appear as its own importee
        init_edges = graph.get("pkg/__init__.py", set())
        assert "pkg/__init__.py" not in init_edges

    def test_parent_relative_import(self, tmp_path: Path) -> None:
        write_file(tmp_path / "pkg" / "__init__.py", "")
        write_file(tmp_path / "pkg" / "sub" / "__init__.py", "")
        write_file(tmp_path / "pkg" / "base.py", "x = 1\n")
        write_file(tmp_path / "pkg" / "sub" / "child.py", "from .. import base\n")
        graph = build_import_graph(tmp_path)
        assert "pkg/base.py" in graph.get("pkg/sub/child.py", set())


class TestComputeImportDistances:
    def test_direct_importer(self, tmp_path: Path) -> None:
        write_file(tmp_path / "src.py", "x = 1\n")
        write_file(tmp_path / "tests" / "test_src.py", "import src\n")
        result = compute_import_distances(
            tmp_path,
            ["tests/test_src.py::test_x"],
            ["src.py"],
        )
        assert result[0].import_distance == 1

    def test_transitive_import(self, tmp_path: Path) -> None:
        write_file(tmp_path / "core.py", "x = 1\n")
        write_file(tmp_path / "lib.py", "import core\n")
        write_file(tmp_path / "tests" / "test_lib.py", "import lib\n")
        result = compute_import_distances(
            tmp_path,
            ["tests/test_lib.py::test_y"],
            ["core.py"],
        )
        assert result[0].import_distance == 2

    def test_unreachable_test(self, tmp_path: Path) -> None:
        write_file(tmp_path / "src.py", "x = 1\n")
        write_file(tmp_path / "tests" / "test_other.py", "# no imports\n")
        result = compute_import_distances(
            tmp_path,
            ["tests/test_other.py::test_z"],
            ["src.py"],
        )
        assert result[0].import_distance == UNREACHABLE

    def test_changed_file_itself(self, tmp_path: Path) -> None:
        """A test file that IS the changed file has distance 0."""
        write_file(tmp_path / "tests" / "test_foo.py", "# test\n")
        result = compute_import_distances(
            tmp_path,
            ["tests/test_foo.py::test_a"],
            ["tests/test_foo.py"],
        )
        assert result[0].import_distance == 0

    def test_empty_changed_list(self, tmp_path: Path) -> None:
        write_file(tmp_path / "tests" / "test_foo.py", "# test\n")
        result = compute_import_distances(
            tmp_path,
            ["tests/test_foo.py::test_a"],
            [],
        )
        assert result[0].import_distance == UNREACHABLE

    def test_multiple_node_ids_ordering(self, tmp_path: Path) -> None:
        write_file(tmp_path / "src.py", "x = 1\n")
        write_file(tmp_path / "tests" / "test_a.py", "import src\n")
        write_file(tmp_path / "tests" / "test_b.py", "# no imports\n")
        result = compute_import_distances(
            tmp_path,
            ["tests/test_a.py::test_x", "tests/test_b.py::test_y"],
            ["src.py"],
        )
        assert result[0].import_distance == 1
        assert result[1].import_distance == UNREACHABLE

    def test_pre_built_graph_accepted(self, tmp_path: Path) -> None:
        """If a pre-built graph is passed, build_import_graph is not called."""
        graph: dict[str, set[str]] = {
            "tests/test_a.py": {"src.py"},
            "src.py": set(),
        }
        result = compute_import_distances(
            tmp_path,  # repo not used when graph is provided
            ["tests/test_a.py::test_x"],
            ["src.py"],
            graph=graph,
        )
        assert result[0].import_distance == 1

    def test_import_graph_features_are_frozen(self) -> None:
        feat = ImportGraphFeatures(node_id="t::t", import_distance=5)
        with pytest.raises(Exception):
            feat.import_distance = 0  # type: ignore[misc]
