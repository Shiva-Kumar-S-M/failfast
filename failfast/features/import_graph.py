"""Import-graph distance features for test prioritization.

This module walks a project's Python source files using the ``ast`` module to
build a directed import graph, then measures the shortest-path distance from
each test file to any changed source file.  A test that imports (directly or
transitively) a module that changed is more likely to be affected by the diff.

Graph construction
------------------
Each ``.py`` file in the project is a node.  An edge ``A -> B`` exists when
file *A* contains an import that resolves to file *B*.  The module handles:

- **Absolute imports**: ``import foo.bar`` → resolves via ``sys.path``-style
  lookup inside the project root.
- **Relative imports**: ``from . import utils`` or ``from ..core import x`` →
  resolved relative to the importing file's package.
- **From-imports**: ``from foo import bar`` → tries ``foo/bar.py`` first, then
  ``foo.py`` (bar may be a name inside the module).

Sentinel value
--------------
When a test has no path (direct or transitive) to any changed file, the
distance is set to ``UNREACHABLE = 999``.  This large sentinel ensures that
unreachable tests sort to the bottom when features are used for ranking.
"""

from __future__ import annotations

import ast
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Distance assigned when no import path exists between the test and any changed file.
UNREACHABLE: int = 999


@dataclass(frozen=True)
class ImportGraphFeatures:
    """Import-graph distance for one test.

    Attributes:
        node_id: Pytest node ID of the test.
        import_distance: Shortest hop count from the test file to any changed
            file.  ``UNREACHABLE`` (999) when no path exists.
    """

    node_id: str
    import_distance: int


def build_import_graph(repo: Path) -> dict[str, set[str]]:
    """Parse every ``.py`` file in *repo* and return a directed import graph.

    The graph is represented as ``{importer_path: {importee_path, ...}}``
    where paths are POSIX strings relative to *repo*.

    Args:
        repo: Absolute path to the project root.

    Returns:
        Adjacency dict mapping each Python file path (relative, POSIX) to the
        set of files it imports.
    """
    graph: dict[str, set[str]] = {}
    py_files = list(repo.rglob("*.py"))

    # Build a lookup from dotted module name -> relative POSIX path
    module_to_path: dict[str, str] = {}
    for p in py_files:
        rel = p.relative_to(repo)
        parts = list(rel.parts)
        # Strip .py extension from last part
        parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            dotted = ".".join(parts)
            module_to_path[dotted] = rel.as_posix()

    for p in py_files:
        rel = p.relative_to(repo).as_posix()
        graph.setdefault(rel, set())
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(p))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _resolve_absolute(alias.name, module_to_path)
                    if target:
                        graph[rel].add(target)

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Relative import
                    targets = _resolve_relative(
                        node.module or "",
                        node.names,
                        node.level,
                        p,
                        repo,
                        module_to_path,
                    )
                    graph[rel].update(targets)
                elif node.module:
                    # Absolute from-import
                    targets = _resolve_from_import(
                        node.module, node.names, module_to_path
                    )
                    graph[rel].update(targets)

    return graph


def _resolve_absolute(
    name: str,
    module_to_path: dict[str, str],
) -> str | None:
    """Resolve ``import name`` to a repo-relative path, or None."""
    # Try exact match, then progressively shorter prefixes
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in module_to_path:
            return module_to_path[candidate]
    return None


def _resolve_from_import(
    module: str,
    names: list[ast.alias],
    module_to_path: dict[str, str],
) -> set[str]:
    """Resolve ``from module import names`` to repo-relative paths."""
    results: set[str] = set()
    # The module itself
    if module in module_to_path:
        results.add(module_to_path[module])
    # Each name might be a submodule: e.g. ``from foo import bar`` where bar.py exists
    for alias in names:
        candidate = f"{module}.{alias.name}"
        if candidate in module_to_path:
            results.add(module_to_path[candidate])
    return results


def _resolve_relative(
    module: str,
    names: list[ast.alias],
    level: int,
    importing_file: Path,
    repo: Path,
    module_to_path: dict[str, str],
) -> set[str]:
    """Resolve a relative import to repo-relative paths.

    Args:
        module: The ``from`` part after the dots (may be empty for ``from . import x``).
        names: The names being imported.
        level: Number of leading dots (1 = current package, 2 = parent, …).
        importing_file: Absolute path of the file containing the import.
        repo: Repository root.
        module_to_path: Module-name to path mapping.

    Returns:
        Set of resolved repo-relative POSIX paths.
    """
    results: set[str] = set()

    # Determine the base package by going ``level`` directories up
    package_parts = list(importing_file.relative_to(repo).parts[:-1])
    if not package_parts:
        return results

    # Go up (level - 1) more times for level > 1
    for _ in range(level - 1):
        if package_parts:
            package_parts.pop()

    # Append the module suffix if present
    if module:
        package_parts.extend(module.split("."))

    base_dotted = ".".join(package_parts)

    # The base module itself
    if base_dotted in module_to_path:
        results.add(module_to_path[base_dotted])

    # Each name may be a submodule
    for alias in names:
        candidate = f"{base_dotted}.{alias.name}" if base_dotted else alias.name
        if candidate in module_to_path:
            results.add(module_to_path[candidate])

    return results


def compute_import_distances(
    repo: Path,
    node_ids: list[str],
    changed_paths: list[str],
    *,
    graph: dict[str, set[str]] | None = None,
) -> list[ImportGraphFeatures]:
    """Compute the shortest import-path distance from each test to any changed file.

    Uses a multi-source BFS from all changed files simultaneously so that the
    distance returned is the minimum across all changed files.

    Args:
        repo: Repository root (used to build the graph when *graph* is None).
        node_ids: Pytest node IDs to score (e.g. ``['tests/test_foo.py::test_x']``).
        changed_paths: Repo-relative paths of files that changed.
        graph: Pre-built adjacency dict.  Built from *repo* when not supplied.

    Returns:
        One :class:`ImportGraphFeatures` per entry in *node_ids*, in order.
    """
    if graph is None:
        graph = build_import_graph(repo)

    # Build reverse graph: importee -> set of importers
    reverse: dict[str, set[str]] = {}
    for src, dests in graph.items():
        for dest in dests:
            reverse.setdefault(dest, set()).add(src)

    # Multi-source BFS from changed files through the reverse graph
    # so we find: "which test files (importers) can reach a changed file?"
    visited: dict[str, int] = {}  # path -> min distance
    queue: deque[tuple[str, int]] = deque()

    for cp in changed_paths:
        if cp not in visited:
            visited[cp] = 0
            queue.append((cp, 0))

    while queue:
        current, dist = queue.popleft()
        for importer in reverse.get(current, set()):
            if importer not in visited:
                visited[importer] = dist + 1
                queue.append((importer, dist + 1))

    features: list[ImportGraphFeatures] = []
    for node_id in node_ids:
        # Extract the file portion of the node ID (before ::)
        test_file = node_id.split("::")[0]
        distance = visited.get(test_file, UNREACHABLE)
        features.append(ImportGraphFeatures(node_id=node_id, import_distance=distance))

    return features
