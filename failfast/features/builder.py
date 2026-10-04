"""Feature builder – assembles all feature groups into a single DataFrame.

This module is the single source of truth for the feature schema.  Every column
name and its definition is documented here so that the model training code and
the README can reference one authoritative list.

Feature schema (stable column order)
-------------------------------------

Coverage-overlap group:

- ``overlap_lines``  – count of changed lines executed by the test.
- ``overlap_files``  – count of changed files covered by the test.
- ``overlap_ratio``  – ``overlap_lines / total_changed_lines`` (0.0 if no changes).

Import-graph group:

- ``import_distance`` – shortest hop count from the test file to any changed
  file in the project import graph.  ``999`` (``UNREACHABLE``) when no path
  exists.

History group (computed from runs *before* the scored commit):

- ``failure_rate``          – fraction of past runs that were failures.
- ``recent_failure_count``  – failures in the last 10 runs.
- ``runs_since_last_failure`` – consecutive passing runs since last failure
  (``999`` if never failed).
- ``flakiness``             – fraction of consecutive run-pairs with a
  flipped outcome.
- ``avg_duration``          – mean execution time (seconds).
- ``last_duration``         – duration of the most recent run (seconds).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from failfast.coverage_map import CoverageEntry
from failfast.features.coverage_overlap import compute_coverage_overlap
from failfast.features.history import HistoryFeatures, compute_history_features
from failfast.features.import_graph import UNREACHABLE, compute_import_distances
from failfast.models import ChangedFile
from failfast.store import Store

if TYPE_CHECKING:
    import pandas as pd

# Stable, documented column order used for every output DataFrame.
FEATURE_COLUMNS: list[str] = [
    # Coverage overlap
    "overlap_lines",
    "overlap_files",
    "overlap_ratio",
    # Import graph
    "import_distance",
    # History
    "failure_rate",
    "recent_failure_count",
    "runs_since_last_failure",
    "flakiness",
    "avg_duration",
    "last_duration",
]


def build_feature_row(
    node_id: str,
    overlap_map: dict[str, dict],
    graph_map: dict[str, int],
    history_map: dict[str, HistoryFeatures],
) -> dict[str, float | int | str]:
    """Assemble one feature row for *node_id* from pre-computed group maps.

    Args:
        node_id: Pytest node ID.
        overlap_map: Mapping from node_id to overlap feature values.
        graph_map: Mapping from node_id to import distance.
        history_map: Mapping from node_id to :class:`HistoryFeatures`.

    Returns:
        Dict with ``node_id`` plus all feature columns from
        :data:`FEATURE_COLUMNS`.
    """
    ov = overlap_map.get(node_id, {})
    dist = graph_map.get(node_id, UNREACHABLE)
    hist = history_map.get(node_id)

    return {
        "node_id": node_id,
        "overlap_lines": ov.get("overlap_lines", 0),
        "overlap_files": ov.get("overlap_files", 0),
        "overlap_ratio": ov.get("overlap_ratio", 0.0),
        "import_distance": dist,
        "failure_rate": hist.failure_rate if hist else 0.0,
        "recent_failure_count": hist.recent_failure_count if hist else 0,
        "runs_since_last_failure": hist.runs_since_last_failure if hist else 999,
        "flakiness": hist.flakiness if hist else 0.0,
        "avg_duration": hist.avg_duration if hist else 0.0,
        "last_duration": hist.last_duration if hist else 0.0,
    }


def build_features(
    repo: Path,
    node_ids: list[str],
    coverage_entries: list[CoverageEntry],
    changed_files: list[ChangedFile],
    store: Store,
    before_timestamp: float,
    *,
    recent_n: int = 10,
    import_graph: dict[str, set[str]] | None = None,
) -> "pd.DataFrame":
    """Build a feature DataFrame for a set of tests against a given diff.

    Args:
        repo: Repository root (used to build the import graph).
        node_ids: Pytest node IDs to include in the output.
        coverage_entries: Per-test coverage from
            :func:`failfast.coverage_map.build_coverage_map`.
        changed_files: Parsed diff from :func:`failfast.diff.diff_refs`.
        store: Open :class:`failfast.store.Store` for history queries.
        before_timestamp: Unix timestamp upper bound for history features
            (prevents label leakage – only data before this point is used).
        recent_n: Window for ``recent_failure_count``.
        import_graph: Pre-built import graph; built from *repo* if ``None``.

    Returns:
        A :class:`pandas.DataFrame` with one row per node ID, columns in
        :data:`FEATURE_COLUMNS` order, indexed by ``node_id``.
    """
    import pandas as pd

    # 1. Coverage overlap
    overlap_results = compute_coverage_overlap(coverage_entries, changed_files)
    overlap_map: dict[str, dict] = {
        r.node_id: {
            "overlap_lines": r.overlap_lines,
            "overlap_files": r.overlap_files,
            "overlap_ratio": r.overlap_ratio,
        }
        for r in overlap_results
    }

    # 2. Import distances
    changed_paths = [cf.path for cf in changed_files]
    graph_results = compute_import_distances(
        repo, node_ids, changed_paths, graph=import_graph
    )
    graph_map: dict[str, int] = {r.node_id: r.import_distance for r in graph_results}

    # 3. History features
    history_results = compute_history_features(
        store, node_ids, before_timestamp, recent_n=recent_n
    )
    history_map: dict[str, HistoryFeatures] = {r.node_id: r for r in history_results}

    # 4. Assemble rows
    rows = [
        build_feature_row(nid, overlap_map, graph_map, history_map)
        for nid in node_ids
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["node_id"] + FEATURE_COLUMNS)
    else:
        # Enforce stable column order
        df = df[["node_id"] + FEATURE_COLUMNS]

    return df.set_index("node_id")
