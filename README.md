# FailFast

FailFast reorders a project's test suite so the tests most likely to fail run first.

---

## The problem

On large code bases, CI can wait minutes—or tens of minutes—before surfacing the
first test failure. Every minute of delay is a minute a developer sits idle or
switches context. The bottleneck is almost always a fixed test order that ignores
what actually changed.

## How it works

```
git diff
   │
   ▼
Feature extractor
   ├── coverage overlap  (which tests touch the changed lines?)
   ├── import-graph distance  (how many hops from the changed module?)
   └── test history  (failure rate, recency, runtime, flakiness)
   │
   ▼
ML model  (LightGBM or baseline heuristic)
   │
   ▼
Scheduler  (rank by P(fail)/runtime, or select within time budget)
   │
   ▼
Parallel test runner  →  SQLite + REST API  →  Dashboard / GitHub Action
```

## Features

FailFast extracts three groups of numeric features for every (commit, test) pair.
All features are computed before the test runs; none require the test outcome.

### Coverage overlap

Compares the diff's changed lines against the per-test coverage map produced by
`coverage.py` contexts.

| Feature | Description |
|---|---|
| `overlap_lines` | Count of changed lines the test executes |
| `overlap_files` | Count of changed files the test covers |
| `overlap_ratio` | `overlap_lines / total_changed_lines` (0 when no lines changed) |

A test that directly executes recently changed code is strongly associated with
the change and therefore a high-priority candidate to run first.

### Import-graph distance

Parses every `.py` file in the project using Python's `ast` module to build a
directed import graph. The graph handles absolute imports, relative imports
(`from . import utils`, `from ..core import x`), and from-imports
(`from foo import bar`). Then it performs a multi-source BFS from the changed
files to each test, measuring the shortest path in import hops.

| Feature | Description |
|---|---|
| `import_distance` | Shortest hop count from the test file to any changed file. `999` when no path exists (sentinel for unreachable tests) |

The sentinel value `999` is large enough that unreachable tests sort to the
bottom of any ranking that uses this feature.

### Test history

Derived from the SQLite run store. Only runs recorded **before** the commit
being scored are used—this strict cutoff prevents label leakage: the model
never sees the outcome it is trying to predict.

| Feature | Description |
|---|---|
| `failure_rate` | `failures / total_runs` over all history (0 if no history) |
| `recent_failure_count` | Failures in the last 10 runs |
| `runs_since_last_failure` | Consecutive passing runs since the most recent failure (`999` if never failed) |
| `flakiness` | Fraction of consecutive run-pairs where the outcome flipped |
| `avg_duration` | Mean wall-clock execution time in seconds |
| `last_duration` | Duration of the most recent run in seconds |

## Current status (Day 2 – feature extraction and dataset builder)

| Module | What it does |
|---|---|
| `failfast/diff.py` | Parses `git diff` between two refs into changed files and line ranges |
| `failfast/discovery.py` | Discovers pytest test node IDs via `--collect-only` |
| `failfast/coverage_map.py` | Runs the suite with `coverage.py` contexts and builds a test-to-file/lines map |
| `failfast/store.py` + `failfast/schema.sql` | SQLite store for runs, tests, and outcomes |
| `failfast/features/coverage_overlap.py` | Coverage overlap features |
| `failfast/features/import_graph.py` | AST-based import graph builder and distance computation |
| `failfast/features/history.py` | Historical failure rate, recency, flakiness, and duration features |
| `failfast/features/builder.py` | Assembles all feature groups into a pandas DataFrame with stable column order |
| `failfast/dataset.py` | Labeled dataset builder, time-based train/test split, Parquet/CSV persistence |
| `failfast/simulate.py` | Synthetic data generator (fault injection); all output is tagged synthetic |
| `failfast/cli.py` | `failfast discover`, `failfast diff`, `failfast map`, `failfast features`, `failfast dataset build`, `failfast simulate` |
| `tests/` | 121+ unit and integration tests (≥ 85% coverage) |
| `.github/workflows/ci.yml` | Lint (ruff + black) and pytest on Python 3.11 and 3.12 |

Model training, the scheduler, the parallel runner, FastAPI endpoints, the
dashboard, and the GitHub Action are **planned** (Days 3–5).

## Quick start

```bash
# Install (requires Python 3.11+)
pip install -e ".[dev]"

# Discover tests in the current directory
failfast discover .

# Show files changed between main and HEAD
failfast diff main HEAD .

# Collect per-test coverage map (requires pytest-cov in the target env)
failfast map .

# Print the feature table for the current diff
failfast features --base main --head HEAD .

# Build a labeled training dataset from recorded run history
failfast dataset build . --db failfast.db --out data

# Generate a small synthetic dataset for pipeline validation
failfast simulate --commits 20 --out data/synthetic

# Run the test suite
pytest

# Lint
ruff check .
black --check .
```

## Important note on synthetic data

`failfast simulate` generates training data by injecting trivial mutations into
a small fixture project. This data is useful for validating the feature pipeline
end-to-end and for demos, but it does not reflect real-world failure
distributions. Every file it produces carries `is_synthetic=1` in every row and
`synthetic: true` in its manifest. Do not use synthetic data as evidence of
real-world FailFast performance.

## Project layout

```
failfast/
├── __init__.py
├── cli.py              entry point (Click commands)
├── coverage_map.py     per-test coverage collection and parsing
├── dataset.py          labeled dataset builder and time-based split
├── diff.py             git diff parser
├── discovery.py        pytest test discovery
├── features/
│   ├── __init__.py
│   ├── builder.py      combines all feature groups into a DataFrame
│   ├── coverage_overlap.py  coverage-line overlap features
│   ├── history.py      historical run features (failure rate, flakiness, …)
│   └── import_graph.py AST-based import graph and distance computation
├── models.py           ChangedFile and Hunk dataclasses
├── schema.sql          SQLite DDL
├── simulate.py         synthetic training data generator
└── store.py            database layer

tests/
├── conftest.py               shared fixtures (mini git repo in tmp_path)
├── test_coverage_map.py
├── test_coverage_overlap.py
├── test_dataset.py
├── test_diff.py
├── test_discovery.py
├── test_feature_builder.py
├── test_history_features.py
├── test_import_graph.py
└── test_store.py

.github/workflows/
└── ci.yml          lint + test on push and pull request
```

## Roadmap

- [x] Day 1 — Foundation: diff parser, test discovery, coverage map, SQLite store, CLI, CI
- [x] Day 2 — Feature extraction: coverage overlap, import-graph distance, history features, dataset builder, synthetic data simulator
- [ ] Day 3 — Model training: baseline heuristic vs LightGBM, APFD and time-to-first-failure metrics, model persistence *(planned)*
- [ ] Day 4 — Scheduler and runner: P(fail)/runtime ranking, time-budget mode, parallel execution, FastAPI endpoints *(planned)*
- [ ] Day 5 — Dashboard, GitHub Action, Dockerfile, benchmark, README polish *(planned)*

## License

MIT. See [LICENSE](LICENSE).
