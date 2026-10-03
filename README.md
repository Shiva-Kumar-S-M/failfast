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

## Current status (Day 1 – foundation)

The following exists today:

| Module | What it does |
|---|---|
| `failfast/diff.py` | Parses `git diff` between two refs into changed files and line ranges |
| `failfast/discovery.py` | Discovers pytest test node IDs via `--collect-only` |
| `failfast/coverage_map.py` | Runs the suite with `coverage.py` contexts and builds a test-to-file/lines map |
| `failfast/store.py` + `failfast/schema.sql` | SQLite store for runs, tests, and outcomes |
| `failfast/cli.py` | `failfast discover`, `failfast diff`, `failfast map` |
| `tests/` | Unit tests for each module (≥ 85% coverage) |
| `.github/workflows/ci.yml` | Lint (ruff + black) and pytest on Python 3.11 and 3.12 |

Feature extraction, model training, the scheduler, the parallel runner, FastAPI
endpoints, the dashboard, and the GitHub Action are **planned** (Days 2–5).

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

# Run the test suite
pytest

# Lint
ruff check .
black --check .
```

## Project layout

```
failfast/
├── __init__.py
├── cli.py          entry point (Click commands)
├── coverage_map.py per-test coverage collection and parsing
├── diff.py         git diff parser
├── discovery.py    pytest test discovery
├── models.py       ChangedFile and Hunk dataclasses
├── schema.sql      SQLite DDL
└── store.py        database layer

tests/
├── conftest.py     shared fixtures (mini git repo in tmp_path)
├── test_coverage_map.py
├── test_diff.py
├── test_discovery.py
└── test_store.py

.github/workflows/
└── ci.yml          lint + test on push and pull request
```

## Roadmap

- [x] Day 1 — Foundation: diff parser, test discovery, coverage map, SQLite store, CLI, CI
- [ ] Day 2 — Feature extraction: coverage overlap, import-graph distance, history features (failure rate, recency, flakiness, runtime), dataset builder *(planned)*
- [ ] Day 3 — Model training: baseline heuristic vs LightGBM, APFD and time-to-first-failure metrics, model persistence *(planned)*
- [ ] Day 4 — Scheduler and runner: P(fail)/runtime ranking, time-budget mode, parallel execution, FastAPI endpoints *(planned)*
- [ ] Day 5 — Dashboard, GitHub Action, Dockerfile, benchmark, README polish *(planned)*

## License

MIT. See [LICENSE](LICENSE).
