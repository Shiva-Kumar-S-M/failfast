-- FailFast SQLite schema
-- Stores test run history for feature extraction.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per CI invocation of FailFast.
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_sha  TEXT    NOT NULL,
    base_sha    TEXT    NOT NULL,
    created_at  REAL    NOT NULL  -- Unix timestamp (seconds since epoch)
);

-- All tests discovered in the repo; identified by their pytest node ID.
CREATE TABLE IF NOT EXISTS tests (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT    NOT NULL UNIQUE
);

-- Outcome of one test in one run.
CREATE TABLE IF NOT EXISTS results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    test_id     INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    outcome     TEXT    NOT NULL CHECK(outcome IN ('passed', 'failed', 'error', 'skipped', 'xfailed', 'xpassed')),
    duration_s  REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(run_id, test_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run    ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_test   ON results(test_id);
CREATE INDEX IF NOT EXISTS idx_results_outcome ON results(outcome);
CREATE INDEX IF NOT EXISTS idx_runs_commit    ON runs(commit_sha);
