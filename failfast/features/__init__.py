"""Feature extraction modules for FailFast.

Each sub-module produces a specific group of numeric features that feed the
ML-based test prioritization model:

- ``coverage_overlap``: line- and file-level overlap between a diff and the
  per-test coverage map.
- ``import_graph``: shortest import-path distance between changed source files
  and each test module.
- ``history``: historical failure rate, recency, flakiness, and duration
  features drawn from the SQLite run store.
- ``builder``: combines the groups above into a single pandas DataFrame with a
  stable column order.
"""
