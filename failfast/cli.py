"""Command-line interface for FailFast."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from failfast.diff import diff_refs, parse_diff_text
from failfast.discovery import discover_tests


@click.group()
@click.version_option(package_name="failfast")
def main() -> None:
    """FailFast – run the tests most likely to fail first."""


# ---------------------------------------------------------------------------
# failfast discover
# ---------------------------------------------------------------------------


@main.command("discover")
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON array.")
@click.option(
    "--path",
    "paths",
    multiple=True,
    help="Restrict discovery to these sub-paths (repeatable).",
)
def cmd_discover(repo: str, as_json: bool, paths: tuple[str, ...]) -> None:
    """List every pytest test node ID found in REPO.

    REPO defaults to the current directory.
    """
    repo_path = Path(repo).resolve()
    try:
        node_ids = discover_tests(repo_path, paths=list(paths) if paths else None)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(node_ids, indent=2))
    else:
        for nid in node_ids:
            click.echo(nid)


# ---------------------------------------------------------------------------
# failfast diff
# ---------------------------------------------------------------------------


@main.command("diff")
@click.argument("base")
@click.argument("head", default="HEAD")
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--stdin",
    "from_stdin",
    is_flag=True,
    help="Read diff text from stdin instead of running git.",
)
def cmd_diff(
    base: str,
    head: str,
    repo: str,
    as_json: bool,
    from_stdin: bool,
) -> None:
    """Show files changed between BASE and HEAD in REPO.

    BASE is the only required argument (e.g. 'main' or a commit SHA).
    HEAD defaults to HEAD. REPO defaults to the current directory.

    Example:

        failfast diff main HEAD .

        git diff main | failfast diff --stdin main
    """
    try:
        if from_stdin:
            text = sys.stdin.read()
            changed = parse_diff_text(text)
        else:
            repo_path = Path(repo).resolve()
            changed = diff_refs(repo_path, base=base, head=head)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        payload = [
            {
                "path": f.path,
                "old_path": f.old_path,
                "change_type": f.change_type,
                "added_lines": sorted(f.added_lines),
                "removed_lines": sorted(f.removed_lines),
            }
            for f in changed
        ]
        click.echo(json.dumps(payload, indent=2))
    else:
        for f in changed:
            tag = f.change_type[0].upper()
            click.echo(f"[{tag}] {f.path}")


# ---------------------------------------------------------------------------
# failfast map
# ---------------------------------------------------------------------------


@main.command("map")
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--source", default=None, help="Package/directory to measure coverage.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def cmd_map(repo: str, source: str | None, as_json: bool) -> None:
    """Collect per-test coverage and print the test-to-file mapping.

    Runs the full test suite with coverage.py contexts enabled. Requires
    pytest-cov to be installed in the target repo's environment.

    This command can take several minutes for large suites.
    """
    # Import here to avoid making coverage_map a hard dependency at import time
    from failfast.coverage_map import build_coverage_map

    repo_path = Path(repo).resolve()
    try:
        entries = build_coverage_map(repo_path, source=source)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        payload = [
            {
                "node_id": e.node_id,
                "files": {
                    path: sorted(lines) for path, lines in e.executed_lines.items()
                },
            }
            for e in entries
        ]
        click.echo(json.dumps(payload, indent=2))
    else:
        for entry in entries:
            click.echo(f"{entry.node_id}: {', '.join(sorted(entry.executed_lines))}")


# ---------------------------------------------------------------------------
# failfast features
# ---------------------------------------------------------------------------


@main.command("features")
@click.option("--base", required=True, help="Base git ref for the diff.")
@click.option("--head", default="HEAD", show_default=True, help="Head git ref.")
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--db", "db_path", default=None, help="Path to FailFast SQLite database.")
def cmd_features(base: str, head: str, repo: str, db_path: str | None) -> None:
    """Print the feature table for all tests affected by a diff.

    BASE is the only required option (e.g. 'main' or a commit SHA).
    HEAD defaults to HEAD.  REPO defaults to the current directory.

    Example:

        failfast features --base main --head HEAD .
    """
    from failfast.coverage_map import build_coverage_map
    from failfast.features.builder import build_features
    from failfast.store import Store

    repo_path = Path(repo).resolve()
    db = Path(db_path) if db_path else repo_path / "failfast.db"

    try:
        changed = diff_refs(repo_path, base=base, head=head)
    except Exception as exc:
        click.echo(f"error computing diff: {exc}", err=True)
        sys.exit(1)

    if not changed:
        click.echo("No changed files found.")
        return

    try:
        cov_entries = build_coverage_map(repo_path)
    except Exception as exc:
        click.echo(f"error building coverage map: {exc}", err=True)
        sys.exit(1)

    node_ids = [e.node_id for e in cov_entries]
    if not node_ids:
        click.echo("No tests found.")
        return

    import time

    store = Store(db)
    try:
        df = build_features(
            repo_path,
            node_ids,
            cov_entries,
            changed,
            store,
            before_timestamp=time.time(),
        )
    except Exception as exc:
        click.echo(f"error building features: {exc}", err=True)
        sys.exit(1)
    finally:
        store.close()

    click.echo(df.to_string())


# ---------------------------------------------------------------------------
# failfast dataset
# ---------------------------------------------------------------------------


@main.group("dataset")
def cmd_dataset() -> None:
    """Build and manage labeled training datasets."""


@cmd_dataset.command("build")
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--db", "db_path", default=None, help="Path to FailFast SQLite database.")
@click.option(
    "--out",
    "output_dir",
    default="data",
    show_default=True,
    help="Output directory for the dataset.",
)
@click.option(
    "--name",
    default="dataset",
    show_default=True,
    help="Base name for output files.",
)
def cmd_dataset_build(
    repo: str, db_path: str | None, output_dir: str, name: str
) -> None:
    """Build a labeled dataset from the run history in DB.

    REPO defaults to the current directory.

    Example:

        failfast dataset build . --db failfast.db --out data --name v1
    """

    import pandas as pd

    from failfast.coverage_map import build_coverage_map
    from failfast.dataset import build_labeled_dataset, save_dataset
    from failfast.features.builder import build_features
    from failfast.store import Store

    repo_path = Path(repo).resolve()
    db = Path(db_path) if db_path else repo_path / "failfast.db"

    if not db.exists():
        click.echo(f"error: database not found: {db}", err=True)
        sys.exit(1)

    store = Store(db)
    runs = store.list_runs()
    if not runs:
        click.echo("No runs found in the database.")
        store.close()
        return

    try:
        cov_entries = build_coverage_map(repo_path)
    except Exception as exc:
        click.echo(f"error building coverage map: {exc}", err=True)
        store.close()
        sys.exit(1)

    all_frames: list[pd.DataFrame] = []

    for run in runs:
        results = store.get_results_for_run(run.id)
        if not results:
            continue

        test_ids = {r.test_id for r in results}
        node_id_rows = store._conn.execute(
            f"SELECT id, node_id FROM tests WHERE id IN ({','.join('?' * len(test_ids))})",
            list(test_ids),
        ).fetchall()
        id_to_node = {r["id"]: r["node_id"] for r in node_id_rows}

        outcome_map = {
            id_to_node[r.test_id]: r.outcome for r in results if r.test_id in id_to_node
        }
        node_ids = list(outcome_map.keys())

        try:
            feat_df = build_features(
                repo_path,
                node_ids,
                cov_entries,
                [],
                store,
                before_timestamp=run.created_at,
            )
            labeled = build_labeled_dataset(feat_df, outcome_map)
            labeled["commit_sha"] = run.commit_sha
            labeled["commit_timestamp"] = run.created_at
            all_frames.append(labeled.reset_index())
        except Exception:
            continue

    store.close()

    if not all_frames:
        click.echo(
            "No data rows assembled. Run 'failfast map' first to collect history."
        )
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    out_path = save_dataset(
        full_df.set_index("node_id"),
        Path(output_dir),
        name,
    )
    click.echo(f"Saved {len(full_df)} rows to {out_path}")


# ---------------------------------------------------------------------------
# failfast simulate
# ---------------------------------------------------------------------------


@main.command("simulate")
@click.option(
    "--commits",
    "n_commits",
    default=20,
    show_default=True,
    help="Number of synthetic commits to generate.",
)
@click.option(
    "--out",
    "output_dir",
    default="data/synthetic",
    show_default=True,
    help="Output directory for synthetic dataset.",
)
@click.option("--seed", default=42, show_default=True, help="Random seed.")
def cmd_simulate(n_commits: int, output_dir: str, seed: int) -> None:
    """Generate a synthetic training dataset by injecting code faults.

    WARNING: Output is entirely synthetic (fabricated).  It is only useful for
    end-to-end testing of the feature pipeline and for demos.  The saved
    manifest marks the data as synthetic=True and every row contains
    is_synthetic=1.  Do not use this data to benchmark real performance.
    """
    from failfast.simulate import run_simulation

    click.echo("Generating synthetic dataset (this may take a minute)…")
    out_path = run_simulation(
        output_dir=Path(output_dir),
        n_commits=n_commits,
        seed=seed,
    )
    click.echo(f"Synthetic dataset saved to: {out_path}")
    click.echo(
        "NOTE: This data is synthetic and must not be used as evidence of "
        "real-world FailFast performance."
    )
