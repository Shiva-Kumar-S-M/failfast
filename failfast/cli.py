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
                    path: sorted(lines)
                    for path, lines in e.executed_lines.items()
                },
            }
            for e in entries
        ]
        click.echo(json.dumps(payload, indent=2))
    else:
        for entry in entries:
            click.echo(f"{entry.node_id}: {', '.join(sorted(entry.executed_lines))}")
