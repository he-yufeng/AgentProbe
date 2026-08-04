"""CLI — thin wrapper around pytest with AgentProbe flags."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from difflib import unified_diff
from pathlib import Path

import click

from agentprobe.storage import (
    delete_last_run,
    list_last_runs,
    load_last_run,
    load_snapshot,
    save_snapshot,
)


@click.group()
@click.version_option()
def main():
    """AgentProbe — regression-testing for AI agents."""


@main.command()
@click.option("--mode", default="exact", type=click.Choice(["exact", "semantic"]))
@click.option("--threshold", default=0.85, type=float, help="Semantic similarity threshold.")
@click.argument("pytest_args", nargs=-1)
def run(mode: str, threshold: float, pytest_args: tuple[str, ...]):
    """Run agent tests via pytest."""
    cmd = [
        sys.executable, "-m", "pytest",
        f"--agentprobe-mode={mode}",
        f"--agentprobe-threshold={threshold}",
        *pytest_args,
    ]
    raise SystemExit(subprocess.call(cmd))


@main.command()
@click.argument("pytest_args", nargs=-1)
def update(pytest_args: tuple[str, ...]):
    """Re-run tests and update all snapshots."""
    cmd = [
        sys.executable, "-m", "pytest",
        "--agentprobe-update",
        *pytest_args,
    ]
    raise SystemExit(subprocess.call(cmd))


def _diff_lines(baseline: object, actual: object) -> str:
    expected = json.dumps(baseline, indent=2, sort_keys=True, default=str).splitlines()
    current = json.dumps(actual, indent=2, sort_keys=True, default=str).splitlines()
    return "\n".join(
        unified_diff(expected, current, fromfile="baseline", tofile="last_run", lineterm="")
    )


@main.command()
@click.argument("name", required=False)
@click.option("--html", "html_path", default=None, type=click.Path(dir_okay=False, path_type=Path),
              help="Write the diffs as a self-contained HTML report instead of printing.")
def diff(name: str | None, html_path: Path | None):
    """Show baseline vs last failing run, one name or all of them."""
    names = [name] if name else list_last_runs()
    if not names:
        click.echo("No saved failing runs.")
        return
    items = []
    for snap_name in names:
        last = load_last_run(snap_name)
        if last is None:
            raise click.ClickException(f"No saved failing run for '{snap_name}'.")
        baseline = load_snapshot(snap_name)
        baseline_output = baseline.get("output") if baseline else None
        diff_text = _diff_lines(baseline_output, last.get("output"))
        if html_path is not None:
            items.append({
                "name": snap_name,
                "diff": diff_text,
                "similarity": last.get("similarity"),
                "mode": last.get("mode"),
                "threshold": last.get("threshold"),
            })
            continue
        click.echo(f"--- {snap_name} ---")
        if last.get("similarity") is not None:
            click.echo(f"similarity: {last['similarity']:.4f} (mode={last.get('mode')}, threshold={last.get('threshold')})")
        click.echo(diff_text)
        click.echo()
    if html_path is not None:
        from .report import render_diff_report
        html_path.write_text(render_diff_report(items), encoding="utf-8")
        click.echo(f"Wrote diff report to {html_path}")


@main.command()
@click.argument("name", required=False)
def accept(name: str | None):
    """Promote the last failing run to the baseline, one name or all of them."""
    names = [name] if name else list_last_runs()
    if not names:
        click.echo("No saved failing runs to accept.")
        return
    for snap_name in names:
        last = load_last_run(snap_name)
        if last is None:
            raise click.ClickException(f"No saved failing run for '{snap_name}'.")
        save_snapshot(snap_name, {"output": last.get("output"), "timestamp": time.time()})
        delete_last_run(snap_name)
        click.echo(f"Accepted new baseline for '{snap_name}'.")
