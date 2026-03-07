"""CLI — thin wrapper around pytest with AgentProbe flags."""

from __future__ import annotations

import subprocess
import sys

import click


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
