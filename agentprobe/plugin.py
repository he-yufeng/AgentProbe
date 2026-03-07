"""Pytest plugin — registers the agentprobe fixture and CLI flags."""

from __future__ import annotations

import pytest

from agentprobe.snapshot import Snapshot


def pytest_addoption(parser):
    group = parser.getgroup("agentprobe", "AgentProbe snapshot testing")
    group.addoption(
        "--agentprobe-update",
        action="store_true",
        default=False,
        help="Update all AgentProbe snapshots instead of comparing.",
    )
    group.addoption(
        "--agentprobe-mode",
        default="exact",
        choices=["exact", "semantic"],
        help="Comparison mode: 'exact' or 'semantic' (default: exact).",
    )
    group.addoption(
        "--agentprobe-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for semantic mode (default: 0.85).",
    )


@pytest.fixture
def agentprobe(request) -> Snapshot:
    """Fixture that provides an AgentProbe Snapshot instance."""
    update = request.config.getoption("--agentprobe-update", default=False)
    mode = request.config.getoption("--agentprobe-mode", default="exact")
    threshold = request.config.getoption("--agentprobe-threshold", default=0.85)
    return Snapshot(update=update, mode=mode, threshold=threshold)
