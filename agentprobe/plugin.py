"""Pytest plugin — registers the agentprobe fixture and CLI flags."""

from __future__ import annotations

import pytest

from agentprobe.snapshot import Snapshot, _start_session_recording, _stop_session_recording


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
        choices=["exact", "semantic", "fuzzy"],
        help="Comparison mode: 'exact', 'semantic' or 'fuzzy' (default: exact).",
    )
    group.addoption(
        "--agentprobe-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for semantic/fuzzy mode (default: 0.85).",
    )
    group.addoption(
        "--agentprobe-redact-secrets",
        action="store_true",
        default=False,
        help="Mask API keys, tokens, and emails in snapshots before storing/comparing.",
    )
    group.addoption(
        "--agentprobe-evalport",
        metavar="PATH",
        default=None,
        help="Write the run's snapshot results to PATH as an EvalPort ResultSet JSON.",
    )


def pytest_configure(config):
    if config.getoption("--agentprobe-evalport"):
        from agentprobe import evalport

        config._agentprobe_evalport_results = []
        config._agentprobe_evalport_started = evalport._utcnow()
        _start_session_recording(config._agentprobe_evalport_results.append)


def pytest_sessionfinish(session):
    config = session.config
    path = config.getoption("--agentprobe-evalport")
    if not path:
        return
    _stop_session_recording()
    results = getattr(config, "_agentprobe_evalport_results", [])
    if not results:
        return
    from agentprobe.evalport import write_resultset

    write_resultset(results, path, started_at=config._agentprobe_evalport_started)


def pytest_terminal_summary(terminalreporter, config):
    path = config.getoption("--agentprobe-evalport")
    if not path:
        return
    results = getattr(config, "_agentprobe_evalport_results", [])
    if results:
        terminalreporter.write_line(
            f"agentprobe: wrote EvalPort ResultSet ({len(results)} result(s)) to {path}"
        )
    else:
        terminalreporter.write_line("agentprobe: no snapshot results recorded, nothing exported")


@pytest.fixture
def agentprobe(request) -> Snapshot:
    """Fixture that provides an AgentProbe Snapshot instance."""
    update = request.config.getoption("--agentprobe-update", default=False)
    mode = request.config.getoption("--agentprobe-mode", default="exact")
    threshold = request.config.getoption("--agentprobe-threshold", default=0.85)
    redact_secrets = request.config.getoption("--agentprobe-redact-secrets", default=False)
    return Snapshot(update=update, mode=mode, threshold=threshold, redact_secrets=redact_secrets)
