"""Tests for snapshot capture and comparison."""

import asyncio
import shutil

import pytest

from agentprobe.snapshot import Snapshot, _redact, snapshot
from agentprobe.storage import DEFAULT_DIR


@pytest.fixture(autouse=True)
def clean_snapshots():
    """Remove snapshot dir before and after each test."""
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR)
    yield
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR)


def test_snapshot_creates_baseline():
    snap = Snapshot(update=False)
    result = snap.capture("test_create", {"answer": "42"})
    assert result.passed
    assert result.message == "snapshot created"


def test_snapshot_matches_same_output():
    snap = Snapshot(update=False)
    snap.capture("test_match", "hello world")
    result = snap.capture("test_match", "hello world")
    assert result.passed


def test_snapshot_detects_mismatch():
    snap = Snapshot(update=False)
    snap.capture("test_mismatch", "original output")
    result = snap.capture("test_mismatch", "completely different output")
    assert not result.passed


def test_snapshot_update_overwrites():
    snap = Snapshot(update=False)
    snap.capture("test_update", "v1")

    snap_update = Snapshot(update=True)
    result = snap_update.capture("test_update", "v2")
    assert result.passed
    assert result.message == "snapshot updated"

    snap_check = Snapshot(update=False)
    result = snap_check.capture("test_update", "v2")
    assert result.passed


def test_snapshot_handles_dict():
    snap = Snapshot()
    snap.capture("test_dict", {"key": "value", "nested": {"a": 1}})
    result = snap.capture("test_dict", {"key": "value", "nested": {"a": 1}})
    assert result.passed


def test_snapshot_handles_pydantic():
    from pydantic import BaseModel

    class Result(BaseModel):
        answer: str
        confidence: float

    snap = Snapshot()
    snap.capture("test_pydantic", Result(answer="yes", confidence=0.95))
    result = snap.capture("test_pydantic", Result(answer="yes", confidence=0.95))
    assert result.passed


def test_snapshot_decorator():
    call_count = 0

    @snapshot("test_decorator")
    def my_agent():
        nonlocal call_count
        call_count += 1
        return "stable output"

    my_agent()
    my_agent()
    assert call_count == 2


def test_snapshot_decorator_raises_on_mismatch():
    @snapshot("test_decorator_fail")
    def changing_agent():
        return changing_agent._counter

    changing_agent._counter = "first"
    changing_agent()

    changing_agent._counter = "second"
    with pytest.raises(AssertionError, match="mismatch"):
        changing_agent()


def test_snapshot_mismatch_error_includes_unified_diff():
    @snapshot("test_diff_output")
    def changing_agent():
        return {"answer": changing_agent._answer, "confidence": 0.8}

    changing_agent._answer = "old answer"
    changing_agent()

    changing_agent._answer = "new answer"
    with pytest.raises(AssertionError) as exc:
        changing_agent()

    message = str(exc.value)
    assert "--- snapshot" in message
    assert "+++ current" in message
    assert '-  "answer": "old answer"' in message
    assert '+  "answer": "new answer"' in message


def test_snapshot_decorator_supports_async_functions():
    calls = 0

    @snapshot("test_async_decorator")
    async def async_agent():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return {"answer": "stable"}

    assert asyncio.run(async_agent()) == {"answer": "stable"}
    assert asyncio.run(async_agent()) == {"answer": "stable"}
    assert calls == 2


def test_snapshot_decorator_raises_on_async_mismatch():
    @snapshot("test_async_decorator_fail")
    async def changing_agent():
        await asyncio.sleep(0)
        return changing_agent._value

    changing_agent._value = "first"
    asyncio.run(changing_agent())

    changing_agent._value = "second"
    with pytest.raises(AssertionError, match="mismatch"):
        asyncio.run(changing_agent())


# -- redaction of non-deterministic fields --


def test_redact_replaces_specified_keys_recursively():
    out = _redact(
        {"answer": "hi", "ts": 123, "nested": {"id": "x", "keep": 1}, "items": [{"id": "y"}]},
        {"ts", "id"},
    )
    assert out == {
        "answer": "hi",
        "ts": "<redacted>",
        "nested": {"id": "<redacted>", "keep": 1},
        "items": [{"id": "<redacted>"}],
    }


def test_snapshot_redact_ignores_nondeterministic_fields():
    snap = Snapshot()
    snap.capture("test_redact", {"answer": "42", "timestamp": 1000}, redact=["timestamp"])
    # a different timestamp must not break the snapshot when it is redacted
    result = snap.capture("test_redact", {"answer": "42", "timestamp": 9999}, redact=["timestamp"])
    assert result.passed


def test_snapshot_redact_still_detects_real_changes():
    snap = Snapshot()
    snap.capture("test_redact_real", {"answer": "42", "timestamp": 1}, redact=["timestamp"])
    # the redacted timestamp differs but so does the answer — that real change must fail
    result = snap.capture(
        "test_redact_real", {"answer": "CHANGED", "timestamp": 2}, redact=["timestamp"]
    )
    assert not result.passed


def test_snapshot_decorator_accepts_redact():
    @snapshot("test_decorator_redact", redact=["ts"])
    def agent():
        agent._n = getattr(agent, "_n", 0) + 1
        return {"answer": "ok", "ts": agent._n}

    agent()  # baseline
    agent()  # ts changed but redacted -> no AssertionError
