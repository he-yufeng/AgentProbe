"""Tests for snapshot capture and comparison."""

import asyncio
import shutil

import pytest

from agentprobe.snapshot import Snapshot, snapshot
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
