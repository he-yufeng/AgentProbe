"""Tests for the multi-step Trace recorder."""

from __future__ import annotations

from agentprobe import Trace, assert_tool_called, assert_tool_sequence
from agentprobe.trace import EVENT, LLM, TOOL_CALL


def test_records_steps_in_order():
    trace = Trace()
    trace.record_llm("thinking about it")
    trace.record_tool_call("search", {"query": "weather"})
    trace.record_event("retry", attempt=2)
    assert len(trace) == 3
    assert trace.names == ["llm", "search", "retry"]
    assert [s.kind for s in trace] == [LLM, TOOL_CALL, EVENT]


def test_tool_calls_feed_existing_assertions():
    trace = Trace()
    trace.record_tool_call("search", {"query": "x"})
    trace.record_llm("got results")
    trace.record_tool_call("fetch", {"url": "http://e"})
    # the recorded tool calls drop straight into the assertion helpers
    assert_tool_called(trace.tool_calls, "search", times=1)
    assert_tool_sequence(trace.tool_calls, ["search", "fetch"])
    assert trace.tool_calls == [
        {"name": "search", "arguments": {"query": "x"}},
        {"name": "fetch", "arguments": {"url": "http://e"}},
    ]


def test_of_kind_filters():
    trace = Trace()
    trace.record_tool_call("a")
    trace.record_event("e1")
    trace.record_tool_call("b")
    assert [s.name for s in trace.of_kind(TOOL_CALL)] == ["a", "b"]
    assert [s.name for s in trace.of_kind(EVENT)] == ["e1"]


def test_to_dict_is_snapshot_friendly():
    trace = Trace()
    trace.record_tool_call("search", {"query": "x"})
    snap = trace.to_dict()
    assert snap == {
        "steps": [{"kind": "tool_call", "name": "search", "data": {"arguments": {"query": "x"}}}]
    }


def test_tool_call_defaults_empty_arguments():
    trace = Trace()
    trace.record_tool_call("noop")
    assert trace.tool_calls == [{"name": "noop", "arguments": {}}]


def test_indexing_and_returned_step():
    trace = Trace()
    step = trace.record_llm("hi", tokens=5)
    assert trace[0] is step
    assert step.data["content"] == "hi"
    assert step.data["tokens"] == 5
