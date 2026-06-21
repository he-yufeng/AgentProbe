"""Tests for the multi-step Trace recorder."""

from __future__ import annotations

import pytest

from agentprobe import (
    Trace,
    assert_cost_under,
    assert_tool_called,
    assert_tool_sequence,
)
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


# pricing: {model: (input_per_1k_usd, output_per_1k_usd)}
_PRICING = {"gpt-4o": (0.005, 0.015)}


def test_token_usage_totals():
    trace = Trace()
    trace.record_llm("a", model="gpt-4o", input_tokens=100, output_tokens=40)
    trace.record_llm("b", model="gpt-4o", input_tokens=200, output_tokens=10)
    trace.record_tool_call("search")  # non-llm step ignored
    assert trace.token_usage() == (300, 50)


def test_estimate_cost_with_dict_pricing():
    trace = Trace()
    trace.record_llm("a", model="gpt-4o", input_tokens=1000, output_tokens=500)
    # 1000/1000*0.005 + 500/1000*0.015 = 0.005 + 0.0075
    assert trace.estimate_cost(_PRICING) == pytest.approx(0.0125)


def test_estimate_cost_with_callable_pricing():
    trace = Trace()
    trace.record_llm("a", model="any", input_tokens=10, output_tokens=20)
    cost = trace.estimate_cost(lambda model, inp, out: (inp + out) * 0.001)
    assert cost == pytest.approx(0.03)


def test_estimate_cost_unknown_model_in_table_is_none():
    trace = Trace()
    trace.record_llm("a", model="mystery-model", input_tokens=100, output_tokens=50)
    # model not in the pricing table -> nothing priceable -> None
    assert trace.estimate_cost(_PRICING) is None


def test_estimate_cost_no_llm_data_is_none():
    trace = Trace()
    trace.record_tool_call("search")
    assert trace.estimate_cost(_PRICING) is None


def test_assert_cost_under_passes_and_fails():
    trace = Trace()
    trace.record_llm("a", model="gpt-4o", input_tokens=1000, output_tokens=500)  # $0.0125
    assert_cost_under(trace, 0.02, pricing=_PRICING)  # under budget -> ok
    with pytest.raises(AssertionError, match="exceeds the budget"):
        assert_cost_under(trace, 0.01, pricing=_PRICING)


def test_assert_cost_under_raises_without_pricing_source():
    trace = Trace()
    trace.record_llm("a", model="gpt-4o", input_tokens=100, output_tokens=50)
    # an explicit empty table prices nothing -> can't estimate -> AssertionError
    with pytest.raises(AssertionError, match="Cannot estimate cost"):
        assert_cost_under(trace, 1.0, pricing={})


def test_estimate_cost_with_model_but_no_tokens_is_none():
    trace = Trace()
    # A model recorded without token counts can't be priced — that should be
    # None ("don't know"), not a $0 step that silently understates the run.
    trace.record_llm("a", model="gpt-4o")
    assert trace.estimate_cost(_PRICING) is None
