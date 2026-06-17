"""Tests for assertion helpers."""

import pytest
from pydantic import BaseModel

from agentprobe.assertions import (
    assert_no_tool_called,
    assert_schema,
    assert_tool_called,
    assert_tool_sequence,
)


class SummaryResult(BaseModel):
    title: str
    bullet_points: list[str]
    confidence: float


# -- assert_tool_called --

def test_tool_called_passes():
    calls = [{"name": "search", "arguments": {"query": "AI"}}]
    assert_tool_called(calls, "search")


def test_tool_called_with_times():
    calls = [
        {"name": "search", "arguments": {"query": "a"}},
        {"name": "search", "arguments": {"query": "b"}},
    ]
    assert_tool_called(calls, "search", times=2)


def test_tool_called_with_args():
    calls = [{"name": "search", "arguments": {"query": "AI", "limit": 10}}]
    assert_tool_called(calls, "search", with_args={"query": "AI"})


def test_tool_not_called_raises():
    calls = [{"name": "other"}]
    with pytest.raises(AssertionError, match="never called"):
        assert_tool_called(calls, "search")


def test_tool_called_wrong_times_raises():
    calls = [{"name": "search"}]
    with pytest.raises(AssertionError, match="1 time"):
        assert_tool_called(calls, "search", times=3)


def test_tool_called_wrong_args_raises():
    calls = [{"name": "search", "arguments": {"query": "other"}}]
    with pytest.raises(AssertionError, match="never called with args"):
        assert_tool_called(calls, "search", with_args={"query": "AI"})


def test_tool_called_accepts_openai_function_shape():
    calls = [{"function": {"name": "search", "arguments": {"query": "AI"}}}]
    assert_tool_called(calls, "search", with_args={"query": "AI"})


def _n_calls(name: str, n: int) -> list[dict[str, object]]:
    return [{"name": name} for _ in range(n)]


def test_tool_called_min_times_passes_when_at_or_above_floor():
    assert_tool_called(_n_calls("search", 3), "search", min_times=2)
    assert_tool_called(_n_calls("search", 2), "search", min_times=2)


def test_tool_called_min_times_raises_when_below_floor():
    with pytest.raises(AssertionError, match="at least 3"):
        assert_tool_called(_n_calls("search", 2), "search", min_times=3)


def test_tool_called_max_times_passes_within_ceiling():
    # "retried at most 3 times" — called once or up to the ceiling, both fine
    assert_tool_called(_n_calls("api_call", 1), "api_call", max_times=3)
    assert_tool_called(_n_calls("api_call", 3), "api_call", max_times=3)


def test_tool_called_max_times_raises_above_ceiling():
    with pytest.raises(AssertionError, match="at most 3"):
        assert_tool_called(_n_calls("api_call", 4), "api_call", max_times=3)


def test_tool_called_min_and_max_define_a_range():
    assert_tool_called(_n_calls("search", 2), "search", min_times=1, max_times=3)
    with pytest.raises(AssertionError, match="at most 3"):
        assert_tool_called(_n_calls("search", 5), "search", min_times=1, max_times=3)


def test_tool_called_times_conflicts_with_min_or_max():
    with pytest.raises(ValueError, match="times.*min_times.*max_times|cannot"):
        assert_tool_called(_n_calls("search", 2), "search", times=2, min_times=1)


def test_tool_called_min_above_max_is_a_usage_error():
    with pytest.raises(ValueError, match="min_times.*max_times|greater"):
        assert_tool_called(_n_calls("search", 2), "search", min_times=3, max_times=1)


def test_tool_called_accepts_json_string_arguments_and_nested_subset():
    calls = [
        {
            "function": {
                "name": "write_file",
                "arguments": '{"path": "README.md", "metadata": {"mode": "safe", "attempt": 2}}',
            }
        }
    ]

    assert_tool_called(
        calls,
        "write_file",
        with_args={"metadata": {"mode": "safe"}},
    )


def test_no_tool_called_passes_and_fails():
    calls = [{"name": "search"}]
    assert_no_tool_called(calls, "delete_file")

    with pytest.raises(AssertionError, match="expected zero"):
        assert_no_tool_called(calls, "search")


def test_tool_sequence_allows_gaps_by_default():
    calls = [
        {"name": "search"},
        {"name": "summarize"},
        {"name": "write_report"},
    ]
    assert_tool_sequence(calls, ["search", "write_report"])


def test_tool_sequence_can_require_contiguous_order():
    calls = [
        {"name": "search"},
        {"name": "summarize"},
        {"name": "write_report"},
    ]
    assert_tool_sequence(calls, ["search", "summarize"], contiguous=True)

    with pytest.raises(AssertionError, match="contiguous"):
        assert_tool_sequence(calls, ["search", "write_report"], contiguous=True)


def test_tool_sequence_reports_remaining_expected_tools():
    calls = [{"name": "search"}]
    with pytest.raises(AssertionError, match="Still waiting"):
        assert_tool_sequence(calls, ["search", "write_report"])


# -- assert_schema --

def test_schema_from_dict():
    data = {"title": "Test", "bullet_points": ["a", "b"], "confidence": 0.9}
    result = assert_schema(data, SummaryResult)
    assert result.title == "Test"
    assert result.confidence == 0.9


def test_schema_from_json_string():
    import json
    data = json.dumps({"title": "T", "bullet_points": [], "confidence": 0.5})
    result = assert_schema(data, SummaryResult)
    assert isinstance(result, SummaryResult)


def test_schema_from_model_instance():
    obj = SummaryResult(title="X", bullet_points=["y"], confidence=1.0)
    result = assert_schema(obj, SummaryResult)
    assert result is obj


def test_schema_invalid_raises():
    with pytest.raises(AssertionError, match="does not match"):
        assert_schema({"title": "no confidence"}, SummaryResult)


def test_schema_bad_json_raises():
    with pytest.raises(AssertionError, match="not valid JSON"):
        assert_schema("not json {{{", SummaryResult)
