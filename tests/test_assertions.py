"""Tests for assertion helpers."""

import pytest
from pydantic import BaseModel

from agentprobe.assertions import (
    assert_no_repeated_calls,
    assert_no_tool_called,
    assert_only_tools_used,
    assert_schema,
    assert_tool_called,
    assert_tool_not_called_with,
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


# -- assert_tool_not_called_with --


def test_tool_not_called_with_passes_when_args_absent():
    # The tool is used, but never with the forbidden argument.
    calls = [
        {"name": "run", "arguments": {"cmd": "ls"}},
        {"name": "run", "arguments": {"cmd": "cat f", "sudo": False}},
    ]
    assert_tool_not_called_with(calls, "run", {"sudo": True})


def test_tool_not_called_with_passes_when_tool_absent():
    calls = [{"name": "search", "arguments": {"q": "x"}}]
    assert_tool_not_called_with(calls, "delete_file", {"path": "/"})


def test_tool_not_called_with_raises_on_forbidden_args():
    calls = [
        {"name": "delete_file", "arguments": {"path": "/tmp/x"}},
        {"name": "delete_file", "arguments": {"path": "/", "recursive": True}},
    ]
    with pytest.raises(AssertionError, match="forbidden args"):
        assert_tool_not_called_with(calls, "delete_file", {"path": "/"})


def test_tool_not_called_with_matches_json_and_function_shape():
    # JSON-string arguments and the OpenAI function shape are both inspected.
    calls = [
        {"function": {"name": "run", "arguments": '{"cmd": "rm", "sudo": true}'}},
    ]
    with pytest.raises(AssertionError, match="forbidden args"):
        assert_tool_not_called_with(calls, "run", {"sudo": True})


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


# -- assert_only_tools_used --

def test_only_tools_used_passes_within_allowlist():
    calls = [{"name": "search"}, {"name": "read_file"}]
    assert_only_tools_used(calls, {"search", "read_file", "list_dir"})


# -- with_args nested list subset --

def test_with_args_list_subset_allows_extra_actual_items():
    # A list in with_args is a prefix-subset: extra trailing items in the actual
    # call must match without crashing the matcher.
    calls = [{"name": "search", "arguments": {"tags": ["a", "b", "c"]}}]
    assert_tool_called(calls, "search", with_args={"tags": ["a"]})


def test_with_args_list_subset_mismatch_fails_cleanly():
    calls = [{"name": "search", "arguments": {"tags": ["a", "b"]}}]
    with pytest.raises(AssertionError):
        assert_tool_called(calls, "search", with_args={"tags": ["z"]})


def test_only_tools_used_passes_with_no_calls():
    assert_only_tools_used([], {"search"})


def test_only_tools_used_raises_on_out_of_scope_tool():
    calls = [{"name": "search"}, {"name": "delete_file"}, {"name": "shell"}]
    with pytest.raises(AssertionError, match="delete_file"):
        assert_only_tools_used(calls, ["search", "read_file"])


# -- assert_no_repeated_calls --

def test_no_repeated_calls_passes_for_varied_calls():
    calls = [
        {"name": "search", "arguments": {"query": "a"}},
        {"name": "search", "arguments": {"query": "b"}},  # same tool, new args: fine
        {"name": "open", "arguments": {"path": "x"}},
        {"name": "search", "arguments": {"query": "a"}},  # repeat but not consecutive
    ]
    assert_no_repeated_calls(calls)


def test_no_repeated_calls_raises_on_back_to_back_duplicate():
    calls = [
        {"name": "search", "arguments": {"query": "a"}},
        {"name": "search", "arguments": {"query": "a"}},
    ]
    with pytest.raises(AssertionError, match="stuck in a loop"):
        assert_no_repeated_calls(calls)


def test_no_repeated_calls_allows_bounded_retries():
    calls = [
        {"name": "fetch", "arguments": {"url": "u"}},
        {"name": "fetch", "arguments": {"url": "u"}},
        {"name": "fetch", "arguments": {"url": "u"}},
    ]
    # three identical in a row is tolerated when retries are allowed...
    assert_no_repeated_calls(calls, max_consecutive=3)
    # ...but the same run trips a tighter ceiling.
    with pytest.raises(AssertionError):
        assert_no_repeated_calls(calls, max_consecutive=2)


def test_no_repeated_calls_rejects_bad_max_consecutive():
    with pytest.raises(ValueError):
        assert_no_repeated_calls([], max_consecutive=0)
