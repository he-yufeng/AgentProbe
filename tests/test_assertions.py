"""Tests for assertion helpers."""

import pytest
from pydantic import BaseModel

from agentprobe.assertions import assert_tool_called, assert_schema


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
