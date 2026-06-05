"""Assertion helpers for agent testing."""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, ValidationError


def assert_tool_called(
    calls: list[dict[str, Any]],
    tool_name: str,
    times: int | None = None,
    with_args: dict[str, Any] | None = None,
) -> None:
    """Assert that a tool was called in the recorded interactions.

    Args:
        calls: List of tool call records. Each should have at minimum a "name" key.
              Works with MockLLM response dicts that have tool_calls, or any list
              of dicts with "name" and optionally "arguments".
        tool_name: The expected tool/function name.
        times: If set, assert the tool was called exactly this many times.
        with_args: If set, assert at least one call had these argument key-value pairs.

    Raises:
        AssertionError: If the assertion fails.
    """
    matching = [call for call in calls if _tool_name(call) == tool_name]

    if not matching:
        available = [_tool_name(call) for call in calls]
        raise AssertionError(
            f"Tool '{tool_name}' was never called. "
            f"Called tools: {available}"
        )

    if times is not None and len(matching) != times:
        raise AssertionError(
            f"Tool '{tool_name}' was called {len(matching)} time(s), expected {times}"
        )

    if with_args is not None:
        for call in matching:
            args = _tool_arguments(call)
            if all(args.get(k) == v for k, v in with_args.items()):
                return
        raise AssertionError(
            f"Tool '{tool_name}' was never called with args {with_args}. "
            f"Actual calls: {[_tool_arguments(call) for call in matching]}"
        )


def assert_no_tool_called(calls: list[dict[str, Any]], tool_name: str) -> None:
    """Assert that a tool was not called."""
    matching = [call for call in calls if _tool_name(call) == tool_name]
    if matching:
        raise AssertionError(
            f"Tool '{tool_name}' was called {len(matching)} time(s), expected zero"
        )


def assert_tool_sequence(
    calls: list[dict[str, Any]],
    expected: list[str],
    *,
    contiguous: bool = False,
) -> None:
    """Assert that tool calls appear in a required order.

    Args:
        calls: Tool call records.
        expected: Tool names that must appear in order.
        contiguous: If true, the expected tools must be adjacent.
    """
    actual = [_tool_name(call) for call in calls]
    if not expected:
        return

    if contiguous:
        window = len(expected)
        for start in range(0, len(actual) - window + 1):
            if actual[start : start + window] == expected:
                return
        raise AssertionError(
            f"Tool sequence {expected} was not found as a contiguous block. "
            f"Actual order: {actual}"
        )

    position = 0
    for name in actual:
        if name == expected[position]:
            position += 1
            if position == len(expected):
                return

    missing = expected[position:]
    raise AssertionError(
        f"Tool sequence {expected} was not found in order. "
        f"Still waiting for: {missing}. Actual order: {actual}"
    )


def assert_schema(output: Any, schema: Type[BaseModel]) -> BaseModel:
    """Validate that output conforms to a Pydantic model.

    Args:
        output: The data to validate. Can be a dict, a JSON string, or a Pydantic model.
        schema: A Pydantic BaseModel subclass.

    Returns:
        The validated Pydantic model instance.

    Raises:
        AssertionError: If validation fails.
    """
    if isinstance(output, schema):
        return output

    if isinstance(output, str):
        import json

        try:
            output = json.loads(output)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Output is not valid JSON: {e}") from None

    if not isinstance(output, dict):
        raise AssertionError(
            f"Expected dict or {schema.__name__}, got {type(output).__name__}"
        )

    try:
        return schema.model_validate(output)
    except ValidationError as e:
        raise AssertionError(f"Output does not match {schema.__name__}:\n{e}") from None


def _tool_name(call: dict[str, Any]) -> str | None:
    if "name" in call:
        return call.get("name")
    function = call.get("function")
    if isinstance(function, dict):
        return function.get("name")
    return None


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if isinstance(args, dict):
        return args
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("arguments"), dict):
        return function["arguments"]
    return {}
