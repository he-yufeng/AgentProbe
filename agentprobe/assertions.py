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
    matching = [c for c in calls if c.get("name") == tool_name]

    if not matching:
        available = [c.get("name") for c in calls]
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
            args = call.get("arguments", {})
            if all(args.get(k) == v for k, v in with_args.items()):
                return
        raise AssertionError(
            f"Tool '{tool_name}' was never called with args {with_args}. "
            f"Actual calls: {[c.get('arguments') for c in matching]}"
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
