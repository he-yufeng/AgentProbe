"""Assertion helpers for agent testing."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Type

from pydantic import BaseModel, ValidationError


def assert_tool_called(
    calls: list[dict[str, Any]],
    tool_name: str,
    times: int | None = None,
    with_args: dict[str, Any] | None = None,
    *,
    min_times: int | None = None,
    max_times: int | None = None,
) -> None:
    """Assert that a tool was called in the recorded interactions.

    Args:
        calls: List of tool call records. Each should have at minimum a "name" key.
              Works with MockLLM response dicts that have tool_calls, or any list
              of dicts with "name" and optionally "arguments".
        tool_name: The expected tool/function name.
        times: If set, assert the tool was called exactly this many times.
        with_args: If set, assert at least one call had these argument key-value pairs.
        min_times: If set, assert the tool was called at least this many times.
        max_times: If set, assert the tool was called at most this many times.

    ``min_times``/``max_times`` are for non-deterministic agents where an exact
    count is too brittle — e.g. "retried at most 3 times" or "searched at least
    twice". The tool must still have been called at least once. ``times`` is
    mutually exclusive with ``min_times``/``max_times``.

    Raises:
        AssertionError: If the assertion fails.
        ValueError: If the count arguments are combined incoherently.
    """
    if times is not None and (min_times is not None or max_times is not None):
        raise ValueError("times cannot be combined with min_times/max_times")
    if min_times is not None and max_times is not None and min_times > max_times:
        raise ValueError(f"min_times ({min_times}) is greater than max_times ({max_times})")

    matching = [call for call in calls if _tool_name(call) == tool_name]

    if not matching:
        available = [_tool_name(call) for call in calls]
        raise AssertionError(
            f"Tool '{tool_name}' was never called. "
            f"Called tools: {available}"
        )

    count = len(matching)
    if times is not None and count != times:
        raise AssertionError(
            f"Tool '{tool_name}' was called {count} time(s), expected {times}"
        )
    if min_times is not None and count < min_times:
        raise AssertionError(
            f"Tool '{tool_name}' was called {count} time(s), expected at least {min_times}"
        )
    if max_times is not None and count > max_times:
        raise AssertionError(
            f"Tool '{tool_name}' was called {count} time(s), expected at most {max_times}"
        )

    if with_args is not None:
        for call in matching:
            args = _tool_arguments(call)
            if _contains_subset(args, with_args):
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


def assert_only_tools_used(
    calls: list[dict[str, Any]],
    allowed_tools: Iterable[str],
) -> None:
    """Assert that every tool call uses a tool on the allowlist.

    The allowlist complement to the per-tool checks: useful for scoping and
    safety tests where an agent must stay within a permitted set (for example
    read-only tools) and never reach for an out-of-scope one such as write,
    delete or shell. Calls with no recognizable tool name are ignored, and
    using no tools at all trivially passes.

    Args:
        calls: List of tool call records, each with a "name"/"function" key.
        allowed_tools: The permitted tool names.

    Raises:
        AssertionError: If any call used a tool outside the allowlist.
    """
    allowed = set(allowed_tools)
    offending = sorted(
        {name for call in calls if (name := _tool_name(call)) and name not in allowed}
    )
    if offending:
        raise AssertionError(
            f"Agent used tool(s) outside the allowlist: {offending}. "
            f"Allowed: {sorted(allowed)}"
        )


def assert_tool_not_called_with(
    calls: list[dict[str, Any]],
    tool_name: str,
    with_args: dict[str, Any],
) -> None:
    """Assert that a tool was never called with a given set of arguments.

    Where ``assert_no_tool_called`` forbids a tool entirely, this allows the
    tool but fails if any call carried the given argument subset — the negative
    counterpart to ``assert_tool_called(..., with_args=...)``. Useful for safety
    checks such as "the agent may run shell commands but must never call
    ``run`` with ``sudo=True``", or "may delete files but never with
    ``path='/'``".

    Args:
        calls: Tool call records.
        tool_name: The tool/function name to inspect.
        with_args: Argument key-value pairs that must not appear together in any
            call to ``tool_name`` (matched as a nested-aware subset).

    Raises:
        AssertionError: If any call to ``tool_name`` matched ``with_args``.
    """
    for call in calls:
        if _tool_name(call) != tool_name:
            continue
        args = _tool_arguments(call)
        if _contains_subset(args, with_args):
            raise AssertionError(
                f"Tool '{tool_name}' was called with forbidden args {with_args}. "
                f"Offending call args: {args}"
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
    if isinstance(args, str):
        return _json_object(args)
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("arguments"), dict):
        return function["arguments"]
    if isinstance(function, dict) and isinstance(function.get("arguments"), str):
        return _json_object(function["arguments"])
    return {}


def _json_object(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual or not _contains_subset(actual[key], value):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) < len(expected):
            return False
        return all(_contains_subset(item, expected[index]) for index, item in enumerate(actual))
    return actual == expected
