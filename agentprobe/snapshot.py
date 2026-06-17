"""Snapshot decorator — captures agent outputs and compares against stored baselines."""

from __future__ import annotations

import functools
import inspect
import json
import time
from difflib import unified_diff
from dataclasses import dataclass, field
from typing import Any, Callable

from agentprobe.storage import load_snapshot, save_snapshot


@dataclass
class SnapshotResult:
    """Result of a snapshot comparison."""

    name: str
    output: Any
    baseline: dict[str, Any] | None
    passed: bool
    similarity: float | None = None
    message: str = ""


@dataclass
class Snapshot:
    """Manages snapshot state for a test session."""

    update: bool = False
    mode: str = "exact"
    threshold: float = 0.85
    redact: tuple[str, ...] = ()
    results: list[SnapshotResult] = field(default_factory=list)

    def capture(
        self, name: str, output: Any, redact: list[str] | tuple[str, ...] | None = None
    ) -> SnapshotResult:
        """Capture an output and compare it against the stored snapshot.

        ``redact`` lists keys whose values are non-deterministic (timestamps,
        request ids, ...). Their values are replaced with ``"<redacted>"``
        anywhere they appear before the snapshot is saved or compared, so they
        don't cause spurious mismatches. A per-call ``redact`` overrides the
        instance default.
        """
        serialized = _serialize(output)
        redact_keys = tuple(redact) if redact is not None else self.redact
        if redact_keys:
            serialized = _redact(serialized, set(redact_keys))
        current = {"output": serialized, "timestamp": time.time()}
        baseline = load_snapshot(name)

        if self.update or baseline is None:
            save_snapshot(name, current)
            result = SnapshotResult(
                name=name,
                output=serialized,
                baseline=baseline,
                passed=True,
                message="snapshot created" if baseline is None else "snapshot updated",
            )
            self.results.append(result)
            return result

        baseline_output = baseline.get("output", "")
        passed, similarity = _compare(
            json.dumps(serialized, sort_keys=True, default=str),
            json.dumps(baseline_output, sort_keys=True, default=str),
            self.mode,
            self.threshold,
        )
        result = SnapshotResult(
            name=name,
            output=serialized,
            baseline=baseline,
            passed=passed,
            similarity=similarity,
            message="" if passed else f"snapshot mismatch (similarity={similarity})",
        )
        self.results.append(result)
        return result


def _serialize(obj: Any) -> Any:
    """Convert an object to a JSON-safe representation."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _serialize(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _serialize(vars(obj))
    return str(obj)


def _redact(obj: Any, keys: set[str], placeholder: str = "<redacted>") -> Any:
    """Replace the values of ``keys`` with ``placeholder`` anywhere in ``obj``.

    Operates on the already-serialized (JSON-safe) structure, recursing through
    dicts and lists so a redacted key is masked at any depth.
    """
    if isinstance(obj, dict):
        return {
            k: placeholder if k in keys else _redact(v, keys, placeholder) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v, keys, placeholder) for v in obj]
    return obj


def _compare(current: str, baseline: str, mode: str, threshold: float) -> tuple[bool, float | None]:
    if mode == "exact":
        return current == baseline, 1.0 if current == baseline else 0.0
    if mode == "semantic":
        from agentprobe.similarity import semantic_similarity

        score = semantic_similarity(current, baseline)
        return score >= threshold, score
    raise ValueError(f"Unknown comparison mode: {mode!r}. Use 'exact' or 'semantic'.")


def snapshot(
    name: str | None = None,
    mode: str = "exact",
    threshold: float = 0.85,
    update: bool = False,
    redact: list[str] | None = None,
) -> Callable:
    """Decorator that captures the return value of a function and compares it to a snapshot.

    Usage::

        @snapshot("my_agent_test")
        def test_summarize():
            return agent.run("Summarize this document")

    Pass ``redact=["timestamp", "request_id"]`` to mask non-deterministic fields
    so they don't cause spurious snapshot mismatches.
    """

    def decorator(fn: Callable) -> Callable:
        snap_name = name or fn.__qualname__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                output = await fn(*args, **kwargs)
                _assert_snapshot(
                    snap_name,
                    output,
                    update=update,
                    mode=mode,
                    threshold=threshold,
                    redact=redact,
                )
                return output

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            output = fn(*args, **kwargs)
            _assert_snapshot(
                snap_name,
                output,
                update=update,
                mode=mode,
                threshold=threshold,
                redact=redact,
            )
            return output

        return wrapper

    return decorator


def _assert_snapshot(
    snap_name: str,
    output: Any,
    *,
    update: bool,
    mode: str,
    threshold: float,
    redact: list[str] | None = None,
) -> None:
    snap = Snapshot(update=update, mode=mode, threshold=threshold)
    result = snap.capture(snap_name, output, redact=redact)
    if result.passed:
        return

    expected = result.baseline.get("output") if result.baseline else None
    raise AssertionError(_format_mismatch(snap_name, expected, result.output, result.message))


def _format_mismatch(name: str, expected: Any, actual: Any, message: str) -> str:
    expected_lines = json.dumps(expected, indent=2, sort_keys=True, default=str).splitlines()
    actual_lines = json.dumps(actual, indent=2, sort_keys=True, default=str).splitlines()
    diff = "\n".join(
        unified_diff(
            expected_lines,
            actual_lines,
            fromfile="snapshot",
            tofile="current",
            lineterm="",
        )
    )
    if len(diff) > 4000:
        diff = f"{diff[:4000]}\n... diff truncated ..."
    return f"Snapshot '{name}' mismatch: {message}\n{diff}"
