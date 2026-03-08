"""Snapshot decorator — captures agent outputs and compares against stored baselines."""

from __future__ import annotations

import functools
import json
import time
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
    results: list[SnapshotResult] = field(default_factory=list)

    def capture(self, name: str, output: Any) -> SnapshotResult:
        """Capture an output and compare it against the stored snapshot."""
        serialized = _serialize(output)
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


def _compare(
    current: str, baseline: str, mode: str, threshold: float
) -> tuple[bool, float | None]:
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
) -> Callable:
    """Decorator that captures the return value of a function and compares it to a snapshot.

    Usage::

        @snapshot("my_agent_test")
        def test_summarize():
            return agent.run("Summarize this document")
    """

    def decorator(fn: Callable) -> Callable:
        snap_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            output = fn(*args, **kwargs)
            snap = Snapshot(update=update, mode=mode, threshold=threshold)
            result = snap.capture(snap_name, output)
            if not result.passed:
                expected = result.baseline.get("output") if result.baseline else None
                raise AssertionError(
                    f"Snapshot '{snap_name}' mismatch: {result.message}\n"
                    f"  Expected: {json.dumps(expected, indent=2)[:500]}\n"
                    f"  Got:      {json.dumps(result.output, indent=2)[:500]}"
                )
            return output

        return wrapper

    return decorator
