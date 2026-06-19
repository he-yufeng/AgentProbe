"""Record an agent's intermediate steps for assertions and snapshots.

AgentProbe's assertions check a *list of tool calls* — but something has to
produce that list. ``Trace`` is the missing recorder: capture each step as the
agent runs (tool calls, LLM turns, custom events), then assert over the result
or snapshot it. ``trace.tool_calls`` is shaped to drop straight into
``assert_tool_called`` and friends; ``trace.to_dict()`` is snapshot-friendly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL = "tool_call"
LLM = "llm"
EVENT = "event"


@dataclass
class Step:
    """One recorded step in an agent run."""

    kind: str  # TOOL_CALL | LLM | EVENT
    name: str
    data: dict[str, Any] = field(default_factory=dict)


class Trace:
    """A growing, ordered record of what an agent did during one run."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def record_tool_call(self, name: str, arguments: dict[str, Any] | None = None) -> Step:
        """Record a tool/function invocation."""
        step = Step(TOOL_CALL, name, {"arguments": dict(arguments or {})})
        self.steps.append(step)
        return step

    def record_llm(self, content: str = "", *, name: str = "llm", **data: Any) -> Step:
        """Record an LLM turn (its text and any metadata, e.g. token counts)."""
        step = Step(LLM, name, {"content": content, **data})
        self.steps.append(step)
        return step

    def record_event(self, name: str, **data: Any) -> Step:
        """Record an arbitrary intermediate event (retry, handoff, error, ...)."""
        step = Step(EVENT, name, dict(data))
        self.steps.append(step)
        return step

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Tool-call steps as dicts compatible with ``agentprobe.assertions``."""
        return [
            {"name": s.name, "arguments": s.data.get("arguments", {})}
            for s in self.steps
            if s.kind == TOOL_CALL
        ]

    @property
    def names(self) -> list[str]:
        """All step names in order (across every kind)."""
        return [s.name for s in self.steps]

    def of_kind(self, kind: str) -> list[Step]:
        """All steps of a given kind, in order."""
        return [s for s in self.steps if s.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        """Serializable form — useful for snapshot-testing the whole trace."""
        return {"steps": [{"kind": s.kind, "name": s.name, "data": s.data} for s in self.steps]}

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> Step:
        return self.steps[index]
