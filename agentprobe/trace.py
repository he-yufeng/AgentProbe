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

    def record_llm(
        self,
        content: str = "",
        *,
        name: str = "llm",
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **data: Any,
    ) -> Step:
        """Record an LLM turn — its text, the model, and token counts.

        ``model`` / ``input_tokens`` / ``output_tokens`` feed
        :meth:`estimate_cost`; any extra metadata goes through ``**data``.
        """
        payload: dict[str, Any] = {"content": content, **data}
        if model is not None:
            payload["model"] = model
        if input_tokens:
            payload["input_tokens"] = input_tokens
        if output_tokens:
            payload["output_tokens"] = output_tokens
        step = Step(LLM, name, payload)
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

    def token_usage(self) -> tuple[int, int]:
        """Total (input, output) tokens summed across recorded LLM steps."""
        inp = sum(s.data.get("input_tokens", 0) or 0 for s in self.steps if s.kind == LLM)
        out = sum(s.data.get("output_tokens", 0) or 0 for s in self.steps if s.kind == LLM)
        return inp, out

    def estimate_cost(self, pricing: Any = None) -> float | None:
        """Estimate the total USD cost of the run's LLM steps.

        ``pricing`` resolves prices, and may be:

        * a callable ``(model, input_tokens, output_tokens) -> float``;
        * a dict ``{model: (input_per_1k_usd, output_per_1k_usd)}``;
        * ``None`` — fall back to TokenTracker's price table if it's installed
          (``pip install tokentracker``).

        Returns ``None`` when no price source is available, or when no LLM step
        carried a model/token count to price.
        """
        price = _resolve_pricing(pricing)
        if price is None:
            return None
        total = 0.0
        priced_any = False
        for step in self.steps:
            if step.kind != LLM:
                continue
            model = step.data.get("model")
            inp = step.data.get("input_tokens", 0) or 0
            out = step.data.get("output_tokens", 0) or 0
            # A step needs token counts to be priced. A model name with no token
            # data isn't a $0 step — it's unpriceable, so skip it rather than
            # quietly counting it as free and understating the run's cost.
            if not (inp or out):
                continue
            cost = price(model, inp, out)
            if cost is not None:
                total += cost
                priced_any = True
        return total if priced_any else None

    def to_dict(self) -> dict[str, Any]:
        """Serializable form — useful for snapshot-testing the whole trace."""
        return {"steps": [{"kind": s.kind, "name": s.name, "data": s.data} for s in self.steps]}

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, index: int) -> Step:
        return self.steps[index]


def _resolve_pricing(pricing: Any) -> Any:
    """Turn a pricing spec into a ``(model, in, out) -> float | None`` callable.

    ``None`` falls back to TokenTracker's price table when installed; a dict maps
    ``{model: (input_per_1k, output_per_1k)}``; a callable is used directly.
    Returns ``None`` if no usable price source exists.
    """
    if pricing is None:
        try:
            from tokentracker.pricing import estimate_cost as tt_estimate_cost
        except ImportError:
            return None
        return lambda model, inp, out: tt_estimate_cost(model, inp, out) if model else None
    if callable(pricing):
        return pricing
    if isinstance(pricing, dict):

        def from_table(model: Any, inp: int, out: int) -> float | None:
            rates = pricing.get(model)
            if rates is None:
                return None
            in_rate, out_rate = rates
            return inp / 1000 * in_rate + out / 1000 * out_rate

        return from_table
    return None
