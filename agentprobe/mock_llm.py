"""MockLLM — deterministic mock for OpenAI-compatible chat completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockMessage:
    role: str = "assistant"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class MockChoice:
    index: int = 0
    message: MockMessage = field(default_factory=MockMessage)
    finish_reason: str = "stop"


@dataclass
class MockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MockCompletion:
    id: str = "mock-completion"
    object: str = "chat.completion"
    model: str = "mock"
    choices: list[MockChoice] = field(default_factory=list)
    usage: MockUsage = field(default_factory=MockUsage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content,
                        **({"tool_calls": c.message.tool_calls} if c.message.tool_calls else {}),
                    },
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class MockLLM:
    """A deterministic mock that replaces OpenAI's chat completions API.

    Usage::

        mock = MockLLM(responses=["Hello!", "How can I help?"])
        result = mock.chat("What's up?")
        assert result.choices[0].message.content == "Hello!"

        # Or use as a drop-in for openai.Client:
        mock = MockLLM(responses=["Done."])
        # Pass mock.create to anything expecting client.chat.completions.create
    """

    def __init__(
        self,
        responses: list[str | dict[str, Any]] | None = None,
        default_response: str = "Mock response",
    ):
        self._responses = list(responses or [])
        self._default = default_response
        self._call_log: list[dict[str, Any]] = []
        self._index = 0
        self.chat = self  # so mock.chat.completions.create works
        self.completions = self

    def create(self, messages: list[dict[str, str]] | None = None, **kwargs) -> MockCompletion:
        """Mimics openai.chat.completions.create()."""
        self._call_log.append({"messages": messages, **kwargs})

        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
        else:
            resp = self._default

        if isinstance(resp, str):
            msg = MockMessage(content=resp)
        elif isinstance(resp, dict):
            tool_calls = resp.get("tool_calls")
            msg = MockMessage(
                content=resp.get("content", ""),
                tool_calls=tool_calls,
            )
            if tool_calls:
                return MockCompletion(
                    choices=[MockChoice(message=msg, finish_reason="tool_calls")]
                )
        else:
            msg = MockMessage(content=str(resp))

        return MockCompletion(choices=[MockChoice(message=msg)])

    @property
    def calls(self) -> list[dict[str, Any]]:
        """All recorded calls."""
        return list(self._call_log)

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    def reset(self):
        """Clear call log and reset response index."""
        self._call_log.clear()
        self._index = 0
