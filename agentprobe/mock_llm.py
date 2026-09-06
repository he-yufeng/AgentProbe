"""MockLLM — deterministic mock for OpenAI-compatible chat completions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self


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


@dataclass
class MockDelta:
    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class MockChunkChoice:
    index: int = 0
    delta: MockDelta = field(default_factory=MockDelta)
    finish_reason: str | None = None


@dataclass
class MockChunk:
    id: str = "mock-completion"
    object: str = "chat.completion.chunk"
    model: str = "mock"
    choices: list[MockChunkChoice] = field(default_factory=list)


class MockAPIError(Exception):
    """Base for scriptable API failures; carries an HTTP-style status code."""

    def __init__(self, message: str = "mock API error", status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class MockRateLimitError(MockAPIError):
    def __init__(self, message: str = "rate limit exceeded"):
        super().__init__(message, status_code=429)


class MockTimeoutError(MockAPIError):
    def __init__(self, message: str = "request timed out"):
        super().__init__(message, status_code=408)


class MockServerError(MockAPIError):
    def __init__(self, message: str = "internal server error"):
        super().__init__(message, status_code=500)


class Flaky:
    """Script entry that raises `error` for the first `times` calls, then yields `then`.

    Occupies one slot in the responses list but consumes `times + 1` calls, so
    retry and fallback logic can be exercised offline.
    """

    def __init__(self, error: BaseException, times: int = 1, then: str | dict[str, Any] = ""):
        if times < 1:
            raise ValueError("times must be >= 1")
        self.error = error
        self.times = times
        self.then = then
        self._remaining = times

    def reset(self):
        self._remaining = self.times


class MockStream:
    """Sync iterator over delta chunks, mirroring openai's Stream."""

    def __init__(self, chunks: list[MockChunk]):
        self._chunks = chunks
        self._pos = 0

    def __iter__(self) -> MockStream:
        return self

    def __next__(self) -> MockChunk:
        if self._pos >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._pos]
        self._pos += 1
        return chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._pos = len(self._chunks)


class AsyncMockStream:
    """Async iterator over delta chunks, mirroring openai's AsyncStream."""

    def __init__(self, chunks: list[MockChunk]):
        self._chunks = chunks
        self._pos = 0

    def __aiter__(self) -> AsyncMockStream:
        return self

    async def __anext__(self) -> MockChunk:
        if self._pos >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._pos]
        self._pos += 1
        return chunk

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._pos = len(self._chunks)


# Roughly token-shaped pieces, the way real streams split content.
_WORD = re.compile(r" ?\S+")


def _stream_chunks(completion: MockCompletion) -> list[MockChunk]:
    choice = completion.choices[0] if completion.choices else MockChoice()
    chunks = [MockChunk(choices=[MockChunkChoice(delta=MockDelta(role=choice.message.role))])]
    if choice.message.tool_calls:
        chunks.append(
            MockChunk(choices=[MockChunkChoice(delta=MockDelta(tool_calls=choice.message.tool_calls))])
        )
    else:
        for piece in _WORD.findall(choice.message.content):
            chunks.append(MockChunk(choices=[MockChunkChoice(delta=MockDelta(content=piece))]))
    chunks.append(
        MockChunk(choices=[MockChunkChoice(delta=MockDelta(), finish_reason=choice.finish_reason)])
    )
    return chunks


class _Scripted:
    """Shared scripting engine: ordered responses, call log, failure entries."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        default_response: str = "Mock response",
    ):
        self._responses = list(responses or [])
        self._default = default_response
        self._call_log: list[dict[str, Any]] = []
        self._index = 0
        self.chat = self  # so mock.chat.completions.create works
        self.completions = self

    def _next_response(self) -> Any:
        if self._index >= len(self._responses):
            return self._default
        entry = self._responses[self._index]
        if isinstance(entry, Flaky):
            # A still-failing Flaky keeps its slot until it succeeds once.
            if entry._remaining > 0:
                entry._remaining -= 1
                raise entry.error
            self._index += 1
            return entry.then
        self._index += 1
        if isinstance(entry, BaseException):
            raise entry
        return entry

    @staticmethod
    def _build_completion(resp: Any) -> MockCompletion:
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
        """Clear call log and reset response index (Flaky entries fail again)."""
        self._call_log.clear()
        self._index = 0
        for entry in self._responses:
            if isinstance(entry, Flaky):
                entry.reset()


class MockLLM(_Scripted):
    """A deterministic mock that replaces OpenAI's chat completions API.

    Usage::

        mock = MockLLM(responses=["Hello!", "How can I help?"])
        result = mock.chat("What's up?")
        assert result.choices[0].message.content == "Hello!"

        # Or use as a drop-in for openai.Client:
        mock = MockLLM(responses=["Done."])
        # Pass mock.create to anything expecting client.chat.completions.create

        # stream=True yields delta chunks; exceptions in the script raise at call time:
        mock = MockLLM(responses=[Flaky(MockRateLimitError(), times=2, then="recovered")])
    """

    def create(
        self, messages: list[dict[str, str]] | None = None, **kwargs
    ) -> MockCompletion | MockStream:
        """Mimics openai.chat.completions.create()."""
        self._call_log.append({"messages": messages, **kwargs})
        completion = self._build_completion(self._next_response())
        if kwargs.get("stream"):
            return MockStream(_stream_chunks(completion))
        return completion


class AsyncMockLLM(_Scripted):
    """Drop-in for openai.AsyncOpenAI: same scripting as MockLLM, but create() is awaited."""

    async def create(
        self, messages: list[dict[str, str]] | None = None, **kwargs
    ) -> MockCompletion | AsyncMockStream:
        """Mimics AsyncOpenAI's chat.completions.create()."""
        self._call_log.append({"messages": messages, **kwargs})
        completion = self._build_completion(self._next_response())
        if kwargs.get("stream"):
            return AsyncMockStream(_stream_chunks(completion))
        return completion
