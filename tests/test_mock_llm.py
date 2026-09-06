"""Tests for MockLLM."""

import asyncio
import inspect

import pytest

from agentprobe.mock_llm import (
    AsyncMockLLM,
    Flaky,
    MockLLM,
    MockRateLimitError,
    MockServerError,
    MockTimeoutError,
)


def test_mock_returns_scripted_responses():
    mock = MockLLM(responses=["Hello!", "Goodbye!"])
    r1 = mock.create(messages=[{"role": "user", "content": "Hi"}])
    assert r1.choices[0].message.content == "Hello!"
    r2 = mock.create(messages=[{"role": "user", "content": "Bye"}])
    assert r2.choices[0].message.content == "Goodbye!"


def test_mock_falls_back_to_default():
    mock = MockLLM(responses=["Only one"], default_response="fallback")
    mock.create()
    r = mock.create()
    assert r.choices[0].message.content == "fallback"


def test_mock_records_calls():
    mock = MockLLM(responses=["ok"])
    mock.create(messages=[{"role": "user", "content": "test"}], temperature=0.5)
    assert mock.call_count == 1
    assert mock.calls[0]["messages"][0]["content"] == "test"
    assert mock.calls[0]["temperature"] == 0.5


def test_mock_tool_calls():
    mock = MockLLM(responses=[
        {"tool_calls": [{"id": "1", "function": {"name": "search", "arguments": "{}"}}]}
    ])
    r = mock.create()
    assert r.choices[0].finish_reason == "tool_calls"
    assert r.choices[0].message.tool_calls is not None


def test_mock_reset():
    mock = MockLLM(responses=["a", "b"])
    mock.create()
    mock.reset()
    assert mock.call_count == 0
    r = mock.create()
    assert r.choices[0].message.content == "a"


def test_mock_openai_compatible_interface():
    mock = MockLLM(responses=["works"])
    r = mock.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert r.choices[0].message.content == "works"


def test_mock_to_dict():
    mock = MockLLM(responses=["test"])
    r = mock.create()
    d = r.to_dict()
    assert d["choices"][0]["message"]["content"] == "test"
    assert "tool_calls" not in d["choices"][0]["message"]


def test_mock_streaming_chunks():
    mock = MockLLM(responses=["Hello brave new world"])
    chunks = list(mock.create(messages=[{"role": "user", "content": "hi"}], stream=True))

    assert all(c.object == "chat.completion.chunk" for c in chunks)
    # first chunk announces the role, last carries finish_reason, content in between
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content is None
    assert chunks[-1].choices[0].finish_reason == "stop"
    text = "".join(c.choices[0].delta.content or "" for c in chunks[1:-1])
    assert text == "Hello brave new world"


def test_mock_streaming_multiple_responses_in_order():
    mock = MockLLM(responses=["first", "second"])
    s1 = mock.create(stream=True)
    s2 = mock.create(stream=True)
    join = lambda stream: "".join(c.choices[0].delta.content or "" for c in stream)
    assert join(s1) == "first"
    assert join(s2) == "second"


def test_mock_streaming_tool_calls():
    mock = MockLLM(responses=[
        {"tool_calls": [{"id": "1", "function": {"name": "search", "arguments": "{}"}}]}
    ])
    chunks = list(mock.create(stream=True))
    deltas = [c.choices[0].delta for c in chunks]
    assert any(d.tool_calls for d in deltas)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_mock_streaming_context_manager():
    mock = MockLLM(responses=["hi there"])
    with mock.create(stream=True) as stream:
        text = "".join(c.choices[0].delta.content or "" for c in stream)
    assert text == "hi there"


def test_exception_entry_raises_then_script_advances():
    mock = MockLLM(responses=[MockRateLimitError("slow down")], default_response="fallback")
    with pytest.raises(MockRateLimitError) as exc_info:
        mock.create()
    assert exc_info.value.status_code == 429
    # the failing entry was consumed; the next call hits the default
    assert mock.create().choices[0].message.content == "fallback"
    assert mock.call_count == 2


def test_flaky_fails_then_succeeds():
    mock = MockLLM(responses=[Flaky(MockServerError(), times=2, then="recovered")])
    with pytest.raises(MockServerError):
        mock.create()
    with pytest.raises(MockServerError):
        mock.create()
    assert mock.create().choices[0].message.content == "recovered"
    assert mock.call_count == 3


def test_flaky_reset_restores_failures():
    mock = MockLLM(responses=[Flaky(MockTimeoutError(), times=1, then="ok")])
    with pytest.raises(MockTimeoutError):
        mock.create()
    assert mock.create().choices[0].message.content == "ok"
    mock.reset()
    with pytest.raises(MockTimeoutError):
        mock.create()


def test_flaky_requires_at_least_one_failure():
    with pytest.raises(ValueError):
        Flaky(MockServerError(), times=0)


def test_async_mock_mirrors_openai_client_shape():
    mock = AsyncMockLLM(responses=["works"])
    assert mock.chat is mock and mock.completions is mock
    assert inspect.iscoroutinefunction(mock.chat.completions.create)


def test_async_mock_create():
    async def run():
        mock = AsyncMockLLM(responses=["Hello!", "Goodbye!"])
        r1 = await mock.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
        r2 = await mock.create(messages=[{"role": "user", "content": "Bye"}])
        assert r1.choices[0].message.content == "Hello!"
        assert r2.choices[0].message.content == "Goodbye!"
        assert mock.call_count == 2

    asyncio.run(run())


def test_async_mock_streaming():
    async def run():
        mock = AsyncMockLLM(responses=["one two three"])
        stream = await mock.create(messages=[], stream=True)
        parts = []
        async for chunk in stream:
            parts.append(chunk.choices[0].delta.content or "")
        assert "".join(parts) == "one two three"

    asyncio.run(run())


def test_async_mock_failure_injection():
    async def run():
        mock = AsyncMockLLM(responses=[Flaky(MockTimeoutError(), times=1, then="ok")])
        with pytest.raises(MockTimeoutError):
            await mock.create()
        r = await mock.create()
        assert r.choices[0].message.content == "ok"

    asyncio.run(run())
