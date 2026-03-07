"""Tests for MockLLM."""

from agentprobe.mock_llm import MockLLM


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
