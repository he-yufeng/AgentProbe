<div align="center">

# AgentProbe

**Regression-testing for AI agents. Like Jest snapshots, but for LLMs.**

Capture your agent's outputs, store them as baselines, and catch regressions in CI — with one decorator.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/he-yufeng/AgentProbe/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/AgentProbe/actions)

**[English](README.md) | [中文](README_CN.md)**

</div>

---

## The Problem

You ship an AI agent. It works great. Two weeks later, you update a prompt, swap a model, or bump a dependency — and something breaks. But you don't notice until a user complains, because **there's no test that catches agent behavior regressions**.

Traditional unit tests don't work for agents. The outputs are non-deterministic. They're natural language, not exact values. You can't just `assertEqual`. And even if you could, you'd spend more time writing test fixtures than writing the agent itself.

**AgentProbe** fixes this. One decorator captures your agent's output and saves it as a baseline snapshot. On the next run, it compares the new output against the baseline — using exact match or semantic similarity. If something changed, the test fails. Run it in CI, and you'll catch regressions before they hit production.

## Quick Start

```bash
pip install agentprobe
```

### 1. Snapshot Testing

Capture agent outputs and compare them across runs:

```python
from agentprobe import snapshot

@snapshot("summarize_article")
def test_summarize():
    # Your agent code here
    result = my_agent.summarize("The quick brown fox jumps over the lazy dog.")
    return result
```

First run: creates a baseline in `.agentprobe/snapshots/summarize_article.json`.
Next runs: compares the output against the baseline. Fails if they differ.

Async agents work the same way:

```python
@snapshot("async_summarize")
async def test_async_summarize():
    result = await my_agent.summarize_async("The quick brown fox jumps over the lazy dog.")
    return result
```

When the output carries non-deterministic fields like timestamps or request ids, list them in `redact` so they're masked before comparison and don't cause spurious mismatches:

```python
@snapshot("summarize_article", redact=["timestamp", "request_id"])
def test_summarize():
    return my_agent.summarize("...")  # {"summary": "...", "timestamp": 1718...}
```

The named keys are replaced with `"<redacted>"` at any depth before the snapshot is saved and compared. Real changes to other fields still fail the snapshot.

### 2. Mock LLM

Test agent logic without hitting any API:

```python
from agentprobe import MockLLM

def test_agent_with_mock():
    mock = MockLLM(responses=[
        "The document discusses three main topics.",
        "Based on my analysis, the sentiment is positive."
    ])

    # Use mock.chat.completions.create as a drop-in for openai
    result = mock.chat.completions.create(
        messages=[{"role": "user", "content": "Summarize this doc"}]
    )
    assert "three main topics" in result.choices[0].message.content
    assert mock.call_count == 1
```

### 3. Tool Call Assertions

Verify your agent calls the right tools:

```python
from agentprobe import assert_no_tool_called, assert_tool_called, assert_tool_sequence

def test_agent_uses_search():
    tool_calls = [
        {"name": "web_search", "arguments": {"query": "latest news"}},
        {"name": "summarize", "arguments": {"text": "..."}},
    ]
    assert_tool_called(tool_calls, "web_search", times=1)
    assert_tool_called(tool_calls, "web_search", with_args={"query": "latest news"})
    assert_tool_sequence(tool_calls, ["web_search", "summarize"])
    assert_no_tool_called(tool_calls, "delete_file")
```

For multi-step agents, `assert_tool_sequence(..., contiguous=True)` catches accidental planner reorderings where a tool must immediately follow another tool.

When the exact call count is non-deterministic, use `min_times`/`max_times` instead of `times` — for example, assert a flaky API was retried at most three times, or a search ran at least twice:

```python
assert_tool_called(tool_calls, "api_call", max_times=3)   # retried, but bounded
assert_tool_called(tool_calls, "web_search", min_times=2)  # at least two searches
```

For overall efficiency, `assert_max_tool_calls` bounds the whole run rather than one tool — an agent can avoid repeating any single call and still be wastefully chatty. Unlike `max_times`, the budget may be met with zero calls:

```python
assert_max_tool_calls(tool_calls, 10)                      # solve it in at most 10 calls
assert_max_tool_calls(tool_calls, 3, tool_name="api_call")  # at most 3 api_calls (zero is fine)
```

`with_args` is a nested subset match and also accepts OpenAI-style JSON string arguments:

```python
assert_tool_called(
    tool_calls,
    "write_file",
    with_args={"metadata": {"mode": "safe"}},
)
```

For safety checks, `assert_tool_not_called_with` is the negative counterpart: it allows the tool but fails if any call carried a forbidden argument subset — handy when a tool is fine to use except in a dangerous mode:

```python
# the agent may run shell commands, but never with sudo
assert_tool_not_called_with(tool_calls, "run", {"sudo": True})
# and may delete files, but never the filesystem root
assert_tool_not_called_with(tool_calls, "delete_file", {"path": "/"})
```

### 4. Schema Validation

Assert that agent outputs conform to a structure:

```python
from pydantic import BaseModel
from agentprobe import assert_schema

class AgentResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

def test_output_structure():
    output = my_agent.run("What is the capital of France?")
    result = assert_schema(output, AgentResponse)
    assert result.confidence > 0.8
```

### 5. Multi-Step Tracing

Record what an agent did step by step, then assert over the trace or snapshot it. `trace.tool_calls` drops straight into the assertion helpers:

```python
from agentprobe import Trace, assert_tool_sequence

def test_research_flow():
    trace = Trace()
    # record steps as your agent runs (tool calls, LLM turns, custom events)
    trace.record_llm("planning the search")
    trace.record_tool_call("search", {"query": "rainfall 2023"})
    trace.record_event("retry", attempt=2)
    trace.record_tool_call("fetch", {"url": "https://example.com"})

    assert_tool_sequence(trace.tool_calls, ["search", "fetch"])
    assert trace.names == ["llm", "search", "retry", "fetch"]
    # trace.to_dict() is snapshot-friendly for full-run regression tests
```

### 6. Cost Tracking

Record token usage on a trace and assert the run stayed under a USD budget — catching regressions that quietly burn more money (longer prompts, extra turns, a pricier model). Pricing comes from a dict, a callable, or [TokenTracker](https://github.com/he-yufeng/TokenTracker)'s price table when it's installed:

```python
from agentprobe import Trace, assert_cost_under

def test_run_stays_under_budget():
    trace = Trace()
    trace.record_llm("plan", model="gpt-4o", input_tokens=1200, output_tokens=300)
    trace.record_llm("answer", model="gpt-4o", input_tokens=800, output_tokens=500)

    # pricing dict: {model: (input_per_1k_usd, output_per_1k_usd)}
    assert_cost_under(trace, 0.05, pricing={"gpt-4o": (0.005, 0.015)})
    # or pricing=None to use TokenTracker's table (pip install tokentracker)
```

## Pytest Integration

AgentProbe registers as a pytest plugin automatically. Use the `agentprobe` fixture:

```python
def test_with_fixture(agentprobe):
    output = my_agent.run("Hello")
    result = agentprobe.capture("greeting_test", output)
    assert result.passed
```

### CLI Flags

```bash
# Run tests normally
pytest tests/

# Update all snapshots (regenerate baselines)
pytest tests/ --agentprobe-update

# Use semantic comparison instead of exact match
pytest tests/ --agentprobe-mode=semantic --agentprobe-threshold=0.85
```

When a snapshot changes, AgentProbe prints a unified diff between the stored JSON snapshot
and the current output, so CI logs show the exact field or sentence that drifted.

### AgentProbe CLI

```bash
# Run tests
agentprobe run

# Run with semantic comparison
agentprobe run --mode semantic --threshold 0.9

# Update all snapshots
agentprobe update
```

## Comparison Modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| `exact` (default) | String equality after serialization | Deterministic agents, structured outputs |
| `semantic` | Cosine similarity via sentence-transformers | Non-deterministic LLM outputs |

For semantic mode, install the optional dependency:

```bash
pip install agentprobe[semantic]
```

## MockLLM Features

`MockLLM` is a drop-in replacement for `openai.Client` that returns scripted responses:

```python
from agentprobe import MockLLM

# Scripted responses (consumed in order)
mock = MockLLM(responses=["First response", "Second response"])

# Falls back to default after scripted responses are exhausted
mock = MockLLM(responses=["Only one"], default_response="I don't know")

# Simulate tool calls
mock = MockLLM(responses=[
    {"tool_calls": [{"id": "1", "function": {"name": "search", "arguments": '{"q": "test"}'}}]}
])

# Check what was called
mock.create(messages=[{"role": "user", "content": "Hi"}])
print(mock.calls)       # all recorded calls
print(mock.call_count)   # number of calls

# Reset for reuse
mock.reset()
```

## How It Compares

| Feature | AgentProbe | DeepEval | Promptfoo |
|---------|-----------|----------|-----------|
| pytest native | Yes (plugin) | Separate runner | CLI only |
| Snapshot baselines | Yes | No | No |
| Semantic comparison | Yes | Yes | Yes |
| Mock LLM | Yes (built-in) | No | Partial |
| Tool call assertions | Yes | No | No |
| Schema validation | Yes (Pydantic) | Partial | No |
| Cloud required | No | Optional | No |
| Config format | Python code | Python code | YAML |

## GitHub Actions

Add this to your CI pipeline:

```yaml
- name: Run agent tests
  run: |
    pip install agentprobe
    pytest tests/ -v
```

Snapshot files (`.agentprobe/snapshots/`) should be committed to your repo so CI can compare against them.

## FAQ

**Do I need an API key to use AgentProbe?**
No. Use `MockLLM` for deterministic tests without any API calls. If you want to test against a real LLM, you'll need the appropriate API key, but that's your agent's dependency, not AgentProbe's.

**How does semantic comparison work?**
It uses sentence-transformers to embed both the baseline and current output, then computes cosine similarity. If the score is above the threshold (default 0.85), the test passes. This handles cases where the wording changes but the meaning stays the same.

**Can I use this with LangChain / CrewAI / AutoGen?**
Yes. AgentProbe doesn't care what framework you use. It tests the output of your agent, not the internals. Just call your agent inside the test function and return the result.

**What about flaky tests from non-deterministic outputs?**
Use semantic mode with an appropriate threshold. If your agent's outputs vary significantly between runs, lower the threshold. If they should be consistent, raise it. You can also use `MockLLM` to make the underlying LLM deterministic.

**How do snapshot files work?**
Snapshots are stored as JSON in `.agentprobe/snapshots/`. The first time you run a test, it creates the baseline. Subsequent runs compare against it. Use `--agentprobe-update` to regenerate baselines after intentional changes.

## Roadmap

- [x] Async agent support (`async def` tests)
- [x] Multi-step agent tracing (record intermediate steps)
- [x] Cost tracking integration (with TokenTracker)
- [x] Visual diff in terminal for snapshot mismatches
- [x] `pytest-xdist` parallel support (atomic snapshot writes)

## Contributing

Contributions welcome. If you're testing AI agents in production and have ideas for what's missing, open an issue.

## Related projects

- [IssueBenchKit](https://github.com/he-yufeng/IssueBenchKit) — turn real GitHub issues into reproducible coding-agent benchmark tasks
- [LiteBench](https://github.com/he-yufeng/LiteBench) — a pip-installable benchmark runner for LLMs and agents
- [CodeJoust](https://github.com/he-yufeng/CodeJoust) — pit coding agents against the same bug and score the patches
- [FlightBox](https://github.com/he-yufeng/FlightBox) — record, replay, and diff local agent sessions
- [agentcikit](https://github.com/he-yufeng/agentcikit) — CLI tools for AI-agent, MCP, and CI evidence and safety

## License

[MIT](LICENSE)

---

<div align="center">

**Stop shipping untested agents.**

[Report a Bug](https://github.com/he-yufeng/AgentProbe/issues) · [Request a Feature](https://github.com/he-yufeng/AgentProbe/issues)

</div>
