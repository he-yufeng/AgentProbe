"""AgentProbe — regression-testing for AI agents."""

from agentprobe.assertions import (
    assert_no_repeated_calls,
    assert_no_tool_called,
    assert_only_tools_used,
    assert_schema,
    assert_tool_called,
    assert_tool_not_called_with,
    assert_tool_sequence,
)
from agentprobe.mock_llm import MockLLM
from agentprobe.snapshot import Snapshot, snapshot
from agentprobe.trace import Step, Trace

__version__ = "0.1.0"
__all__ = [
    "snapshot",
    "Snapshot",
    "MockLLM",
    "Trace",
    "Step",
    "assert_tool_called",
    "assert_no_tool_called",
    "assert_no_repeated_calls",
    "assert_only_tools_used",
    "assert_tool_not_called_with",
    "assert_tool_sequence",
    "assert_schema",
]
