"""AgentProbe — regression-testing for AI agents."""

from agentprobe.assertions import (
    assert_no_tool_called,
    assert_schema,
    assert_tool_called,
    assert_tool_sequence,
)
from agentprobe.mock_llm import MockLLM
from agentprobe.snapshot import Snapshot, snapshot

__version__ = "0.1.0"
__all__ = [
    "snapshot",
    "Snapshot",
    "MockLLM",
    "assert_tool_called",
    "assert_no_tool_called",
    "assert_tool_sequence",
    "assert_schema",
]
