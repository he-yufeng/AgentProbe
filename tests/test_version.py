"""Keep __version__, pyproject, and the installed dist metadata in lockstep."""

import importlib.metadata
import re
from pathlib import Path

import agentprobe

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


def _pyproject_version() -> str:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    if tomllib is not None:
        return tomllib.loads(text)["project"]["version"]
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match, "no version line in pyproject.toml"
    return match.group(1)


def test_version_is_consistent_everywhere():
    expected = _pyproject_version()
    assert agentprobe.__version__ == expected
    assert importlib.metadata.version("agentpoke") == expected
