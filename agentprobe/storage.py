"""Snapshot storage — reads and writes JSON baselines to .agentprobe/snapshots/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".agentprobe") / "snapshots"


def _snapshot_path(name: str, directory: Path = DEFAULT_DIR) -> Path:
    safe_name = name.replace("/", "__").replace("\\", "__")
    return directory / f"{safe_name}.json"


def load_snapshot(name: str, directory: Path = DEFAULT_DIR) -> dict[str, Any] | None:
    path = _snapshot_path(name, directory)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_snapshot(name: str, data: dict[str, Any], directory: Path = DEFAULT_DIR) -> Path:
    path = _snapshot_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
