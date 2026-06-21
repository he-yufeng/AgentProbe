"""Snapshot storage — reads and writes JSON baselines to .agentprobe/snapshots/."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_DIR = Path(".agentprobe") / "snapshots"

_T = TypeVar("_T")


def _snapshot_path(name: str, directory: Path = DEFAULT_DIR) -> Path:
    safe_name = name.replace("/", "__").replace("\\", "__")
    return directory / f"{safe_name}.json"


def load_snapshot(name: str, directory: Path = DEFAULT_DIR) -> dict[str, Any] | None:
    path = _snapshot_path(name, directory)
    if not path.exists():
        return None
    # Reading needs the same Windows retry as the write side: a reader that opens
    # the file at the instant a concurrent writer is replacing it can hit a
    # PermissionError, so back off and retry rather than surfacing a spurious
    # error under parallel pytest-xdist contention.
    text = _retry_on_windows_lock(lambda: path.read_text(encoding="utf-8"))
    return json.loads(text)


def save_snapshot(name: str, data: dict[str, Any], directory: Path = DEFAULT_DIR) -> Path:
    path = _snapshot_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    # Write to a unique temp file in the same directory, then atomically replace.
    # os.replace is atomic on a single filesystem, so a concurrent reader (e.g. a
    # parallel pytest-xdist worker) always sees either the old or the new complete
    # snapshot, never a half-written file.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        _retry_on_windows_lock(lambda: os.replace(tmp, path))
    except BaseException:
        # Don't leave a temp file behind if the write or replace fails.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _retry_on_windows_lock(
    action: Callable[[], _T], attempts: int = 40, base_delay: float = 0.005
) -> _T:
    """Run a filesystem ``action``, retrying the Windows file-lock PermissionError.

    On POSIX, replacing or reading a snapshot succeeds even while another thread
    holds the file open. On Windows either side can raise ``PermissionError`` if
    a concurrent replace/read overlaps at that instant. Under sustained
    contention — the documented use case of many parallel pytest-xdist workers
    hammering one snapshot — the lock can recur, so back off exponentially
    (capped) to widen the retry window well beyond a single momentary hold before
    giving up. The happy path still runs on the first attempt with no delay.
    """
    delay = base_delay
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)
    raise AssertionError("unreachable")  # pragma: no cover
