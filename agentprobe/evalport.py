"""Export snapshot results as an EvalPort-compatible ResultSet.

EvalPort (https://github.com/adhabnr-ux/evalport) is an open JSON interchange
format for LLM eval results, so a ResultSet written here can be diffed,
archived, or displayed by tooling built for any other framework that speaks
it. Each SnapshotResult becomes one test-case result carrying a single
``agentprobe_snapshot`` GraderResult: ``passed`` carries over, ``similarity``
becomes ``score`` (baseline creates and updates never compared anything, so
they fall back to 1.0/0.0 from ``passed``), and ``message`` becomes ``reason``.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentprobe.storage import _retry_on_windows_lock

if TYPE_CHECKING:
    from agentprobe.snapshot import SnapshotResult

SPEC_VERSION = "1.0.0"
SCHEMA_URL = "https://evalport.org/schema/resultset.json"
GRADER_ID = "agentprobe_snapshot"

# Comparison mode -> EvalPort grader type. fuzzy has no well-known equivalent
# (character n-gram cosine, not embeddings), so it keeps a framework-specific
# name, which the spec's type-openness rule allows.
_GRADER_TYPES = {
    "exact": "exact_match",
    "semantic": "semantic_similarity",
    "fuzzy": "agentprobe_fuzzy",
}


def to_grader_result(result: SnapshotResult) -> dict[str, Any]:
    """Map one SnapshotResult to an EvalPort GraderResult."""
    grader: dict[str, Any] = {
        "grader_id": GRADER_ID,
        "type": _GRADER_TYPES.get(result.mode or "", GRADER_ID),
        "score": result.similarity if result.similarity is not None else float(result.passed),
        "passed": result.passed,
    }
    if result.message:
        grader["reason"] = result.message
    return grader


def to_resultset(
    results: list[SnapshotResult],
    *,
    suite_id: str = "agentprobe",
    run_id: str | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Map a run's SnapshotResults to an EvalPort ResultSet dict."""
    entries = [
        {
            "test_case_id": r.name,
            "actual_output": (
                r.output
                if isinstance(r.output, str)
                else json.dumps(r.output, sort_keys=True, default=str)
            ),
            "grader_results": [to_grader_result(r)],
            "passed": r.passed,
        }
        for r in results
    ]
    resultset: dict[str, Any] = {
        "$schema": SCHEMA_URL,
        "version": SPEC_VERSION,
        "suite_id": suite_id,
        "run_id": run_id or f"agentprobe-{uuid.uuid4().hex[:12]}",
        "started_at": started_at or _utcnow(),
        "completed_at": _utcnow(),
        "runner": {"name": "agentprobe", "version": _runner_version()},
        "results": entries,
    }
    if entries:
        resultset["summary"] = _summary(entries)
    return resultset


def write_resultset(
    results: list[SnapshotResult],
    path: str | Path,
    *,
    suite_id: str = "agentprobe",
    run_id: str | None = None,
    started_at: str | None = None,
) -> Path:
    """Write the ResultSet JSON to ``path``, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        to_resultset(results, suite_id=suite_id, run_id=run_id, started_at=started_at),
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    # Same atomic temp-then-replace write as snapshot storage, so a concurrent
    # reader never sees a half-written ResultSet.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        _retry_on_windows_lock(lambda: os.replace(tmp, path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    passed = sum(1 for e in entries if e["passed"])
    scores = [g["score"] for e in entries for g in e["grader_results"] if g["score"] is not None]
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "skipped": 0,
        "pass_rate": passed / total,
        "avg_score": avg,
        "by_grader": {GRADER_ID: {"passed": passed, "failed": total - passed, "avg_score": avg}},
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _runner_version() -> str:
    try:
        return importlib.metadata.version("agentpoke")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
