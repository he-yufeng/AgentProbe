"""EvalPort ResultSet export: field mapping, file output, and the pytest option."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentprobe.evalport import GRADER_ID, to_grader_result, to_resultset, write_resultset
from agentprobe.snapshot import Snapshot, SnapshotResult


def _make_result(**overrides) -> SnapshotResult:
    fields = {
        "name": "greeting",
        "output": {"text": "hello"},
        "baseline": {"output": {"text": "hello"}},
        "passed": True,
        "similarity": 0.97,
        "message": "",
        "mode": "semantic",
    }
    fields.update(overrides)
    return SnapshotResult(**fields)


def test_grader_result_maps_semantic_score_and_reason():
    result = _make_result(passed=False, similarity=0.72, message="snapshot mismatch")
    grader = to_grader_result(result)
    assert grader == {
        "grader_id": GRADER_ID,
        "type": "semantic_similarity",
        "score": 0.72,
        "passed": False,
        "reason": "snapshot mismatch",
    }


def test_grader_result_omits_reason_without_message():
    assert "reason" not in to_grader_result(_make_result())


def test_grader_result_type_follows_mode():
    assert to_grader_result(_make_result(mode="exact"))["type"] == "exact_match"
    assert to_grader_result(_make_result(mode="fuzzy"))["type"] == "agentprobe_fuzzy"
    assert to_grader_result(_make_result(mode=None))["type"] == GRADER_ID


def test_grader_result_falls_back_to_passed_when_never_compared():
    created = to_grader_result(_make_result(similarity=None, message="snapshot created"))
    assert created["score"] == 1.0
    assert created["reason"] == "snapshot created"
    failed = to_grader_result(_make_result(passed=False, similarity=None))
    assert failed["score"] == 0.0


def test_resultset_has_required_envelope():
    rs = to_resultset([_make_result()], started_at="2026-08-27T00:00:00Z")
    assert rs["$schema"] == "https://evalport.org/schema/resultset.json"
    assert rs["version"] == "1.0.0"
    assert rs["suite_id"] == "agentprobe"
    assert rs["run_id"].startswith("agentprobe-")
    assert rs["started_at"] == "2026-08-27T00:00:00Z"
    assert rs["completed_at"].endswith("Z")
    assert rs["runner"]["name"] == "agentprobe"
    assert rs["runner"]["version"]


def test_resultset_maps_results():
    rs = to_resultset(
        [
            _make_result(name="str_out", output="hello", similarity=1.0, mode="exact"),
            _make_result(name="dict_out"),
        ],
        suite_id="my-suite",
        run_id="run-1",
    )
    assert rs["suite_id"] == "my-suite"
    assert rs["run_id"] == "run-1"
    first, second = rs["results"]
    assert first["test_case_id"] == "str_out"
    assert first["actual_output"] == "hello"
    assert first["passed"] is True
    assert len(first["grader_results"]) == 1
    assert second["actual_output"] == json.dumps({"text": "hello"}, sort_keys=True)


def test_resultset_summary_aggregates():
    rs = to_resultset(
        [
            _make_result(name="a", similarity=1.0),
            _make_result(name="b", passed=False, similarity=0.5),
            _make_result(name="c", similarity=None),
        ]
    )
    summary = rs["summary"]
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["skipped"] == 0
    assert summary["pass_rate"] == 2 / 3
    assert summary["avg_score"] == (1.0 + 0.5 + 1.0) / 3
    assert summary["by_grader"][GRADER_ID] == {
        "passed": 2,
        "failed": 1,
        "avg_score": summary["avg_score"],
    }


def test_resultset_empty_has_no_summary():
    rs = to_resultset([])
    assert rs["results"] == []
    assert "summary" not in rs


def test_write_resultset_creates_parents_and_roundtrips(tmp_path: Path):
    path = write_resultset([_make_result()], tmp_path / "ci" / "resultset.json", run_id="run-1")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["results"][0]["test_case_id"] == "greeting"


def test_export_from_real_captures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    snap = Snapshot()
    snap.capture("greeting", {"text": "hello"})
    result = snap.capture("greeting", {"text": "hello"})
    assert result.passed

    path = write_resultset(snap.results, "resultset.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    created, compared = data["results"]
    assert created["grader_results"][0]["reason"] == "snapshot created"
    assert compared["grader_results"][0]["type"] == "exact_match"
    assert compared["grader_results"][0]["score"] == 1.0


AGENT_TEST = '''from agentprobe import snapshot


def test_greeting(agentprobe):
    agentprobe.capture("greeting", "hello")


@snapshot("decorated_call")
def test_decorated():
    return {"answer": "42"}
'''


def _run_pytest(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_evalport_option_writes_resultset(tmp_path: Path):
    (tmp_path / "test_agent.py").write_text(AGENT_TEST)
    proc = _run_pytest(tmp_path, "test_agent.py", "--agentprobe-evalport=result.json", "-q")
    assert proc.returncode == 0, proc.stderr

    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    by_id = {r["test_case_id"]: r for r in data["results"]}
    assert set(by_id) == {"greeting", "decorated_call"}
    assert all(r["passed"] for r in by_id.values())
    assert by_id["greeting"]["grader_results"][0]["reason"] == "snapshot created"
    assert "wrote EvalPort ResultSet" in proc.stdout


def test_evalport_option_records_comparisons_on_second_run(tmp_path: Path):
    (tmp_path / "test_agent.py").write_text(AGENT_TEST)
    first = _run_pytest(tmp_path, "test_agent.py", "-q")
    assert first.returncode == 0, first.stderr

    proc = _run_pytest(tmp_path, "test_agent.py", "--agentprobe-evalport=result.json", "-q")
    assert proc.returncode == 0, proc.stderr
    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    for entry in data["results"]:
        grader = entry["grader_results"][0]
        assert grader["type"] == "exact_match"
        assert grader["score"] == 1.0
        assert grader["passed"] is True
    assert data["summary"]["pass_rate"] == 1.0


def test_no_option_writes_nothing(tmp_path: Path):
    (tmp_path / "test_agent.py").write_text(AGENT_TEST)
    proc = _run_pytest(tmp_path, "test_agent.py", "-q")
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "result.json").exists()
    assert "EvalPort" not in proc.stdout
