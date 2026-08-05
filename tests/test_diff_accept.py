"""The failing-run workflow: _assert_snapshot persists last_run, diff shows it, accept promotes it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentprobe.cli import main
from agentprobe.snapshot import _assert_snapshot
from agentprobe.storage import (
    delete_last_run,
    list_last_runs,
    load_last_run,
    load_snapshot,
    save_last_run,
    save_snapshot,
)


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _seed_baseline_and_last_run() -> None:
    save_snapshot("summarize", {"output": {"text": "old summary", "ts": "<redacted>"}, "timestamp": 1.0})
    save_last_run(
        "summarize",
        {
            "output": {"text": "new summary", "ts": "<redacted>"},
            "timestamp": 2.0,
            "mode": "exact",
            "threshold": 0.85,
            "similarity": 0.72,
        },
    )


def test_mismatch_persists_last_run(workdir: Path) -> None:
    save_snapshot("case", {"output": "before", "timestamp": 1.0})
    with pytest.raises(AssertionError):
        _assert_snapshot("case", "after", update=False, mode="exact", threshold=0.85)

    saved = load_last_run("case")
    assert saved is not None
    assert saved["output"] == "after"
    assert saved["mode"] == "exact"


def test_mismatch_last_run_is_redacted(workdir: Path) -> None:
    save_snapshot("case", {"output": {"token": "old-token"}, "timestamp": 1.0})
    with pytest.raises(AssertionError):
        _assert_snapshot(
            "case", {"token": "sk-live-123456789"}, update=False, mode="exact", threshold=0.85, redact=["token"]
        )

    saved = load_last_run("case")
    assert saved is not None
    assert "sk-live-123456789" not in json.dumps(saved)


def test_list_last_runs(workdir: Path) -> None:
    assert list_last_runs() == []
    save_last_run("b_case", {"output": 2, "timestamp": 1.0})
    save_last_run("a_case", {"output": 1, "timestamp": 1.0})
    assert list_last_runs() == ["a_case", "b_case"]
    delete_last_run("a_case")
    assert list_last_runs() == ["b_case"]


def test_diff_shows_similarity_and_changes(workdir: Path) -> None:
    _seed_baseline_and_last_run()
    result = CliRunner().invoke(main, ["diff", "summarize"])

    assert result.exit_code == 0
    assert "--- summarize ---" in result.output
    assert "similarity: 0.7200" in result.output
    assert '-  "text": "old summary",' in result.output
    assert '+  "text": "new summary",' in result.output


def test_diff_all_and_empty(workdir: Path) -> None:
    assert CliRunner().invoke(main, ["diff"]).output.startswith("No saved failing runs.")

    _seed_baseline_and_last_run()
    assert "--- summarize ---" in CliRunner().invoke(main, ["diff"]).output


def test_diff_missing_name_errors(workdir: Path) -> None:
    result = CliRunner().invoke(main, ["diff", "nope"])
    assert result.exit_code != 0
    assert "No saved failing run for 'nope'" in result.output


def test_accept_promotes_and_clears(workdir: Path) -> None:
    _seed_baseline_and_last_run()
    result = CliRunner().invoke(main, ["accept", "summarize"])

    assert result.exit_code == 0, result.output
    baseline = load_snapshot("summarize")
    assert baseline is not None
    assert baseline["output"] == {"text": "new summary", "ts": "<redacted>"}
    assert baseline["timestamp"] > 2.0
    assert load_last_run("summarize") is None


def test_accept_all_and_empty(workdir: Path) -> None:
    assert CliRunner().invoke(main, ["accept"]).output.startswith("No saved failing runs to accept.")

    _seed_baseline_and_last_run()
    result = CliRunner().invoke(main, ["accept"])
    assert result.exit_code == 0
    assert load_last_run("summarize") is None


def test_diff_html_writes_escaped_report(workdir: Path, tmp_path: Path) -> None:
    _seed_baseline_and_last_run()
    out = tmp_path / "report.html"

    result = CliRunner().invoke(main, ["diff", "--html", str(out)])

    assert result.exit_code == 0, result.output
    body = out.read_text(encoding="utf-8")
    assert "<h1>AgentProbe diff report</h1>" in body
    assert "summarize" in body
    assert "old summary" in body and "new summary" in body
    assert "similarity" in body
    assert "<span class=\"ins\">" in body and "<span class=\"del\">" in body


def test_diff_html_escapes_snapshot_content(workdir: Path, tmp_path: Path) -> None:
    save_snapshot("evil", {"output": {"text": "<script>alert(1)</script>"}, "timestamp": 1.0})
    save_last_run("evil", {"output": {"text": "<b>ok</b>"}, "timestamp": 2.0})
    out = tmp_path / "report.html"

    result = CliRunner().invoke(main, ["diff", "--html", str(out)])

    assert result.exit_code == 0, result.output
    body = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


def test_render_diff_report_hunk_headers_not_colored() -> None:
    from agentprobe.report import render_diff_report

    body = render_diff_report([{
        "name": "s",
        "diff": "--- baseline\n+++ last_run\n@@ -1 +1 @@\n-old\n+new",
        "similarity": None,
    }])
    assert '<span class="ctx">--- baseline</span>' in body
    assert '<span class="ctx">+++ last_run</span>' in body
    assert '<span class="ins">+new</span>' in body
    assert '<span class="del">-old</span>' in body


def test_diff_stat_summary(workdir: Path) -> None:
    _seed_baseline_and_last_run()
    result = CliRunner().invoke(main, ["diff", "--stat"])

    assert result.exit_code == 0
    assert "summarize: +1 -1" in result.output
    assert "similarity: 0.7200" in result.output
    # no diff body in stat mode
    assert '"text": "old summary"' not in result.output


def test_diff_stat_and_html_conflict(workdir: Path, tmp_path: Path) -> None:
    _seed_baseline_and_last_run()
    result = CliRunner().invoke(main, ["diff", "--stat", "--html", str(tmp_path / "r.html")])

    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_diff_stat_counts_skip_headers() -> None:
    from agentprobe.cli import _diff_stat

    diff_text = "--- baseline\n+++ last_run\n@@ -1,3 +1,3 @@\n ctx\n-old\n+new\n+extra"
    assert _diff_stat(diff_text) == (2, 1)
