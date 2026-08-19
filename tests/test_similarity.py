"""Tests for the pure-stdlib fuzzy similarity mode (no extra dependencies)."""

import shutil

import pytest

from agentprobe.similarity import fuzzy_similarity, texts_match
from agentprobe.snapshot import Snapshot
from agentprobe.storage import DEFAULT_DIR


@pytest.fixture(autouse=True)
def clean_snapshots():
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR)
    yield
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR)


PARAGRAPH = "The agent searched the knowledge base, found three matching documents, and summarized them into a short answer."


def test_fuzzy_identical_scores_one():
    assert fuzzy_similarity(PARAGRAPH, PARAGRAPH) == pytest.approx(1.0)


def test_fuzzy_unrelated_scores_low():
    other = "Quantum entanglement enables correlations that classical systems cannot reproduce."
    assert fuzzy_similarity(PARAGRAPH, other) < 0.3


def test_fuzzy_small_edit_scores_high():
    edited = "The agent searched the knowledge base, found 3 matching documents, then summarized them into a short answer."
    assert fuzzy_similarity(PARAGRAPH, edited) > 0.8


def test_fuzzy_ignores_case_and_whitespace():
    assert fuzzy_similarity(PARAGRAPH, PARAGRAPH.upper()) == pytest.approx(1.0)
    assert fuzzy_similarity(PARAGRAPH, "  " + PARAGRAPH + "\n") == pytest.approx(1.0)


def test_fuzzy_empty_inputs():
    assert fuzzy_similarity("", "") == 1.0
    assert fuzzy_similarity("", "nonempty") == 0.0
    assert fuzzy_similarity("hi", "") == 0.0


def test_texts_match_fuzzy_mode():
    edited = PARAGRAPH.replace("three", "3")
    assert texts_match(PARAGRAPH, edited, threshold=0.85, mode="fuzzy")
    assert not texts_match(
        PARAGRAPH, "Something entirely unrelated happened.", threshold=0.85, mode="fuzzy"
    )


def test_texts_match_unknown_mode():
    with pytest.raises(ValueError, match="Unknown comparison mode"):
        texts_match("a", "a", mode="levenshtein")


def test_snapshot_fuzzy_mode_passes_on_wording_drift():
    snap = Snapshot(update=False, mode="fuzzy", threshold=0.85)
    snap.capture("fuzzy_drift", PARAGRAPH)
    result = snap.capture("fuzzy_drift", PARAGRAPH.replace("three", "3"))
    assert result.passed
    assert result.similarity is not None and result.similarity >= 0.85


def test_snapshot_fuzzy_mode_fails_on_real_change():
    snap = Snapshot(update=False, mode="fuzzy", threshold=0.85)
    snap.capture("fuzzy_change", PARAGRAPH)
    result = snap.capture(
        "fuzzy_change", "The database migration failed halfway and left the schema inconsistent."
    )
    assert not result.passed
