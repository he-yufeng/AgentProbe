"""Tests for pattern-based secret scrubbing in snapshots."""

import shutil

import pytest

from agentprobe.snapshot import Snapshot
from agentprobe.storage import DEFAULT_DIR


@pytest.fixture(autouse=True)
def clean_snapshots():
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR)
    yield
    if DEFAULT_DIR.exists():
        shutil.rmtree(DEFAULT_DIR)


def test_api_keys_masked_when_redact_secrets_on():
    snap = Snapshot(redact_secrets=True)
    result = snap.capture("sk", {"reply": "use key sk-abcdefghijklmnop1234 for this"})
    assert "sk-abcdefghijklmnop1234" not in str(result.output)
    assert "<redacted>" in result.output["reply"]


def test_github_pat_and_jwt_masked():
    snap = Snapshot(redact_secrets=True)
    result = snap.capture(
        "tokens",
        {"gh": "ghp_abcdefghijklmnopqrstuvwxyz", "jwt": "eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM"},
    )
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in str(result.output)
    assert "eyJhbGciOi.eyJzdWIi.SflKxwRJSM" not in str(result.output)


def test_email_masked_but_normal_prose_untouched():
    snap = Snapshot(redact_secrets=True)
    result = snap.capture(
        "prose",
        {"note": "email me at dev@example.com about the token rotation", "plain": "sk short token"},
    )
    assert "dev@example.com" not in str(result.output)
    # "sk" alone is not a key, and the word "token" is not a secret
    assert "sk short token" in result.output["plain"]


def test_secrets_survive_when_redaction_off():
    snap = Snapshot()
    result = snap.capture("off", {"reply": "key sk-abcdefghijklmnop1234"})
    assert "sk-abcdefghijklmnop1234" in result.output["reply"]


def test_custom_redact_patterns():
    snap = Snapshot(redact_patterns=(r"internal-\d{4}",))
    result = snap.capture("custom", {"host": "dial internal-4512 for staging"})
    assert "internal-4512" not in str(result.output)
    assert "<redacted>" in result.output["host"]


def test_redaction_makes_comparison_secret_agnostic():
    snap = Snapshot()
    first = snap.capture("agnostic", {"reply": "call with sk-aaaaaaaaaaaaaaaa1111"})
    assert first.passed
    snap2 = Snapshot(redact_secrets=True)
    # a different key in the same text would normally mismatch; with scrubbing
    # both sides are "<redacted>", so it still matches
    second = snap2.capture("agnostic", {"reply": "call with sk-bbbbbbbbbbbb2222"})
    assert not second.passed  # baseline stored the unredacted key
    snap3 = Snapshot(update=True, redact_secrets=True)
    snap3.capture("agnostic", {"reply": "call with sk-bbbbbbbbbbbb2222"})
    snap4 = Snapshot(redact_secrets=True)
    assert snap4.capture("agnostic", {"reply": "call with sk-cccccccccccc3333"}).passed


def test_decorator_path_scrubs_secrets():
    from agentprobe.snapshot import snapshot

    @snapshot("deco_secret", redact_secrets=True)
    def run():
        return {"reply": "key sk-abcdefghijklmnop1234"}

    run()  # creates the baseline
    from agentprobe.storage import load_snapshot

    stored = load_snapshot("deco_secret")
    assert "sk-abcdefghijklmnop1234" not in str(stored)


def test_decorator_path_custom_patterns():
    from agentprobe.snapshot import snapshot

    @snapshot("deco_pattern", redact_patterns=[r"internal-\d{4}"])
    def run():
        return {"host": "dial internal-4512"}

    run()
    from agentprobe.storage import load_snapshot

    stored = load_snapshot("deco_pattern")
    assert "internal-4512" not in str(stored)


def test_decorator_path_async_scrubs_secrets():
    import asyncio

    from agentprobe.snapshot import snapshot

    @snapshot("deco_async_secret", redact_secrets=True)
    async def run():
        return {"reply": "key sk-abcdefghijklmnop1234"}

    asyncio.run(run())
    from agentprobe.storage import load_snapshot

    stored = load_snapshot("deco_async_secret")
    assert "sk-abcdefghijklmnop1234" not in str(stored)
