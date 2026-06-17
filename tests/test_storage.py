"""Tests for snapshot storage, including parallel-safe atomic writes."""

import threading

from agentprobe.storage import load_snapshot, save_snapshot


def test_save_and_load_roundtrip(tmp_path):
    save_snapshot("roundtrip", {"output": "hello", "timestamp": 1.0}, directory=tmp_path)
    loaded = load_snapshot("roundtrip", directory=tmp_path)
    assert loaded == {"output": "hello", "timestamp": 1.0}


def test_load_missing_returns_none(tmp_path):
    assert load_snapshot("never-written", directory=tmp_path) is None


def test_atomic_write_leaves_no_temp_files(tmp_path):
    save_snapshot("clean", {"output": "x" * 10_000}, directory=tmp_path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    # The real snapshot is present and complete.
    loaded = load_snapshot("clean", directory=tmp_path)
    assert loaded is not None and loaded["output"] == "x" * 10_000


def test_overwrite_replaces_atomically(tmp_path):
    save_snapshot("ow", {"output": "v1"}, directory=tmp_path)
    save_snapshot("ow", {"output": "v2"}, directory=tmp_path)
    loaded = load_snapshot("ow", directory=tmp_path)
    assert loaded is not None and loaded["output"] == "v2"
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_concurrent_writes_never_yield_partial_reads(tmp_path):
    # Simulate pytest-xdist workers hammering the same snapshot: readers must
    # never observe a truncated file (which a non-atomic write would expose).
    big = {"output": "y" * 50_000, "timestamp": 0.0}
    save_snapshot("race", big, directory=tmp_path)
    errors: list[Exception] = []

    def writer():
        for _ in range(40):
            try:
                save_snapshot("race", big, directory=tmp_path)
            except Exception as exc:  # noqa: BLE001 - record for the assertion
                errors.append(exc)

    def reader():
        for _ in range(40):
            try:
                loaded = load_snapshot("race", directory=tmp_path)
                assert loaded is None or loaded["output"] == "y" * 50_000
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_concurrent_writes_to_distinct_snapshots(tmp_path):
    # The real pytest-xdist shape: workers each own different snapshots.
    errors: list[Exception] = []

    def worker(i: int):
        try:
            save_snapshot(f"snap-{i}", {"output": f"value-{i}"}, directory=tmp_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    for i in range(16):
        loaded = load_snapshot(f"snap-{i}", directory=tmp_path)
        assert loaded is not None and loaded["output"] == f"value-{i}"
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
