"""Unit tests for history module."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from nexi.history import (
    HistoryEntry,
    add_history_entry,
    clear_history,
    create_entry,
    format_entry_full,
    format_entry_preview,
    format_time_ago,
    get_entry_by_id,
    get_history_path,
    get_last_n_entries,
    get_latest_entry,
)


def test_history_entry_dataclass():
    """Test HistoryEntry dataclass."""
    entry = HistoryEntry(
        id="abc123",
        ts="2024-01-01T00:00:00+00:00",
        query="test query",
        answer="test answer",
        urls=["http://example.com"],
        effort="m",
        iterations=5,
        duration_s=10.5,
        tokens=1000,
    )

    assert entry.id == "abc123"
    assert entry.query == "test query"
    assert entry.iterations == 5


def test_history_entry_to_dict():
    """Test HistoryEntry.to_dict()."""
    entry = HistoryEntry(
        id="abc123",
        ts="2024-01-01T00:00:00+00:00",
        query="test query",
        answer="test answer",
        urls=["http://example.com"],
        effort="m",
        iterations=5,
        duration_s=10.5,
        tokens=1000,
    )

    data = entry.to_dict()
    assert data["id"] == "abc123"
    assert data["query"] == "test query"
    assert data["urls"] == ["http://example.com"]


def test_history_entry_from_dict():
    """Test HistoryEntry.from_dict()."""
    data = {
        "id": "abc123",
        "ts": "2024-01-01T00:00:00+00:00",
        "query": "test query",
        "answer": "test answer",
        "urls": ["http://example.com"],
        "effort": "m",
        "iterations": 5,
        "duration_s": 10.5,
        "tokens": 1000,
    }

    entry = HistoryEntry.from_dict(data)
    assert entry.id == "abc123"
    assert entry.query == "test query"


def test_get_history_path():
    """Test get_history_path returns Path."""
    path = get_history_path()
    assert isinstance(path, Path)
    assert path.name == "history.jsonl"


def test_add_history_entry(tmp_path, monkeypatch):
    """Test add_history_entry appends to file."""
    # Mock history file location
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    entry = HistoryEntry(
        id="abc123",
        ts="2024-01-01T00:00:00+00:00",
        query="test query",
        answer="test answer",
        urls=["http://example.com"],
        effort="m",
        iterations=5,
        duration_s=10.5,
        tokens=1000,
    )

    add_history_entry(entry)

    # Verify file was created and contains entry
    assert test_file.exists()
    content = test_file.read_text()
    assert "abc123" in content
    assert "test query" in content


def test_get_last_n_entries(tmp_path, monkeypatch):
    """Test get_last_n_entries returns entries in correct order."""
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    # Create test entries
    entries = [
        {
            "id": "1",
            "ts": "2024-01-01T00:00:00+00:00",
            "query": "q1",
            "answer": "a1",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
        {
            "id": "2",
            "ts": "2024-01-02T00:00:00+00:00",
            "query": "q2",
            "answer": "a2",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
        {
            "id": "3",
            "ts": "2024-01-03T00:00:00+00:00",
            "query": "q3",
            "answer": "a3",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
    ]

    # Write entries to file
    with open(test_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Get last 2 entries
    result = get_last_n_entries(2)

    assert len(result) == 2
    # Should be most recent first (reversed)
    assert result[0].id == "3"
    assert result[1].id == "2"


def test_get_last_n_entries_empty(tmp_path, monkeypatch):
    """Test get_last_n_entries with no history file."""
    test_file = tmp_path / "nonexistent.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    result = get_last_n_entries(5)
    assert result == []


def test_get_entry_by_id(tmp_path, monkeypatch):
    """Test get_entry_by_id finds specific entry."""
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    entries = [
        {
            "id": "abc",
            "ts": "2024-01-01T00:00:00+00:00",
            "query": "q1",
            "answer": "a1",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
        {
            "id": "def",
            "ts": "2024-01-02T00:00:00+00:00",
            "query": "q2",
            "answer": "a2",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
    ]

    with open(test_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_entry_by_id("def")
    assert result is not None
    assert result.id == "def"
    assert result.query == "q2"


def test_get_entry_by_id_not_found(tmp_path, monkeypatch):
    """Test get_entry_by_id returns None for missing ID."""
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    result = get_entry_by_id("nonexistent")
    assert result is None


def test_get_latest_entry(tmp_path, monkeypatch):
    """Test get_latest_entry returns most recent."""
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    entries = [
        {
            "id": "1",
            "ts": "2024-01-01T00:00:00+00:00",
            "query": "q1",
            "answer": "a1",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
        {
            "id": "2",
            "ts": "2024-01-02T00:00:00+00:00",
            "query": "q2",
            "answer": "a2",
            "urls": [],
            "effort": "m",
            "iterations": 1,
            "duration_s": 1.0,
            "tokens": 100,
        },
    ]

    with open(test_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_latest_entry()
    assert result is not None
    assert result.id == "2"


def test_get_latest_entry_empty(tmp_path, monkeypatch):
    """Test get_latest_entry returns None for empty history."""
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    result = get_latest_entry()
    assert result is None


def test_clear_history(tmp_path, monkeypatch):
    """Test clear_history deletes history file."""
    test_file = tmp_path / "history.jsonl"
    monkeypatch.setattr("nexi.history.HISTORY_FILE", test_file)

    # Create file
    test_file.write_text('{"test": "data"}\n')
    assert test_file.exists()

    clear_history()

    assert not test_file.exists()


def test_create_entry():
    """Test create_entry creates entry with generated ID and timestamp."""
    entry = create_entry(
        query="test query",
        answer="test answer",
        urls=["http://example.com"],
        effort="m",
        iterations=5,
        duration_s=10.5,
        tokens=1000,
    )

    assert entry.query == "test query"
    assert entry.answer == "test answer"
    assert entry.effort == "m"
    assert entry.iterations == 5
    assert len(entry.id) == 6  # Generated hex ID
    assert entry.ts is not None  # ISO timestamp


def test_format_time_ago():
    """Test format_time_ago formats relative time."""
    # Recent timestamp
    now = datetime.now(UTC)
    ts = now.isoformat()
    result = format_time_ago(ts)
    assert "s ago" in result or "m ago" in result or "h ago" in result


def test_format_time_ago_old():
    """Test format_time_ago with old timestamp."""
    old_ts = "2020-01-01T00:00:00+00:00"
    result = format_time_ago(old_ts)
    assert "d ago" in result


def test_format_time_ago_invalid():
    """Test format_time_ago with invalid timestamp."""
    result = format_time_ago("invalid")
    assert result == "unknown"


def test_format_entry_preview():
    """Test format_entry_preview creates truncated preview."""
    entry = HistoryEntry(
        id="abc123",
        ts="2024-01-01T00:00:00+00:00",
        query="test query",
        answer="This is a long answer that should be truncated in the preview. " * 10,
        urls=["http://example.com", "http://test.com"],
        effort="m",
        iterations=5,
        duration_s=10.5,
        tokens=1000,
    )

    result = format_entry_preview(entry, index=1)

    assert "[1]" in result
    assert "test query" in result
    assert "[truncated]" in result
    assert "http://example.com" in result


def test_format_entry_full():
    """Test format_entry_full creates full output."""
    entry = HistoryEntry(
        id="abc123",
        ts="2024-01-01T00:00:00+00:00",
        query="test query",
        answer="Full answer text",
        urls=["http://example.com"],
        effort="m",
        iterations=5,
        duration_s=10.5,
        tokens=1000,
    )

    result = format_entry_full(entry)

    assert "Query: test query" in result
    assert "Full answer text" in result
    assert "http://example.com" in result
    assert "5 iterations" in result
