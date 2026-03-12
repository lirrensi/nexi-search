"""Unit tests for history pathing and JSONL storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from nexi import history as history_module
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


def _patch_history_dir(monkeypatch, tmp_path: Path) -> Path:
    """Point history storage at a temporary config directory."""
    monkeypatch.setattr(history_module, "CONFIG_DIR", tmp_path)
    return tmp_path / "history.jsonl"


def test_history_entry_dataclass() -> None:
    """HistoryEntry stores the expected fields."""
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


def test_history_entry_to_dict() -> None:
    """HistoryEntry serializes cleanly to a dictionary."""
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


def test_history_entry_from_dict() -> None:
    """HistoryEntry deserializes from a dictionary."""
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


def test_get_history_path_uses_config_dir(monkeypatch, tmp_path: Path) -> None:
    """History follows the shared ~/.config/nexi layout."""
    history_path = _patch_history_dir(monkeypatch, tmp_path)

    assert get_history_path() == history_path
    assert str(get_history_path()).endswith("history.jsonl")


def test_add_history_entry(tmp_path: Path, monkeypatch) -> None:
    """add_history_entry appends JSONL data under the config directory."""
    history_file = _patch_history_dir(monkeypatch, tmp_path)
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

    assert history_file.exists()
    content = history_file.read_text(encoding="utf-8")
    assert "abc123" in content
    assert "test query" in content


def test_get_last_n_entries(tmp_path: Path, monkeypatch) -> None:
    """get_last_n_entries returns most recent entries first."""
    history_file = _patch_history_dir(monkeypatch, tmp_path)
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
    with open(history_file, "w", encoding="utf-8") as file_obj:
        for entry in entries:
            file_obj.write(json.dumps(entry) + "\n")

    result = get_last_n_entries(2)

    assert len(result) == 2
    assert result[0].id == "3"
    assert result[1].id == "2"


def test_get_last_n_entries_empty(tmp_path: Path, monkeypatch) -> None:
    """Missing history files return an empty list."""
    _patch_history_dir(monkeypatch, tmp_path)

    assert get_last_n_entries(5) == []


def test_get_entry_by_id(tmp_path: Path, monkeypatch) -> None:
    """Specific entries can be looked up by ID."""
    history_file = _patch_history_dir(monkeypatch, tmp_path)
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
    with open(history_file, "w", encoding="utf-8") as file_obj:
        for entry in entries:
            file_obj.write(json.dumps(entry) + "\n")

    result = get_entry_by_id("def")

    assert result is not None
    assert result.id == "def"
    assert result.query == "q2"


def test_get_entry_by_id_not_found(tmp_path: Path, monkeypatch) -> None:
    """Unknown IDs return None."""
    _patch_history_dir(monkeypatch, tmp_path)
    assert get_entry_by_id("nonexistent") is None


def test_get_latest_entry(tmp_path: Path, monkeypatch) -> None:
    """The latest entry returns the most recently stored result."""
    history_file = _patch_history_dir(monkeypatch, tmp_path)
    with open(history_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(
            '{"id": "1", "ts": "2024-01-01T00:00:00+00:00", "query": "q1", "answer": "a1", "urls": [], "effort": "m", "iterations": 1, "duration_s": 1.0, "tokens": 100}\n'
        )
        file_obj.write(
            '{"id": "2", "ts": "2024-01-02T00:00:00+00:00", "query": "q2", "answer": "a2", "urls": [], "effort": "m", "iterations": 1, "duration_s": 1.0, "tokens": 100}\n'
        )

    result = get_latest_entry()

    assert result is not None
    assert result.id == "2"


def test_get_latest_entry_empty(tmp_path: Path, monkeypatch) -> None:
    """An empty history returns None for the latest entry."""
    _patch_history_dir(monkeypatch, tmp_path)
    assert get_latest_entry() is None


def test_clear_history(tmp_path: Path, monkeypatch) -> None:
    """clear_history removes the JSONL file when present."""
    history_file = _patch_history_dir(monkeypatch, tmp_path)
    history_file.write_text('{"test": "data"}\n', encoding="utf-8")

    clear_history()

    assert not history_file.exists()


def test_create_entry() -> None:
    """create_entry fills generated metadata fields."""
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
    assert len(entry.id) == 6
    assert entry.ts is not None


def test_format_time_ago() -> None:
    """format_time_ago returns a relative timestamp string."""
    result = format_time_ago(datetime.now(UTC).isoformat())
    assert any(unit in result for unit in ("s ago", "m ago", "h ago"))


def test_format_time_ago_old() -> None:
    """Older timestamps are represented in days."""
    assert "d ago" in format_time_ago("2020-01-01T00:00:00+00:00")


def test_format_time_ago_invalid() -> None:
    """Invalid timestamps fall back to an unknown label."""
    assert format_time_ago("invalid") == "unknown"


def test_format_entry_preview() -> None:
    """Preview formatting truncates long answers."""
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


def test_format_entry_full() -> None:
    """Full formatting includes query, answer, and URLs."""
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
