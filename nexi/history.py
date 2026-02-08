"""History management for NEXI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexi.config import CONFIG_DIR, generate_id

HISTORY_FILE = CONFIG_DIR / "history.jsonl"


@dataclass
class HistoryEntry:
    """A single search history entry."""

    id: str
    ts: str
    query: str
    answer: str
    urls: list[str]
    effort: str
    iterations: int
    duration_s: float
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        """Create entry from dictionary."""
        return cls(**data)


def get_history_path() -> Path:
    """Get path to history file."""
    return HISTORY_FILE


def add_history_entry(entry: HistoryEntry) -> None:
    """Append entry to history file.

    Args:
        entry: History entry to add
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        json_line = json.dumps(entry.to_dict(), ensure_ascii=False)
        f.write(json_line + "\n")


def get_last_n_entries(n: int) -> list[HistoryEntry]:
    """Get last N history entries (most recent first).

    Args:
        n: Number of entries to retrieve

    Returns:
        List of history entries, most recent first
    """
    if not HISTORY_FILE.exists():
        return []

    entries = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    entries.append(HistoryEntry.from_dict(data))
                except (json.JSONDecodeError, TypeError):
                    # Skip corrupted lines
                    continue

    # Return last N entries, most recent first
    return entries[-n:][::-1]


def get_entry_by_id(entry_id: str) -> HistoryEntry | None:
    """Get specific entry by ID.

    Args:
        entry_id: Entry ID to find

    Returns:
        History entry or None if not found
    """
    if not HISTORY_FILE.exists():
        return None

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if data.get("id") == entry_id:
                        return HistoryEntry.from_dict(data)
                except (json.JSONDecodeError, TypeError):
                    continue

    return None


def get_latest_entry() -> HistoryEntry | None:
    """Get most recent history entry.

    Returns:
        Most recent entry or None if history is empty
    """
    entries = get_last_n_entries(1)
    return entries[0] if entries else None


def clear_history() -> None:
    """Delete all history."""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


def create_entry(
    query: str,
    answer: str,
    urls: list[str],
    effort: str,
    iterations: int,
    duration_s: float,
    tokens: int = 0,
) -> HistoryEntry:
    """Create a new history entry.

    Args:
        query: User's query
        answer: Final answer
        urls: URLs fetched
        effort: Effort level used
        iterations: Number of iterations
        duration_s: Total duration in seconds
        tokens: Output tokens

    Returns:
        New history entry
    """
    return HistoryEntry(
        id=generate_id(),
        ts=datetime.now(timezone.utc).isoformat(),
        query=query,
        answer=answer,
        urls=urls,
        effort=effort,
        iterations=iterations,
        duration_s=duration_s,
        tokens=tokens,
    )


def format_time_ago(ts: str) -> str:
    """Format timestamp as relative time.

    Args:
        ts: ISO 8601 timestamp

    Returns:
        Human-readable relative time (e.g., "2m ago", "1h ago")
    """
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        seconds = int(diff.total_seconds())

        if seconds < 60:
            return f"{seconds}s ago"
        elif seconds < 3600:
            return f"{seconds // 60}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        else:
            return f"{seconds // 86400}d ago"
    except (ValueError, TypeError):
        return "unknown"


def format_entry_preview(entry: HistoryEntry, index: int = 0) -> str:
    """Format entry for --last display (truncated).

    Args:
        entry: History entry
        index: Entry index for display

    Returns:
        Formatted string for display
    """
    time_ago = format_time_ago(entry.ts)
    answer_preview = entry.answer[:200]
    if len(entry.answer) > 200:
        answer_preview += " [truncated]"

    lines = [
        f'[{index}] {time_ago} ({entry.effort}, {entry.iterations} iter, {entry.duration_s:.0f}s): "{entry.query}"',
        f"    Answer: {answer_preview}",
    ]

    if entry.urls:
        urls_str = ", ".join(entry.urls[:3])
        if len(entry.urls) > 3:
            urls_str += f" (+{len(entry.urls) - 3} more)"
        lines.append(f"    URLs: {urls_str}")

    return "\n".join(lines)


def format_entry_full(entry: HistoryEntry) -> str:
    """Format entry for --prev/--show display (full).

    Args:
        entry: History entry

    Returns:
        Formatted string for display
    """
    lines = [
        f"Query: {entry.query}",
        f"Effort: {entry.effort} ({entry.iterations} iterations, {entry.duration_s:.1f}s, {entry.tokens} tokens)",
        f"Timestamp: {entry.ts}",
        "",
    ]

    if entry.urls:
        lines.append("URLs fetched:")
        for url in entry.urls:
            lines.append(f"- {url}")
        lines.append("")

    lines.append("Answer:")
    lines.append(entry.answer)

    return "\n".join(lines)
