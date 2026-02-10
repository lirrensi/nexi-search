"""Unit tests for compaction module."""

from __future__ import annotations

from nexi.compaction import (
    CompactionMetadata,
    extract_metadata,
    rebuild_context,
    should_compact,
)
from nexi.config import Config


def test_should_compact_true() -> None:
    """Test compaction trigger when threshold exceeded."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        time_target=600,
        max_output_tokens=8192,
        max_context=100000,
        auto_compact_thresh=0.9,
    )
    # 90000 + 10000 = 100000 > 90000 (threshold)
    assert should_compact(90000, 10000, config)


def test_should_compact_false() -> None:
    """Test no compaction when threshold not exceeded."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        time_target=600,
        max_output_tokens=8192,
        max_context=100000,
        auto_compact_thresh=0.9,
    )
    # 80000 + 5000 = 85000 < 90000 (threshold)
    assert not should_compact(80000, 5000, config)


def test_should_compact_at_threshold() -> None:
    """Test compaction at exact threshold."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        time_target=600,
        max_output_tokens=8192,
        max_context=100000,
        auto_compact_thresh=0.9,
    )
    # 90000 + 1 = 90001 > 90000 (threshold)
    assert should_compact(90000, 1, config)


def test_should_compact_disabled() -> None:
    """Test compaction disabled when threshold is 1.0."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        time_target=600,
        max_output_tokens=8192,
        max_context=100000,
        auto_compact_thresh=1.0,
    )
    # 90000 + 5000 = 95000 < 100000 (threshold)
    assert not should_compact(90000, 5000, config)


def test_extract_metadata_simple() -> None:
    """Test extracting metadata from simple conversation."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"queries": ["test1", "test2"]}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "web_get",
                        "arguments": '{"urls": ["https://example.com"]}',
                    },
                }
            ],
        },
    ]
    metadata = extract_metadata(messages)
    assert metadata.original_query == "test query"
    assert "test1" in metadata.search_queries
    assert "test2" in metadata.search_queries
    assert "https://example.com" in metadata.urls_fetched


def test_extract_metadata_multiple_searches() -> None:
    """Test extracting metadata from multiple web_search calls."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"queries": ["q1"]}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"queries": ["q2", "q3"]}',
                    },
                }
            ],
        },
    ]
    metadata = extract_metadata(messages)
    assert len(metadata.search_queries) == 3
    assert "q1" in metadata.search_queries
    assert "q2" in metadata.search_queries
    assert "q3" in metadata.search_queries


def test_extract_metadata_multiple_gets() -> None:
    """Test extracting metadata from multiple web_get calls."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_get",
                        "arguments": '{"urls": ["https://example1.com"]}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "web_get",
                        "arguments": '{"urls": ["https://example2.com", "https://example3.com"]}',
                    },
                }
            ],
        },
    ]
    metadata = extract_metadata(messages)
    assert len(metadata.urls_fetched) == 3
    assert "https://example1.com" in metadata.urls_fetched
    assert "https://example2.com" in metadata.urls_fetched
    assert "https://example3.com" in metadata.urls_fetched


def test_extract_metadata_empty() -> None:
    """Test extracting metadata from empty conversation."""
    messages = []
    metadata = extract_metadata(messages)
    assert metadata.original_query == ""
    assert len(metadata.search_queries) == 0
    assert len(metadata.urls_fetched) == 0


def test_extract_metadata_no_tool_calls() -> None:
    """Test extracting metadata from conversation without tool calls."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "Hello!"},
    ]
    metadata = extract_metadata(messages)
    assert metadata.original_query == "test query"
    assert len(metadata.search_queries) == 0
    assert len(metadata.urls_fetched) == 0


def test_extract_metadata_invalid_json() -> None:
    """Test extracting metadata with invalid JSON in arguments."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "invalid json",
                    },
                }
            ],
        },
    ]
    metadata = extract_metadata(messages)
    assert metadata.original_query == "test query"
    assert len(metadata.search_queries) == 0


def test_rebuild_context() -> None:
    """Test rebuilding context with summary."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "old message 1"},
        {"role": "assistant", "content": "old message 2"},
        {"role": "assistant", "content": "recent message"},
    ]
    metadata = CompactionMetadata(
        search_queries=["test"],
        urls_fetched=["https://example.com"],
        original_query="test query",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=1)

    # Should have: system, user, summary, recent message
    assert len(new_messages) == 4
    assert new_messages[0]["role"] == "system"
    assert new_messages[1]["role"] == "user"
    assert new_messages[2]["role"] == "assistant"
    assert "Test summary" in new_messages[2]["content"]
    assert new_messages[3]["content"] == "recent message"


def test_rebuild_context_preserve_multiple() -> None:
    """Test rebuilding context preserving multiple messages."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "old message"},
        {"role": "assistant", "content": "recent 1"},
        {"role": "assistant", "content": "recent 2"},
        {"role": "assistant", "content": "recent 3"},
    ]
    metadata = CompactionMetadata(
        search_queries=["test"],
        urls_fetched=["https://example.com"],
        original_query="test query",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=2)

    # Should have: system, user, summary, recent 2, recent 3
    assert len(new_messages) == 5
    assert new_messages[3]["content"] == "recent 2"
    assert new_messages[4]["content"] == "recent 3"


def test_rebuild_context_preserve_zero() -> None:
    """Test rebuilding context preserving zero messages."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "old message"},
        {"role": "assistant", "content": "recent message"},
    ]
    metadata = CompactionMetadata(
        search_queries=["test"],
        urls_fetched=["https://example.com"],
        original_query="test query",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=0)

    # Should have: system, user, summary only
    assert len(new_messages) == 3
    assert new_messages[0]["role"] == "system"
    assert new_messages[1]["role"] == "user"
    assert new_messages[2]["role"] == "assistant"


def test_rebuild_context_with_metadata() -> None:
    """Test rebuilding context includes metadata in summary."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "old message"},
    ]
    metadata = CompactionMetadata(
        search_queries=["query1", "query2"],
        urls_fetched=["https://example1.com", "https://example2.com"],
        original_query="test query",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=0)

    summary_content = new_messages[2]["content"]
    assert "Original query: test query" in summary_content
    assert "query1" in summary_content
    assert "query2" in summary_content
    assert "https://example1.com" in summary_content
    assert "https://example2.com" in summary_content
    assert "Test summary" in summary_content


def test_compaction_metadata() -> None:
    """Test CompactionMetadata dataclass."""
    metadata = CompactionMetadata(
        search_queries=["q1", "q2"],
        urls_fetched=["url1", "url2"],
        original_query="test",
    )
    assert metadata.search_queries == ["q1", "q2"]
    assert metadata.urls_fetched == ["url1", "url2"]
    assert metadata.original_query == "test"


def test_rebuild_context_empty_messages() -> None:
    """Test rebuilding context with empty messages."""
    messages = []
    metadata = CompactionMetadata(
        search_queries=[],
        urls_fetched=[],
        original_query="",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=0)

    # Should only have summary message
    assert len(new_messages) == 1
    assert new_messages[0]["role"] == "assistant"
    assert "Test summary" in new_messages[0]["content"]


def test_rebuild_context_no_system() -> None:
    """Test rebuilding context without system message."""
    messages = [
        {"role": "user", "content": "test query"},
        {"role": "assistant", "content": "old message"},
    ]
    metadata = CompactionMetadata(
        search_queries=["test"],
        urls_fetched=["https://example.com"],
        original_query="test query",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=0)

    # Should have: summary only (user message not preserved when no system message)
    assert len(new_messages) == 1
    assert new_messages[0]["role"] == "assistant"


def test_rebuild_context_no_user() -> None:
    """Test rebuilding context without user message."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "old message"},
    ]
    metadata = CompactionMetadata(
        search_queries=["test"],
        urls_fetched=["https://example.com"],
        original_query="",
    )
    summary = "Test summary"

    new_messages = rebuild_context(messages, metadata, summary, preserve_last_n=0)

    # Should have: system, summary
    assert len(new_messages) == 2
    assert new_messages[0]["role"] == "system"
    assert new_messages[1]["role"] == "assistant"
