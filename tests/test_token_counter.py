"""Unit tests for token_counter module."""

from __future__ import annotations

import pytest

from nexi.token_counter import (
    count_messages_tokens,
    count_tokens,
    estimate_page_tokens,
    get_encoding,
)


def test_count_tokens_empty() -> None:
    """Test counting tokens in empty string."""
    assert count_tokens("") == 0


def test_count_tokens_simple() -> None:
    """Test counting tokens in simple text."""
    text = "Hello, world!"
    tokens = count_tokens(text)
    assert tokens > 0
    assert tokens < len(text)  # Tokens < chars


def test_count_tokens_long_text() -> None:
    """Test counting tokens in longer text."""
    text = "This is a longer text. " * 100
    tokens = count_tokens(text)
    assert tokens > 0


def test_count_messages_empty() -> None:
    """Test counting tokens in empty message list."""
    assert count_messages_tokens([]) == 0


def test_count_messages_simple() -> None:
    """Test counting tokens in simple messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    tokens = count_messages_tokens(messages)
    assert tokens > 0


def test_count_messages_with_tool_calls() -> None:
    """Test counting tokens in messages with tool calls."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"queries": ["test"]}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '{"results": []}'},
    ]
    tokens = count_messages_tokens(messages)
    assert tokens > 0


def test_count_messages_with_multiple_tool_calls() -> None:
    """Test counting tokens with multiple tool calls."""
    messages = [
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
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "web_get",
                        "arguments": '{"urls": ["https://example.com"]}',
                    },
                },
            ],
        }
    ]
    tokens = count_messages_tokens(messages)
    assert tokens > 0


def test_estimate_page_tokens() -> None:
    """Test estimating tokens for page content."""
    content = '{"pages": [{"url": "https://example.com", "content": "Test content"}]}'
    tokens = estimate_page_tokens(content)
    assert tokens > 0


def test_estimate_page_tokens_empty() -> None:
    """Test estimating tokens for empty content."""
    assert estimate_page_tokens("") == 0


def test_get_encoding_default() -> None:
    """Test getting default encoding."""
    enc = get_encoding()
    assert enc is not None
    assert hasattr(enc, "encode")


def test_get_encoding_cached() -> None:
    """Test that encoding is cached."""
    enc1 = get_encoding("cl100k_base")
    enc2 = get_encoding("cl100k_base")
    assert enc1 is enc2  # Same object reference


def test_get_encoding_invalid() -> None:
    """Test getting invalid encoding raises error."""
    with pytest.raises(ValueError, match="Failed to load encoding"):
        get_encoding("invalid_encoding_name")


def test_count_tokens_with_custom_encoding() -> None:
    """Test counting tokens with custom encoding."""
    text = "Hello, world!"
    tokens = count_tokens(text, "cl100k_base")
    assert tokens > 0


def test_count_messages_with_none_content() -> None:
    """Test counting messages with None content."""
    messages = [
        {"role": "assistant", "content": None},
    ]
    tokens = count_messages_tokens(messages)
    assert tokens > 0  # Should count role + formatting tokens


def test_count_messages_with_empty_tool_calls() -> None:
    """Test counting messages with empty tool_calls list."""
    messages = [
        {"role": "assistant", "content": "test", "tool_calls": []},
    ]
    tokens = count_messages_tokens(messages)
    assert tokens > 0
