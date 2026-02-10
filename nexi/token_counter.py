"""Token counting utilities for NEXI.

This module provides pure functions for counting tokens in text and messages
using tiktoken. These functions are used for context window management.
"""

from __future__ import annotations

from typing import Any

import tiktoken

# Module-level cache for encodings
_encoding_cache: dict[str, tiktoken.Encoding] = {}

__all__ = [
    "count_tokens",
    "count_messages_tokens",
    "estimate_page_tokens",
    "get_encoding",
]


def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken encoding with caching.

    Args:
        encoding_name: Name of the encoding (default: cl100k_base)

    Returns:
        tiktoken.Encoding object

    Raises:
        ValueError: If encoding cannot be loaded
    """
    if encoding_name in _encoding_cache:
        return _encoding_cache[encoding_name]

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        _encoding_cache[encoding_name] = encoding
        return encoding
    except Exception as e:
        raise ValueError(f"Failed to load encoding '{encoding_name}': {e}") from e


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in a text string.

    Args:
        text: Text to count tokens for
        encoding: tiktoken encoding name (default: cl100k_base)

    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0

    enc = get_encoding(encoding)
    tokens = enc.encode(text)
    return len(tokens)


def count_messages_tokens(messages: list[dict[str, Any]], encoding: str = "cl100k_base") -> int:
    """Count total tokens in a list of chat messages.

    Args:
        messages: List of chat messages (OpenAI format)
        encoding: tiktoken encoding name (default: cl100k_base)

    Returns:
        Total token count for all messages
    """
    if not messages:
        return 0

    enc = get_encoding(encoding)
    total_tokens = 0

    for message in messages:
        # Count role
        role = message.get("role", "")
        total_tokens += len(enc.encode(role))

        # Count content if present
        content = message.get("content")
        if content is not None:
            total_tokens += len(enc.encode(str(content)))

        # Count tool_calls if present
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                # Count tool_call id
                tool_call_id = tool_call.get("id", "")
                total_tokens += len(enc.encode(tool_call_id))

                # Count function name and arguments
                function = tool_call.get("function", {})
                if function:
                    func_name = function.get("name", "")
                    func_args = function.get("arguments", "")
                    total_tokens += len(enc.encode(func_name))
                    total_tokens += len(enc.encode(func_args))

        # Count tool_call_id if present (for tool messages)
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            total_tokens += len(enc.encode(tool_call_id))

        # Add 3 tokens per message for formatting (OpenAI convention)
        total_tokens += 3

    return total_tokens


def estimate_page_tokens(content: str, encoding: str = "cl100k_base") -> int:
    """Estimate tokens for a page content (tool result).

    Args:
        content: Page content (typically JSON string)
        encoding: tiktoken encoding name (default: cl100k_base)

    Returns:
        Estimated token count
    """
    return count_tokens(content, encoding)
