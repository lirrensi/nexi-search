"""Conversation compaction utilities for NEXI.

This module provides functions for compacting conversation history when
approaching context limits, preserving search quality while preventing
token overflow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from nexi.config import Config, get_compaction_prompt
from nexi.token_counter import count_messages_tokens

__all__ = [
    "CompactionMetadata",
    "should_compact",
    "extract_metadata",
    "generate_summary",
    "rebuild_context",
    "compact_conversation",
]


@dataclass
class CompactionMetadata:
    """Metadata extracted from conversation for compaction."""

    search_queries: list[str]
    urls_fetched: list[str]
    original_query: str


def should_compact(
    current_tokens: int,
    estimated_next: int,
    config: Config,
) -> bool:
    """Check if compaction should be triggered.

    Args:
        current_tokens: Current token count in conversation
        estimated_next: Estimated tokens for next addition
        config: NEXI configuration

    Returns:
        True if compaction should be triggered
    """
    threshold = int(config.max_context * config.auto_compact_thresh)
    return (current_tokens + estimated_next) > threshold


def extract_metadata(messages: list[dict[str, Any]]) -> CompactionMetadata:
    """Extract metadata from conversation messages.

    Args:
        messages: List of chat messages

    Returns:
        CompactionMetadata with search queries and URLs
    """
    search_queries: list[str] = []
    urls_fetched: list[str] = []
    original_query = ""

    for message in messages:
        # Extract original query from first user message
        if message.get("role") == "user" and not original_query:
            original_query = message.get("content", "")

        # Extract from tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                arguments_str = function.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    continue

                if tool_name == "web_search":
                    queries = arguments.get("queries", [])
                    search_queries.extend(queries)
                elif tool_name == "web_get":
                    urls = arguments.get("urls", [])
                    urls_fetched.extend(urls)

    return CompactionMetadata(
        search_queries=search_queries,
        urls_fetched=urls_fetched,
        original_query=original_query,
    )


async def generate_summary(
    content: str,
    original_query: str,
    client: AsyncOpenAI,
    model: str,
    target_words: int = 5000,
    verbose: bool = False,
) -> str:
    """Generate summary of conversation content using LLM.

    Args:
        content: Content to summarize (web_fetch results + assistant messages)
        original_query: User's original search query
        client: OpenAI client
        model: Model name
        target_words: Target word count for summary
        verbose: Show detailed progress

    Returns:
        Generated summary text
    """
    if verbose:
        print("[Compaction] Generating summary...")

    prompt = get_compaction_prompt(
        original_query=original_query,
        content=content,
        target_words=target_words,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=target_words * 2,  # Rough estimate: 2 tokens per word
        )

        summary = response.choices[0].message.content or ""

        if verbose:
            print(f"[Compaction] Summary generated ({len(summary)} chars)")

        return summary
    except Exception as e:
        if verbose:
            print(f"[Compaction] ❌ Error generating summary: {e}")
        return ""


def rebuild_context(
    messages: list[dict[str, Any]],
    metadata: CompactionMetadata,
    summary: str,
    preserve_last_n: int = 3,
) -> list[dict[str, Any]]:
    """Rebuild conversation context with compaction.

    Args:
        messages: Original conversation messages
        metadata: Extracted metadata
        summary: Generated summary
        preserve_last_n: Number of recent assistant messages to preserve

    Returns:
        New compacted message list
    """
    new_messages: list[dict[str, Any]] = []

    # Preserve system prompt (index 0)
    if messages and messages[0].get("role") == "system":
        new_messages.append(messages[0])

    # Preserve original user message (index 1)
    if len(messages) > 1 and messages[1].get("role") == "user":
        new_messages.append(messages[1])

    # Build summary message
    summary_parts = [
        f"Original query: {metadata.original_query}",
        "",
        "Search queries performed:",
    ]
    for query in metadata.search_queries:
        summary_parts.append(f"- {query}")

    summary_parts.append("")
    summary_parts.append("Links navigated:")
    for url in metadata.urls_fetched:
        summary_parts.append(f"- {url}")

    summary_parts.append("")
    summary_parts.append("Findings:")
    summary_parts.append(summary)

    summary_message = {
        "role": "assistant",
        "content": "\n".join(summary_parts),
    }
    new_messages.append(summary_message)

    # Preserve last N assistant messages
    assistant_messages = [
        msg for msg in messages if msg.get("role") == "assistant" and msg.get("content")
    ]
    preserved_messages = assistant_messages[-preserve_last_n:] if preserve_last_n > 0 else []

    new_messages.extend(preserved_messages)

    return new_messages


async def compact_conversation(
    messages: list[dict[str, Any]],
    original_query: str,
    client: AsyncOpenAI,
    model: str,
    config: Config,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Compact conversation to reduce token count.

    Args:
        messages: Original conversation messages
        original_query: User's original search query
        client: OpenAI client
        model: Model name
        config: NEXI configuration
        verbose: Show detailed progress

    Returns:
        New compacted message list
    """
    if verbose:
        print("[Compaction] Starting conversation compaction...")

    # Extract metadata
    metadata = extract_metadata(messages)

    if verbose:
        print(f"[Compaction] Found {len(metadata.search_queries)} search queries")
        print(f"[Compaction] Found {len(metadata.urls_fetched)} URLs")

    # Collect content to summarize
    content_parts: list[str] = []

    for message in messages:
        # Include tool messages from web_get
        if message.get("role") == "tool":
            content_parts.append(message.get("content", ""))

        # Include assistant messages (except last N)
        if message.get("role") == "assistant" and message.get("content"):
            content_parts.append(message.get("content", ""))

    content = "\n\n".join(content_parts)

    if verbose:
        print(f"[Compaction] Content to summarize: {len(content)} chars")

    # Generate summary
    summary = await generate_summary(
        content=content,
        original_query=original_query,
        client=client,
        model=model,
        target_words=config.compact_target_words,
        verbose=verbose,
    )

    # If summary generation failed, return original messages
    if not summary:
        if verbose:
            print("[Compaction] ⚠️ Summary generation failed, keeping original messages")
        return messages

    # Rebuild context
    new_messages = rebuild_context(
        messages=messages,
        metadata=metadata,
        summary=summary,
        preserve_last_n=config.preserve_last_n_messages,
    )

    if verbose:
        old_tokens = count_messages_tokens(messages, config.tokenizer_encoding)
        new_tokens = count_messages_tokens(new_messages, config.tokenizer_encoding)
        reduction = old_tokens - new_tokens
        reduction_pct = (reduction / old_tokens * 100) if old_tokens > 0 else 0
        print(
            f"[Compaction] Reduced from {old_tokens} to {new_tokens} tokens ({reduction_pct:.1f}% reduction)"
        )

    return new_messages
