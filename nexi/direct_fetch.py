"""Direct fetch output helpers for NEXI."""

# FILE: nexi/direct_fetch.py
# PURPOSE: Cap direct-fetch output and spill oversized pages to temp files.
# OWNS: Token-aware truncation and post-processing for direct fetch surfaces.
# EXPORTS: truncate_to_token_cap, spill_text_to_temp_file, post_process_direct_fetch_payload
# DOCS: agent_chat/plan_fetch_resilience_2026-04-24.md

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from nexi.token_counter import get_encoding

DEFAULT_DIRECT_FETCH_MAX_TOKENS = 8000


def truncate_to_token_cap(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
) -> tuple[str, bool]:
    """Truncate text to a token budget.

    Args:
        text: Text to cap.
        max_tokens: Maximum number of tokens allowed.
        encoding_name: tiktoken encoding name.

    Returns:
        Tuple of (possibly truncated text, whether truncation occurred).
    """
    if max_tokens <= 0 or not text:
        return "", bool(text)

    encoding = get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, False

    return encoding.decode(tokens[:max_tokens]), True


def spill_text_to_temp_file(text: str) -> str:
    """Write text to a temp file and return its absolute path."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        suffix=".txt",
        prefix="nexi-fetch-",
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)

    return str(temp_path.resolve())


def post_process_direct_fetch_payload(
    payload: dict[str, Any],
    max_tokens: int = DEFAULT_DIRECT_FETCH_MAX_TOKENS,
    encoding_name: str = "cl100k_base",
) -> dict[str, Any]:
    """Apply direct-fetch truncation and spillover to a payload.

    Args:
        payload: Structured fetch payload.
        max_tokens: Maximum emitted tokens per page.
        encoding_name: tiktoken encoding name.

    Returns:
        New payload with capped page content and spillover paths.
    """
    processed_payload = dict(payload)
    processed_pages: list[Any] = []

    for page in payload.get("pages", []):
        if not isinstance(page, dict):
            processed_pages.append(page)
            continue

        processed_page = dict(page)
        content = processed_page.get("content")
        if isinstance(content, str) and content:
            truncated_content, was_truncated = truncate_to_token_cap(
                content,
                max_tokens,
                encoding_name,
            )
            if was_truncated:
                processed_page["content"] = truncated_content
                processed_page["full_content_path"] = spill_text_to_temp_file(content)

        processed_pages.append(processed_page)

    processed_payload["pages"] = processed_pages
    return processed_payload


__all__ = [
    "DEFAULT_DIRECT_FETCH_MAX_TOKENS",
    "post_process_direct_fetch_payload",
    "spill_text_to_temp_file",
    "truncate_to_token_cap",
]
