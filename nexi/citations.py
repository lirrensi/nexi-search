"""Citation post-processing for NEXI answers.

This module provides utilities for:
- Extracting citation markers from answers
- Mapping markers to URLs
- Auto-adding citations when model doesn't cite
- Formatting final citation sections
"""

from __future__ import annotations

import re
from typing import Any


def extract_citation_markers(text: str) -> set[int]:
    """Extract all citation marker numbers from text.

    Args:
        text: Answer text containing citations like [1], [2], etc.

    Returns:
        Set of citation numbers found in text
    """
    # Match [1], [2], etc. but not [1,2] or other patterns
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return {int(m) for m in matches}


def map_markers_to_urls(
    markers: set[int],
    number_to_url: dict[int, str],
) -> dict[int, str]:
    """Map citation markers to their URLs.

    Args:
        markers: Set of citation numbers found in answer
        number_to_url: Mapping of number -> URL

    Returns:
        Mapping of citation number -> URL for valid markers
    """
    return {num: number_to_url[num] for num in markers if num in number_to_url}


def format_citations_section(
    marker_url_map: dict[int, str],
    url_to_title: dict[str, str] | None = None,
    plain: bool = False,
) -> str:
    """Format the citations section for the answer.

    Args:
        marker_url_map: Mapping of citation number -> URL
        url_to_title: Mapping of citation number -> title (for formatting)
        plain: Use plain text format (no emojis)

    Returns:
        Formatted citations section
    """
    if not marker_url_map:
        return ""

    lines = []
    if plain:
        lines.append("Sources:")
    else:
        lines.append("📚 Sources:")

    for num in sorted(marker_url_map.keys()):
        url = marker_url_map[num]
        # Get title if available
        title = (
            (url_to_title.get(url) if url_to_title else "") if url in (url_to_title or {}) else ""
        )
        if title:
            lines.append(f"[{num}] {title} - {url}")
        else:
            lines.append(f"[{num}] {url}")

    return "\n" + "\n".join(lines)


def add_citations_to_answer(
    answer: str,
    number_to_url: dict[int, str],
    url_to_title: dict[str, str] | None = None,
    plain: bool = False,
) -> str:
    """Add citations section to answer if not already present.

    Args:
        answer: Original answer text
        number_to_url: Mapping of citation number -> URL
        url_to_title: Mapping of citation number -> title (for formatting)
        plain: Use plain text format

    Returns:
        Answer with citations section appended
    """
    # Check if answer already has a sources/citations section
    if re.search(r"(?i)^(sources|references|citations):", answer, re.MULTILINE):
        # Already has citations section, don't add another
        return answer

    # Extract existing citation markers from answer
    existing_markers = extract_citation_markers(answer)

    if not existing_markers:
        # No citations found - try to auto-detect referenced URLs
        # This is a simple heuristic: if URLs are mentioned in text, cite them
        referenced_urls = _detect_referenced_urls(answer, list(number_to_url.values()))

        # Map detected URLs to their citation numbers
        url_to_number = {url: num for num, url in number_to_url.items()}
        for url in referenced_urls:
            if url in url_to_number:
                existing_markers.add(url_to_number[url])

    if not existing_markers:
        # No citations to add
        return answer

    # Get URLs for the markers that exist
    marker_url_map = {num: number_to_url[num] for num in existing_markers if num in number_to_url}

    # Format and append citations section
    citations_section = format_citations_section(marker_url_map, url_to_title, plain)

    # Check if answer already ends with this exact citations section
    if answer.rstrip().endswith(citations_section.strip()):
        return answer

    return answer + citations_section


def _detect_referenced_urls(text: str, urls: list[str]) -> set[str]:
    """Detect which URLs are referenced in the text.

    This is a simple heuristic that checks if URL fragments appear in text.

    Args:
        text: Answer text
        urls: List of URLs to check

    Returns:
        Set of URLs that appear to be referenced
    """
    referenced = set()
    text_lower = text.lower()

    for url in urls:
        # Check if URL or its components appear in text
        url_lower = url.lower()

        # Try different matching strategies
        # 1. Full URL
        if url_lower in text_lower:
            referenced.add(url)
            continue

        # 2. Domain + path fragment
        if "/" in url:
            domain, path = url_lower.split("/", 1)
            if domain in text_lower and len(path) > 3 and path in text_lower:
                referenced.add(url)
                continue

        # 3. Just domain
        domain = url_lower.split("/")[0]
        if domain in text_lower:
            # Only add if it's not a common word
            if not _is_common_word(domain):
                referenced.add(url)

    return referenced


def _is_common_word(word: str) -> bool:
    """Check if a word is too common to be a meaningful citation.

    Args:
        word: Word to check

    Returns:
        True if word is too common
    """
    common = {
        "com",
        "org",
        "net",
        "http",
        "https",
        "www",
        "wikipedia",
        "github",
        "stackoverflow",
    }
    return word in common


def process_answer_with_citations(
    answer: str,
    url_citations: dict[str, int],
    url_to_title: dict[str, str] | None = None,
    include_citations_section: bool = True,
    plain: bool = False,
) -> str:
    """Process an answer with citation information.

    This is the main entry point for citation post-processing.

    Args:
        answer: Original answer text
        url_citations: URL -> citation number mapping
        url_to_title: URL -> title mapping (for formatting sources list)
        include_citations_section: Whether to append citations section
        plain: Use plain text format

    Returns:
        Processed answer with citations
    """
    # Build number -> URL mapping
    number_to_url = {num: url for url, num in url_citations.items()}

    if include_citations_section:
        return add_citations_to_answer(answer, number_to_url, url_to_title, plain)

    return answer


def extract_citations_from_tool_result(result: dict[str, Any]) -> dict[str, int]:
    """Extract URL citations from a web_get tool result.

    Args:
        result: Tool result from web_get

    Returns:
        URL -> citation number mapping
    """
    citations: dict[str, int] = {}

    pages = result.get("pages", [])
    for page in pages:
        url = page.get("url", "")
        content = page.get("content", "")

        # Extract citation number from content header like "[1] https://..."
        match = re.match(r"\[(\d+)\]\s+", content)
        if match:
            citations[url] = int(match.group(1))

    return citations


__all__ = [
    "extract_citation_markers",
    "map_markers_to_urls",
    "format_citations_section",
    "add_citations_to_answer",
    "process_answer_with_citations",
    "extract_citations_from_tool_result",
]
