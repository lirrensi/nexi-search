"""Tests for citation post-processing."""

from __future__ import annotations

import pytest

from nexi.citations import (
    extract_citation_markers,
    map_markers_to_urls,
    format_citations_section,
    add_citations_to_answer,
    process_answer_with_citations,
    extract_citations_from_tool_result,
)


class TestExtractCitationMarkers:
    """Tests for extract_citation_markers function."""

    def test_extract_single_marker(self):
        """Test extracting a single citation marker."""
        text = "According to [1], the answer is yes."
        markers = extract_citation_markers(text)
        assert markers == {1}

    def test_extract_multiple_markers(self):
        """Test extracting multiple citation markers."""
        text = "According to [1], the answer is [2]. Also see [3] for more info."
        markers = extract_citation_markers(text)
        assert markers == {1, 2, 3}

    def test_extract_duplicate_markers(self):
        """Test that duplicate markers are deduplicated."""
        text = "According to [1], the answer is [1]. Also see [2] and [1]."
        markers = extract_citation_markers(text)
        assert markers == {1, 2}

    def test_no_markers(self):
        """Test when no markers are present."""
        text = "According to the source, the answer is yes."
        markers = extract_citation_markers(text)
        assert markers == set()

    def test_ignore_non_citation_brackets(self):
        """Test that non-citation brackets are ignored."""
        text = "The answer is [1], but not [1,2] or [see note 1]."
        markers = extract_citation_markers(text)
        assert markers == {1}


class TestMapMarkersToUrls:
    """Tests for map_markers_to_urls function."""

    def test_map_valid_markers(self):
        """Test mapping valid markers to URLs."""
        markers = {1, 2, 3}
        number_to_url = {
            1: "https://example.com/1",
            2: "https://example.com/2",
            3: "https://example.com/3",
        }
        result = map_markers_to_urls(markers, number_to_url)
        assert result == {
            1: "https://example.com/1",
            2: "https://example.com/2",
            3: "https://example.com/3",
        }

    def test_ignore_invalid_markers(self):
        """Test that markers without URLs are ignored."""
        markers = {1, 2, 3}
        number_to_url = {
            1: "https://example.com/1",
            # 2 is missing
            3: "https://example.com/3",
        }
        result = map_markers_to_urls(markers, number_to_url)
        assert result == {
            1: "https://example.com/1",
            3: "https://example.com/3",
        }


class TestFormatCitationsSection:
    """Tests for format_citations_section function."""

    def test_format_plain(self):
        """Test formatting citations in plain mode."""
        marker_url_map = {
            1: "https://example.com/1",
            2: "https://example.com/2",
        }
        result = format_citations_section(marker_url_map, plain=True)
        assert "Sources:" in result
        assert "[1] https://example.com/1" in result
        assert "[2] https://example.com/2" in result

    def test_format_with_emoji(self):
        """Test formatting citations with emoji."""
        marker_url_map = {
            1: "https://example.com/1",
        }
        result = format_citations_section(marker_url_map, plain=False)
        assert "📚" in result
        assert "[1] https://example.com/1" in result

    def test_empty_map(self):
        """Test with empty mapping."""
        result = format_citations_section({}, plain=True)
        assert result == ""


class TestAddCitationsToAnswer:
    """Tests for add_citations_to_answer function."""

    def test_add_citations_to_answer(self):
        """Test adding citations section to answer."""
        answer = "The answer is yes [1]."
        number_to_url = {1: "https://example.com"}
        result = add_citations_to_answer(answer, number_to_url, plain=True)
        assert "Sources:" in result
        assert "[1] https://example.com" in result

    def test_no_duplicates(self):
        """Test that duplicate citations sections are not added."""
        answer = "The answer is yes.\n\nSources:\n[1] https://example.com"
        number_to_url = {1: "https://example.com"}
        result = add_citations_to_answer(answer, number_to_url, plain=True)
        # Should not add another Sources section
        assert result.count("Sources:") == 1

    def test_no_citations_to_add(self):
        """Test when no citations are referenced."""
        answer = "The answer is yes."
        number_to_url = {1: "https://example.com"}
        result = add_citations_to_answer(answer, number_to_url, plain=True)
        # Should not add citations section if no markers found
        assert result == answer


class TestProcessAnswerWithCitations:
    """Tests for process_answer_with_citations function."""

    def test_process_with_citations(self):
        """Test processing answer with citation mapping."""
        answer = "The answer is [1]."
        url_citations = {"https://example.com": 1}
        result = process_answer_with_citations(
            answer,
            url_citations,
            include_citations_section=True,
            plain=True,
        )
        assert "Sources:" in result
        assert "[1] https://example.com" in result

    def test_process_without_citations_section(self):
        """Test processing without adding citations section."""
        answer = "The answer is [1]."
        url_citations = {"https://example.com": 1}
        result = process_answer_with_citations(
            answer,
            url_citations,
            include_citations_section=False,
            plain=True,
        )
        assert result == answer


class TestExtractCitationsFromToolResult:
    """Tests for extract_citations_from_tool_result function."""

    def test_extract_from_tool_result(self):
        """Test extracting citations from web_get tool result."""
        result = {
            "pages": [
                {
                    "url": "https://example.com/1",
                    "content": "[1] https://example.com/1\n---\nContent here",
                },
                {
                    "url": "https://example.com/2",
                    "content": "[2] https://example.com/2\n---\nMore content",
                },
            ]
        }
        citations = extract_citations_from_tool_result(result)
        assert citations == {
            "https://example.com/1": 1,
            "https://example.com/2": 2,
        }

    def test_extract_with_errors(self):
        """Test extracting citations when some pages have errors."""
        result = {
            "pages": [
                {
                    "url": "https://example.com/1",
                    "content": "[1] https://example.com/1\n---\nContent",
                },
                {
                    "url": "https://example.com/error",
                    "error": "Failed to fetch",
                    "content": "",
                },
            ]
        }
        citations = extract_citations_from_tool_result(result)
        assert citations == {"https://example.com/1": 1}

    def test_extract_no_citations(self):
        """Test when no citations are found."""
        result = {
            "pages": [
                {
                    "url": "https://example.com/1",
                    "content": "https://example.com/1\n---\nContent without marker",
                },
            ]
        }
        citations = extract_citations_from_tool_result(result)
        assert citations == {}
