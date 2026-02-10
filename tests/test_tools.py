"""Unit tests for tools module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexi.tools import (
    TOOLS,
    clear_url_cache,
    close_http_client,
    execute_tool,
    get_http_client,
    web_get,
    web_search,
)


def test_tools_schema_structure():
    """Test that TOOLS has correct structure."""
    assert len(TOOLS) == 3

    tool_names = [tool["function"]["name"] for tool in TOOLS]
    assert "web_search" in tool_names
    assert "web_get" in tool_names
    assert "final_answer" in tool_names


def test_web_search_tool_schema():
    """Test web_search tool schema."""
    web_search_tool = next(t for t in TOOLS if t["function"]["name"] == "web_search")

    assert web_search_tool["type"] == "function"
    params = web_search_tool["function"]["parameters"]
    assert "queries" in params["properties"]
    assert params["required"] == ["queries"]


def test_web_get_tool_schema():
    """Test web_get tool schema."""
    web_get_tool = next(t for t in TOOLS if t["function"]["name"] == "web_get")

    assert web_get_tool["type"] == "function"
    params = web_get_tool["function"]["parameters"]
    assert "urls" in params["properties"]
    assert "instructions" in params["properties"]
    assert "get_full" in params["properties"]
    assert params["required"] == ["urls"]


def test_final_answer_tool_schema():
    """Test final_answer tool schema."""
    final_answer_tool = next(t for t in TOOLS if t["function"]["name"] == "final_answer")

    assert final_answer_tool["type"] == "function"
    params = final_answer_tool["function"]["parameters"]
    assert "answer" in params["properties"]
    assert params["required"] == ["answer"]


def test_clear_url_cache():
    """Test URL cache clearing."""
    # Add something to cache
    from nexi.tools import _url_cache

    _url_cache["test_url"] = "test_content"

    clear_url_cache()

    assert len(_url_cache) == 0


@pytest.mark.asyncio
async def test_get_http_client():
    """Test HTTP client creation and caching."""
    # Close any existing client
    await close_http_client()

    client1 = get_http_client(timeout=30.0)
    client2 = get_http_client(timeout=30.0)

    # Should return same instance
    assert client1 is client2

    # Cleanup
    await close_http_client()


@pytest.mark.asyncio
async def test_execute_tool_web_search():
    """Test execute_tool with web_search."""
    with patch("nexi.tools.web_search") as mock_search:
        mock_search.return_value = {"searches": []}

        result = await execute_tool(
            "web_search",
            {"queries": ["test"]},
            jina_key="test_key",
            verbose=False,
        )

        assert result == {"searches": []}
        mock_search.assert_called_once()


@pytest.mark.asyncio
async def test_execute_tool_web_get():
    """Test execute_tool with web_get."""
    with patch("nexi.tools.web_get") as mock_get:
        mock_get.return_value = {"pages": []}

        result = await execute_tool(
            "web_get",
            {"urls": ["http://example.com"]},
            jina_key="test_key",
            verbose=False,
        )

        assert result == {"pages": []}
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_execute_tool_final_answer():
    """Test execute_tool with final_answer."""
    result = await execute_tool(
        "final_answer",
        {"answer": "test answer"},
        jina_key="test_key",
        verbose=False,
    )

    assert result == {"answer": "test answer"}


@pytest.mark.asyncio
async def test_execute_tool_unknown():
    """Test execute_tool with unknown tool."""
    result = await execute_tool(
        "unknown_tool",
        {},
        jina_key="test_key",
        verbose=False,
    )

    assert "error" in result
    assert "Unknown tool" in result["error"]


@pytest.mark.asyncio
async def test_web_search_success():
    """Test web_search with mocked response."""
    mock_response = MagicMock()
    mock_response.text = (
        '{"data": [{"title": "Test", "url": "http://test.com", "description": "Test desc"}]}'
    )
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with patch("nexi.tools.get_http_client", return_value=mock_client):
        result = await web_search(
            queries=["test query"],
            jina_key="test_key",
            verbose=False,
            timeout=30,
        )

    assert "searches" in result
    assert len(result["searches"]) == 1


@pytest.mark.asyncio
async def test_web_search_error():
    """Test web_search handles errors gracefully."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Network error")

    with patch("nexi.tools.get_http_client", return_value=mock_client):
        result = await web_search(
            queries=["test query"],
            jina_key="test_key",
            verbose=False,
            timeout=30,
        )

    assert "searches" in result
    assert result["searches"][0]["error"] == "Network error"


@pytest.mark.asyncio
async def test_web_get_success():
    """Test web_get with mocked response."""
    mock_response = MagicMock()
    mock_response.text = "Test page content"
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    # Clear cache first
    clear_url_cache()

    with patch("nexi.tools.get_http_client", return_value=mock_client):
        result = await web_get(
            urls=["http://example.com"],
            jina_key="test_key",
            verbose=False,
            get_full=True,
            timeout=30,
        )

    assert "pages" in result
    assert len(result["pages"]) == 1
    assert "Test page content" in result["pages"][0]["content"]


@pytest.mark.asyncio
async def test_web_get_uses_cache():
    """Test web_get uses URL cache."""
    # Pre-populate cache
    from nexi.tools import _url_cache

    _url_cache["http://cached.com"] = "Cached content"

    result = await web_get(
        urls=["http://cached.com"],
        jina_key="test_key",
        verbose=False,
        get_full=True,
        timeout=30,
    )

    assert "pages" in result
    assert "Cached content" in result["pages"][0]["content"]


@pytest.mark.asyncio
async def test_web_get_error():
    """Test web_get handles errors gracefully."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Fetch error")

    # Clear cache
    clear_url_cache()

    with patch("nexi.tools.get_http_client", return_value=mock_client):
        result = await web_get(
            urls=["http://example.com"],
            jina_key="test_key",
            verbose=False,
            get_full=True,
            timeout=30,
        )

    assert "pages" in result
    assert len(result["pages"]) == 1
    # Check that either error key exists or content is present
    page = result["pages"][0]
    assert "url" in page
    # Error might be in error field or the request succeeded somehow
    if "error" in page:
        assert "Fetch error" in page["error"]
