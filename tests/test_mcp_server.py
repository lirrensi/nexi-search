"""Tests for MCP server config handling and tool execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nexi.config import Config, ConfigCreatedError
from nexi.search import SearchResult

fastmcp = pytest.importorskip("fastmcp")


def _build_config() -> Config:
    """Create a usable config fixture for MCP tests."""
    return Config(
        llm_backends=["openrouter"],
        search_backends=["jina"],
        fetch_backends=["crawl4ai_local", "markdown_new"],
        providers={
            "openrouter": {
                "type": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
                "model": "test-model",
            },
            "jina": {
                "type": "jina",
                "api_key": "test-jina-key",
            },
            "crawl4ai_local": {
                "type": "crawl4ai",
                "headless": True,
            },
            "markdown_new": {
                "type": "markdown_new",
                "method": "auto",
                "retain_images": False,
            },
        },
        default_effort="m",
    )


def test_mcp_server_module_exists() -> None:
    """MCP server module imports when fastmcp is present."""
    from nexi.mcp_server import mcp, nexi_agent, nexi_fetch, nexi_search, run

    assert mcp is not None
    assert nexi_agent is not None
    assert nexi_search is not None
    assert nexi_fetch is not None
    assert run is not None


@patch("nexi.mcp_server.check_command_readiness", return_value=[])
@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_nexi_agent_basic(mock_run_search, mock_ensure_config, _mock_readiness) -> None:
    """nexi_agent returns answer text and metadata."""
    from nexi.mcp_server import nexi_agent

    mock_ensure_config.return_value = _build_config()
    mock_run_search.return_value = SearchResult(
        answer="Test answer",
        urls=["http://example.com"],
        iterations=3,
        duration_s=5.5,
        tokens=1000,
    )

    result = await nexi_agent(
        query="test query",
        effort="m",
        verbose=False,
    )

    assert "Test answer" in result
    assert "http://example.com" in result
    assert "Iterations: 3" in result
    assert "Duration: 5.5s" in result

    mock_run_search.assert_awaited_once()
    call_kwargs = mock_run_search.call_args.kwargs
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["effort"] == "m"
    assert call_kwargs["config"].default_effort == "m"


@patch("nexi.mcp_server.ensure_config")
@pytest.mark.asyncio
async def test_nexi_agent_missing_config_message(mock_ensure_config) -> None:
    """Missing config returns the bootstrap message with the canonical path."""
    from nexi.mcp_server import nexi_agent

    mock_ensure_config.side_effect = ConfigCreatedError(Path("/tmp/config.toml"))

    result = await nexi_agent(query="test query")

    assert "Config template created at ~/.config/nexi/config.toml" in result


@patch("nexi.mcp_server.check_command_readiness")
@patch("nexi.mcp_server.ensure_config")
@pytest.mark.asyncio
async def test_nexi_agent_readiness_failure(mock_ensure_config, mock_readiness) -> None:
    """Readiness errors are returned as user-facing strings."""
    from nexi.mcp_server import nexi_agent

    mock_ensure_config.return_value = _build_config()
    mock_readiness.return_value = ["Activate at least one search provider in search_backends"]

    result = await nexi_agent(query="test query")

    assert "NEXI config is not ready" in result
    assert "Activate at least one search provider in search_backends" in result


@patch("nexi.mcp_server.check_command_readiness", return_value=[])
@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_nexi_agent_runtime_error(
    mock_run_search, mock_ensure_config, _mock_readiness
) -> None:
    """nexi_agent handles runtime search errors."""
    from nexi.mcp_server import nexi_agent

    mock_ensure_config.return_value = _build_config()
    mock_run_search.side_effect = Exception("Search failed")

    result = await nexi_agent(query="test query")

    assert "Search failed" in result


@patch("nexi.mcp_server.check_command_readiness", return_value=[])
@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search_chain", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_nexi_search_returns_direct_payload(
    mock_run_search_chain,
    mock_ensure_config,
    _mock_readiness,
) -> None:
    """nexi_search returns the direct search payload."""
    from nexi.mcp_server import nexi_search

    payload = {
        "searches": [{"query": "test query", "results": [{"title": "Result"}]}],
        "provider_failures": [],
    }
    mock_ensure_config.return_value = _build_config()
    mock_run_search_chain.return_value = payload

    result = await nexi_search(query="test query")

    assert result == payload
    mock_run_search_chain.assert_awaited_once_with(
        ["test query"], mock_ensure_config.return_value, False
    )


@patch("nexi.mcp_server.ensure_config")
@pytest.mark.asyncio
async def test_nexi_search_missing_config_returns_error_object(mock_ensure_config) -> None:
    """Missing config returns a structured error for direct MCP search."""
    from nexi.mcp_server import nexi_search

    mock_ensure_config.side_effect = ConfigCreatedError(Path("/tmp/config.toml"))

    result = await nexi_search(query="test query")

    assert "error" in result
    assert "Config template created at ~/.config/nexi/config.toml" in result["error"]


@patch("nexi.mcp_server.check_command_readiness", return_value=[])
@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.web_get", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_nexi_fetch_returns_direct_payload(
    mock_web_get, mock_ensure_config, _mock_readiness
) -> None:
    """nexi_fetch returns the direct fetch payload."""
    from nexi.mcp_server import nexi_fetch

    payload = {
        "pages": [{"url": "https://example.com", "content": "body"}],
        "provider_failures": [],
    }
    mock_ensure_config.return_value = _build_config()
    mock_web_get.return_value = payload

    result = await nexi_fetch(urls=["https://example.com"], full=True)

    assert result == payload
    mock_web_get.assert_awaited_once()


@pytest.mark.asyncio
async def test_nexi_fetch_requires_one_url() -> None:
    """nexi_fetch rejects empty URL lists."""
    from nexi.mcp_server import nexi_fetch

    result = await nexi_fetch(urls=[])

    assert result == {"error": "At least one URL is required"}
