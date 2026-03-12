"""Tests for MCP server functionality."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nexi.config import Config
from nexi.search import SearchResult

fastmcp = pytest.importorskip("fastmcp")


def _build_config() -> Config:
    """Create a canonical config fixture."""
    return Config(
        llm_backends=["openai_default"],
        search_backends=["jina"],
        fetch_backends=["jina"],
        providers={
            "openai_default": {
                "type": "openai_compatible",
                "base_url": "https://api.test.com/v1",
                "api_key": "test-key",
                "model": "test-model",
            },
            "jina": {
                "type": "jina",
                "api_key": "test-jina-key",
            },
        },
        default_effort="m",
        max_output_tokens=4000,
        time_target=600,
    )


def test_mcp_server_module_exists() -> None:
    """MCP server module imports when fastmcp is present."""
    from nexi.mcp_server import mcp, nexi_search, run

    assert mcp is not None
    assert nexi_search is not None
    assert run is not None


@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search_sync")
def test_nexi_search_basic(mock_run_search, mock_ensure_config) -> None:
    """nexi_search returns answer text and metadata."""
    from nexi.mcp_server import nexi_search

    mock_ensure_config.return_value = _build_config()
    mock_run_search.return_value = SearchResult(
        answer="Test answer",
        urls=["http://example.com"],
        iterations=3,
        duration_s=5.5,
        tokens=1000,
        reached_max_iter=False,
    )

    result = nexi_search(
        query="test query",
        effort="m",
        max_iter=5,
        time_target=120,
        verbose=False,
    )

    assert "Test answer" in result
    assert "http://example.com" in result
    assert "Iterations: 3" in result
    assert "Duration: 5.5s" in result

    mock_run_search.assert_called_once()
    call_kwargs = mock_run_search.call_args.kwargs
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["effort"] == "m"
    assert call_kwargs["max_iter"] == 5
    assert call_kwargs["time_target"] == 120
    assert call_kwargs["config"].default_effort == "m"
    assert call_kwargs["config"].time_target == 120


@patch("nexi.mcp_server.ensure_config")
def test_nexi_search_config_error(mock_ensure_config) -> None:
    """nexi_search handles config loading errors."""
    from nexi.mcp_server import nexi_search

    mock_ensure_config.side_effect = Exception("Config error")

    result = nexi_search(query="test query")

    assert "Error loading NEXI config" in result
    assert "Config error" in result


@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search_sync")
def test_nexi_search_runtime_error(mock_run_search, mock_ensure_config) -> None:
    """nexi_search handles runtime search errors."""
    from nexi.mcp_server import nexi_search

    mock_ensure_config.return_value = _build_config()
    mock_run_search.side_effect = Exception("Search failed")

    result = nexi_search(query="test query")

    assert "Search failed" in result
