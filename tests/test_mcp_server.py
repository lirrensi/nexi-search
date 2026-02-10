"""Tests for MCP server functionality."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest


def test_mcp_server_module_exists():
    """Test that MCP server module can be imported."""
    try:
        from nexi.mcp_server import mcp, nexi_search, run

        assert mcp is not None
        assert nexi_search is not None
        assert run is not None
    except ImportError:
        pytest.skip("fastmcp not installed - install with: uv sync --group mcp")


@pytest.mark.skipif(
    True, reason="Requires fastmcp to be installed - install with: uv sync --group mcp"
)
def test_nexi_search_tool_signature():
    """Test that nexi_search tool has correct signature."""
    import inspect

    from nexi.mcp_server import nexi_search

    sig = inspect.signature(nexi_search)
    params = list(sig.parameters.keys())

    assert "query" in params
    assert "effort" in params
    assert "max_iter" in params
    assert "time_target" in params
    assert "verbose" in params

    # Check defaults
    assert sig.parameters["effort"].default == "m"
    assert sig.parameters["max_iter"].default is None
    assert sig.parameters["time_target"].default is None
    assert sig.parameters["verbose"].default is False


@pytest.mark.skipif(
    True, reason="Requires fastmcp to be installed - install with: uv sync --group mcp"
)
@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search_sync")
def test_nexi_search_basic(mock_run_search, mock_ensure_config):
    """Test basic nexi_search functionality."""
    from nexi.mcp_server import nexi_search
    from nexi.search import SearchResult

    # Mock config
    mock_config = Mock()
    mock_config.base_url = "http://test.com"
    mock_config.api_key = "test-key"
    mock_config.model = "test-model"
    mock_config.jina_key = "test-jina-key"
    mock_config.default_effort = "m"
    mock_config.time_target = 600
    mock_config.max_output_tokens = 4000
    mock_ensure_config.return_value = mock_config

    # Mock search result
    mock_result = SearchResult(
        answer="Test answer",
        urls=["http://example.com"],
        iterations=3,
        duration_s=5.5,
        tokens=1000,
        reached_max_iter=False,
    )
    mock_run_search.return_value = mock_result

    # Call the tool
    result = nexi_search(
        query="test query",
        effort="m",
        max_iter=5,
        time_target=120,
        verbose=False,
    )

    # Verify result
    assert "Test answer" in result
    assert "http://example.com" in result
    assert "3" in result  # iterations
    assert "5.5" in result  # duration

    # Verify run_search_sync was called correctly
    mock_run_search.assert_called_once()
    call_kwargs = mock_run_search.call_args.kwargs
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["effort"] == "m"
    assert call_kwargs["max_iter"] == 5
    assert call_kwargs["time_target"] == 120


@pytest.mark.skipif(
    True, reason="Requires fastmcp to be installed - install with: uv sync --group mcp"
)
@patch("nexi.mcp_server.ensure_config")
def test_nexi_search_config_error(mock_ensure_config):
    """Test nexi_search handles config errors."""
    from nexi.mcp_server import nexi_search

    mock_ensure_config.side_effect = Exception("Config error")

    result = nexi_search(query="test query")

    assert "Error loading NEXI config" in result
    assert "Config error" in result


@pytest.mark.skipif(
    True, reason="Requires fastmcp to be installed - install with: uv sync --group mcp"
)
@patch("nexi.mcp_server.ensure_config")
@patch("nexi.mcp_server.run_search_sync")
def test_nexi_search_runtime_error(mock_run_search, mock_ensure_config):
    """Test nexi_search handles runtime errors."""
    from nexi.mcp_server import nexi_search

    # Mock config
    mock_config = Mock()
    mock_config.base_url = "http://test.com"
    mock_config.api_key = "test-key"
    mock_config.model = "test-model"
    mock_config.jina_key = "test-jina-key"
    mock_config.default_effort = "m"
    mock_config.time_target = 600
    mock_config.max_output_tokens = 4000
    mock_ensure_config.return_value = mock_config

    # Mock search error
    mock_run_search.side_effect = Exception("Search failed")

    result = nexi_search(query="test query")

    assert "Search failed" in result
