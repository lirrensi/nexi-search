"""Unit tests for tools module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nexi.config import Config
from nexi.tools import TOOLS, execute_tool, web_get, web_search


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
                "api_key": "test_key",
                "model": "test_model",
            },
            "jina": {
                "type": "jina",
                "api_key": "test_jina",
            },
        },
        default_effort="m",
        max_output_tokens=8192,
    )


def test_tools_schema_structure() -> None:
    """Tool schema exposes the expected tool names."""
    assert len(TOOLS) == 3
    assert [tool["function"]["name"] for tool in TOOLS] == [
        "web_search",
        "web_get",
        "final_answer",
    ]


@pytest.mark.asyncio
async def test_execute_tool_web_search_routes_through_orchestrator() -> None:
    """execute_tool delegates search requests to web_search."""
    config = _build_config()
    with patch(
        "nexi.tools.web_search", new=AsyncMock(return_value={"searches": []})
    ) as mock_search:
        result = await execute_tool("web_search", {"queries": ["test"]}, config)

    assert result == {"searches": []}
    mock_search.assert_awaited_once_with(["test"], config, False)


@pytest.mark.asyncio
async def test_execute_tool_web_get_routes_through_orchestrator() -> None:
    """execute_tool delegates fetch requests to web_get."""
    config = _build_config()
    with patch("nexi.tools.web_get", new=AsyncMock(return_value={"pages": []})) as mock_get:
        result = await execute_tool("web_get", {"urls": ["https://example.com"]}, config)

    assert result == {"pages": []}
    mock_get.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_tool_final_answer() -> None:
    """execute_tool returns final answers directly."""
    result = await execute_tool("final_answer", {"answer": "done"}, _build_config())

    assert result == {"answer": "done"}


@pytest.mark.asyncio
async def test_web_search_returns_provider_failures() -> None:
    """web_search returns orchestrator payloads unchanged."""
    config = _build_config()
    payload = {
        "searches": [{"query": "test", "results": []}],
        "provider_failures": [{"provider": "jina", "error": "boom"}],
    }

    with patch("nexi.tools.run_search_chain", new=AsyncMock(return_value=payload)) as mock_chain:
        result = await web_search(["test"], config, verbose=True)

    assert result == payload
    mock_chain.assert_awaited_once_with(["test"], config, True)


@pytest.mark.asyncio
async def test_web_get_formats_full_content_and_preserves_failures() -> None:
    """web_get keeps fetch failure metadata and formats page content."""
    config = _build_config()
    payload = {
        "pages": [
            {"url": "https://example.com", "content": "Example body"},
            {"url": "https://broken.example", "content": "", "error": "Network error"},
        ],
        "provider_failures": [{"provider": "jina", "error": "timeout"}],
    }

    with patch("nexi.tools.run_fetch_chain", new=AsyncMock(return_value=payload)) as mock_chain:
        result = await web_get(
            urls=["https://example.com", "https://broken.example"],
            config=config,
            get_full=True,
            url_numbers={"https://example.com": 1},
        )

    assert result["provider_failures"] == payload["provider_failures"]
    assert result["pages"][0]["content"].startswith("[1] https://example.com")
    assert result["pages"][1]["error"] == "Network error"
    mock_chain.assert_awaited_once_with(
        ["https://example.com", "https://broken.example"],
        config,
        False,
    )
