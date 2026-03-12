"""Tests for direct search and fetch CLI entrypoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from nexi.config import Config
from nexi.fetch_cli import main as fetch_main
from nexi.search_cli import main as search_main


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
                "api_key": "test-jina",
            },
        },
        default_effort="m",
        max_output_tokens=4000,
    )


def test_nexi_search_json() -> None:
    """nexi-search --json prints orchestrator JSON."""
    runner = CliRunner()
    payload = {
        "searches": [
            {
                "query": "test query",
                "results": [
                    {"title": "Result", "url": "https://example.com", "description": "desc"}
                ],
            }
        ],
        "provider_failures": [],
    }

    with (
        patch("nexi.search_cli.ensure_config", return_value=_build_config()),
        patch(
            "nexi.search_cli.run_search_chain",
            new=AsyncMock(return_value=payload),
        ) as mock_chain,
    ):
        result = runner.invoke(search_main, ["--json", "test query"])

    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    mock_chain.assert_awaited_once()


def test_nexi_fetch_json() -> None:
    """nexi-fetch --json prints fetch payload JSON."""
    runner = CliRunner()
    payload = {
        "pages": [{"url": "https://example.com", "content": "https://example.com\n---\nBody"}],
        "provider_failures": [],
    }

    with (
        patch("nexi.fetch_cli.ensure_config", return_value=_build_config()),
        patch(
            "nexi.fetch_cli.web_get",
            new=AsyncMock(return_value=payload),
        ) as mock_get,
    ):
        result = runner.invoke(fetch_main, ["--json", "https://example.com"])

    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    mock_get.assert_awaited_once()
