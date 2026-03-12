"""Tests for direct search and fetch CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from nexi.config import Config, ConfigCreatedError
from nexi.fetch_cli import main as fetch_main
from nexi.search_cli import main as search_main


def _build_config() -> Config:
    """Create a usable config fixture for direct commands."""
    return Config(
        llm_backends=["openrouter"],
        search_backends=["jina"],
        fetch_backends=["jina"],
        providers={
            "openrouter": {
                "type": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
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
        patch("nexi.search_cli.check_command_readiness", return_value=[]),
        patch(
            "nexi.search_cli.run_search_chain", new=AsyncMock(return_value=payload)
        ) as mock_chain,
    ):
        result = runner.invoke(search_main, ["--json", "test query"])

    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    mock_chain.assert_awaited_once()


def test_nexi_search_missing_config_bootstraps_cleanly() -> None:
    """nexi-search shows the bootstrap message instead of a traceback."""
    runner = CliRunner()

    with patch(
        "nexi.search_cli.ensure_config",
        side_effect=ConfigCreatedError(Path("/tmp/config.toml")),
    ):
        result = runner.invoke(search_main, ["test query"])

    assert result.exit_code != 0
    assert "Config template created at" in result.output


def test_nexi_search_readiness_failure() -> None:
    """nexi-search exits non-zero when no usable search provider is active."""
    runner = CliRunner()

    with (
        patch("nexi.search_cli.ensure_config", return_value=_build_config()),
        patch(
            "nexi.search_cli.check_command_readiness",
            return_value=["Activate at least one search provider in search_backends"],
        ),
    ):
        result = runner.invoke(search_main, ["test query"])

    assert result.exit_code != 0
    assert "Activate at least one search provider in search_backends" in result.output


def test_nexi_fetch_json() -> None:
    """nexi-fetch --json prints fetch payload JSON."""
    runner = CliRunner()
    payload = {
        "pages": [{"url": "https://example.com", "content": "https://example.com\n---\nBody"}],
        "provider_failures": [],
    }

    with (
        patch("nexi.fetch_cli.ensure_config", return_value=_build_config()),
        patch("nexi.fetch_cli.check_command_readiness", return_value=[]),
        patch("nexi.fetch_cli.web_get", new=AsyncMock(return_value=payload)) as mock_get,
    ):
        result = runner.invoke(fetch_main, ["--json", "https://example.com"])

    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    mock_get.assert_awaited_once()


def test_nexi_fetch_missing_config_bootstraps_cleanly() -> None:
    """nexi-fetch shows the bootstrap message instead of a traceback."""
    runner = CliRunner()

    with patch(
        "nexi.fetch_cli.ensure_config",
        side_effect=ConfigCreatedError(Path("/tmp/config.toml")),
    ):
        result = runner.invoke(fetch_main, ["https://example.com"])

    assert result.exit_code != 0
    assert "Config template created at" in result.output


def test_nexi_fetch_readiness_failure() -> None:
    """nexi-fetch exits non-zero when no usable fetch provider is active."""
    runner = CliRunner()

    with (
        patch("nexi.fetch_cli.ensure_config", return_value=_build_config()),
        patch(
            "nexi.fetch_cli.check_command_readiness",
            return_value=["Activate at least one fetch provider in fetch_backends"],
        ),
    ):
        result = runner.invoke(fetch_main, ["https://example.com"])

    assert result.exit_code != 0
    assert "Activate at least one fetch provider in fetch_backends" in result.output
