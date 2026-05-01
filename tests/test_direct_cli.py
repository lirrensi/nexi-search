"""Tests for direct search and fetch CLI entrypoints."""

# FILE: tests/test_direct_cli.py
# PURPOSE: Verify direct CLI surfaces, including provider override behavior.
# OWNS: Click-level tests for nexi-search and nexi-fetch.
# EXPORTS: none
# DOCS: agent_chat/plan_direct_provider_override_2026-04-24.md

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
            "brave": {
                "type": "brave",
                "api_key": "test-brave",
            },
            "special_trafilatura": {
                "type": "special_trafilatura",
            },
            "special_playwright": {
                "type": "special_playwright",
            },
        },
        default_effort="m",
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


def test_nexi_search_non_verbose_prints_human_output() -> None:
    """nexi-search prints formatted text output in non-verbose mode."""
    runner = CliRunner()
    payload = {
        "searches": [
            {
                "query": "test query",
                "results": [
                    {
                        "title": "Result Title",
                        "url": "https://example.com",
                        "description": "Result description",
                    }
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
        result = runner.invoke(search_main, ["test query"])

    assert result.exit_code == 0
    assert "Query: test query" in result.output
    assert "1. Result Title" in result.output
    assert "https://example.com" in result.output
    mock_chain.assert_awaited_once()


def test_nexi_search_provider_override_uses_single_backend() -> None:
    """nexi-search --provider narrows the chain to one provider."""
    runner = CliRunner()
    payload = {"searches": [{"query": "test query", "results": []}], "provider_failures": []}

    with (
        patch("nexi.search_cli.ensure_config", return_value=_build_config()),
        patch("nexi.search_cli.check_command_readiness") as mock_readiness,
        patch(
            "nexi.search_cli.run_search_chain", new=AsyncMock(return_value=payload)
        ) as mock_chain,
    ):
        result = runner.invoke(search_main, ["--provider", "jina", "test query"])

    assert result.exit_code == 0
    mock_readiness.assert_not_called()
    mock_chain.assert_awaited_once()
    call_config = mock_chain.await_args.args[1]
    assert call_config.search_backends == ["jina"]


def test_nexi_search_provider_override_capability_mismatch_fails() -> None:
    """nexi-search rejects fetch-only provider overrides."""
    runner = CliRunner()

    with patch("nexi.search_cli.ensure_config", return_value=_build_config()):
        result = runner.invoke(search_main, ["--provider", "special_trafilatura", "test query"])

    assert result.exit_code != 0
    assert "Unsupported search provider type" in result.output


def test_nexi_search_provider_override_missing_provider_fails() -> None:
    """nexi-search rejects unknown provider overrides."""
    runner = CliRunner()

    with patch("nexi.search_cli.ensure_config", return_value=_build_config()):
        result = runner.invoke(search_main, ["--provider", "missing", "test query"])

    assert result.exit_code != 0
    assert "Missing provider instance: missing" in result.output


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
        patch(
            "nexi.fetch_cli.post_process_direct_fetch_payload",
            side_effect=lambda payload, **kwargs: payload,
        ) as mock_post,
        patch("nexi.fetch_cli.web_get", new=AsyncMock(return_value=payload)) as mock_get,
    ):
        result = runner.invoke(fetch_main, ["--json", "https://example.com"])

    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    mock_get.assert_awaited_once()
    mock_post.assert_called_once()


def test_nexi_fetch_non_verbose_prints_content_output() -> None:
    """nexi-fetch prints fetched content in non-verbose mode."""
    runner = CliRunner()
    payload = {
        "pages": [{"url": "https://example.com", "content": "https://example.com\n---\nBody"}],
        "provider_failures": [],
    }

    with (
        patch("nexi.fetch_cli.ensure_config", return_value=_build_config()),
        patch("nexi.fetch_cli.check_command_readiness", return_value=[]),
        patch(
            "nexi.fetch_cli.post_process_direct_fetch_payload",
            side_effect=lambda payload, **kwargs: payload,
        ),
        patch("nexi.fetch_cli.web_get", new=AsyncMock(return_value=payload)) as mock_get,
    ):
        result = runner.invoke(fetch_main, ["https://example.com"])

    assert result.exit_code == 0
    assert "https://example.com" in result.output
    assert "Body" in result.output
    mock_get.assert_awaited_once()


def test_nexi_fetch_provider_override_uses_single_backend() -> None:
    """nexi-fetch --provider narrows the chain to one provider."""
    runner = CliRunner()
    payload = {
        "pages": [{"url": "https://example.com", "content": "body"}],
        "provider_failures": [],
    }

    with (
        patch("nexi.fetch_cli.ensure_config", return_value=_build_config()),
        patch("nexi.fetch_cli.check_command_readiness") as mock_readiness,
        patch(
            "nexi.fetch_cli.post_process_direct_fetch_payload",
            side_effect=lambda payload, **kwargs: payload,
        ),
        patch("nexi.fetch_cli.web_get", new=AsyncMock(return_value=payload)) as mock_get,
    ):
        result = runner.invoke(
            fetch_main,
            ["--provider", "special_trafilatura", "https://example.com"],
        )

    assert result.exit_code == 0
    mock_readiness.assert_not_called()
    mock_get.assert_awaited_once()
    call_config = mock_get.await_args.kwargs["config"]
    assert call_config.fetch_backends == ["special_trafilatura"]


def test_nexi_fetch_provider_override_capability_mismatch_fails() -> None:
    """nexi-fetch rejects search-only provider overrides."""
    runner = CliRunner()

    with patch("nexi.fetch_cli.ensure_config", return_value=_build_config()):
        result = runner.invoke(fetch_main, ["--provider", "brave", "https://example.com"])

    assert result.exit_code != 0
    assert "Unsupported fetch provider type" in result.output


def test_nexi_fetch_provider_override_missing_provider_fails() -> None:
    """nexi-fetch rejects unknown provider overrides."""
    runner = CliRunner()

    with patch("nexi.fetch_cli.ensure_config", return_value=_build_config()):
        result = runner.invoke(fetch_main, ["--provider", "missing", "https://example.com"])

    assert result.exit_code != 0
    assert "Missing provider instance: missing" in result.output


def test_nexi_fetch_text_includes_spillover_path() -> None:
    """nexi-fetch text output shows the spillover file path when present."""
    runner = CliRunner()
    payload = {
        "pages": [
            {
                "url": "https://example.com",
                "content": "https://example.com\n---\nBody",
                "full_content_path": r"C:\Temp\nexi-fetch-full.txt",
            }
        ],
        "provider_failures": [],
    }

    with (
        patch("nexi.fetch_cli.ensure_config", return_value=_build_config()),
        patch("nexi.fetch_cli.check_command_readiness", return_value=[]),
        patch(
            "nexi.fetch_cli.post_process_direct_fetch_payload",
            side_effect=lambda payload, **kwargs: payload,
        ),
        patch("nexi.fetch_cli.web_get", new=AsyncMock(return_value=payload)),
    ):
        result = runner.invoke(fetch_main, ["https://example.com"])

    assert result.exit_code == 0
    assert r"Full content saved to: C:\Temp\nexi-fetch-full.txt" in result.output


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
