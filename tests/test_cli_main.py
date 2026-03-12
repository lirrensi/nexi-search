"""Tests for the main `nexi` CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from nexi import cli as cli_module
from nexi import config as config_module
from nexi import history as history_module
from nexi.cli import main
from nexi.config import Config
from nexi.config_template import write_default_template
from nexi.search import SearchResult


def _build_ready_config() -> Config:
    """Create a usable config fixture for the main CLI."""
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
                "api_key": "test-jina",
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
        max_output_tokens=4096,
        provider_timeout=30,
        search_provider_retries=2,
        fetch_provider_retries=2,
    )


def _patch_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point config and history paths at a temp directory."""
    config_path = tmp_path / "config.toml"
    history_path = tmp_path / "history.jsonl"

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_path)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", config_path)
    monkeypatch.setattr(history_module, "CONFIG_DIR", tmp_path)

    return config_path, history_path


def test_missing_config_search_bootstraps_and_exits_nonzero(monkeypatch, tmp_path: Path) -> None:
    """A normal search creates config.toml and stops with instructions."""
    config_path, _history_path = _patch_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["test query"])

    assert result.exit_code == 1
    assert config_path.exists()
    assert "Config template created at" in result.output
    assert "config.toml" in result.output


def test_nexi_init_creates_once_and_leaves_existing_file(monkeypatch, tmp_path: Path) -> None:
    """`nexi init` only creates the template when the file is missing."""
    config_path, _history_path = _patch_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    first = runner.invoke(main, ["init"])

    assert first.exit_code == 0
    assert config_path.exists()
    assert "Created config template" in first.output

    config_path.write_text("sentinel", encoding="utf-8")

    second = runner.invoke(main, ["init"])

    assert second.exit_code == 0
    assert "Config already exists" in second.output
    assert config_path.read_text(encoding="utf-8") == "sentinel"


def test_nexi_doctor_reports_template_readiness_failures(monkeypatch, tmp_path: Path) -> None:
    """`nexi doctor` reports the template as not ready for search commands."""
    config_path, _history_path = _patch_paths(monkeypatch, tmp_path)
    write_default_template(config_path)
    runner = CliRunner()

    result = runner.invoke(main, ["doctor"])

    assert result.exit_code == 1
    assert "nexi: FAIL" in result.output
    assert "nexi-search: FAIL" in result.output
    assert "nexi-fetch: PASS" in result.output


def test_nexi_clean_deletes_history_and_rewrites_template(monkeypatch, tmp_path: Path) -> None:
    """`nexi clean` resets local state and recreates the default template."""
    config_path, history_path = _patch_paths(monkeypatch, tmp_path)
    config_path.write_text("invalid", encoding="utf-8")
    history_path.write_text('{"query": "old"}\n', encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(main, ["clean"])

    assert result.exit_code == 0
    assert not history_path.exists()
    assert config_path.exists()
    assert "[providers.markdown_new]" in config_path.read_text(encoding="utf-8")
    assert "Recreated config template" in result.output


def test_nexi_config_opens_path_via_editor(monkeypatch, tmp_path: Path) -> None:
    """`nexi config` opens the config path with the chosen editor command."""
    config_path, _history_path = _patch_paths(monkeypatch, tmp_path)
    runner = CliRunner()

    with patch("nexi.cli.subprocess.run") as mock_run:
        result = runner.invoke(main, ["config"], env={"EDITOR": "code --wait"})

    assert result.exit_code == 0
    assert config_path.exists()
    mock_run.assert_called_once_with(["code", "--wait", str(config_path)], check=False)


def test_default_search_uses_search_path_when_dependencies_are_mocked() -> None:
    """Positional queries still route through the normal search path."""
    runner = CliRunner()

    with (
        patch("nexi.cli.ensure_config", return_value=_build_ready_config()),
        patch("nexi.cli.check_command_readiness", return_value=[]),
        patch("nexi.cli.is_tty", return_value=True),
        patch(
            "nexi.cli.run_search_sync",
            return_value=SearchResult(
                answer="Mock answer",
                urls=["https://example.com"],
                iterations=2,
                duration_s=1.5,
                tokens=123,
                reached_max_iter=False,
            ),
        ) as mock_run_search,
        patch("nexi.cli.add_history_entry") as mock_add_history,
    ):
        result = runner.invoke(main, ["hello world"])

    assert result.exit_code == 0
    assert "Mock answer" in result.output
    mock_run_search.assert_called_once()
    mock_add_history.assert_called_once()
