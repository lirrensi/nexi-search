"""Tests for SnitchMD fetch provider."""

from __future__ import annotations

import json
from typing import Any

import pytest

from nexi.backends.snitchmd import SnitchFetchProvider, _build_command


class FakeSubprocessResult:
    """Fake subprocess result."""

    def __init__(
        self,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.asyncio
async def test_snitchmd_fetch_provider_returns_markdown(monkeypatch) -> None:
    """SnitchMD fetch provider returns markdown from Docker output."""
    captured_commands: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> FakeSubprocessResult:
        captured_commands.append(cmd)
        return FakeSubprocessResult(0, stdout="# Example page content")

    monkeypatch.setattr("nexi.backends.snitchmd.subprocess.run", fake_run)

    provider = SnitchFetchProvider()
    payload = await provider.fetch(
        ["https://example.com"],
        {"type": "snitchmd"},
        timeout=30,
        verbose=False,
    )

    assert payload == {"pages": [{"url": "https://example.com", "content": "# Example page content"}]}
    assert any("syabro/snitchmd" in cmd for cmd in captured_commands)
    assert any("https://example.com" in cmd for cmd in captured_commands)


@pytest.mark.asyncio
async def test_snitchmd_fetch_provider_parses_json_output(monkeypatch) -> None:
    """SnitchMD fetch provider extracts markdown from JSON output."""
    json_data = {
        "url": "https://example.com",
        "markdown": "# Extracted content",
        "title": "Example",
    }

    def fake_run(cmd: list[str], **kwargs: Any) -> FakeSubprocessResult:
        return FakeSubprocessResult(0, stdout=json.dumps(json_data))

    monkeypatch.setattr("nexi.backends.snitchmd.subprocess.run", fake_run)

    provider = SnitchFetchProvider()
    payload = await provider.fetch(
        ["https://example.com"],
        {"type": "snitchmd", "json_output": True},
        timeout=30,
        verbose=False,
    )

    assert payload == {"pages": [{"url": "https://example.com", "content": "# Extracted content"}]}


@pytest.mark.asyncio
async def test_snitchmd_fetch_provider_handles_error(monkeypatch) -> None:
    """SnitchMD fetch provider returns error on non-0 exit code."""
    # Mock Docker as available
    monkeypatch.setattr("nexi.backends.snitchmd._is_docker_available", lambda: True)

    def fake_run(cmd: list[str], **kwargs: Any) -> FakeSubprocessResult:
        return FakeSubprocessResult(1, stderr="docker: not found")

    monkeypatch.setattr("nexi.backends.snitchmd.subprocess.run", fake_run)

    provider = SnitchFetchProvider()
    payload = await provider.fetch(
        ["https://example.com"],
        {"type": "snitchmd"},
        timeout=30,
        verbose=False,
    )

    assert payload["pages"][0]["error"] == "SnitchMD error: docker: not found"


@pytest.mark.asyncio
async def test_snitchmd_fetch_provider_timeout(monkeypatch) -> None:
    """SnitchMD fetch provider returns error on timeout."""
    import subprocess

    # Mock Docker as available
    monkeypatch.setattr("nexi.backends.snitchmd._is_docker_available", lambda: True)

    def fake_run(cmd: list[str], **kwargs: Any) -> FakeSubprocessResult:
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr("nexi.backends.snitchmd.subprocess.run", fake_run)

    provider = SnitchFetchProvider()
    payload = await provider.fetch(
        ["https://example.com"],
        {"type": "snitchmd"},
        timeout=30,
        verbose=False,
    )

    assert payload["pages"][0]["error"] == "SnitchMD timeout"


@pytest.mark.asyncio
async def test_snitchmd_fetch_provider_verbose(monkeypatch, capsys) -> None:
    """SnitchMD fetch provider prints URL when verbose is True."""
    def fake_run(cmd: list[str], **kwargs: Any) -> FakeSubprocessResult:
        return FakeSubprocessResult(0, stdout="# Content")

    monkeypatch.setattr("nexi.backends.snitchmd.subprocess.run", fake_run)

    provider = SnitchFetchProvider()
    await provider.fetch(
        ["https://example.com"],
        {"type": "snitchmd"},
        timeout=30,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "[SnitchMD] URL: https://example.com" in captured.out


def test_snitchmd_validate_config_accepts_valid_options() -> None:
    """SnitchMD provider validates valid config options."""
    provider = SnitchFetchProvider()

    # Should not raise
    provider.validate_config({"wait": 5})
    provider.validate_config({"wait_for_selector": ".content"})
    provider.validate_config({"mode": "precision"})
    provider.validate_config({"mode": "recall"})
    provider.validate_config({"no_cache": True})


def test_snitchmd_validate_config_rejects_invalid_mode() -> None:
    """SnitchMD provider rejects invalid mode values."""
    provider = SnitchFetchProvider()

    with pytest.raises(ValueError, match="mode must be 'precision' or 'recall'"):
        provider.validate_config({"mode": "invalid"})


def test_snitchmd_validate_config_rejects_invalid_wait() -> None:
    """SnitchMD provider rejects invalid wait values."""
    provider = SnitchFetchProvider()

    with pytest.raises(ValueError, match="wait must be a number"):
        provider.validate_config({"wait": ["not", "a", "number"]})


def test_snitchmd_validate_config_rejects_invalid_no_cache() -> None:
    """SnitchMD provider rejects invalid no_cache values."""
    provider = SnitchFetchProvider()

    with pytest.raises(ValueError, match="no_cache must be a boolean"):
        provider.validate_config({"no_cache": "true"})


def test_build_command_includes_options() -> None:
    """SnitchMD command builder includes configured options."""
    config = {
        "json_output": True,
        "no_cache": True,
        "wait": 5,
        "mode": "precision",
    }

    cmd = _build_command("https://example.com", config, verbose=False)

    assert "--json" in cmd
    assert "--no-cache" in cmd
    assert "--wait" in cmd
    assert "5" in cmd
    assert "--favor-precision" in cmd
    assert "https://example.com" in cmd


def test_build_command_recall_mode() -> None:
    """SnitchMD command builder includes recall mode flag."""
    config = {"mode": "recall"}

    cmd = _build_command("https://example.com", config, verbose=False)

    assert "--favor-recall" in cmd
    assert "--favor-precision" not in cmd


def test_is_docker_available_check() -> None:
    """Docker availability check returns bool."""
    from nexi.backends.snitchmd import _is_docker_available

    # Just verify it returns a bool - actual value depends on system
    result = _is_docker_available()
    assert isinstance(result, bool)
