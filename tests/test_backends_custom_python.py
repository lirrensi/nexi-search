"""Tests for local Python provider adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexi import config as config_module
from nexi.backends.orchestrators import run_fetch_chain, run_llm_chain, run_search_chain
from nexi.config import Config


def _build_custom_config() -> Config:
    """Create a config that routes all capabilities through one custom file."""
    return Config(
        llm_backends=["custom_llm"],
        search_backends=["custom_search"],
        fetch_backends=["custom_fetch"],
        providers={
            "custom_llm": {"type": "provider-custom_api", "token": "ok"},
            "custom_search": {"type": "provider-custom_api", "token": "ok"},
            "custom_fetch": {"type": "provider-custom_api", "token": "ok"},
        },
        default_effort="m",
        provider_timeout=5,
        search_provider_retries=1,
        fetch_provider_retries=1,
    )


def _write_custom_provider(provider_dir: Path) -> None:
    """Write a custom provider file into the config directory."""
    (provider_dir / "custom_api.py").write_text(
        """from __future__ import annotations

def validate_config(config: dict, capability: str) -> None:
    if config.get("token") != "ok":
        raise ValueError(f"invalid token for {capability}")

async def search(queries: list[str], **kwargs) -> dict:
    return {
        "searches": [
            {
                "query": query,
                "results": [{"title": f"Result for {query}", "url": f"https://{query}.example"}],
            }
            for query in queries
        ]
    }

def fetch(urls: list[str], **kwargs) -> dict:
    return {
        "pages": [
            {"url": url, "content": f"raw content for {url}"}
            for url in urls
        ]
    }

def complete(messages: list[dict], tools: list[dict], **kwargs) -> dict:
    if tools:
        return {
            "tool_calls": [
                {
                    "name": "final_answer",
                    "arguments": {"answer": "custom answer"},
                }
            ],
            "usage": {"total_tokens": 11, "prompt_tokens": 7, "completion_tokens": 4},
        }
    return {"content": messages[-1]["content"]}
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_custom_python_providers_work_across_capabilities(
    monkeypatch, tmp_path: Path
) -> None:
    """Custom provider files can back search, fetch, and llm chains."""
    _write_custom_provider(tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    config = _build_custom_config()

    search_payload = await run_search_chain(["alpha"], config, verbose=False)
    fetch_payload = await run_fetch_chain(["https://example.com"], config, verbose=False)
    llm_response = await run_llm_chain(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "final_answer"}}],
        config=config,
        verbose=False,
        max_tokens=128,
    )

    assert search_payload["searches"][0]["results"][0]["url"] == "https://alpha.example"
    assert fetch_payload["pages"][0]["content"] == "raw content for https://example.com"
    assert llm_response.choices[0].message.tool_calls[0].function.name == "final_answer"
    assert (
        llm_response.choices[0].message.tool_calls[0].function.arguments
        == '{"answer": "custom answer"}'
    )
    assert llm_response.usage.total_tokens == 11


@pytest.mark.asyncio
async def test_custom_python_provider_validation_failure_is_reported(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Custom provider validate_config hooks flow into provider failures."""
    _write_custom_provider(tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    config = _build_custom_config()
    config.providers["custom_search"]["token"] = "bad"

    payload = await run_search_chain(["alpha"], config, verbose=False)

    assert payload["searches"][0]["error"] == "All configured search providers failed"
    assert payload["provider_failures"][0]["stage"] == "validate"
    assert payload["provider_failures"][0]["error"] == "invalid token for search"


@pytest.mark.asyncio
async def test_custom_python_fetch_requires_canonical_object_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Custom fetch providers must return the canonical pages object."""
    (tmp_path / "custom_api.py").write_text(
        """from __future__ import annotations

def fetch(urls: list[str], **kwargs) -> str:
    return "raw text only"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    config = _build_custom_config()
    payload = await run_fetch_chain(["https://example.com"], config, verbose=False)

    assert payload["pages"][0]["error"] == "All configured fetch providers failed"
    assert payload["provider_failures"][0]["error"] == (
        "Custom fetch provider must return an object with a 'pages' list"
    )
