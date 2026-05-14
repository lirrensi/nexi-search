"""Tests for backend orchestration behavior."""

# FILE: tests/test_backends_orchestrators.py
# PURPOSE: Verify provider-chain failover, retry, and failure metadata.
# OWNS: Orchestrator behavior tests for search, fetch, and LLM provider chains.
# EXPORTS: none
# DOCS: docs/arch.md; agent_chat/plan_search_empty_failover_2026-04-24.md

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from nexi.backends.orchestrators import (
    ProviderChainError,
    run_fetch_chain,
    run_llm_chain,
    run_search_chain,
)
from nexi.config import Config


def _build_config(**overrides: Any) -> Config:
    """Create a canonical config fixture."""
    config = Config(
        llm_backends=["primary_llm", "secondary_llm"],
        search_backends=["primary_search", "secondary_search"],
        fetch_backends=["primary_fetch", "secondary_fetch"],
        providers={
            "primary_llm": {
                "type": "primary_llm",
                "base_url": "https://api.test.com/v1",
                "api_key": "key",
                "model": "model",
            },
            "secondary_llm": {
                "type": "secondary_llm",
                "base_url": "https://api.test.com/v1",
                "api_key": "key",
                "model": "model",
            },
            "primary_search": {"type": "primary_search"},
            "secondary_search": {"type": "secondary_search"},
            "primary_fetch": {"type": "primary_fetch"},
            "secondary_fetch": {"type": "secondary_fetch"},
        },
        default_effort="m",
        provider_timeout=5,
        search_provider_retries=1,
        fetch_provider_retries=1,
    )

    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.mark.asyncio
async def test_run_search_chain_partial_failover(monkeypatch) -> None:
    """Successful queries stay while failed queries move to the next provider."""
    calls: list[tuple[str, list[str]]] = []

    class PrimarySearchProvider:
        name = "primary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            calls.append((self.name, queries.copy()))
            return {
                "searches": [
                    {
                        "query": query,
                        "results": [{"title": f"{query} title", "url": f"https://{query}.example"}],
                    }
                    if query == "q1"
                    else {"query": query, "results": [], "error": "primary failed"}
                    for query in queries
                ]
            }

    class SecondarySearchProvider:
        name = "secondary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            calls.append((self.name, queries.copy()))
            return {
                "searches": [
                    {
                        "query": query,
                        "results": [
                            {"title": f"{query} recovered", "url": f"https://{query}.fallback"}
                        ],
                    }
                    for query in queries
                ]
            }

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_search_provider",
        lambda provider_name, providers: {
            "primary_search": PrimarySearchProvider,
            "secondary_search": SecondarySearchProvider,
        }[provider_name],
    )

    payload = await run_search_chain(["q1", "q2"], _build_config(), verbose=False)

    assert calls == [
        ("primary_search", ["q1", "q2"]),
        ("primary_search", ["q2"]),
        ("secondary_search", ["q2"]),
    ]
    assert payload["searches"][0]["results"][0]["url"] == "https://q1.example"
    assert payload["searches"][1]["results"][0]["url"] == "https://q2.fallback"
    assert payload["provider_failures"][0]["provider"] == "primary_search"
    assert payload["provider_failures"][0]["failed_items"] == ["q2"]
    assert payload["provider_failures"][0]["failure_kind"] == "provider_error"


@pytest.mark.asyncio
async def test_run_search_chain_empty_result_failover(monkeypatch) -> None:
    """Empty search responses move to the next provider without dropping successes."""
    calls: list[tuple[str, list[str]]] = []

    class PrimarySearchProvider:
        name = "primary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            calls.append((self.name, queries.copy()))
            return {
                "searches": [
                    {
                        "query": query,
                        "results": [{"title": f"{query} title", "url": f"https://{query}.example"}],
                    }
                    if query == "q1"
                    else {"query": query, "results": []}
                    for query in queries
                ]
            }

    class SecondarySearchProvider:
        name = "secondary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            calls.append((self.name, queries.copy()))
            return {
                "searches": [
                    {
                        "query": query,
                        "results": [
                            {"title": f"{query} recovered", "url": f"https://{query}.fallback"}
                        ],
                    }
                    for query in queries
                ]
            }

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_search_provider",
        lambda provider_name, providers: {
            "primary_search": PrimarySearchProvider,
            "secondary_search": SecondarySearchProvider,
        }[provider_name],
    )

    payload = await run_search_chain(["q1", "q2"], _build_config(), verbose=False)

    assert calls == [
        ("primary_search", ["q1", "q2"]),
        ("secondary_search", ["q2"]),
    ]
    assert payload["searches"][0]["results"][0]["url"] == "https://q1.example"
    assert payload["searches"][1]["results"][0]["url"] == "https://q2.fallback"
    assert payload["provider_failures"][0]["provider"] == "primary_search"
    assert payload["provider_failures"][0]["failed_items"] == ["q2"]
    assert payload["provider_failures"][0]["failure_kind"] == "empty_results"


@pytest.mark.asyncio
async def test_run_search_chain_all_empty_results_return_clear_error(monkeypatch) -> None:
    """All-empty provider chains end with a clear no-results error."""

    class EmptySearchProvider:
        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            return {"searches": [{"query": query, "results": []} for query in queries]}

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_search_provider",
        lambda provider_name, providers: EmptySearchProvider,
    )

    payload = await run_search_chain(["q1"], _build_config(), verbose=False)

    assert payload["searches"][0]["results"] == []
    assert payload["searches"][0]["error"] == "No results found across configured search providers"
    assert [item["failure_kind"] for item in payload["provider_failures"]] == [
        "empty_results",
        "empty_results",
    ]
    assert payload["provider_failures"][0]["failed_items"] == ["q1"]
    assert payload["provider_failures"][1]["failed_items"] == ["q1"]


@pytest.mark.asyncio
async def test_run_fetch_chain_partial_failover(monkeypatch) -> None:
    """Successful fetches stay while failed URLs move to the next provider."""
    calls: list[tuple[str, list[str]]] = []

    class PrimaryFetchProvider:
        name = "primary_fetch"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def fetch(
            self,
            urls: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            calls.append((self.name, urls.copy()))
            return {
                "pages": [
                    {"url": url, "content": f"content for {url}"}
                    if url == "https://ok.example"
                    else {"url": url, "content": "", "error": "primary failed"}
                    for url in urls
                ]
            }

    class SecondaryFetchProvider:
        name = "secondary_fetch"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def fetch(
            self,
            urls: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            calls.append((self.name, urls.copy()))
            return {"pages": [{"url": url, "content": f"fallback for {url}"} for url in urls]}

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_fetch_provider",
        lambda provider_name, providers: {
            "primary_fetch": PrimaryFetchProvider,
            "secondary_fetch": SecondaryFetchProvider,
        }[provider_name],
    )

    payload = await run_fetch_chain(
        ["https://ok.example", "https://retry.example"],
        _build_config(),
        verbose=False,
    )

    assert calls == [
        ("primary_fetch", ["https://ok.example", "https://retry.example"]),
        ("primary_fetch", ["https://retry.example"]),
        ("secondary_fetch", ["https://retry.example"]),
    ]
    assert payload["pages"][0]["content"] == "content for https://ok.example"
    assert payload["pages"][1]["content"] == "fallback for https://retry.example"
    assert payload["provider_failures"][0]["provider"] == "primary_fetch"


@pytest.mark.asyncio
async def test_run_fetch_chain_suppresses_provider_chatter_when_quiet(
    monkeypatch,
    capsys,
) -> None:
    """Quiet fetch runs suppress provider stdout/stderr chatter."""

    class NoisyFetchProvider:
        name = "primary_fetch"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def fetch(
            self,
            urls: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            print("provider-stdout")
            print("provider-stderr", file=sys.stderr)
            return {"pages": [{"url": url, "content": "quiet content"} for url in urls]}

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_fetch_provider",
        lambda provider_name, providers: NoisyFetchProvider,
    )

    config = _build_config(fetch_backends=["primary_fetch"])
    payload = await run_fetch_chain(["https://example.com"], config, verbose=False)

    captured = capsys.readouterr()

    assert payload["pages"][0]["content"] == "quiet content"
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.asyncio
async def test_run_fetch_chain_keeps_provider_chatter_when_verbose(
    monkeypatch,
    capsys,
) -> None:
    """Verbose fetch runs keep provider chatter visible for debugging."""

    class NoisyFetchProvider:
        name = "primary_fetch"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def fetch(
            self,
            urls: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            print("provider-stdout")
            print("provider-stderr", file=sys.stderr)
            return {"pages": [{"url": url, "content": "loud content"} for url in urls]}

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_fetch_provider",
        lambda provider_name, providers: NoisyFetchProvider,
    )

    config = _build_config(fetch_backends=["primary_fetch"])
    payload = await run_fetch_chain(["https://example.com"], config, verbose=True)

    captured = capsys.readouterr()

    assert payload["pages"][0]["content"] == "loud content"
    assert "provider-stdout" in captured.out
    assert "provider-stderr" in captured.err


@pytest.mark.asyncio
async def test_run_llm_chain_immediate_failover(monkeypatch) -> None:
    """LLM orchestration immediately falls through on hard failure."""
    calls: list[str] = []

    class PrimaryLLMProvider:
        name = "primary_llm"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            config: dict[str, Any],
            verbose: bool,
            max_tokens: int,
        ) -> Any:
            calls.append(self.name)
            raise RuntimeError("model not found")

    class SecondaryLLMProvider:
        name = "secondary_llm"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            config: dict[str, Any],
            verbose: bool,
            max_tokens: int,
        ) -> Any:
            calls.append(self.name)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                usage=None,
            )

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_llm_provider",
        lambda provider_name, providers: {
            "primary_llm": PrimaryLLMProvider,
            "secondary_llm": SecondaryLLMProvider,
        }[provider_name],
    )

    response = await run_llm_chain(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        config=_build_config(),
        verbose=False,
        max_tokens=100,
    )

    assert calls == ["primary_llm", "secondary_llm"]
    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_provider_config_validation_failure(monkeypatch) -> None:
    """Validation failures are captured as provider failures."""

    class InvalidSearchProvider:
        name = "primary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            raise ValueError("missing api_key")

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            raise AssertionError("search should not run when validation fails")

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_search_provider",
        lambda provider_name, providers: InvalidSearchProvider,
    )

    config = _build_config(search_backends=["primary_search"])
    payload = await run_search_chain(["q1"], config, verbose=False)

    assert payload["searches"][0]["error"] == "All configured search providers failed"
    assert payload["provider_failures"][0]["stage"] == "validate"
    assert payload["provider_failures"][0]["error"] == "missing api_key"
    assert payload["provider_failures"][0]["failure_kind"] == "validation_error"


@pytest.mark.asyncio
async def test_run_llm_chain_raises_when_all_providers_fail(monkeypatch) -> None:
    """LLM orchestration raises ProviderChainError after total failure."""

    class BrokenLLMProvider:
        name = "broken"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            config: dict[str, Any],
            verbose: bool,
            max_tokens: int,
        ) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_llm_provider",
        lambda provider_name, providers: BrokenLLMProvider,
    )

    config = _build_config(llm_backends=["primary_llm"])

    with pytest.raises(ProviderChainError, match="All configured llm providers failed"):
        await run_llm_chain(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            config=config,
            verbose=False,
            max_tokens=10,
        )


# ---------------------------------------------------------------------------
# Multi-key (per-key fallback) orchestration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_search_chain_retries_second_key_before_next_provider(monkeypatch) -> None:
    """Search orchestrator retries second key in the same provider before the next.

    provider_primary has api_key = ["bad", "good"] - first key fails, second succeeds.
    provider_secondary must NOT be called.
    """
    key_attempts: list[str] = []

    class MultiKeySearchProvider:
        name = "primary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            key_attempts.append(config.get("api_key", "none"))
            if config.get("api_key") == "good":
                return {
                    "searches": [
                        {"query": q, "results": [{"title": "ok", "url": "https://ok"}]}
                        for q in queries
                    ]
                }
            return {
                "searches": [
                    {"query": q, "results": [], "error": "key rejected"} for q in queries
                ]
            }

    class SecondarySearchProvider:
        name = "secondary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            raise AssertionError("Secondary provider should not be called")

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_search_provider",
        lambda provider_name, providers: {
            "primary_search": MultiKeySearchProvider,
            "secondary_search": SecondarySearchProvider,
        }[provider_name],
    )

    config = _build_config(
        search_backends=["primary_search", "secondary_search"],
        search_provider_retries=0,
        providers={
            "primary_search": {
                "type": "primary_search",
                "api_key": ["bad", "good"],
            },
            "secondary_search": {
                "type": "secondary_search",
            },
        },
    )

    payload = await run_search_chain(["q1"], config, verbose=False)

    # With search_provider_retries=0, "bad" key gets 1 attempt, then "good" key succeeds
    assert key_attempts == ["bad", "good"]
    assert payload["searches"][0]["results"][0]["url"] == "https://ok"
    assert len(payload["provider_failures"]) > 0
    for failure in payload["provider_failures"]:
        assert "api_key" not in failure


@pytest.mark.asyncio
async def test_run_fetch_chain_retries_second_key_before_next_provider(monkeypatch) -> None:
    """Fetch orchestrator retries second key in the same provider before the next.

    provider_primary has api_key = ["bad", "good"] - first key fails, second succeeds.
    provider_secondary must NOT be called.
    """
    key_attempts: list[str] = []

    class MultiKeyFetchProvider:
        name = "primary_fetch"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def fetch(
            self,
            urls: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            key_attempts.append(config.get("api_key", "none"))
            if config.get("api_key") == "good":
                return {"pages": [{"url": url, "content": f"content for {url}"} for url in urls]}
            return {"pages": [{"url": url, "content": "", "error": "key rejected"} for url in urls]}

    class SecondaryFetchProvider:
        name = "secondary_fetch"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def fetch(
            self,
            urls: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            raise AssertionError("Secondary provider should not be called")

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_fetch_provider",
        lambda provider_name, providers: {
            "primary_fetch": MultiKeyFetchProvider,
            "secondary_fetch": SecondaryFetchProvider,
        }[provider_name],
    )

    config = _build_config(
        fetch_backends=["primary_fetch", "secondary_fetch"],
        fetch_provider_retries=0,
        providers={
            "primary_fetch": {
                "type": "primary_fetch",
                "api_key": ["bad", "good"],
            },
            "secondary_fetch": {
                "type": "secondary_fetch",
            },
        },
    )

    payload = await run_fetch_chain(["https://example.com"], config, verbose=False)

    # With fetch_provider_retries=0, "bad" key gets 1 attempt, then "good" key succeeds
    assert key_attempts == ["bad", "good"]
    assert payload["pages"][0]["content"] == "content for https://example.com"
    for failure in payload["provider_failures"]:
        assert "api_key" not in failure


@pytest.mark.asyncio
async def test_run_llm_chain_retries_second_key_before_next_provider(monkeypatch) -> None:
    """LLM orchestrator retries second key in the same provider before the next.

    provider_primary has api_key = ["bad", "good"] - first key raises, second succeeds.
    provider_secondary must NOT be called.
    """
    key_attempts: list[str] = []

    class MultiKeyLLMProvider:
        name = "primary_llm"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            config: dict[str, Any],
            verbose: bool,
            max_tokens: int,
        ) -> Any:
            key_attempts.append(config.get("api_key", "none"))
            if config.get("api_key") == "good":
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
                    usage=None,
                )
            raise RuntimeError("bad key")

    class SecondaryLLMProvider:
        name = "secondary_llm"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            config: dict[str, Any],
            verbose: bool,
            max_tokens: int,
        ) -> Any:
            raise AssertionError("Secondary provider should not be called")

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_llm_provider",
        lambda provider_name, providers: {
            "primary_llm": MultiKeyLLMProvider,
            "secondary_llm": SecondaryLLMProvider,
        }[provider_name],
    )

    config = _build_config(
        llm_backends=["primary_llm", "secondary_llm"],
        providers={
            "primary_llm": {
                "type": "primary_llm",
                "base_url": "https://api.test.com/v1",
                "api_key": ["bad", "good"],
                "model": "model",
            },
            "secondary_llm": {
                "type": "secondary_llm",
                "base_url": "https://api.test.com/v1",
                "api_key": "key",
                "model": "model",
            },
        },
    )

    response = await run_llm_chain(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        config=config,
        verbose=False,
        max_tokens=100,
    )

    assert key_attempts == ["bad", "good"]
    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_run_search_chain_api_key_exhausted_failure_kind(monkeypatch) -> None:
    """When all keys fail for a provider, failure_kind includes api_key_exhausted."""
    key_attempts: list[str] = []

    class AlwaysFailingSearchProvider:
        name = "primary_search"

        def validate_config(self, config: dict[str, Any]) -> None:
            return None

        async def search(
            self,
            queries: list[str],
            config: dict[str, Any],
            timeout: float,
            verbose: bool,
        ) -> dict[str, Any]:
            key_attempts.append(config.get("api_key", "none"))
            return {
                "searches": [
                    {"query": q, "results": [], "error": "always fails"} for q in queries
                ]
            }

    monkeypatch.setattr(
        "nexi.backends.orchestrators.resolve_search_provider",
        lambda provider_name, providers: AlwaysFailingSearchProvider,
    )

    config = _build_config(
        search_backends=["primary_search"],
        search_provider_retries=0,
        providers={
            "primary_search": {
                "type": "primary_search",
                "api_key": ["k1", "k2"],
            },
        },
    )

    payload = await run_search_chain(["q1"], config, verbose=False)

    # Both keys should have been exhausted
    failure_kinds = [f["failure_kind"] for f in payload["provider_failures"]]
    assert "api_key_exhausted" in failure_kinds
    # No raw key in failure metadata
    for failure in payload["provider_failures"]:
        assert "api_key" not in failure
