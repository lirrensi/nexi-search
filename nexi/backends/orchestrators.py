"""Backend orchestration helpers for NEXI."""

# FILE: nexi/backends/orchestrators.py
# PURPOSE: Route search, fetch, and LLM requests through ordered provider chains.
# OWNS: Provider-chain failover, retry, and failure metadata for backend calls.
# EXPORTS: ProviderChainError, run_search_chain, run_fetch_chain, run_llm_chain
# DOCS: docs/arch.md; docs/product.md

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any

from nexi.backends.registry import (
    resolve_fetch_provider,
    resolve_llm_provider,
    resolve_search_provider,
)
from nexi.config import Config


class ProviderChainError(RuntimeError):
    """Raised when every configured provider for a capability fails."""

    def __init__(self, capability: str, provider_failures: list[dict[str, Any]]) -> None:
        self.capability = capability
        self.provider_failures = provider_failures
        super().__init__(f"All configured {capability} providers failed")


def _provider_type(provider_config: Any) -> str | None:
    """Extract provider type for failure metadata."""
    if isinstance(provider_config, dict):
        provider_type = provider_config.get("type")
        if isinstance(provider_type, str) and provider_type.strip():
            return provider_type
    return None


def _provider_failure(
    capability: str,
    provider_name: str,
    provider_type: str | None,
    items: list[str],
    error: str,
    stage: str,
    attempts: int,
    failure_kind: str,
) -> dict[str, Any]:
    """Build structured provider failure metadata."""
    return {
        "capability": capability,
        "provider": provider_name,
        "provider_type": provider_type,
        "failed_items": items,
        "error": error,
        "stage": stage,
        "attempts": attempts,
        "failure_kind": failure_kind,
    }


def _summarize_item_errors(
    items_by_key: dict[str, dict[str, Any]],
    failed_items: list[str],
    fallback: str,
) -> str:
    """Summarize per-item errors for provider failure metadata."""
    errors = []
    for item_key in failed_items:
        item = items_by_key.get(item_key, {})
        error = item.get("error")
        if isinstance(error, str) and error:
            errors.append(f"{item_key}: {error}")
    return "; ".join(errors) if errors else fallback


@contextlib.contextmanager
def _quiet_provider_io(verbose: bool):
    """Silence provider stdout/stderr unless verbose mode is enabled."""
    if verbose:
        yield
        return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        yield


def _search_item_failure_kind(item: dict[str, Any]) -> str | None:
    """Classify a search item failure."""
    if item.get("error"):
        return "provider_error"

    results = item.get("results")
    if isinstance(results, list) and not results:
        return "empty_results"

    if results is None:
        return "provider_error"

    return None


async def run_search_chain(
    queries: list[str],
    config: Config,
    verbose: bool,
) -> dict[str, Any]:
    """Run the configured search provider chain."""
    ordered_queries = list(dict.fromkeys(queries))
    results_by_query: dict[str, dict[str, Any]] = {}
    pending_queries = ordered_queries.copy()
    provider_failures: list[dict[str, Any]] = []
    query_had_empty_results: set[str] = set()

    for provider_name in config.search_backends:
        if not pending_queries:
            break

        provider_config = config.providers.get(provider_name)
        provider_type = _provider_type(provider_config)

        try:
            provider_class = resolve_search_provider(provider_name, config.providers)
            provider = provider_class()
            provider.validate_config(config.providers[provider_name])
        except ValueError as exc:
            provider_failures.append(
                _provider_failure(
                    "search",
                    provider_name,
                    provider_type,
                    pending_queries.copy(),
                    str(exc),
                    "validate",
                    0,
                    "validation_error",
                )
            )
            continue

        retry_queries = pending_queries.copy()
        empty_queries: list[str] = []
        empty_failure_attempt = 0
        final_error = f"Search provider '{provider_name}' failed"
        attempts_made = 0
        max_attempts = config.search_provider_retries + 1

        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            try:
                with _quiet_provider_io(verbose):
                    payload = await provider.search(
                        retry_queries,
                        config.providers[provider_name],
                        float(config.provider_timeout),
                        verbose,
                    )
            except Exception as exc:
                final_error = str(exc)
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                break

            searches = payload.get("searches")
            if not isinstance(searches, list):
                final_error = "Provider returned invalid search payload"
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                break

            searches_by_query: dict[str, dict[str, Any]] = {}
            for item in searches:
                if not isinstance(item, dict):
                    continue
                item_query = item.get("query")
                if isinstance(item_query, str):
                    searches_by_query[item_query] = item

            retryable_queries: list[str] = []
            for query in retry_queries:
                item = searches_by_query.get(query)
                if item is None:
                    retryable_queries.append(query)
                    continue

                failure_kind = _search_item_failure_kind(item)
                if failure_kind == "empty_results":
                    empty_queries.append(query)
                    query_had_empty_results.add(query)
                    if empty_failure_attempt == 0:
                        empty_failure_attempt = attempt
                    continue

                if failure_kind == "provider_error":
                    retryable_queries.append(query)
                    continue

                results_by_query[query] = {
                    "query": query,
                    "results": item.get("results", []),
                    **({"error": item["error"]} if "error" in item else {}),
                }

            if not retryable_queries:
                retry_queries = []
                break

            final_error = _summarize_item_errors(
                searches_by_query,
                retryable_queries,
                f"Search provider '{provider_name}' failed",
            )
            retry_queries = retryable_queries
            if attempt < max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))

        if empty_queries:
            provider_failures.append(
                _provider_failure(
                    "search",
                    provider_name,
                    provider_type,
                    empty_queries.copy(),
                    f"Search provider '{provider_name}' returned no results",
                    "execute",
                    empty_failure_attempt or attempts_made,
                    "empty_results",
                )
            )

        if retry_queries:
            provider_failures.append(
                _provider_failure(
                    "search",
                    provider_name,
                    provider_type,
                    retry_queries.copy(),
                    final_error,
                    "execute",
                    attempts_made,
                    "provider_error",
                )
            )
            pending_queries = retry_queries + empty_queries
            continue

        pending_queries = empty_queries

    for query in pending_queries:
        final_error = (
            "No results found across configured search providers"
            if query in query_had_empty_results
            else "All configured search providers failed"
        )
        results_by_query[query] = {
            "query": query,
            "results": [],
            "error": final_error,
        }

    return {
        "searches": [
            results_by_query.get(
                query, {"query": query, "results": [], "error": "No result returned"}
            )
            for query in ordered_queries
        ],
        "provider_failures": provider_failures,
    }


async def run_fetch_chain(
    urls: list[str],
    config: Config,
    verbose: bool,
) -> dict[str, Any]:
    """Run the configured fetch provider chain."""
    ordered_urls = list(dict.fromkeys(urls))
    pages_by_url: dict[str, dict[str, Any]] = {}
    pending_urls = ordered_urls.copy()
    provider_failures: list[dict[str, Any]] = []

    for provider_name in config.fetch_backends:
        if not pending_urls:
            break

        provider_config = config.providers.get(provider_name)
        provider_type = _provider_type(provider_config)

        try:
            provider_class = resolve_fetch_provider(provider_name, config.providers)
            provider = provider_class()
            provider.validate_config(config.providers[provider_name])
        except ValueError as exc:
            provider_failures.append(
                _provider_failure(
                    "fetch",
                    provider_name,
                    provider_type,
                    pending_urls.copy(),
                    str(exc),
                    "validate",
                    0,
                    "validation_error",
                )
            )
            continue

        remaining_urls = pending_urls.copy()
        final_error = f"Fetch provider '{provider_name}' failed"
        attempts_made = 0
        max_attempts = config.fetch_provider_retries + 1

        for attempt in range(1, max_attempts + 1):
            attempts_made = attempt
            try:
                with _quiet_provider_io(verbose):
                    payload = await provider.fetch(
                        remaining_urls,
                        config.providers[provider_name],
                        float(config.provider_timeout),
                        verbose,
                    )
            except Exception as exc:
                final_error = str(exc)
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                break

            pages = payload.get("pages")
            if not isinstance(pages, list):
                final_error = "Provider returned invalid fetch payload"
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                break

            pages_by_url_batch: dict[str, dict[str, Any]] = {}
            for item in pages:
                if not isinstance(item, dict):
                    continue
                item_url = item.get("url")
                if isinstance(item_url, str):
                    pages_by_url_batch[item_url] = item

            failed_urls: list[str] = []
            for url in remaining_urls:
                item = pages_by_url_batch.get(url)
                if item is None:
                    failed_urls.append(url)
                    continue
                pages_by_url[url] = {
                    "url": url,
                    "content": item.get("content", ""),
                    **({"error": item["error"]} if "error" in item else {}),
                }
                if item.get("error"):
                    failed_urls.append(url)

            if not failed_urls:
                remaining_urls = []
                break

            final_error = _summarize_item_errors(
                pages_by_url_batch,
                failed_urls,
                f"Fetch provider '{provider_name}' failed",
            )
            remaining_urls = failed_urls
            if attempt < max_attempts:
                await asyncio.sleep(2 ** (attempt - 1))

        if remaining_urls:
            provider_failures.append(
                _provider_failure(
                    "fetch",
                    provider_name,
                    provider_type,
                    remaining_urls.copy(),
                    final_error,
                    "execute",
                    attempts_made,
                    "provider_error",
                )
            )
            pending_urls = remaining_urls
            continue

        pending_urls = []

    for url in pending_urls:
        pages_by_url[url] = {
            "url": url,
            "content": "",
            "error": "All configured fetch providers failed",
        }

    return {
        "pages": [
            pages_by_url.get(url, {"url": url, "content": "", "error": "No result returned"})
            for url in ordered_urls
        ],
        "provider_failures": provider_failures,
    }


async def run_llm_chain(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    config: Config,
    verbose: bool,
    max_tokens: int,
) -> Any:
    """Run the configured LLM provider chain."""
    provider_failures: list[dict[str, Any]] = []

    for provider_name in config.llm_backends:
        provider_config = config.providers.get(provider_name)
        provider_type = _provider_type(provider_config)

        try:
            provider_class = resolve_llm_provider(provider_name, config.providers)
            provider = provider_class()
            provider.validate_config(config.providers[provider_name])
        except ValueError as exc:
            provider_failures.append(
                _provider_failure(
                    "llm",
                    provider_name,
                    provider_type,
                    [provider_name],
                    str(exc),
                    "validate",
                    0,
                    "validation_error",
                )
            )
            continue

        try:
            with _quiet_provider_io(verbose):
                return await provider.complete(
                    messages=messages,
                    tools=tools,
                    config=config.providers[provider_name],
                    verbose=verbose,
                    max_tokens=max_tokens,
                )
        except Exception as exc:
            provider_failures.append(
                _provider_failure(
                    "llm",
                    provider_name,
                    provider_type,
                    [provider_name],
                    str(exc),
                    "execute",
                    1,
                    "provider_error",
                )
            )
            if verbose:
                print(f"[LLM] Provider '{provider_name}' failed: {exc}")

    raise ProviderChainError("llm", provider_failures)


__all__ = [
    "ProviderChainError",
    "run_fetch_chain",
    "run_llm_chain",
    "run_search_chain",
]
