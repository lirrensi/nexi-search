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

from nexi.backends.api_keys import build_api_key_attempt_configs
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
    attempt_key: str = "",
) -> dict[str, Any]:
    """Build structured provider failure metadata.

    *attempt_key* is a non-secret label such as ``"key_1"``, ``"key_2"``
    that identifies which API-key attempt failed.  It is omitted when empty.
    """
    result: dict[str, Any] = {
        "capability": capability,
        "provider": provider_name,
        "provider_type": provider_type,
        "failed_items": items,
        "error": error,
        "stage": stage,
        "attempts": attempts,
        "failure_kind": failure_kind,
    }
    if attempt_key:
        result["attempt_key"] = attempt_key
    return result


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


async def _run_search_attempt(
    provider: Any,
    config: Config,
    provider_name: str,
    provider_type: str | None,
    retry_queries: list[str],
    attempt_config: dict[str, Any],
    key_idx: int,
    verbose: bool,
    query_had_empty_results: set[str],
    results_by_query: dict[str, dict[str, Any]],
    provider_failures: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Execute one per-key search attempt with the same-provider retry loop.

    Returns ``(retryable_queries, empty_queries)`` for the next key attempt.
    Preserves successful queries in *results_by_query* and appends failure
    metadata to *provider_failures*.
    """
    empty_queries: list[str] = []
    empty_failure_attempt = 0
    final_error = f"Search provider '{provider_name}' failed"
    attempts_made = 0
    max_attempts = config.search_provider_retries + 1
    remaining = retry_queries.copy()

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            with _quiet_provider_io(verbose):
                payload = await provider.search(
                    remaining,
                    attempt_config,
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

        retryable: list[str] = []
        for query in remaining:
            item = searches_by_query.get(query)
            if item is None:
                retryable.append(query)
                continue

            item_failure_kind = _search_item_failure_kind(item)
            if item_failure_kind == "empty_results":
                empty_queries.append(query)
                query_had_empty_results.add(query)
                if empty_failure_attempt == 0:
                    empty_failure_attempt = attempt
                continue

            if item_failure_kind == "provider_error":
                retryable.append(query)
                continue

            results_by_query[query] = {
                "query": query,
                "results": item.get("results", []),
                **({"error": item["error"]} if "error" in item else {}),
            }

        if not retryable:
            return [], empty_queries

        final_error = _summarize_item_errors(
            searches_by_query,
            retryable,
            f"Search provider '{provider_name}' failed",
        )
        remaining = retryable
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
                attempt_key=f"key_{key_idx + 1}",
            )
        )

    provider_failures.append(
        _provider_failure(
            "search",
            provider_name,
            provider_type,
            remaining,
            final_error,
            "execute",
            attempts_made,
            "provider_error",
            attempt_key=f"key_{key_idx + 1}",
        )
    )

    return remaining, empty_queries


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

        attempt_configs = build_api_key_attempt_configs(provider_config, provider_name)
        if not attempt_configs:
            # Zero-key provider: run once with the original config
            attempt_configs = [config.providers[provider_name]]

        remaining_queries = pending_queries.copy()
        last_empty: list[str] = []

        for key_idx, attempt_config in enumerate(attempt_configs):
            if not remaining_queries:
                break

            try:
                provider.validate_config(attempt_config)
            except ValueError as exc:
                provider_failures.append(
                    _provider_failure(
                        "search",
                        provider_name,
                        provider_type,
                        remaining_queries.copy(),
                        str(exc),
                        "validate",
                        0,
                        "validation_error",
                        attempt_key=f"key_{key_idx + 1}",
                    )
                )
                continue

            retryable, empty = await _run_search_attempt(
                provider,
                config,
                provider_name,
                provider_type,
                remaining_queries,
                attempt_config,
                key_idx,
                verbose,
                query_had_empty_results,
                results_by_query,
                provider_failures,
            )
            remaining_queries = retryable + empty
            last_empty = empty

            if not retryable:
                remaining_queries = empty
                break

        if remaining_queries:
            are_only_empty = (remaining_queries == last_empty)

            if are_only_empty:
                # Purely empty-result queries pass to the next provider.
                provider_failures.append(
                    _provider_failure(
                        "search",
                        provider_name,
                        provider_type,
                        remaining_queries.copy(),
                        f"Search provider '{provider_name}' returned no results",
                        "execute",
                        1,
                        "empty_results",
                    )
                )
                pending_queries = remaining_queries
                continue

            # Queries with actual errors remain after all key attempts.
            if len(attempt_configs) > 1:
                provider_failures.append(
                    _provider_failure(
                        "search",
                        provider_name,
                        provider_type,
                        remaining_queries.copy(),
                        f"All API keys exhausted for '{provider_name}'",
                        "execute",
                        1,
                        "api_key_exhausted",
                    )
                )
            pending_queries = remaining_queries
            continue

        pending_queries = last_empty

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


async def _run_fetch_attempt(
    provider: Any,
    config: Config,
    provider_name: str,
    provider_type: str | None,
    remaining_urls: list[str],
    attempt_config: dict[str, Any],
    key_idx: int,
    verbose: bool,
    pages_by_url: dict[str, dict[str, Any]],
    provider_failures: list[dict[str, Any]],
) -> list[str]:
    """Execute one per-key fetch attempt with the same-provider retry loop.

    Returns the list of URLs still unresolved after this attempt.
    Preserves successful pages in *pages_by_url* and appends failure
    metadata to *provider_failures*.
    """
    urls = remaining_urls.copy()
    final_error = f"Fetch provider '{provider_name}' failed"
    attempts_made = 0
    max_attempts = config.fetch_provider_retries + 1

    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            with _quiet_provider_io(verbose):
                payload = await provider.fetch(
                    urls,
                    attempt_config,
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
        for url in urls:
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
            return []

        final_error = _summarize_item_errors(
            pages_by_url_batch,
            failed_urls,
            f"Fetch provider '{provider_name}' failed",
        )
        urls = failed_urls
        if attempt < max_attempts:
            await asyncio.sleep(2 ** (attempt - 1))

    provider_failures.append(
        _provider_failure(
            "fetch",
            provider_name,
            provider_type,
            urls,
            final_error,
            "execute",
            attempts_made,
            "provider_error",
            attempt_key=f"key_{key_idx + 1}",
        )
    )

    return urls


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

        attempt_configs = build_api_key_attempt_configs(provider_config, provider_name)
        if not attempt_configs:
            attempt_configs = [config.providers[provider_name]]

        remaining_urls = pending_urls.copy()

        for key_idx, attempt_config in enumerate(attempt_configs):
            if not remaining_urls:
                break

            try:
                provider.validate_config(attempt_config)
            except ValueError as exc:
                provider_failures.append(
                    _provider_failure(
                        "fetch",
                        provider_name,
                        provider_type,
                        remaining_urls.copy(),
                        str(exc),
                        "validate",
                        0,
                        "validation_error",
                        attempt_key=f"key_{key_idx + 1}",
                    )
                )
                continue

            remaining_urls = await _run_fetch_attempt(
                provider,
                config,
                provider_name,
                provider_type,
                remaining_urls,
                attempt_config,
                key_idx,
                verbose,
                pages_by_url,
                provider_failures,
            )

            if not remaining_urls:
                break

        if remaining_urls:
            provider_failures.append(
                _provider_failure(
                    "fetch",
                    provider_name,
                    provider_type,
                    remaining_urls.copy(),
                    f"All API keys exhausted for '{provider_name}'",
                    "execute",
                    1,
                    "api_key_exhausted",
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

        attempt_configs = build_api_key_attempt_configs(provider_config, provider_name)
        if not attempt_configs:
            attempt_configs = [config.providers[provider_name]]

        all_keys_failed = True

        for key_idx, attempt_config in enumerate(attempt_configs):
            try:
                provider.validate_config(attempt_config)
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
                        attempt_key=f"key_{key_idx + 1}",
                    )
                )
                continue

            try:
                with _quiet_provider_io(verbose):
                    response = await provider.complete(
                        messages=messages,
                        tools=tools,
                        config=attempt_config,
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
                        attempt_key=f"key_{key_idx + 1}",
                    )
                )
                if verbose:
                    print(f"[LLM] Provider '{provider_name}' key_{key_idx + 1} failed: {exc}")
                continue

            all_keys_failed = False
            return response

        if all_keys_failed:
            provider_failures.append(
                _provider_failure(
                    "llm",
                    provider_name,
                    provider_type,
                    [provider_name],
                    f"All API keys exhausted for '{provider_name}'",
                    "execute",
                    1,
                    "api_key_exhausted",
                )
            )
            if verbose:
                print(f"[LLM] All API keys exhausted for '{provider_name}'")

    raise ProviderChainError("llm", provider_failures)


__all__ = [
    "ProviderChainError",
    "run_fetch_chain",
    "run_llm_chain",
    "run_search_chain",
]
