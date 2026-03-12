"""Jina provider adapters for NEXI."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, cast

import httpx

_url_cache: dict[str, str] = {}
_http_client: httpx.AsyncClient | None = None


def clear_url_cache() -> None:
    """Clear the in-memory URL cache."""
    _url_cache.clear()


def get_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Get or create the shared HTTP client."""
    global _http_client
    if _http_client is None:
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        _http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


class JinaSearchProvider:
    """Jina AI search provider."""

    name = "jina"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Jina search config."""
        api_key = config.get("api_key", "")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("Jina api_key must be a string")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Jina AI."""
        if verbose:
            print(f"[Jina Search] Starting {len(queries)} parallel searches...")

        api_key = str(config.get("api_key", "") or "")
        client = get_http_client(timeout=timeout)
        tasks = [_search_single(client, query, api_key, verbose) for query in queries]
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[dict[str, Any]] = []
        for query, result in zip(queries, search_results, strict=False):
            if isinstance(result, BaseException):
                error_msg = str(result)
                if verbose:
                    print(f"  [Jina Search] Exception: {error_msg}")
                results.append({"query": query, "error": error_msg, "results": []})
                continue
            results.append(cast(dict[str, Any], result))

        if verbose:
            print(f"[Jina Search] Completed {len(results)} searches")

        return {"searches": results}


class JinaFetchProvider:
    """Jina AI fetch provider."""

    name = "jina"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Jina fetch config."""
        api_key = config.get("api_key", "")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("Jina api_key must be a string")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content via Jina Reader."""
        if verbose:
            print(f"[Jina Reader] Starting {len(urls)} parallel fetches...")

        api_key = str(config.get("api_key", "") or "")
        client = get_http_client(timeout=timeout)
        urls_to_fetch = [url for url in urls if url not in _url_cache]

        for url in urls:
            if verbose and url in _url_cache:
                print(f"  [Jina Reader] Cache hit for: {url}")

        fetch_results = await asyncio.gather(
            *[_get_single(client, url, api_key, verbose) for url in urls_to_fetch],
            return_exceptions=True,
        )

        fetched_by_url: dict[str, dict[str, Any]] = {}
        for url, result in zip(urls_to_fetch, fetch_results, strict=False):
            if isinstance(result, BaseException):
                error_msg = str(result)
                if verbose:
                    print(f"  [Jina Reader] Exception: {error_msg}")
                fetched_by_url[url] = {"url": url, "error": error_msg, "content": ""}
                continue

            response_payload = cast(dict[str, Any], result)
            raw_content = response_payload.get("content", "")
            if raw_content and "error" not in response_payload:
                _url_cache[url] = raw_content
            fetched_by_url[url] = response_payload

        pages: list[dict[str, Any]] = []
        for url in urls:
            if url in fetched_by_url:
                pages.append(fetched_by_url[url])
            elif url in _url_cache:
                pages.append({"url": url, "content": _url_cache[url]})
            else:
                pages.append({"url": url, "error": "No content returned", "content": ""})

        if verbose:
            print(f"[Jina Reader] Completed {len(pages)} fetches")

        return {"pages": pages}


def _parse_search_response(raw_text: str) -> list[dict[str, Any]]:
    """Parse Jina search text response into structured results."""
    results: list[dict[str, Any]] = []
    current_result: dict[str, Any] = {}

    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("[") and "] Title:" in stripped:
            if current_result:
                results.append(current_result)
            current_result = {}
            title_part = stripped.split("] Title: ", 1)
            if len(title_part) > 1:
                current_result["title"] = title_part[1].strip()
        elif "] URL Source:" in stripped:
            url_part = stripped.split("] URL Source: ", 1)
            if len(url_part) > 1:
                current_result["url"] = url_part[1].strip()
        elif "] Description:" in stripped:
            desc_part = stripped.split("] Description: ", 1)
            if len(desc_part) > 1:
                current_result["description"] = desc_part[1].strip()
        elif "] Published Time:" in stripped:
            time_part = stripped.split("] Published Time: ", 1)
            if len(time_part) > 1:
                current_result["published_time"] = time_part[1].strip()

    if current_result:
        results.append(current_result)

    return results


async def _search_single(
    client: httpx.AsyncClient,
    query: str,
    api_key: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a single search query."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if verbose:
        print(f"  [Jina Search] Query: {query}")
        print(f"  [Jina Search] URL: https://s.jina.ai/?q={query}")
        print(f"  [Jina Search] Headers: {headers}")

    response = None
    try:
        response = await client.get("https://s.jina.ai/", params={"q": query}, headers=headers)
        response.raise_for_status()
        raw_text = response.text

        if verbose:
            print(f"  [Jina Search] Status: {response.status_code}")
            print(f"  [Jina Search] Raw response (first 500 chars): {raw_text[:500]}")

        try:
            data = response.json()
        except Exception:
            if verbose:
                print("  [Jina Search] Parsing as text format")
            return {"query": query, "results": _parse_search_response(raw_text)}

        if verbose:
            print("  [Jina Search] Parsed as JSON")
        return {"query": query, "results": data if isinstance(data, list) else [data]}
    except httpx.HTTPStatusError as exc:
        error_msg = f"HTTP {exc.response.status_code}: {exc.response.text}"
        if verbose:
            print(f"  [Jina Search] ERROR: {error_msg}")
        return {"query": query, "error": error_msg, "results": []}
    except Exception as exc:
        error_msg = str(exc)
        if verbose:
            print(f"  [Jina Search] ERROR: {error_msg}")
            if response is not None:
                with contextlib.suppress(BaseException):
                    print(f"  [Jina Search] Raw response: {response.text[:500]}")
        return {"query": query, "error": error_msg, "results": []}


async def _get_single(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fetch content from a single URL."""
    headers = {
        "X-Retain-Images": "none",
        "X-Retain-Links": "gpt-oss",
        "X-Timeout": "40",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if verbose:
        print(f"  [Jina Reader] URL: {url}")
        print(f"  [Jina Reader] Headers: {headers}")

    response = None
    try:
        response = await client.get(f"https://r.jina.ai/{url}", headers=headers)
        response.raise_for_status()
        if verbose:
            print(f"  [Jina Reader] Status: {response.status_code}")
            print(f"  [Jina Reader] Content length: {len(response.text)} chars")
        return {"url": url, "content": response.text}
    except httpx.HTTPStatusError as exc:
        error_msg = f"HTTP {exc.response.status_code}: {exc.response.text}"
        if verbose:
            print(f"  [Jina Reader] ERROR: {error_msg}")
        return {"url": url, "error": error_msg, "content": ""}
    except Exception as exc:
        error_msg = str(exc)
        if verbose:
            print(f"  [Jina Reader] ERROR: {error_msg}")
            if response is not None:
                with contextlib.suppress(BaseException):
                    print(f"  [Jina Reader] Raw response: {response.text[:500]}")
        return {"url": url, "error": error_msg, "content": ""}


__all__ = [
    "JinaFetchProvider",
    "JinaSearchProvider",
    "clear_url_cache",
    "close_http_client",
    "get_http_client",
]
