"""Linkup provider adapters for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.http_client import get_http_client

BASE_URL = "https://api.linkup.so/v1"


class LinkupSearchProvider:
    """Linkup search provider."""

    name = "linkup"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Linkup search config."""
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Linkup api_key must be a non-empty string")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Linkup."""
        client = get_http_client(timeout=timeout)
        searches = []
        for query in queries:
            searches.append(await _search_single(client, query, config, verbose))
        return {"searches": searches}


class LinkupFetchProvider:
    """Linkup fetch provider."""

    name = "linkup"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Linkup fetch config."""
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Linkup api_key must be a non-empty string")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content via Linkup."""
        client = get_http_client(timeout=timeout)
        pages = []
        for url in urls:
            pages.append(await _fetch_single(client, url, config, verbose))
        return {"pages": pages}


async def _search_single(
    client: httpx.AsyncClient,
    query: str,
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Execute a single Linkup search."""
    payload: dict[str, Any] = {
        "q": query,
        "depth": config.get("depth", "standard"),
        "outputType": config.get("output_type", "searchResults"),
    }
    if "includeDomains" in config:
        payload["includeDomains"] = config["includeDomains"]
    if "excludeDomains" in config:
        payload["excludeDomains"] = config["excludeDomains"]

    if verbose:
        print(f"  [Linkup Search] Query: {query}")

    try:
        response = await client.post(
            f"{BASE_URL}/search",
            json=payload,
            headers=_headers(config["api_key"]),
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        return {
            "query": query,
            "results": [],
            "error": f"HTTP {exc.response.status_code}: {exc.response.text}",
        }
    except Exception as exc:
        return {"query": query, "results": [], "error": str(exc)}

    raw_results = data.get("results", []) or data.get("sources", [])
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("name") or item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "description": item.get("content") or item.get("snippet") or "",
            }
        )
    return {"query": query, "results": results}


async def _fetch_single(
    client: httpx.AsyncClient,
    url: str,
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Fetch a single page via Linkup."""
    payload: dict[str, Any] = {"url": url}
    if "includeImages" in config:
        payload["includeImages"] = config["includeImages"]

    if verbose:
        print(f"  [Linkup Fetch] URL: {url}")

    try:
        response = await client.post(
            f"{BASE_URL}/fetch",
            json=payload,
            headers=_headers(config["api_key"]),
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        return {
            "url": url,
            "content": "",
            "error": f"HTTP {exc.response.status_code}: {exc.response.text}",
        }
    except Exception as exc:
        return {"url": url, "content": "", "error": str(exc)}

    content = data.get("content") or data.get("markdown") or ""
    if not isinstance(content, str) or not content.strip():
        return {"url": url, "content": "", "error": "No content returned"}
    return {"url": url, "content": content}


def _headers(api_key: str) -> dict[str, str]:
    """Build Linkup headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


__all__ = ["LinkupFetchProvider", "LinkupSearchProvider"]
