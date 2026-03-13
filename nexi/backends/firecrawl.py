"""Firecrawl provider adapters for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.http_client import get_http_client

BASE_URL = "https://api.firecrawl.dev/v2"


class FirecrawlSearchProvider:
    """Firecrawl search provider."""

    name = "firecrawl"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Firecrawl search config."""
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Firecrawl api_key must be a non-empty string")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Firecrawl."""
        client = get_http_client(timeout=timeout)
        searches = []
        for query in queries:
            searches.append(await _search_single(client, query, config, verbose))
        return {"searches": searches}


class FirecrawlFetchProvider:
    """Firecrawl scrape provider."""

    name = "firecrawl"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Firecrawl fetch config."""
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Firecrawl api_key must be a non-empty string")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content via Firecrawl scrape."""
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
    """Execute a single Firecrawl search."""
    payload: dict[str, Any] = {
        "query": query,
        "limit": config.get("limit", 5),
    }
    if "lang" in config:
        payload["lang"] = config["lang"]
    if "country" in config:
        payload["country"] = config["country"]
    if "scrape_options" in config:
        payload["scrapeOptions"] = config["scrape_options"]

    if verbose:
        print(f"  [Firecrawl Search] Query: {query}")

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

    container = data.get("data", {})
    raw_results = []
    if isinstance(container, dict):
        if isinstance(container.get("web"), list):
            raw_results = container["web"]
        elif isinstance(container.get("results"), list):
            raw_results = container["results"]

    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "description": item.get("description") or item.get("markdown") or "",
            }
        )
    return {"query": query, "results": results}


async def _fetch_single(
    client: httpx.AsyncClient,
    url: str,
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Fetch a single page via Firecrawl scrape."""
    payload: dict[str, Any] = {
        "url": url,
        "formats": config.get("formats", ["markdown"]),
        "onlyMainContent": config.get("only_main_content", True),
    }

    if verbose:
        print(f"  [Firecrawl Scrape] URL: {url}")

    try:
        response = await client.post(
            f"{BASE_URL}/scrape",
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

    result_data = data.get("data", {})
    content = ""
    if isinstance(result_data, dict):
        content = (
            result_data.get("markdown")
            or result_data.get("content")
            or result_data.get("html")
            or ""
        )
    if not isinstance(content, str) or not content.strip():
        return {"url": url, "content": "", "error": "No content returned"}
    return {"url": url, "content": content}


def _headers(api_key: str) -> dict[str, str]:
    """Build Firecrawl headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


__all__ = ["FirecrawlFetchProvider", "FirecrawlSearchProvider"]
