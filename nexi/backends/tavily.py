"""Tavily provider adapters for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.api_keys import normalize_api_keys, validate_api_keys
from nexi.backends.http_client import get_http_client

BASE_URL = "https://api.tavily.com"


class TavilySearchProvider:
    """Tavily search provider."""

    name = "tavily"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Tavily search config."""
        validate_api_keys(config, "Tavily")
        if not normalize_api_keys(config):
            raise ValueError("Tavily api_key must be a non-empty string or list")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Tavily."""
        client = get_http_client(timeout=timeout)
        searches = []
        for query in queries:
            searches.append(await _search_single(client, query, config, verbose))
        return {"searches": searches}


class TavilyFetchProvider:
    """Tavily extract provider."""

    name = "tavily"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Tavily fetch config."""
        validate_api_keys(config, "Tavily")
        if not normalize_api_keys(config):
            raise ValueError("Tavily api_key must be a non-empty string or list")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content via Tavily extract."""
        client = get_http_client(timeout=timeout)
        headers = _headers(config["api_key"])
        payload: dict[str, Any] = {"urls": urls}
        if "extract_depth" in config:
            payload["extract_depth"] = config["extract_depth"]
        if "include_images" in config:
            payload["include_images"] = config["include_images"]

        if verbose:
            print(f"[Tavily Extract] Fetching {len(urls)} URLs")

        try:
            response = await client.post(
                f"{BASE_URL}/extract",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}: {exc.response.text}"
            return {"pages": [{"url": url, "content": "", "error": error} for url in urls]}
        except Exception as exc:
            return {"pages": [{"url": url, "content": "", "error": str(exc)} for url in urls]}

        results = data.get("results", [])
        results_by_url = {
            item.get("url"): item
            for item in results
            if isinstance(item, dict) and isinstance(item.get("url"), str)
        }
        pages = []
        for url in urls:
            item = results_by_url.get(url)
            if item is None:
                pages.append({"url": url, "content": "", "error": "No content returned"})
                continue
            content = item.get("raw_content") or item.get("content") or ""
            if not isinstance(content, str) or not content.strip():
                pages.append({"url": url, "content": "", "error": "No content returned"})
                continue
            pages.append({"url": url, "content": content})
        return {"pages": pages}


async def _search_single(
    client: httpx.AsyncClient,
    query: str,
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Execute a single Tavily search."""
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": config.get("search_depth", "basic"),
        "topic": config.get("topic", "general"),
        "max_results": config.get("max_results", 5),
        "include_raw_content": config.get("include_raw_content", False),
    }
    for key in (
        "include_domains",
        "exclude_domains",
        "country",
        "time_range",
        "start_date",
        "end_date",
        "include_favicon",
        "auto_parameters",
        "exact_match",
    ):
        if key in config:
            payload[key] = config[key]

    if verbose:
        print(f"  [Tavily Search] Query: {query}")

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

    raw_results = data.get("results", [])
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        description = item.get("content") or item.get("raw_content") or ""
        results.append(
            {
                "title": item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "description": description,
                **(
                    {"published_time": item["published_date"]}
                    if isinstance(item.get("published_date"), str)
                    else {}
                ),
            }
        )
    return {"query": query, "results": results}


def _headers(api_key: str) -> dict[str, str]:
    """Build Tavily headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


__all__ = ["TavilyFetchProvider", "TavilySearchProvider"]
