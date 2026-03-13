"""Brave Search provider for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.http_client import get_http_client

BASE_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider:
    """Brave Search API provider."""

    name = "brave"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Brave search config."""
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Brave api_key must be a non-empty string")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Brave."""
        client = get_http_client(timeout=timeout)
        searches = []
        for query in queries:
            searches.append(await _search_single(client, query, config, verbose))
        return {"searches": searches}


async def _search_single(
    client: httpx.AsyncClient,
    query: str,
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Execute a single Brave search."""
    params: dict[str, Any] = {
        "q": query,
        "count": config.get("count", 5),
    }
    if "country" in config:
        params["country"] = config["country"]
    if "search_lang" in config:
        params["search_lang"] = config["search_lang"]
    if "safesearch" in config:
        params["safesearch"] = config["safesearch"]

    if verbose:
        print(f"  [Brave Search] Query: {query}")

    try:
        response = await client.get(
            BASE_URL,
            params=params,
            headers={"X-Subscription-Token": config["api_key"]},
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

    web_results = data.get("web", {}).get("results", [])
    results = []
    for item in web_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "description": item.get("description") or "",
                **({"published_time": item["age"]} if isinstance(item.get("age"), str) else {}),
            }
        )
    return {"query": query, "results": results}


__all__ = ["BraveSearchProvider"]
