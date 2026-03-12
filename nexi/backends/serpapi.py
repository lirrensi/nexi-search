"""SerpAPI search provider for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.jina import get_http_client

BASE_URL = "https://serpapi.com/search.json"


class SerpAPISearchProvider:
    """SerpAPI search provider."""

    name = "serpapi"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate SerpAPI search config."""
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("SerpAPI api_key must be a non-empty string")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via SerpAPI."""
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
    """Execute a single SerpAPI search."""
    params: dict[str, Any] = {
        "q": query,
        "api_key": config["api_key"],
        "engine": config.get("engine", "google"),
    }
    for key in ("gl", "hl", "location", "num"):
        if key in config:
            params[key] = config[key]

    if verbose:
        print(f"  [SerpAPI Search] Query: {query}")

    try:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        return {"query": query, "results": [], "error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:
        return {"query": query, "results": [], "error": str(exc)}

    raw_results = data.get("organic_results", [])
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title") or item.get("link") or "Untitled",
                "url": item.get("link", ""),
                "description": item.get("snippet") or "",
                **(
                    {"published_time": item["date"]}
                    if isinstance(item.get("date"), str)
                    else {}
                ),
            }
        )
    return {"query": query, "results": results}


__all__ = ["SerpAPISearchProvider"]
