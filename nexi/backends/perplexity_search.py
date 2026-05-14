"""Perplexity Search API provider for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.api_keys import normalize_api_keys, validate_api_keys
from nexi.backends.http_client import get_http_client

BASE_URL = "https://api.perplexity.ai/search"


class PerplexitySearchProvider:
    """Perplexity Search API provider."""

    name = "perplexity_search"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Perplexity search config."""
        validate_api_keys(config, "Perplexity")
        if not normalize_api_keys(config):
            raise ValueError("Perplexity api_key must be a non-empty string or list")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Perplexity Search API."""
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
    """Execute a single Perplexity search."""
    payload: dict[str, Any] = {
        "query": query,
        "max_results": config.get("max_results", 5),
    }
    for key in ("search_domain_filter", "search_recency_filter", "country", "language"):
        if key in config:
            payload[key] = config[key]

    if verbose:
        print(f"  [Perplexity Search] Query: {query}")

    try:
        response = await client.post(
            BASE_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
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

    raw_results = data.get("results", []) or data.get("citations", [])
    results = []
    for item in raw_results:
        if isinstance(item, str):
            results.append(
                {
                    "title": item,
                    "url": item,
                    "description": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "description": item.get("snippet") or item.get("text") or "",
                **({"published_time": item["date"]} if isinstance(item.get("date"), str) else {}),
            }
        )
    return {"query": query, "results": results}


__all__ = ["PerplexitySearchProvider"]
