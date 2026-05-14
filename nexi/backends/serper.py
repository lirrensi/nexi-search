"""Serper search provider for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.api_keys import normalize_api_keys, validate_api_keys
from nexi.backends.http_client import get_http_client

BASE_URL = "https://google.serper.dev/search"


class SerperSearchProvider:
    """Serper Google search provider."""

    name = "serper"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Serper search config."""
        validate_api_keys(config, "Serper")
        if not normalize_api_keys(config):
            raise ValueError("Serper api_key must be a non-empty string or list")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Serper."""
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
    """Execute a single Serper search."""
    payload: dict[str, Any] = {
        "q": query,
        "num": config.get("num", 5),
    }
    for key in ("gl", "hl", "location", "page", "autocorrect"):
        if key in config:
            payload[key] = config[key]

    if verbose:
        print(f"  [Serper Search] Query: {query}")

    try:
        response = await client.post(
            BASE_URL,
            json=payload,
            headers={
                "X-API-KEY": config["api_key"],
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

    organic = data.get("organic", [])
    results = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title") or item.get("link") or "Untitled",
                "url": item.get("link", ""),
                "description": item.get("snippet") or "",
                **({"published_time": item["date"]} if isinstance(item.get("date"), str) else {}),
            }
        )
    return {"query": query, "results": results}


__all__ = ["SerperSearchProvider"]
