"""Exa provider adapters for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.api_keys import normalize_api_keys, validate_api_keys
from nexi.backends.http_client import get_http_client

BASE_URL = "https://api.exa.ai"


class ExaSearchProvider:
    """Exa search provider."""

    name = "exa"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Exa search config."""
        validate_api_keys(config, "Exa")
        if not normalize_api_keys(config):
            raise ValueError("Exa api_key must be a non-empty string or list")

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via Exa."""
        client = get_http_client(timeout=timeout)
        searches = []
        for query in queries:
            searches.append(await _search_single(client, query, config, verbose))
        return {"searches": searches}


class ExaFetchProvider:
    """Exa contents provider."""

    name = "exa"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Exa fetch config."""
        validate_api_keys(config, "Exa")
        if not normalize_api_keys(config):
            raise ValueError("Exa api_key must be a non-empty string or list")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content via Exa contents."""
        client = get_http_client(timeout=timeout)
        payload: dict[str, Any] = {
            "urls": urls,
            "text": config.get("text", True),
        }
        for key in ("highlights", "summary", "livecrawl"):
            if key in config:
                payload[key] = config[key]

        if verbose:
            print(f"[Exa Contents] Fetching {len(urls)} URLs")

        try:
            response = await client.post(
                f"{BASE_URL}/contents",
                json=payload,
                headers=_headers(config["api_key"]),
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}: {exc.response.text}"
            return {"pages": [{"url": url, "content": "", "error": error} for url in urls]}
        except Exception as exc:
            return {"pages": [{"url": url, "content": "", "error": str(exc)} for url in urls]}

        raw_results = data.get("results", [])
        results_by_url = {
            item.get("url"): item
            for item in raw_results
            if isinstance(item, dict) and isinstance(item.get("url"), str)
        }
        pages = []
        for url in urls:
            item = results_by_url.get(url)
            if item is None:
                pages.append({"url": url, "content": "", "error": "No content returned"})
                continue
            content = item.get("text") or item.get("summary") or ""
            if isinstance(item.get("highlights"), list) and not content:
                content = "\n\n".join(
                    value
                    for value in item["highlights"]
                    if isinstance(value, str) and value.strip()
                )
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
    """Execute a single Exa search."""
    payload: dict[str, Any] = {
        "query": query,
        "numResults": config.get("num_results", 5),
        "type": config.get("search_type", "auto"),
        "contents": {"text": config.get("text", False)},
    }
    for key in (
        "category",
        "includeDomains",
        "excludeDomains",
        "startPublishedDate",
        "endPublishedDate",
    ):
        if key in config:
            payload[key] = config[key]

    if verbose:
        print(f"  [Exa Search] Query: {query}")

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
        description = item.get("text") or item.get("summary") or ""
        results.append(
            {
                "title": item.get("title") or item.get("url") or "Untitled",
                "url": item.get("url", ""),
                "description": description,
                **(
                    {"published_time": item["publishedDate"]}
                    if isinstance(item.get("publishedDate"), str)
                    else {}
                ),
            }
        )
    return {"query": query, "results": results}


def _headers(api_key: str) -> dict[str, str]:
    """Build Exa headers."""
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }


__all__ = ["ExaFetchProvider", "ExaSearchProvider"]
