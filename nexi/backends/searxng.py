"""SearXNG search provider for NEXI."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx

from nexi.backends.http_client import get_http_client

MAX_REQUEST_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {429}


class SearXNGSearchProvider:
    """SearXNG search provider."""

    name = "searxng"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate SearXNG search config."""
        base_url = config.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("SearXNG base_url must be a non-empty string")

        parsed = urlparse(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("SearXNG base_url must be a valid http(s) URL")

        format_value = config.get("format")
        if format_value is not None and format_value != "json":
            raise ValueError("SearXNG format must be 'json' when configured")

        _normalize_csv_param(config.get("engines"), "engines")
        _normalize_csv_param(config.get("categories"), "categories")
        _normalize_optional_string(config.get("language"), "language")
        _normalize_optional_int(config.get("pageno"), "pageno", minimum=1)
        _normalize_optional_int(config.get("safesearch"), "safesearch", minimum=0, maximum=2)
        _normalize_optional_choice(
            config.get("time_range"),
            "time_range",
            {"day", "month", "year"},
        )

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute search via SearXNG."""
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
    """Execute a single SearXNG search."""
    params = _build_params(query, config)
    search_url = _build_search_url(str(config["base_url"]))

    if verbose:
        print(f"  [SearXNG Search] Query: {query}")
        print(f"  [SearXNG Search] URL: {search_url}")
        print(f"  [SearXNG Search] Params: {params}")

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            return _normalize_search_response(query, data)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error_msg = f"HTTP {status_code}: {exc.response.text}"
            if _is_retryable_status(status_code) and attempt < MAX_REQUEST_ATTEMPTS:
                await asyncio.sleep(2 ** (attempt - 1))
                continue
            return {"query": query, "results": [], "error": error_msg}
        except httpx.RequestError as exc:
            error_msg = str(exc)
            if attempt < MAX_REQUEST_ATTEMPTS:
                await asyncio.sleep(2 ** (attempt - 1))
                continue
            return {"query": query, "results": [], "error": error_msg}
        except ValueError as exc:
            return {"query": query, "results": [], "error": f"Invalid JSON response: {exc}"}
        except Exception as exc:
            return {"query": query, "results": [], "error": str(exc)}

    return {"query": query, "results": [], "error": "SearXNG request failed"}


def _build_search_url(base_url: str) -> str:
    """Build the search endpoint URL from a configured base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/search"):
        return normalized
    return f"{normalized}/search"


def _build_params(query: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build SearXNG query parameters."""
    params: dict[str, Any] = {"q": query, "format": "json"}

    optional_values = {
        "engines": _normalize_csv_param(config.get("engines"), "engines"),
        "categories": _normalize_csv_param(config.get("categories"), "categories"),
        "language": _normalize_optional_string(config.get("language"), "language"),
        "pageno": _normalize_optional_int(config.get("pageno"), "pageno", minimum=1),
        "time_range": _normalize_optional_choice(
            config.get("time_range"),
            "time_range",
            {"day", "month", "year"},
        ),
        "safesearch": _normalize_optional_int(
            config.get("safesearch"), "safesearch", minimum=0, maximum=2
        ),
    }

    for key, value in optional_values.items():
        if value is not None:
            params[key] = value

    return params


def _normalize_search_response(query: str, data: Any) -> dict[str, Any]:
    """Normalize a SearXNG response into NEXI's canonical search shape."""
    if not isinstance(data, dict):
        return {"query": query, "results": [], "error": "SearXNG response must be an object"}

    results = data.get("results")
    if not isinstance(results, list):
        return {"query": query, "results": [], "error": "SearXNG response missing results"}

    normalized_results: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue

        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            continue

        title = item.get("title") or url or "Untitled"
        description = item.get("content") or item.get("snippet") or item.get("description") or ""

        normalized_item: dict[str, Any] = {
            "title": title,
            "url": url,
            "description": description if isinstance(description, str) else str(description),
        }

        published_time = item.get("publishedDate") or item.get("published_time") or item.get("age")
        if isinstance(published_time, str) and published_time.strip():
            normalized_item["published_time"] = published_time

        engine = item.get("engine")
        if isinstance(engine, str) and engine.strip():
            normalized_item["engine"] = engine

        normalized_results.append(normalized_item)

    return {"query": query, "results": normalized_results}


def _is_retryable_status(status_code: int) -> bool:
    """Return True for retryable HTTP status codes."""
    return status_code in RETRYABLE_STATUS_CODES or 500 <= status_code < 600


def _normalize_csv_param(value: Any, field_name: str) -> str | None:
    """Normalize a string-or-list config value into a CSV string."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"SearXNG {field_name} must be a non-empty string or list of strings")
        return stripped
    if isinstance(value, (list, tuple)):
        normalized_values = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"SearXNG {field_name} must contain only non-empty strings")
            normalized_values.append(item.strip())
        if not normalized_values:
            raise ValueError(f"SearXNG {field_name} must not be empty")
        return ",".join(normalized_values)
    raise ValueError(f"SearXNG {field_name} must be a string or list of strings")


def _normalize_optional_string(value: Any, field_name: str) -> str | None:
    """Normalize an optional string field."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SearXNG {field_name} must be a non-empty string")
    return value.strip()


def _normalize_optional_int(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Normalize an optional bounded integer field."""
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"SearXNG {field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"SearXNG {field_name} must be greater than or equal to {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"SearXNG {field_name} must be less than or equal to {maximum}")
    return value


def _normalize_optional_choice(value: Any, field_name: str, allowed: set[str]) -> str | None:
    """Normalize an optional string field with a fixed set of allowed values."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SearXNG {field_name} must be a non-empty string")
    normalized = value.strip()
    if normalized not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"SearXNG {field_name} must be one of: {options}")
    return normalized


__all__ = ["SearXNGSearchProvider"]
