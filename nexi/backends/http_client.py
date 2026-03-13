"""Shared HTTP client cache for timeout-sensitive providers."""

from __future__ import annotations

from collections.abc import Awaitable

import httpx

_HTTP_CLIENTS: dict[float, httpx.AsyncClient] = {}
_HTTP_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)


def get_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Get or create a shared HTTP client for one timeout value."""
    normalized_timeout = float(timeout)
    client = _HTTP_CLIENTS.get(normalized_timeout)
    if client is None:
        client = httpx.AsyncClient(timeout=normalized_timeout, limits=_HTTP_LIMITS)
        _HTTP_CLIENTS[normalized_timeout] = client
    return client


def close_http_client() -> Awaitable[None]:
    """Close all cached HTTP clients and clear the cache."""

    async def _close_all() -> None:
        clients = list(_HTTP_CLIENTS.values())
        _HTTP_CLIENTS.clear()
        for client in clients:
            await client.aclose()

    return _close_all()


__all__ = ["close_http_client", "get_http_client"]
