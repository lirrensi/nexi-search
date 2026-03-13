"""Tests for shared HTTP client caching."""

from __future__ import annotations

import pytest

from nexi.backends import http_client as http_client_module


class FakeAsyncClient:
    """Minimal async client test double."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    async def aclose(self) -> None:
        """Record closure calls."""
        self.closed = True


@pytest.mark.asyncio
async def test_get_http_client_caches_per_timeout_and_closes_all(monkeypatch) -> None:
    """Different timeout values get isolated cached clients."""
    await http_client_module.close_http_client()
    monkeypatch.setattr(http_client_module.httpx, "AsyncClient", FakeAsyncClient)

    client_5_a = http_client_module.get_http_client(timeout=5)
    client_30 = http_client_module.get_http_client(timeout=30)
    client_5_b = http_client_module.get_http_client(timeout=5)

    assert client_5_a is client_5_b
    assert client_5_a is not client_30

    await http_client_module.close_http_client()

    assert client_5_a.closed is True
    assert client_30.closed is True

    client_5_c = http_client_module.get_http_client(timeout=5)

    assert client_5_c is not client_5_a

    await http_client_module.close_http_client()
