"""Tests for HTTP-backed provider adapters."""

from __future__ import annotations

from typing import Any

import pytest

from nexi.backends.brave import BraveSearchProvider
from nexi.backends.exa import ExaFetchProvider, ExaSearchProvider
from nexi.backends.firecrawl import FirecrawlFetchProvider, FirecrawlSearchProvider
from nexi.backends.linkup import LinkupFetchProvider, LinkupSearchProvider
from nexi.backends.perplexity_search import PerplexitySearchProvider
from nexi.backends.searxng import SearXNGSearchProvider
from nexi.backends.serpapi import SerpAPISearchProvider
from nexi.backends.serper import SerperSearchProvider
from nexi.backends.tavily import TavilyFetchProvider, TavilySearchProvider


class FakeResponse:
    """Minimal fake HTTP response."""

    def __init__(
        self,
        data: dict[str, Any],
        *,
        text: str = "",
        status_code: int = 200,
    ) -> None:
        self._data = data
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Keep successful response behavior."""
        return None

    def json(self) -> dict[str, Any]:
        """Return the stored JSON payload."""
        return self._data


class FakeHttpClient:
    """Queue-based fake HTTP client."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, Any],
    ) -> FakeResponse:
        """Record POST calls and return the next fake response."""
        self.calls.append(("POST", url, json, headers))
        return self.responses.pop(0)

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        headers: dict[str, Any] | None = None,
    ) -> FakeResponse:
        """Record GET calls and return the next fake response."""
        self.calls.append(("GET", url, params, headers or {}))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_tavily_providers_normalize_search_and_fetch(monkeypatch) -> None:
    """Tavily adapters normalize search results and extract content."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "A",
                            "url": "https://a.example",
                            "content": "Snippet",
                            "published_date": "2026-03-10",
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "results": [
                        {
                            "url": "https://a.example",
                            "raw_content": "# A",
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr("nexi.backends.tavily.get_http_client", lambda timeout=30.0: client)

    search_provider = TavilySearchProvider()
    fetch_provider = TavilyFetchProvider()

    search_payload = await search_provider.search(
        ["alpha"],
        {"api_key": "key"},
        timeout=5,
        verbose=False,
    )
    fetch_payload = await fetch_provider.fetch(
        ["https://a.example"],
        {"api_key": "key"},
        timeout=5,
        verbose=False,
    )

    assert search_payload["searches"][0]["results"][0]["url"] == "https://a.example"
    assert fetch_payload == {"pages": [{"url": "https://a.example", "content": "# A"}]}


@pytest.mark.asyncio
async def test_exa_providers_normalize_search_and_fetch(monkeypatch) -> None:
    """Exa adapters normalize search results and contents."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "B",
                            "url": "https://b.example",
                            "text": "Semantic snippet",
                            "publishedDate": "2026-03-11",
                        }
                    ]
                }
            ),
            FakeResponse(
                {
                    "results": [
                        {
                            "url": "https://b.example",
                            "text": "# B",
                        }
                    ]
                }
            ),
        ]
    )
    monkeypatch.setattr("nexi.backends.exa.get_http_client", lambda timeout=30.0: client)

    search_payload = await ExaSearchProvider().search(["beta"], {"api_key": "key"}, 5, False)
    fetch_payload = await ExaFetchProvider().fetch(
        ["https://b.example"], {"api_key": "key"}, 5, False
    )

    assert search_payload["searches"][0]["results"][0]["description"] == "Semantic snippet"
    assert fetch_payload == {"pages": [{"url": "https://b.example", "content": "# B"}]}


@pytest.mark.asyncio
async def test_firecrawl_providers_normalize_search_and_fetch(monkeypatch) -> None:
    """Firecrawl adapters normalize search results and scraped markdown."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "data": {
                        "web": [
                            {
                                "title": "C",
                                "url": "https://c.example",
                                "description": "Crawler result",
                            }
                        ]
                    }
                }
            ),
            FakeResponse({"data": {"markdown": "# C"}}),
        ]
    )
    monkeypatch.setattr("nexi.backends.firecrawl.get_http_client", lambda timeout=30.0: client)

    search_payload = await FirecrawlSearchProvider().search(["gamma"], {"api_key": "key"}, 5, False)
    fetch_payload = await FirecrawlFetchProvider().fetch(
        ["https://c.example"],
        {"api_key": "key"},
        5,
        False,
    )

    assert search_payload["searches"][0]["results"][0]["title"] == "C"
    assert fetch_payload == {"pages": [{"url": "https://c.example", "content": "# C"}]}


@pytest.mark.asyncio
async def test_linkup_providers_normalize_search_and_fetch(monkeypatch) -> None:
    """Linkup adapters normalize search results and fetch content."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "name": "D",
                            "url": "https://d.example",
                            "content": "Linkup snippet",
                        }
                    ]
                }
            ),
            FakeResponse({"content": "# D"}),
        ]
    )
    monkeypatch.setattr("nexi.backends.linkup.get_http_client", lambda timeout=30.0: client)

    search_payload = await LinkupSearchProvider().search(["delta"], {"api_key": "key"}, 5, False)
    fetch_payload = await LinkupFetchProvider().fetch(
        ["https://d.example"], {"api_key": "key"}, 5, False
    )

    assert search_payload["searches"][0]["results"][0]["title"] == "D"
    assert fetch_payload == {"pages": [{"url": "https://d.example", "content": "# D"}]}


@pytest.mark.asyncio
async def test_brave_search_provider_normalizes_results(monkeypatch) -> None:
    """Brave search adapter normalizes web results."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "web": {
                        "results": [
                            {
                                "title": "E",
                                "url": "https://e.example",
                                "description": "Brave snippet",
                                "age": "2026-03-12",
                            }
                        ]
                    }
                }
            )
        ]
    )
    monkeypatch.setattr("nexi.backends.brave.get_http_client", lambda timeout=30.0: client)

    payload = await BraveSearchProvider().search(["epsilon"], {"api_key": "key"}, 5, False)

    assert payload["searches"][0]["results"][0]["url"] == "https://e.example"


@pytest.mark.asyncio
async def test_serpapi_search_provider_normalizes_results(monkeypatch) -> None:
    """SerpAPI search adapter normalizes organic results."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "organic_results": [
                        {
                            "title": "F",
                            "link": "https://f.example",
                            "snippet": "Serp snippet",
                            "date": "Mar 12, 2026",
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr("nexi.backends.serpapi.get_http_client", lambda timeout=30.0: client)

    payload = await SerpAPISearchProvider().search(["zeta"], {"api_key": "key"}, 5, False)

    assert payload["searches"][0]["results"][0]["title"] == "F"


@pytest.mark.asyncio
async def test_serper_search_provider_normalizes_results(monkeypatch) -> None:
    """Serper search adapter normalizes organic results."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "organic": [
                        {
                            "title": "G",
                            "link": "https://g.example",
                            "snippet": "Serper snippet",
                            "date": "Mar 12, 2026",
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr("nexi.backends.serper.get_http_client", lambda timeout=30.0: client)

    payload = await SerperSearchProvider().search(["eta"], {"api_key": "key"}, 5, False)

    assert payload["searches"][0]["results"][0]["url"] == "https://g.example"


@pytest.mark.asyncio
async def test_perplexity_search_provider_normalizes_results(monkeypatch) -> None:
    """Perplexity search adapter normalizes search results."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "H",
                            "url": "https://h.example",
                            "snippet": "Perplexity snippet",
                            "date": "2026-03-12",
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr(
        "nexi.backends.perplexity_search.get_http_client",
        lambda timeout=30.0: client,
    )

    payload = await PerplexitySearchProvider().search(
        ["theta"],
        {"api_key": "key"},
        5,
        False,
    )

    assert payload["searches"][0]["results"][0]["title"] == "H"


@pytest.mark.asyncio
async def test_searxng_search_provider_normalizes_results_and_params(monkeypatch) -> None:
    """SearXNG search adapter normalizes results and passes query params."""
    client = FakeHttpClient(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "title": "I",
                            "url": "https://i.example",
                            "content": "SearXNG snippet",
                            "publishedDate": "2026-03-13",
                            "engine": "google",
                        }
                    ]
                }
            )
        ]
    )
    monkeypatch.setattr("nexi.backends.searxng.get_http_client", lambda timeout=30.0: client)

    payload = await SearXNGSearchProvider().search(
        ["iota"],
        {
            "base_url": "https://search.example.org",
            "categories": ["general", "science"],
            "engines": "google,bing",
            "language": "en",
            "safesearch": 1,
            "pageno": 2,
            "format": "json",
        },
        5,
        False,
    )

    assert client.calls[0][1] == "https://search.example.org/search"
    assert client.calls[0][2]["q"] == "iota"
    assert client.calls[0][2]["format"] == "json"
    assert client.calls[0][2]["categories"] == "general,science"
    assert client.calls[0][2]["engines"] == "google,bing"
    assert client.calls[0][2]["language"] == "en"
    assert client.calls[0][2]["safesearch"] == 1
    assert client.calls[0][2]["pageno"] == 2
    assert payload["searches"][0]["results"][0]["description"] == "SearXNG snippet"
    assert payload["searches"][0]["results"][0]["engine"] == "google"


def test_searxng_search_provider_rejects_missing_base_url() -> None:
    """SearXNG search config requires a base_url."""
    with pytest.raises(ValueError, match="base_url"):
        SearXNGSearchProvider().validate_config({"format": "json"})
