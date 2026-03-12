"""Tests for concrete fetch provider adapters."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nexi.backends.crawl4ai import Crawl4AIFetchProvider
from nexi.backends.markdown_new import MarkdownNewFetchProvider


class _FakeResponse:
    """Minimal fake HTTP response."""

    def __init__(
        self,
        text: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {"content-type": "text/markdown"}

    def raise_for_status(self) -> None:
        """Raise nothing for successful responses."""
        return None

    def json(self) -> dict[str, str]:
        """Return a JSON payload when requested."""
        return {"markdown": self.text}


class _FakeHttpClient:
    """Minimal fake HTTP client."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, str]]] = []

    async def post(
        self,
        url: str,
        json: dict[str, str],
        headers: dict[str, str],
    ) -> _FakeResponse:
        """Record calls and return a fake response."""
        self.calls.append((url, json, headers))
        return _FakeResponse("# Example")


@pytest.mark.asyncio
async def test_markdown_new_fetch_provider_uses_post_api(monkeypatch) -> None:
    """markdown.new fetch uses the HTTP POST API and returns markdown."""
    client = _FakeHttpClient()
    monkeypatch.setattr("nexi.backends.markdown_new.get_http_client", lambda timeout=30.0: client)

    provider = MarkdownNewFetchProvider()
    payload = await provider.fetch(
        ["https://example.com"],
        {"type": "markdown_new", "method": "browser", "retain_images": False},
        timeout=5,
        verbose=False,
    )

    assert payload == {"pages": [{"url": "https://example.com", "content": "# Example"}]}
    assert client.calls == [
        (
            "https://markdown.new/",
            {
                "url": "https://example.com",
                "method": "browser",
                "images": "false",
            },
            {"Accept": "text/markdown"},
        )
    ]


@pytest.mark.asyncio
async def test_crawl4ai_fetch_provider_returns_markdown(monkeypatch) -> None:
    """Crawl4AI fetch provider converts crawl results into NEXI pages."""

    class FakeCrawler:
        """Async crawler test double."""

        def __init__(self, config: object = None) -> None:
            self.config = config

        async def __aenter__(self) -> FakeCrawler:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def arun(self, url: str, config: object = None) -> object:
            return SimpleNamespace(success=True, markdown=SimpleNamespace(raw_markdown="# Local"))

    class FakeBrowserConfig:
        """Browser config test double."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeRunConfig:
        """Run config test double."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_module = ModuleType("crawl4ai")
    fake_module.AsyncWebCrawler = FakeCrawler
    fake_module.BrowserConfig = FakeBrowserConfig
    fake_module.CrawlerRunConfig = FakeRunConfig
    fake_module.CacheMode = SimpleNamespace(BYPASS="BYPASS")
    monkeypatch.setitem(sys.modules, "crawl4ai", fake_module)

    provider = Crawl4AIFetchProvider()
    payload = await provider.fetch(
        ["https://example.com"],
        {"type": "crawl4ai", "headless": True, "cache_mode": "BYPASS"},
        timeout=5,
        verbose=False,
    )

    assert payload == {"pages": [{"url": "https://example.com", "content": "# Local"}]}


@pytest.mark.asyncio
async def test_crawl4ai_fetch_provider_missing_dependency_is_explicit(monkeypatch) -> None:
    """Missing Crawl4AI dependency yields a clear runtime error."""
    monkeypatch.delitem(sys.modules, "crawl4ai", raising=False)
    monkeypatch.setattr(
        "nexi.backends.crawl4ai.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing crawl4ai")),
    )

    provider = Crawl4AIFetchProvider()

    with pytest.raises(RuntimeError, match="Crawl4AI is not installed"):
        await provider.fetch(
            ["https://example.com"],
            {"type": "crawl4ai"},
            timeout=5,
            verbose=False,
        )
