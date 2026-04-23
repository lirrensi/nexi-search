"""Tests for special resilient fetch providers."""

from __future__ import annotations

from typing import Any

import pytest

from nexi.backends.special_fetch import (
    SpecialPlaywrightFetchProvider,
    SpecialTrafilaturaFetchProvider,
)


class FakeResponse:
    """Minimal fake HTTP response."""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class FakeHttpClient:
    """Queue-based fake HTTP client."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str, *, follow_redirects: bool = True) -> FakeResponse:
        """Record GET calls and return the next fake response."""
        self.calls.append((url, {"follow_redirects": follow_redirects}))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_special_trafilatura_falls_back_to_best_text(monkeypatch) -> None:
    """Trafilatura provider falls back when extraction is empty or blocked."""
    client = FakeHttpClient([FakeResponse("<html><body><h1>Blocked page</h1></body></html>", 403)])
    monkeypatch.setattr(
        "nexi.backends.special_fetch.get_http_client",
        lambda timeout=30.0: client,
    )
    monkeypatch.setattr("nexi.backends.special_fetch._extract_trafilatura_text", lambda *args: "")

    payload = await SpecialTrafilaturaFetchProvider().fetch(
        ["https://blocked.example"],
        {},
        timeout=5,
        verbose=False,
    )

    assert payload["pages"][0]["content"] == "Blocked page"


class FakeLocator:
    """Fake Playwright locator."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def inner_text(self) -> str:
        return self.text


class FakePage:
    """Fake Playwright page."""

    def __init__(self) -> None:
        self.goto_calls: list[tuple[str, str, int]] = []

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_calls.append((url, wait_until, timeout))

    def locator(self, selector: str) -> FakeLocator:
        texts = {
            "main": "",
            "article": "Rendered article text with useful details",
            "[role='main']": "",
            "body": "Short body text",
        }
        return FakeLocator(texts.get(selector, ""))

    async def title(self) -> str:
        return "Example title"

    async def close(self) -> None:
        return None


class FakeContext:
    """Fake Playwright browser context."""

    def __init__(self) -> None:
        self.page = FakePage()

    async def new_page(self) -> FakePage:
        return self.page

    async def close(self) -> None:
        return None


class FakeBrowser:
    """Fake Playwright browser."""

    def __init__(self) -> None:
        self.launch_kwargs: dict[str, Any] = {}
        self.context = FakeContext()

    async def new_context(self, **kwargs: Any) -> FakeContext:
        self.launch_kwargs["context_kwargs"] = kwargs
        return self.context

    async def close(self) -> None:
        return None


class FakeChromium:
    """Fake chromium launcher."""

    def __init__(self) -> None:
        self.launch_kwargs: dict[str, Any] = {}
        self.browser = FakeBrowser()

    async def launch(self, **kwargs: Any) -> FakeBrowser:
        self.launch_kwargs = kwargs
        return self.browser


class FakePlaywright:
    """Fake async Playwright object."""

    def __init__(self) -> None:
        self.chromium = FakeChromium()

    async def __aenter__(self) -> FakePlaywright:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeAsyncPlaywrightFactory:
    """Factory returning a fake Playwright context manager."""

    def __init__(self) -> None:
        self.playwright = FakePlaywright()

    def __call__(self) -> FakePlaywright:
        return self.playwright


@pytest.mark.asyncio
async def test_special_playwright_uses_headed_browser_and_rendered_text(monkeypatch) -> None:
    """Playwright provider launches headed and returns rendered text."""
    factory = FakeAsyncPlaywrightFactory()
    monkeypatch.setattr(
        "nexi.backends.special_fetch._import_async_playwright",
        lambda: factory,
    )

    payload = await SpecialPlaywrightFetchProvider().fetch(
        ["https://example.com"],
        {},
        timeout=5,
        verbose=False,
    )

    assert factory.playwright.chromium.launch_kwargs["headless"] is False
    assert "--start-minimized" in factory.playwright.chromium.launch_kwargs["args"]
    assert payload["pages"][0]["content"] == (
        "Example title\n\nRendered article text with useful details"
    )
