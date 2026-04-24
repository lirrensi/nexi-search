"""Resilient special fetch providers for NEXI."""

# FILE: nexi/backends/special_fetch.py
# PURPOSE: Provide resilient trafilatura and Playwright-backed fetch adapters.
# OWNS: Special fetch provider implementations and their local fallback text extraction.
# EXPORTS: SpecialTrafilaturaFetchProvider, SpecialPlaywrightFetchProvider
# DOCS: agent_chat/plan_fetch_resilience_2026-04-24.md

from __future__ import annotations

import html
import os
from html.parser import HTMLParser
from typing import Any

import httpx

from nexi.backends.http_client import get_http_client


class SpecialTrafilaturaFetchProvider:
    """Resilient fetch provider backed by trafilatura when available."""

    name = "special_trafilatura"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate special trafilatura config."""
        pass

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content with Trafilatura-first extraction and text fallback."""
        client = get_http_client(timeout=timeout)
        pages = []
        for url in urls:
            pages.append(await _fetch_with_trafilatura(client, url, verbose))
        return {"pages": pages}


class SpecialPlaywrightFetchProvider:
    """Headed Playwright fetch provider with rendered text extraction."""

    name = "special_playwright"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate special Playwright config."""
        pass

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content using headed Playwright and rendered text extraction."""
        async_playwright = _import_async_playwright()
        if not verbose:
            os.environ.setdefault("NODE_NO_WARNINGS", "1")
        pages = []

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=False,
                args=[
                    "--start-minimized",
                    "--disable-infobars",
                    "--disable-popup-blocking",
                    "--no-default-browser-check",
                ],
            )
            try:
                context = await browser.new_context(viewport={"width": 1280, "height": 720})
                try:
                    for url in urls:
                        try:
                            pages.append(
                                await _fetch_with_playwright_page(
                                    context,
                                    url,
                                    timeout,
                                    verbose,
                                )
                            )
                        except Exception as exc:
                            pages.append({"url": url, "content": "", "error": str(exc)})
                finally:
                    await context.close()
            finally:
                await browser.close()

        return {"pages": pages}


async def _fetch_with_trafilatura(
    client: httpx.AsyncClient,
    url: str,
    verbose: bool,
) -> dict[str, Any]:
    """Fetch one URL with trafilatura extraction and resilient fallbacks."""
    if verbose:
        print(f"  [Special Trafilatura] URL: {url}")

    try:
        response = await client.get(url, follow_redirects=True)
    except Exception as exc:
        return {"url": url, "content": "", "error": str(exc)}

    raw_text = response.text or ""
    if response.status_code >= 400:
        fallback_text = _best_effort_text(raw_text)
        if fallback_text:
            return {"url": url, "content": fallback_text}

        if raw_text.strip():
            return {"url": url, "content": raw_text.strip()}

        return {"url": url, "content": "", "error": "No content returned"}

    extracted = _extract_trafilatura_text(raw_text, url)
    if extracted:
        return {"url": url, "content": extracted}

    fallback_text = _best_effort_text(raw_text)
    if fallback_text:
        return {"url": url, "content": fallback_text}

    if raw_text.strip():
        return {"url": url, "content": raw_text.strip()}

    return {"url": url, "content": "", "error": "No content returned"}


async def _fetch_with_playwright_page(
    context: Any,
    url: str,
    timeout: float,
    verbose: bool,
) -> dict[str, Any]:
    """Fetch one URL with a headed Playwright browser page."""
    if verbose:
        print(f"  [Special Playwright] URL: {url}")

    page = await context.new_page()
    navigation_error: Exception | None = None
    try:
        try:
            await page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))
        except Exception as exc:
            navigation_error = exc

        rendered_text = await _extract_playwright_text(page)
        if rendered_text:
            return {"url": url, "content": rendered_text}

        if navigation_error is not None:
            return {"url": url, "content": "", "error": str(navigation_error)}

        return {"url": url, "content": "", "error": "No content returned"}
    finally:
        await page.close()


def _extract_trafilatura_text(raw_text: str, url: str) -> str:
    """Extract text via Trafilatura when the optional dependency is available."""
    try:
        import trafilatura
    except ImportError:
        return ""

    try:
        extracted = trafilatura.extract(
            raw_text,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:
        return ""

    return extracted.strip() if isinstance(extracted, str) else ""


async def _extract_playwright_text(page: Any) -> str:
    """Extract rendered text from a Playwright page without returning HTML."""
    candidates: list[str] = []
    for selector in ("main", "article", "[role='main']", "body"):
        try:
            locator = page.locator(selector)
            text = await locator.inner_text()
        except Exception:
            continue

        normalized = _normalize_whitespace(text)
        if normalized:
            candidates.append(normalized)

    if not candidates:
        return ""

    best_text = max(candidates, key=len)
    if len(best_text) >= 120:
        return best_text

    try:
        title = _normalize_whitespace(await page.title())
    except Exception:
        title = ""

    if title and title != best_text:
        return "\n\n".join(part for part in (title, best_text) if part)

    return best_text


def _best_effort_text(raw_text: str) -> str:
    """Convert HTML-ish content to readable text as a fallback."""
    if not raw_text:
        return ""

    parser = _TextExtractor()
    parser.feed(html.unescape(raw_text))
    parser.close()
    return _normalize_whitespace(parser.get_text())


def _import_async_playwright() -> Any:
    """Import async Playwright lazily so the provider remains optional."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Install the optional 'playwright' package to use "
            "the special_playwright fetch provider."
        ) from exc

    return async_playwright


def _normalize_whitespace(text: str) -> str:
    """Collapse excess whitespace and trim text."""
    return " ".join(text.split()).strip()


class _TextExtractor(HTMLParser):
    """Lightweight HTML-to-text extractor for fallback rendering."""

    _BLOCK_TAGS = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


__all__ = [
    "SpecialPlaywrightFetchProvider",
    "SpecialTrafilaturaFetchProvider",
]
