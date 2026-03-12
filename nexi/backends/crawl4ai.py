"""Crawl4AI fetch provider for NEXI."""

from __future__ import annotations

import importlib
from typing import Any


class Crawl4AIFetchProvider:
    """Optional local fetch provider backed by Crawl4AI."""

    name = "crawl4ai"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Crawl4AI provider config."""
        browser_type = config.get("browser_type")
        if browser_type is not None and not isinstance(browser_type, str):
            raise ValueError("crawl4ai browser_type must be a string")

        headless = config.get("headless")
        if headless is not None and not isinstance(headless, bool):
            raise ValueError("crawl4ai headless must be a boolean")

        cache_mode = config.get("cache_mode")
        if cache_mode is not None and not isinstance(cache_mode, str):
            raise ValueError("crawl4ai cache_mode must be a string")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content from Crawl4AI."""
        crawl4ai = _import_crawl4ai()
        browser_config = _build_browser_config(crawl4ai, config)
        run_config = _build_run_config(crawl4ai, config)

        pages = []
        async with crawl4ai.AsyncWebCrawler(config=browser_config) as crawler:
            for url in urls:
                if verbose:
                    print(f"  [Crawl4AI] URL: {url}")
                try:
                    result = await crawler.arun(url=url, config=run_config)
                except Exception as exc:
                    pages.append({"url": url, "content": "", "error": str(exc)})
                    continue
                pages.append(_result_to_page(url, result))

        return {"pages": pages}


def _import_crawl4ai() -> Any:
    """Import Crawl4AI lazily so it remains optional."""
    try:
        return importlib.import_module("crawl4ai")
    except ImportError as exc:
        raise RuntimeError(
            "Crawl4AI is not installed. Install the optional 'crawl4ai' package to use "
            "the local crawl4ai fetch provider."
        ) from exc


def _build_browser_config(crawl4ai: Any, config: dict[str, Any]) -> Any:
    """Build BrowserConfig when available."""
    browser_config_class = getattr(crawl4ai, "BrowserConfig", None)
    if browser_config_class is None:
        return None

    kwargs: dict[str, Any] = {}
    if isinstance(config.get("browser_type"), str) and config["browser_type"].strip():
        kwargs["browser_type"] = config["browser_type"]
    if isinstance(config.get("headless"), bool):
        kwargs["headless"] = config["headless"]

    return browser_config_class(**kwargs)


def _build_run_config(crawl4ai: Any, config: dict[str, Any]) -> Any:
    """Build CrawlerRunConfig when available."""
    run_config_class = getattr(crawl4ai, "CrawlerRunConfig", None)
    if run_config_class is None:
        return None

    kwargs: dict[str, Any] = {}
    cache_mode_name = config.get("cache_mode")
    cache_mode_enum = getattr(crawl4ai, "CacheMode", None)
    if isinstance(cache_mode_name, str) and cache_mode_name.strip() and cache_mode_enum is not None:
        enum_value = getattr(cache_mode_enum, cache_mode_name, None)
        if enum_value is not None:
            kwargs["cache_mode"] = enum_value

    return run_config_class(**kwargs)


def _result_to_page(url: str, result: Any) -> dict[str, Any]:
    """Convert a Crawl4AI result object into NEXI page payload."""
    success = getattr(result, "success", True)
    if success is False:
        error_message = getattr(result, "error_message", None) or "Crawl4AI fetch failed"
        return {"url": url, "content": "", "error": str(error_message)}

    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, str):
        content = markdown
    else:
        content = (
            getattr(markdown, "raw_markdown", None)
            or getattr(markdown, "fit_markdown", None)
            or getattr(result, "cleaned_html", None)
            or getattr(result, "html", None)
            or ""
        )

    if not isinstance(content, str) or not content.strip():
        return {"url": url, "content": "", "error": "No content returned"}

    return {"url": url, "content": content}


__all__ = ["Crawl4AIFetchProvider"]
