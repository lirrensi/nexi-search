"""Crawl4AI fetch provider for NEXI."""

# FILE: nexi/backends/crawl4ai.py
# PURPOSE: Adapt Crawl4AI into NEXI fetch payloads while keeping non-verbose runs quiet.
# OWNS: Crawl4AI-backed fetch configuration, execution, fallback handling, and noise suppression.
# EXPORTS: Crawl4AIFetchProvider
# DOCS: agent_chat/plan_crawl4ai_quiet_2026-04-24.md

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import os
import warnings
from html import unescape
from typing import Any

import httpx
from bs4 import BeautifulSoup

from nexi.backends.http_client import get_http_client


class Crawl4AIFetchProvider:
    """Optional local fetch provider backed by Crawl4AI."""

    name = "crawl4ai"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate Crawl4AI provider config."""
        browser_type = config.get("browser_type")
        if browser_type is not None and not isinstance(browser_type, str):
            raise ValueError("crawl4ai browser_type must be a string")

        cdp_url = config.get("cdp_url")
        if cdp_url is not None and not isinstance(cdp_url, str):
            raise ValueError("crawl4ai cdp_url must be a string")

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
        with _crawl4ai_noise_suppressed(verbose):
            crawl4ai = _import_crawl4ai()
            browser_config = _build_browser_config(crawl4ai, config, verbose)
            run_config = _build_run_config(crawl4ai, config, verbose)

            pages = []
            async with crawl4ai.AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls:
                    if verbose:
                        print(f"  [Crawl4AI] URL: {url}")
                    try:
                        result = await crawler.arun(url=url, config=run_config)
                    except Exception as exc:
                        fallback_page = await _fetch_via_http(url, timeout, verbose)
                        pages.append(
                            fallback_page
                            if fallback_page.get("content")
                            else {"url": url, "content": "", "error": str(exc)}
                        )
                        continue

                    page = _result_to_page(url, result)
                    if page.get("error"):
                        fallback_page = await _fetch_via_http(url, timeout, verbose)
                        pages.append(fallback_page if fallback_page.get("content") else page)
                        continue

                    pages.append(page)

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


def _build_browser_config(crawl4ai: Any, config: dict[str, Any], verbose: bool) -> Any:
    """Build BrowserConfig when available."""
    browser_config_class = getattr(crawl4ai, "BrowserConfig", None)
    if browser_config_class is None:
        return None

    kwargs: dict[str, Any] = {}
    if isinstance(config.get("browser_type"), str) and config["browser_type"].strip():
        kwargs["browser_type"] = config["browser_type"]
    if isinstance(config.get("cdp_url"), str) and config["cdp_url"].strip():
        kwargs["cdp_url"] = config["cdp_url"].strip()
    if isinstance(config.get("headless"), bool):
        kwargs["headless"] = config["headless"]

    if _supports_kwarg(browser_config_class, "verbose"):
        kwargs["verbose"] = verbose

    return browser_config_class(**kwargs)


def _build_run_config(crawl4ai: Any, config: dict[str, Any], verbose: bool) -> Any:
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

    if _supports_kwarg(run_config_class, "verbose"):
        kwargs["verbose"] = verbose

    return run_config_class(**kwargs)


def _supports_kwarg(callable_obj: Any, kwarg_name: str) -> bool:
    """Return True when a callable accepts the named keyword argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == kwarg_name:
            return True

    return False


@contextlib.contextmanager
def _crawl4ai_noise_suppressed(verbose: bool):
    """Silence Crawl4AI chatter for quiet runs without changing results."""
    if verbose:
        yield
        return

    os.environ.setdefault("NODE_NO_WARNINGS", "1")
    previous_disable_level = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with (
                open(os.devnull, "w", encoding="utf-8") as devnull,
                contextlib.redirect_stdout(devnull),
                contextlib.redirect_stderr(devnull),
            ):
                yield
    finally:
        logging.disable(previous_disable_level)


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


async def _fetch_via_http(url: str, timeout: float, verbose: bool) -> dict[str, Any]:
    """Fallback to direct HTTP fetch when browser crawling fails."""
    client = get_http_client(timeout=timeout)

    if verbose:
        print(f"  [Crawl4AI HTTP Fallback] URL: {url}")

    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "url": url,
            "content": "",
            "error": f"HTTP {exc.response.status_code}: {exc.response.text}",
        }
    except Exception as exc:
        return {"url": url, "content": "", "error": str(exc)}

    content = _html_to_text(response.text)
    if not content.strip():
        return {"url": url, "content": "", "error": "No content returned"}

    return {"url": url, "content": content}


def _html_to_text(html: str) -> str:
    """Convert HTML to readable text."""
    soup = BeautifulSoup(unescape(html), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


__all__ = ["Crawl4AIFetchProvider"]
