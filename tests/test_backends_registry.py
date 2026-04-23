"""Tests for backend registry resolution."""

from __future__ import annotations

import pytest

from nexi.backends.brave import BraveSearchProvider
from nexi.backends.crawl4ai import Crawl4AIFetchProvider
from nexi.backends.exa import ExaFetchProvider, ExaSearchProvider
from nexi.backends.firecrawl import FirecrawlFetchProvider, FirecrawlSearchProvider
from nexi.backends.jina import JinaFetchProvider, JinaSearchProvider
from nexi.backends.linkup import LinkupFetchProvider, LinkupSearchProvider
from nexi.backends.markdown_new import MarkdownNewFetchProvider
from nexi.backends.openai_compatible import OpenAICompatibleLLMProvider
from nexi.backends.perplexity_search import PerplexitySearchProvider
from nexi.backends.registry import (
    resolve_fetch_provider,
    resolve_llm_provider,
    resolve_search_provider,
)
from nexi.backends.searxng import SearXNGSearchProvider
from nexi.backends.serpapi import SerpAPISearchProvider
from nexi.backends.serper import SerperSearchProvider
from nexi.backends.special_fetch import (
    SpecialPlaywrightFetchProvider,
    SpecialTrafilaturaFetchProvider,
)
from nexi.backends.tavily import TavilyFetchProvider, TavilySearchProvider


def test_resolve_search_provider_success() -> None:
    """Search provider resolution returns the registered adapter."""
    provider = resolve_search_provider("jina", {"jina": {"type": "jina"}})

    assert provider is JinaSearchProvider


def test_resolve_fetch_provider_success() -> None:
    """Fetch provider resolution returns the registered adapter."""
    provider = resolve_fetch_provider("jina", {"jina": {"type": "jina"}})

    assert provider is JinaFetchProvider


def test_resolve_tavily_search_provider_success() -> None:
    """Tavily search provider resolves correctly."""
    provider = resolve_search_provider("tavily", {"tavily": {"type": "tavily"}})

    assert provider is TavilySearchProvider


def test_resolve_tavily_fetch_provider_success() -> None:
    """Tavily fetch provider resolves correctly."""
    provider = resolve_fetch_provider("tavily", {"tavily": {"type": "tavily"}})

    assert provider is TavilyFetchProvider


def test_resolve_exa_providers_success() -> None:
    """Exa search and fetch providers resolve correctly."""
    search_provider = resolve_search_provider("exa", {"exa": {"type": "exa"}})
    fetch_provider = resolve_fetch_provider("exa", {"exa": {"type": "exa"}})

    assert search_provider is ExaSearchProvider
    assert fetch_provider is ExaFetchProvider


def test_resolve_firecrawl_providers_success() -> None:
    """Firecrawl search and fetch providers resolve correctly."""
    search_provider = resolve_search_provider("firecrawl", {"firecrawl": {"type": "firecrawl"}})
    fetch_provider = resolve_fetch_provider("firecrawl", {"firecrawl": {"type": "firecrawl"}})

    assert search_provider is FirecrawlSearchProvider
    assert fetch_provider is FirecrawlFetchProvider


def test_resolve_linkup_providers_success() -> None:
    """Linkup search and fetch providers resolve correctly."""
    search_provider = resolve_search_provider("linkup", {"linkup": {"type": "linkup"}})
    fetch_provider = resolve_fetch_provider("linkup", {"linkup": {"type": "linkup"}})

    assert search_provider is LinkupSearchProvider
    assert fetch_provider is LinkupFetchProvider


def test_resolve_brave_and_serpapi_search_providers_success() -> None:
    """Brave and SerpAPI search providers resolve correctly."""
    brave_provider = resolve_search_provider("brave", {"brave": {"type": "brave"}})
    serpapi_provider = resolve_search_provider("serpapi", {"serpapi": {"type": "serpapi"}})

    assert brave_provider is BraveSearchProvider
    assert serpapi_provider is SerpAPISearchProvider


def test_resolve_serper_and_perplexity_search_providers_success() -> None:
    """Serper and Perplexity search providers resolve correctly."""
    serper_provider = resolve_search_provider("serper", {"serper": {"type": "serper"}})
    perplexity_provider = resolve_search_provider(
        "perplexity_search",
        {"perplexity_search": {"type": "perplexity_search"}},
    )

    assert serper_provider is SerperSearchProvider
    assert perplexity_provider is PerplexitySearchProvider


def test_resolve_searxng_search_provider_success() -> None:
    """SearXNG search provider resolves correctly."""
    provider = resolve_search_provider(
        "searxng",
        {"searxng": {"type": "searxng", "base_url": "https://search.example.org"}},
    )

    assert provider is SearXNGSearchProvider


def test_resolve_markdown_new_fetch_provider_success() -> None:
    """markdown.new fetch provider resolves correctly."""
    provider = resolve_fetch_provider(
        "markdown_new",
        {"markdown_new": {"type": "markdown_new"}},
    )

    assert provider is MarkdownNewFetchProvider


def test_resolve_special_trafilatura_fetch_provider_success() -> None:
    """Special Trafilatura fetch provider resolves correctly."""
    provider = resolve_fetch_provider(
        "special_trafilatura",
        {"special_trafilatura": {"type": "special_trafilatura"}},
    )

    assert provider is SpecialTrafilaturaFetchProvider


def test_resolve_special_playwright_fetch_provider_success() -> None:
    """Special Playwright fetch provider resolves correctly."""
    provider = resolve_fetch_provider(
        "special_playwright",
        {"special_playwright": {"type": "special_playwright"}},
    )

    assert provider is SpecialPlaywrightFetchProvider


def test_resolve_crawl4ai_fetch_provider_success() -> None:
    """Crawl4AI fetch provider resolves correctly."""
    provider = resolve_fetch_provider(
        "crawl4ai_local",
        {"crawl4ai_local": {"type": "crawl4ai"}},
    )

    assert provider is Crawl4AIFetchProvider


def test_resolve_llm_provider_success() -> None:
    """LLM provider resolution returns the registered adapter."""
    provider = resolve_llm_provider(
        "openai_default",
        {"openai_default": {"type": "openai_compatible"}},
    )

    assert provider is OpenAICompatibleLLMProvider


def test_resolve_provider_missing_instance_raises_value_error() -> None:
    """Missing provider instances raise ValueError instead of KeyError."""
    with pytest.raises(ValueError, match="Missing provider instance"):
        resolve_search_provider("missing", {})


def test_resolve_provider_missing_type_raises_value_error() -> None:
    """Provider configs require an explicit type field."""
    with pytest.raises(ValueError, match="missing required field: type"):
        resolve_fetch_provider("jina", {"jina": {}})


def test_resolve_provider_unsupported_capability_raises_value_error() -> None:
    """Using an unsupported provider type for a capability raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported search provider type"):
        resolve_search_provider(
            "openai_default",
            {"openai_default": {"type": "openai_compatible"}},
        )
