"""Provider registry for NEXI backends."""

from __future__ import annotations

from typing import Any

from nexi.backends.base import FetchProvider, LLMProvider, SearchProvider
from nexi.backends.brave import BraveSearchProvider
from nexi.backends.crawl4ai import Crawl4AIFetchProvider
from nexi.backends.custom_python import build_custom_provider_class, is_custom_provider_type
from nexi.backends.exa import ExaFetchProvider, ExaSearchProvider
from nexi.backends.firecrawl import FirecrawlFetchProvider, FirecrawlSearchProvider
from nexi.backends.jina import JinaFetchProvider, JinaSearchProvider
from nexi.backends.linkup import LinkupFetchProvider, LinkupSearchProvider
from nexi.backends.markdown_new import MarkdownNewFetchProvider
from nexi.backends.openai_compatible import OpenAICompatibleLLMProvider
from nexi.backends.perplexity_search import PerplexitySearchProvider
from nexi.backends.searxng import SearXNGSearchProvider
from nexi.backends.serpapi import SerpAPISearchProvider
from nexi.backends.serper import SerperSearchProvider
from nexi.backends.snitchmd import SnitchFetchProvider
from nexi.backends.special_fetch import (
    SpecialPlaywrightFetchProvider,
    SpecialTrafilaturaFetchProvider,
)
from nexi.backends.tavily import TavilyFetchProvider, TavilySearchProvider

SEARCH_PROVIDER_REGISTRY: dict[str, type[SearchProvider]] = {
    "brave": BraveSearchProvider,
    "exa": ExaSearchProvider,
    "firecrawl": FirecrawlSearchProvider,
    "jina": JinaSearchProvider,
    "linkup": LinkupSearchProvider,
    "perplexity_search": PerplexitySearchProvider,
    "searxng": SearXNGSearchProvider,
    "serpapi": SerpAPISearchProvider,
    "serper": SerperSearchProvider,
    "tavily": TavilySearchProvider,
}
FETCH_PROVIDER_REGISTRY: dict[str, type[FetchProvider]] = {
    "crawl4ai": Crawl4AIFetchProvider,
    "exa": ExaFetchProvider,
    "firecrawl": FirecrawlFetchProvider,
    "jina": JinaFetchProvider,
    "linkup": LinkupFetchProvider,
    "markdown_new": MarkdownNewFetchProvider,
    "snitchmd": SnitchFetchProvider,
    "special_playwright": SpecialPlaywrightFetchProvider,
    "special_trafilatura": SpecialTrafilaturaFetchProvider,
    "tavily": TavilyFetchProvider,
}
LLM_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "openai_compatible": OpenAICompatibleLLMProvider,
}


def _resolve_provider(
    provider_name: str,
    providers: dict[str, dict[str, Any]],
    registry: dict[str, type[Any]],
    capability: str,
) -> type[Any]:
    """Resolve a provider adapter class for a capability."""
    provider_config = providers.get(provider_name)
    if provider_config is None:
        raise ValueError(f"Missing provider instance: {provider_name}")
    if not isinstance(provider_config, dict):
        raise ValueError(f"Provider config must be an object: {provider_name}")

    provider_type = provider_config.get("type")
    if not isinstance(provider_type, str) or not provider_type.strip():
        raise ValueError(f"Provider '{provider_name}' is missing required field: type")

    provider_class = registry.get(provider_type)
    if provider_class is None:
        if is_custom_provider_type(provider_type):
            return build_custom_provider_class(capability, provider_name, provider_type)
        raise ValueError(
            f"Unsupported {capability} provider type '{provider_type}' for provider '{provider_name}'"
        )

    return provider_class


def resolve_search_provider(
    provider_name: str,
    providers: dict[str, dict[str, Any]],
) -> type[SearchProvider]:
    """Resolve a search provider adapter class by provider instance name."""
    return _resolve_provider(provider_name, providers, SEARCH_PROVIDER_REGISTRY, "search")


def resolve_fetch_provider(
    provider_name: str,
    providers: dict[str, dict[str, Any]],
) -> type[FetchProvider]:
    """Resolve a fetch provider adapter class by provider instance name."""
    return _resolve_provider(provider_name, providers, FETCH_PROVIDER_REGISTRY, "fetch")


def resolve_llm_provider(
    provider_name: str,
    providers: dict[str, dict[str, Any]],
) -> type[LLMProvider]:
    """Resolve an LLM provider adapter class by provider instance name."""
    return _resolve_provider(provider_name, providers, LLM_PROVIDER_REGISTRY, "llm")


__all__ = [
    "FETCH_PROVIDER_REGISTRY",
    "LLM_PROVIDER_REGISTRY",
    "SEARCH_PROVIDER_REGISTRY",
    "resolve_fetch_provider",
    "resolve_llm_provider",
    "resolve_search_provider",
]
