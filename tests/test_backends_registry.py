"""Tests for backend registry resolution."""

from __future__ import annotations

import pytest

from nexi.backends.jina import JinaFetchProvider, JinaSearchProvider
from nexi.backends.openai_compatible import OpenAICompatibleLLMProvider
from nexi.backends.registry import (
    resolve_fetch_provider,
    resolve_llm_provider,
    resolve_search_provider,
)


def test_resolve_search_provider_success() -> None:
    """Search provider resolution returns the registered adapter."""
    provider = resolve_search_provider("jina", {"jina": {"type": "jina"}})

    assert provider is JinaSearchProvider


def test_resolve_fetch_provider_success() -> None:
    """Fetch provider resolution returns the registered adapter."""
    provider = resolve_fetch_provider("jina", {"jina": {"type": "jina"}})

    assert provider is JinaFetchProvider


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
