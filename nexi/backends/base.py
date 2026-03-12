"""Provider contracts for NEXI backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """Base provider contract."""

    name: str

    def validate_config(self, config: dict[str, Any]) -> None:
        """Raise ValueError if required config is missing or invalid."""


@runtime_checkable
class SearchProvider(Provider, Protocol):
    """Search provider contract."""

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute one batch of search queries."""
        ...


@runtime_checkable
class FetchProvider(Provider, Protocol):
    """Fetch provider contract."""

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute one batch of fetch requests."""
        ...


@runtime_checkable
class LLMProvider(Provider, Protocol):
    """LLM provider contract."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        verbose: bool,
        max_tokens: int,
    ) -> Any:
        """Execute one completion request."""
        ...


__all__ = ["FetchProvider", "LLMProvider", "Provider", "SearchProvider"]
