"""Readiness checks for NEXI commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nexi.backends.registry import (
    resolve_fetch_provider,
    resolve_llm_provider,
    resolve_search_provider,
)
from nexi.config import Config

Resolver = Callable[[str, dict[str, dict[str, Any]]], type[Any]]

COMMAND_REQUIREMENTS: dict[str, list[tuple[str, Resolver, str]]] = {
    "nexi": [
        (
            "llm_backends",
            resolve_llm_provider,
            "Activate at least one LLM provider in llm_backends",
        ),
        (
            "search_backends",
            resolve_search_provider,
            "Activate at least one search provider in search_backends",
        ),
    ],
    "nexi-search": [
        (
            "search_backends",
            resolve_search_provider,
            "Activate at least one search provider in search_backends",
        ),
    ],
    "nexi-fetch": [
        (
            "fetch_backends",
            resolve_fetch_provider,
            "Activate at least one fetch provider in fetch_backends",
        ),
    ],
}


def check_command_readiness(config: Config, command_name: str) -> list[str]:
    """Check whether a command has usable configured providers."""
    requirements = COMMAND_REQUIREMENTS.get(command_name)
    if requirements is None:
        raise ValueError(f"Unsupported command readiness check: {command_name}")

    errors: list[str] = []
    providers = config.providers

    for field_name, resolver, empty_message in requirements:
        chain = getattr(config, field_name)
        if not chain:
            errors.append(empty_message)
            continue

        for provider_name in chain:
            try:
                provider_class = resolver(provider_name, providers)
                provider = provider_class()
                provider_config = providers.get(provider_name)
                if not isinstance(provider_config, dict):
                    raise ValueError(f"Provider config must be an object: {provider_name}")
                provider.validate_config(provider_config)
            except Exception as exc:
                errors.append(f"Active provider '{provider_name}' is not ready: {exc}")

    return errors


def build_doctor_report(config: Config) -> dict[str, list[str]]:
    """Build readiness results for each public command."""
    return {
        command_name: check_command_readiness(config, command_name)
        for command_name in ("nexi", "nexi-search", "nexi-fetch")
    }


__all__ = ["build_doctor_report", "check_command_readiness"]
