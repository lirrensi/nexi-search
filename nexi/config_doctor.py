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


def build_doctor_summary(config: Config) -> list[str]:
    """Build a human-readable summary of configured chains and providers."""
    llm_models = _collect_llm_models(config)
    summary = [
        f"Configured providers: {len(config.providers)}",
        f"llm_backends ({len(config.llm_backends)}): {_format_chain(config.llm_backends)}",
        f"search_backends ({len(config.search_backends)}): {_format_chain(config.search_backends)}",
        f"fetch_backends ({len(config.fetch_backends)}): {_format_chain(config.fetch_backends)}",
    ]
    if llm_models:
        summary.append(f"LLM models: {', '.join(llm_models)}")
    return summary


def build_doctor_warnings(config: Config) -> list[str]:
    """Build non-fatal config warnings for doctor output."""
    warnings: list[str] = []
    for chain_name in ("llm_backends", "search_backends", "fetch_backends"):
        chain = getattr(config, chain_name)
        if len(chain) == 1:
            warnings.append(
                f"Only one provider in {chain_name}; failover is disabled for that capability"
            )
    return warnings


def _format_chain(chain: list[str]) -> str:
    """Format an active provider chain for doctor output."""
    if not chain:
        return "none"
    return ", ".join(chain)


def _collect_llm_models(config: Config) -> list[str]:
    """Collect configured model names from active LLM backends."""
    models: list[str] = []
    for provider_name in config.llm_backends:
        provider_config = config.providers.get(provider_name)
        if not isinstance(provider_config, dict):
            continue
        model = provider_config.get("model")
        if isinstance(model, str) and model.strip():
            models.append(f"{provider_name}={model.strip()}")
    return models


__all__ = [
    "build_doctor_report",
    "build_doctor_summary",
    "build_doctor_warnings",
    "check_command_readiness",
]
