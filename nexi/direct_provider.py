"""Helpers for direct provider overrides."""

# FILE: nexi/direct_provider.py
# PURPOSE: Validate and narrow direct CLI provider overrides before execution.
# OWNS: Shared direct-provider validation and single-provider config shaping.
# EXPORTS: build_direct_provider_config
# DOCS: agent_chat/plan_direct_provider_override_2026-04-24.md

from __future__ import annotations

from dataclasses import replace

from nexi.backends.registry import resolve_fetch_provider, resolve_search_provider
from nexi.config import Config


def build_direct_provider_config(
    config: Config,
    provider_name: str,
    capability: str,
) -> Config:
    """Return a config narrowed to one direct-provider instance.

    Args:
        config: Active NEXI configuration.
        provider_name: Provider instance name requested by the user.
        capability: Required capability, either ``search`` or ``fetch``.

    Returns:
        A copy of ``config`` whose chain for the requested capability contains only
        the named provider.

    Raises:
        ValueError: If the provider name is missing, unknown, or incompatible.
    """
    cleaned_name = provider_name.strip()
    if not cleaned_name:
        raise ValueError("Missing provider name for --provider")

    if capability == "search":
        resolve_search_provider(cleaned_name, config.providers)
        return replace(config, search_backends=[cleaned_name])

    if capability == "fetch":
        resolve_fetch_provider(cleaned_name, config.providers)
        return replace(config, fetch_backends=[cleaned_name])

    raise ValueError(f"Unsupported direct-provider capability: {capability}")


__all__ = ["build_direct_provider_config"]
