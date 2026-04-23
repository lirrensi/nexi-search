"""Interactive onboarding for NEXI config activation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import click
import questionary

from nexi.config import CONFIG_FILE, Config, load_config, save_config
from nexi.config_template import (
    ACTIVE_FETCH_PROVIDER_DEFAULTS,
    DEFAULT_CHAIN_CONFIG,
    PROVIDER_EXAMPLES,
    write_default_template,
)

LLM_CHOICES = ["openrouter", "openai", "local_openai", "custom_llm"]
SEARCH_CHOICES = [
    "jina",
    "searxng",
    "tavily",
    "exa",
    "firecrawl",
    "linkup",
    "brave",
    "serpapi",
    "serper",
    "perplexity",
    "custom_search",
]
FETCH_CHOICES = [
    "crawl4ai_local",
    "special_trafilatura",
    "special_playwright",
    "markdown_new",
    "jina",
    "tavily",
    "exa",
    "firecrawl",
    "linkup",
    "custom_fetch",
]


def run_onboarding() -> None:
    """Run the guided config onboarding flow."""
    if not CONFIG_FILE.exists():
        write_default_template(CONFIG_FILE, force=False)

    current_config = load_config().to_dict()

    llm_choice = _ask_select("Choose an LLM setup:", LLM_CHOICES)
    if llm_choice is None:
        _print_cancelled()
        return
    llm_setup = _build_setup(llm_choice)
    if llm_setup is None:
        _print_cancelled()
        return

    search_choice = _ask_select("Choose a search setup:", SEARCH_CHOICES)
    if search_choice is None:
        _print_cancelled()
        return
    search_setup = _build_setup(search_choice)
    if search_setup is None:
        _print_cancelled()
        return

    keep_default_fetch = questionary.confirm(
        "Keep the default fetch chain [crawl4ai_local, special_trafilatura, special_playwright, markdown_new]?",
        default=True,
    ).ask()
    if keep_default_fetch is None:
        _print_cancelled()
        return

    providers: dict[str, dict[str, Any]] = {}
    providers[llm_setup[0]] = llm_setup[1]
    providers[search_setup[0]] = search_setup[1]

    if keep_default_fetch:
        fetch_backends = deepcopy(DEFAULT_CHAIN_CONFIG["fetch_backends"])
        for provider_name in fetch_backends:
            providers.setdefault(
                provider_name,
                deepcopy(ACTIVE_FETCH_PROVIDER_DEFAULTS[provider_name]),
            )
    else:
        selected_fetch = questionary.checkbox(
            "Choose fetch providers:",
            choices=FETCH_CHOICES,
        ).ask()
        if selected_fetch is None:
            _print_cancelled()
            return

        fetch_backends = list(selected_fetch)
        for provider_name in fetch_backends:
            if provider_name in ACTIVE_FETCH_PROVIDER_DEFAULTS:
                providers.setdefault(
                    provider_name,
                    deepcopy(ACTIVE_FETCH_PROVIDER_DEFAULTS[provider_name]),
                )
                continue
            if provider_name in providers:
                continue
            fetch_setup = _build_setup(provider_name)
            if fetch_setup is None:
                _print_cancelled()
                return
            providers[fetch_setup[0]] = fetch_setup[1]
            if fetch_setup[0] != provider_name:
                fetch_backends = [
                    fetch_setup[0] if item == provider_name else item for item in fetch_backends
                ]

    updated_config = current_config.copy()
    updated_config["llm_backends"] = [llm_setup[0]]
    updated_config["search_backends"] = [search_setup[0]]
    updated_config["fetch_backends"] = fetch_backends
    updated_config["providers"] = providers

    save_config(Config.from_dict(updated_config))
    click.echo(f"Saved config to {CONFIG_FILE}")


def _build_setup(choice: str) -> tuple[str, dict[str, Any]] | None:
    """Build provider config for a selected onboarding choice."""
    if choice == "openrouter":
        api_key = _ask_password("OpenRouter API key:")
        if api_key is None:
            return None
        model = _ask_text("OpenRouter model:", PROVIDER_EXAMPLES["openrouter"]["model"])
        if model is None:
            return None
        return (
            "openrouter",
            {
                "type": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": api_key,
                "model": model,
            },
        )

    if choice == "openai":
        api_key = _ask_password("OpenAI API key:")
        if api_key is None:
            return None
        model = _ask_text("OpenAI model:", PROVIDER_EXAMPLES["openai"]["model"])
        if model is None:
            return None
        return (
            "openai",
            {
                "type": "openai_compatible",
                "base_url": "https://api.openai.com/v1",
                "api_key": api_key,
                "model": model,
            },
        )

    if choice == "local_openai":
        base_url = _ask_text(
            "Local OpenAI-compatible base URL:", PROVIDER_EXAMPLES["local_openai"]["base_url"]
        )
        if base_url is None:
            return None
        api_key = _ask_password("Local API key:")
        if api_key is None:
            return None
        model = _ask_text("Local model:", PROVIDER_EXAMPLES["local_openai"]["model"])
        if model is None:
            return None
        return (
            "local_openai",
            {
                "type": "openai_compatible",
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            },
        )

    if choice in {"custom_llm", "custom_search", "custom_fetch"}:
        return _build_custom_setup(choice)

    if choice in {
        "jina",
        "searxng",
        "tavily",
        "exa",
        "firecrawl",
        "linkup",
        "brave",
        "serpapi",
        "serper",
        "perplexity",
    }:
        provider_config = deepcopy(PROVIDER_EXAMPLES[choice])
        if choice == "searxng":
            base_url = _ask_text(
                "SearXNG base URL:",
                str(PROVIDER_EXAMPLES[choice]["base_url"]),
            )
            if base_url is None:
                return None
            provider_config["base_url"] = base_url
            return choice, provider_config

        api_key = _ask_password(f"{choice} API key:")
        if api_key is None:
            return None
        provider_config["api_key"] = api_key
        return choice, provider_config

    raise ValueError(f"Unsupported onboarding choice: {choice}")


def _build_custom_setup(choice: str) -> tuple[str, dict[str, Any]] | None:
    """Collect settings for a custom provider setup."""
    default_type = str(PROVIDER_EXAMPLES[choice]["type"])
    provider_name = _ask_text("Custom provider name:", choice)
    if provider_name is None:
        return None
    provider_type = _ask_text("Custom provider type:", default_type)
    if provider_type is None:
        return None
    return provider_name, {"type": provider_type}


def _ask_select(message: str, choices: list[str]) -> str | None:
    """Prompt for one choice."""
    return questionary.select(message, choices=choices).ask()


def _ask_text(message: str, default: str = "") -> str | None:
    """Prompt for required text input."""
    return questionary.text(
        message,
        default=default,
        validate=lambda value: bool(str(value).strip()) or "This field is required",
    ).ask()


def _ask_password(message: str) -> str | None:
    """Prompt for required password input."""
    return questionary.password(
        message,
        validate=lambda value: bool(str(value).strip()) or "This field is required",
    ).ask()


def _print_cancelled() -> None:
    """Print the standard onboarding cancellation message."""
    click.echo("Onboarding cancelled")


__all__ = ["run_onboarding"]
