"""TOML config template rendering for NEXI."""

# FILE: nexi/config_template.py
# PURPOSE: Render the default TOML config template and bundled provider examples.
# OWNS: Bootstrap template shape, default chains, and commented provider examples.
# DOCS: docs/product.md, docs/arch.md, docs/provider-matrix.md, agent_chat/plan_crawl4ai_opt_in_2026-04-24.md

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_CHAIN_CONFIG: dict[str, list[str]] = {
    "llm_backends": [],
    "search_backends": [],
    "fetch_backends": [
        "special_trafilatura",
        "special_playwright",
        "markdown_new",
    ],
}

DEFAULT_SCALAR_CONFIG: dict[str, Any] = {
    "default_effort": "m",
    "max_context": 128000,
    "auto_compact_thresh": 0.9,
    "compact_target_words": 5000,
    "preserve_last_n_messages": 3,
    "tokenizer_encoding": "cl100k_base",
    "provider_timeout": 30,
    "direct_fetch_max_tokens": 8000,
    "search_provider_retries": 2,
    "fetch_provider_retries": 2,
}

ACTIVE_FETCH_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "special_trafilatura": {
        "type": "special_trafilatura",
    },
    "special_playwright": {
        "type": "special_playwright",
    },
    "markdown_new": {
        "type": "markdown_new",
        "method": "auto",
        "retain_images": False,
    },
}

PROVIDER_EXAMPLES: dict[str, dict[str, Any]] = {
    "openrouter": {
        "type": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "<your_api_key>",
        "model": "google/gemini-2.5-flash-lite",
    },
    "openai": {
        "type": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "api_key": "<your_api_key>",
        "model": "gpt-4.1-mini",
    },
    "local_openai": {
        "type": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
        "api_key": "local-key",
        "model": "your-model",
    },
    "custom_llm": {
        "type": "provider-custom_llm",
    },
    "jina": {
        "type": "jina",
        "api_key": "<your_api_key>",
    },
    "searxng": {
        "type": "searxng",
        "base_url": "https://search.example.org",
        "language": "en",
        "categories": ["general"],
        "safesearch": 0,
        "format": "json",
    },
    "tavily": {
        "type": "tavily",
        "api_key": "<your_api_key>",
        "search_depth": "basic",
        "topic": "general",
        "max_results": 5,
    },
    "exa": {
        "type": "exa",
        "api_key": "<your_api_key>",
        "num_results": 5,
        "text": True,
    },
    "firecrawl": {
        "type": "firecrawl",
        "api_key": "<your_api_key>",
        "only_main_content": True,
        "formats": ["markdown"],
        "limit": 5,
    },
    "linkup": {
        "type": "linkup",
        "api_key": "<your_api_key>",
        "depth": "standard",
        "output_type": "searchResults",
    },
    "brave": {
        "type": "brave",
        "api_key": "<your_api_key>",
        "count": 5,
    },
    "serpapi": {
        "type": "serpapi",
        "api_key": "<your_api_key>",
        "engine": "google",
    },
    "serper": {
        "type": "serper",
        "api_key": "<your_api_key>",
        "num": 5,
    },
    "perplexity": {
        "type": "perplexity_search",
        "api_key": "<your_api_key>",
        "max_results": 5,
    },
    "custom_search": {
        "type": "provider-custom_search",
    },
    "custom_fetch": {
        "type": "provider-custom_fetch",
    },
    "crawl4ai_local": {
        "type": "crawl4ai",
        "headless": True,
        "cdp_url": "http://localhost:9222",
    },
    "special_trafilatura": {
        "type": "special_trafilatura",
    },
    "special_playwright": {
        "type": "special_playwright",
    },
}

LLM_EXAMPLE_ORDER = ["openrouter", "openai", "local_openai", "custom_llm"]
SEARCH_EXAMPLE_ORDER = [
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
FETCH_EXAMPLE_ORDER = [
    "jina",
    "searxng",
    "tavily",
    "exa",
    "firecrawl",
    "linkup",
    "crawl4ai_local",
    "special_trafilatura",
    "special_playwright",
    "custom_fetch",
]


def render_config_toml(active_config: dict[str, Any] | None = None) -> str:
    """Render the canonical NEXI config template as TOML text."""
    config = _build_render_config(active_config)
    lines = [
        "# Activate at least one LLM provider and one search provider before running `nexi`.",
        "# The default fetch chain uses the quiet providers only.",
        "# Provider instances are shared across chains.",
        "# Define each [providers.<name>] table only once, then reuse that name in search_backends and fetch_backends.",
        '# If you need different settings for search and fetch, use different names like "jina_search" and "jina_fetch".',
        _format_assignment("llm_backends", config["llm_backends"]),
        _format_assignment("search_backends", config["search_backends"]),
        _format_assignment("fetch_backends", config["fetch_backends"]),
        "",
    ]

    scalar_order = [
        "default_effort",
        "max_context",
        "auto_compact_thresh",
        "compact_target_words",
        "preserve_last_n_messages",
        "tokenizer_encoding",
        "provider_timeout",
        "direct_fetch_max_tokens",
        "search_provider_retries",
        "fetch_provider_retries",
    ]
    for key in scalar_order:
        lines.extend(_render_scalar_line(key, config.get(key)))
    lines.append("")

    active_provider_names = set(config["providers"].keys())
    if config["providers"]:
        lines.append("# Active provider configs")
        for provider_name, provider_config in config["providers"].items():
            lines.extend(_render_provider_block(provider_name, provider_config))
        lines.append("")

    lines.extend(
        _render_chain_example_section("# LLM chain examples", "llm_backends", LLM_EXAMPLE_ORDER)
    )
    lines.append("")
    lines.extend(
        _render_chain_example_section(
            "# Search chain examples",
            "search_backends",
            SEARCH_EXAMPLE_ORDER,
        )
    )
    lines.append("")
    lines.extend(
        _render_chain_example_section(
            "# Fetch chain examples",
            "fetch_backends",
            FETCH_EXAMPLE_ORDER,
        )
    )
    lines.append("")
    lines.extend(
        _render_provider_example_section(
            heading="# Shared provider definition examples",
            provider_names=_unique_provider_example_order(),
            active_provider_names=active_provider_names,
        )
    )

    return "\n".join(_trim_trailing_blank_lines(lines)) + "\n"


def write_default_template(config_path: Path, force: bool = False) -> bool:
    """Write the default TOML template when missing or forced."""
    if config_path.exists() and not force:
        return False

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_config_toml(), encoding="utf-8")
    return True


def _build_render_config(active_config: dict[str, Any] | None) -> dict[str, Any]:
    """Build the config payload used for rendering."""
    config: dict[str, Any] = {
        **deepcopy(DEFAULT_CHAIN_CONFIG),
        **deepcopy(DEFAULT_SCALAR_CONFIG),
        "providers": deepcopy(ACTIVE_FETCH_PROVIDER_DEFAULTS),
    }

    if not isinstance(active_config, dict):
        return config

    for field_name in DEFAULT_CHAIN_CONFIG:
        value = active_config.get(field_name)
        if isinstance(value, list):
            config[field_name] = deepcopy(value)

    for field_name in DEFAULT_SCALAR_CONFIG:
        if field_name in active_config:
            config[field_name] = deepcopy(active_config[field_name])

    providers = active_config.get("providers")
    if isinstance(providers, dict):
        config["providers"] = deepcopy(providers)
    else:
        config["providers"] = {}

    for provider_name in config["fetch_backends"]:
        if (
            provider_name in ACTIVE_FETCH_PROVIDER_DEFAULTS
            and provider_name not in config["providers"]
        ):
            config["providers"][provider_name] = deepcopy(
                ACTIVE_FETCH_PROVIDER_DEFAULTS[provider_name]
            )

    return config


def _render_scalar_line(key: str, value: Any) -> list[str]:
    """Render a scalar config line, commenting nullable defaults when needed."""
    if value is None:
        return []
    return [_format_assignment(key, value)]


def _render_chain_example_section(
    heading: str,
    chain_field: str,
    provider_names: list[str],
) -> list[str]:
    """Render a commented chain example section."""
    lines = [heading]
    lines.append(
        f"# Edit {chain_field} at the top of the file. Do not add another {chain_field} assignment later in the file."
    )
    for provider_name in provider_names:
        lines.append(f"# - add {_format_toml_value(provider_name)} to {chain_field}")
    return _trim_trailing_blank_lines(lines)


def _render_provider_example_section(
    heading: str,
    provider_names: list[str],
    active_provider_names: set[str],
) -> list[str]:
    """Render commented provider definition examples without duplicates."""
    lines = [heading]
    lines.append(
        "# Reuse the same provider name in multiple chains. Do not declare the same [providers.<name>] table twice."
    )
    for provider_name in provider_names:
        if provider_name in active_provider_names:
            continue
        lines.extend(
            _render_provider_block(provider_name, PROVIDER_EXAMPLES[provider_name], commented=True)
        )
        lines.append("")
    return _trim_trailing_blank_lines(lines)


def _unique_provider_example_order() -> list[str]:
    """Return provider example names in a stable order without duplicates."""
    ordered_names = [*LLM_EXAMPLE_ORDER, *SEARCH_EXAMPLE_ORDER, *FETCH_EXAMPLE_ORDER]
    deduplicated: list[str] = []
    seen: set[str] = set()
    for provider_name in ordered_names:
        if provider_name in seen:
            continue
        seen.add(provider_name)
        deduplicated.append(provider_name)
    return deduplicated


def _render_provider_block(
    provider_name: str,
    provider_config: dict[str, Any],
    commented: bool = False,
) -> list[str]:
    """Render a provider table and any nested tables."""
    return _render_table(
        table_path=["providers", provider_name],
        table_data=provider_config,
        commented=commented,
    )


def _render_table(
    table_path: list[str],
    table_data: dict[str, Any],
    commented: bool,
) -> list[str]:
    """Render a TOML table recursively."""
    prefix = "# " if commented else ""
    lines = [f"{prefix}[{'.'.join(table_path)}]"]
    nested_tables: list[tuple[str, dict[str, Any]]] = []

    for key, value in table_data.items():
        if isinstance(value, dict):
            nested_tables.append((key, value))
            continue
        lines.append(f"{prefix}{_format_assignment(key, value)}")

    for key, nested_value in nested_tables:
        lines.append("")
        lines.extend(
            _render_table(
                table_path=[*table_path, key],
                table_data=nested_value,
                commented=commented,
            )
        )

    return lines


def _format_assignment(key: str, value: Any) -> str:
    """Format a TOML assignment."""
    return f"{key} = {_format_toml_value(value)}"


def _format_toml_value(value: Any) -> str:
    """Format a Python value as TOML."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return f"[{', '.join(_format_toml_value(item) for item in value)}]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _trim_trailing_blank_lines(lines: list[str]) -> list[str]:
    """Remove trailing blank lines from rendered output chunks."""
    trimmed = list(lines)
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    return trimmed


__all__ = [
    "ACTIVE_FETCH_PROVIDER_DEFAULTS",
    "DEFAULT_CHAIN_CONFIG",
    "DEFAULT_SCALAR_CONFIG",
    "FETCH_EXAMPLE_ORDER",
    "LLM_EXAMPLE_ORDER",
    "PROVIDER_EXAMPLES",
    "SEARCH_EXAMPLE_ORDER",
    "render_config_toml",
    "write_default_template",
]
