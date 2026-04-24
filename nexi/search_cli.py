"""Direct search CLI for NEXI backends."""

# FILE: nexi/search_cli.py
# PURPOSE: Run direct search requests with optional provider override support.
# OWNS: Direct search CLI parsing, readiness checks, and result formatting.
# EXPORTS: main
# DOCS: agent_chat/plan_direct_provider_override_2026-04-24.md

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from nexi.backends.orchestrators import run_search_chain
from nexi.config import ConfigCreatedError, ensure_config, format_config_created_message
from nexi.config_doctor import check_command_readiness
from nexi.direct_provider import build_direct_provider_config
from nexi.runtime_noise import configure_runtime_noise, suppress_runtime_chatter


def _all_searches_failed(payload: dict[str, Any]) -> bool:
    """Return True when every search item failed."""
    searches = payload.get("searches", [])
    return bool(searches) and all(item.get("error") for item in searches if isinstance(item, dict))


def _format_search_payload(payload: dict[str, Any]) -> str:
    """Format search payload for human-readable output."""
    lines: list[str] = []
    for search in payload.get("searches", []):
        if not isinstance(search, dict):
            continue
        query = search.get("query", "")
        lines.append(f"Query: {query}")

        if search.get("error"):
            lines.append(f"Error: {search['error']}")
            lines.append("")
            continue

        results = search.get("results", [])
        if not results:
            lines.append("No results")
            lines.append("")
            continue

        for index, result in enumerate(results, 1):
            if not isinstance(result, dict):
                continue
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            description = result.get("description", "")
            lines.append(f"{index}. {title}")
            if url:
                lines.append(url)
            if description:
                lines.append(description)
            lines.append("")

    return "\n".join(lines).strip()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("query")
@click.option("--json", "json_output", is_flag=True, help="Print structured JSON output")
@click.option(
    "--provider",
    default=None,
    help="Use only the named provider instance and bypass the fallback chain",
)
@click.option("-v", "--verbose", is_flag=True, help="Show provider debug output")
def main(query: str, json_output: bool, provider: str | None, verbose: bool) -> None:
    """Run direct search using configured backend orchestration."""
    configure_runtime_noise(verbose)
    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        raise click.ClickException(format_config_created_message(exc.config_path)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if provider is None:
        readiness_errors = check_command_readiness(config, "nexi-search")
        if readiness_errors:
            raise click.ClickException("; ".join(readiness_errors))
    else:
        try:
            config = build_direct_provider_config(config, provider, "search")
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    with suppress_runtime_chatter(verbose):
        payload = asyncio.run(run_search_chain([query], config, verbose))

    if json_output:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        click.echo(_format_search_payload(payload))

    if _all_searches_failed(payload):
        if provider is not None:
            raise click.ClickException(f"Provider '{provider}' failed")
        raise click.ClickException("All configured search providers failed")


__all__ = ["main"]


if __name__ == "__main__":
    main()
