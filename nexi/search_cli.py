"""Direct search CLI for NEXI backends."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from nexi.backends.orchestrators import run_search_chain
from nexi.config import ensure_config


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
@click.option("-v", "--verbose", is_flag=True, help="Show provider debug output")
def main(query: str, json_output: bool, verbose: bool) -> None:
    """Run direct search using configured backend orchestration."""
    config = ensure_config()
    payload = asyncio.run(run_search_chain([query], config, verbose))

    if json_output:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        click.echo(_format_search_payload(payload))

    if _all_searches_failed(payload):
        raise click.ClickException("All configured search providers failed")


__all__ = ["main"]


if __name__ == "__main__":
    main()
