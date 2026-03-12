"""Direct fetch CLI for NEXI backends."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from nexi.config import ConfigCreatedError, ensure_config, format_config_created_message
from nexi.config_doctor import check_command_readiness
from nexi.tools import web_get


def _all_pages_failed(payload: dict[str, Any]) -> bool:
    """Return True when every requested page failed."""
    pages = payload.get("pages", [])
    return bool(pages) and all(page.get("error") for page in pages if isinstance(page, dict))


def _format_fetch_payload(payload: dict[str, Any]) -> str:
    """Format fetch payload for human-readable output."""
    blocks: list[str] = []
    for page in payload.get("pages", []):
        if not isinstance(page, dict):
            continue
        if page.get("error"):
            blocks.append(f"{page.get('url', '')}\n---\nError: {page['error']}")
            continue
        blocks.append(str(page.get("content", "")))
    return "\n\n".join(block for block in blocks if block)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("urls", nargs=-1, required=True)
@click.option("--json", "json_output", is_flag=True, help="Print structured JSON output")
@click.option("--full", is_flag=True, help="Return full fetched content without extraction")
@click.option("--chunks", is_flag=True, help="Use chunk selection instead of summarization")
@click.option("--instructions", default="", help="Custom extraction instructions")
@click.option("-v", "--verbose", is_flag=True, help="Show provider debug output")
def main(
    urls: tuple[str, ...],
    json_output: bool,
    full: bool,
    chunks: bool,
    instructions: str,
    verbose: bool,
) -> None:
    """Run direct fetch/extraction using configured backend orchestration."""
    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        raise click.ClickException(format_config_created_message(exc.config_path)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    readiness_errors = check_command_readiness(config, "nexi-fetch")
    if readiness_errors:
        raise click.ClickException("; ".join(readiness_errors))

    payload = asyncio.run(
        web_get(
            urls=list(urls),
            config=config,
            verbose=verbose,
            instructions=instructions,
            get_full=full,
            use_chunks=chunks,
        )
    )

    if json_output:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        click.echo(_format_fetch_payload(payload))

    if _all_pages_failed(payload):
        raise click.ClickException("All configured fetch providers failed")


__all__ = ["main"]


if __name__ == "__main__":
    main()
