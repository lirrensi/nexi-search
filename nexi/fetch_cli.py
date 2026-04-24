"""Direct fetch CLI for NEXI backends."""

# FILE: nexi/fetch_cli.py
# PURPOSE: Run direct fetch requests with optional provider override support.
# OWNS: Direct fetch CLI parsing, readiness checks, and output formatting.
# EXPORTS: main
# DOCS: agent_chat/plan_direct_provider_override_2026-04-24.md

from __future__ import annotations

import asyncio
import json
from typing import Any

import click

from nexi.config import ConfigCreatedError, ensure_config, format_config_created_message
from nexi.config_doctor import check_command_readiness
from nexi.direct_fetch import post_process_direct_fetch_payload
from nexi.direct_provider import build_direct_provider_config
from nexi.runtime_noise import configure_runtime_noise, suppress_runtime_chatter
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
        content = str(page.get("content", ""))
        full_content_path = page.get("full_content_path")
        if isinstance(full_content_path, str) and full_content_path.strip():
            content = f"{content}\n\nFull content saved to: {full_content_path}"
        blocks.append(content)
    return "\n\n".join(block for block in blocks if block)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("urls", nargs=-1, required=True)
@click.option("--json", "json_output", is_flag=True, help="Print structured JSON output")
@click.option(
    "--provider",
    default=None,
    help="Use only the named provider instance and bypass the fallback chain",
)
@click.option("--full", is_flag=True, help="Return full fetched content without extraction")
@click.option("--chunks", is_flag=True, help="Use chunk selection instead of summarization")
@click.option("--instructions", default="", help="Custom extraction instructions")
@click.option("-v", "--verbose", is_flag=True, help="Show provider debug output")
def main(
    urls: tuple[str, ...],
    json_output: bool,
    provider: str | None,
    full: bool,
    chunks: bool,
    instructions: str,
    verbose: bool,
) -> None:
    """Run direct fetch/extraction using configured backend orchestration."""
    configure_runtime_noise(verbose)
    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        raise click.ClickException(format_config_created_message(exc.config_path)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if provider is None:
        readiness_errors = check_command_readiness(config, "nexi-fetch")
        if readiness_errors:
            raise click.ClickException("; ".join(readiness_errors))
    else:
        try:
            config = build_direct_provider_config(config, provider, "fetch")
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    with suppress_runtime_chatter(verbose):
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
    payload = post_process_direct_fetch_payload(
        payload,
        max_tokens=config.direct_fetch_max_tokens,
        encoding_name=config.tokenizer_encoding,
    )

    if json_output:
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        click.echo(_format_fetch_payload(payload))

    if _all_pages_failed(payload):
        if provider is not None:
            raise click.ClickException(f"Provider '{provider}' failed")
        raise click.ClickException("All configured fetch providers failed")


__all__ = ["main"]


if __name__ == "__main__":
    main()
