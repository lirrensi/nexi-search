#!/usr/bin/env python
"""Run the NEXI MCP server."""

from __future__ import annotations

import click

from nexi.mcp_server import run


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("transport", required=False, default="stdio", type=click.Choice(["stdio", "http"]))
@click.argument("host", required=False, default="0.0.0.0")
@click.argument("port", required=False, default=8000, type=int)
def main(transport: str, host: str, port: int) -> None:
    """Run the NEXI MCP server over STDIO or HTTP."""
    run(transport=transport, host=host, port=port)


__all__ = ["main"]


if __name__ == "__main__":
    main()
