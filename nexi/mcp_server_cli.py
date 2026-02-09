#!/usr/bin/env python
"""Run NEXI MCP server.

Usage:
    # STDIO transport (for local MCP clients)
    python -m nexi.mcp_server

    # HTTP transport (for network access)
    python -m nexi.mcp_server http 0.0.0.0 8000

    # Or using uv
    uv run python -m nexi.mcp_server
"""

from __future__ import annotations

import sys

from nexi.mcp_server import run


def main() -> None:
    """Main entry point for running the MCP server."""
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

    if transport not in ("stdio", "http"):
        print(f"Invalid transport: {transport}")
        print("Valid options: stdio, http")
        sys.exit(1)

    run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
