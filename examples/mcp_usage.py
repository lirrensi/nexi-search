"""Example script demonstrating NEXI MCP server usage."""

from __future__ import annotations

import asyncio

try:
    from fastmcp import FastMCP
except ImportError:
    print("fastmcp is required. Install with: pip install fastmcp")
    exit(1)

from nexi.mcp_server import nexi_search


async def example_usage():
    """Demonstrate using the nexi_search tool directly."""
    print("=" * 60)
    print("NEXI MCP Server - Example Usage")
    print("=" * 60)
    print()

    # Example 1: Quick search
    print("Example 1: Quick search (effort='s')")
    print("-" * 60)
    result = nexi_search(
        query="What is Python?",
        effort="s",
        verbose=False,
    )
    print(result)
    print()

    # Example 2: Deep search with custom parameters
    print("Example 2: Deep search with custom parameters")
    print("-" * 60)
    result = nexi_search(
        query="Explain quantum computing in simple terms",
        effort="l",
        max_iter=8,
        max_timeout=180,
        verbose=True,
    )
    print(result)
    print()

    # Example 3: Medium search with timeout
    print("Example 3: Medium search with timeout")
    print("-" * 60)
    result = nexi_search(
        query="Latest developments in AI",
        effort="m",
        max_timeout=60,
        verbose=False,
    )
    print(result)
    print()


def example_mcp_client():
    """Demonstrate using FastMCP client to connect to NEXI server."""
    print("=" * 60)
    print("NEXI MCP Client Example")
    print("=" * 60)
    print()

    # This would connect to a running MCP server
    # For demonstration, we'll use the tool directly
    print("Note: This example uses the tool directly.")
    print("To connect to a running server, use FastMCP client.")
    print()

    # Simulate calling the tool
    result = nexi_search(
        query="What is the MCP protocol?",
        effort="m",
    )

    print("Result from nexi_search tool:")
    print(result)


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_usage())
    print()
    example_mcp_client()
