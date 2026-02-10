"""MCP server for NEXI search functionality."""

from __future__ import annotations

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "fastmcp is required to run the MCP server. Install it with: pip install fastmcp"
    )

from nexi.config import Config, ensure_config
from nexi.search import run_search_sync

# Initialize MCP server
mcp = FastMCP(name="nexi-search")


@mcp.tool
def nexi_search(
    query: str,
    effort: str = "m",
    max_iter: int | None = None,
    time_target: int | None = None,
    verbose: bool = False,
) -> str:
    """Perform an intelligent web search using NEXI.

    NEXI uses an agentic search loop that can:
    - Search the web using multiple parallel queries
    - Fetch and process web pages automatically
    - Extract relevant information from multiple sources
    - Synthesize comprehensive answers

    Args:
        query: The search query to investigate
        effort: Search depth/effort level. Options:
            - "s": Quick search (8 iterations)
            - "m": Medium search (16 iterations, default)
            - "l": Deep search (32 iterations)
        max_iter: Override maximum search iterations (optional)
        time_target: Soft limit: force final answer after N seconds (optional, overrides config)
        verbose: Show detailed progress including tool calls and debug info

    Returns:
        Comprehensive answer with sources cited in markdown format
    """
    # Load config
    try:
        config = ensure_config()
    except Exception as e:
        return f"Error loading NEXI config: {e}"

    # Override config with tool parameters
    search_effort = effort or config.default_effort
    search_time_target = time_target if time_target is not None else config.time_target

    # Create temporary config with overrides
    search_config = Config(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        jina_key=config.jina_key,
        default_effort=search_effort,
        time_target=search_time_target,
        max_output_tokens=config.max_output_tokens,
    )

    try:
        # Run search
        result = run_search_sync(
            query=query,
            config=search_config,
            effort=search_effort,
            max_iter=max_iter,
            time_target=search_time_target,
            verbose=verbose,
            progress_callback=None,
        )

        # Build response with metadata
        response = f"""# Search Results

{result.answer}

---

**Metadata:**
- Iterations: {result.iterations}
- Duration: {result.duration_s:.1f}s
- Tokens: {result.tokens}
- Sources: {len(result.urls)} pages

**Sources:**
"""

        for i, url in enumerate(result.urls, 1):
            response += f"{i}. {url}\n"

        return response

    except Exception as e:
        return f"Search failed: {e}"


def run(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the MCP server.

    Args:
        transport: Transport type - "stdio" (default) or "http"
        host: Host address for HTTP transport
        port: Port number for HTTP transport
    """
    if transport == "http":
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    import sys

    # Simple CLI for running the server
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

    run(transport=transport, host=host, port=port)
