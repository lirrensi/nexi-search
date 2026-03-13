"""MCP server for NEXI runtime surfaces."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError as err:
    raise ImportError(
        "fastmcp is required to run the MCP server. Install it with: pip install fastmcp"
    ) from err

from nexi.backends.orchestrators import run_search_chain
from nexi.config import Config, ConfigCreatedError, ensure_config, format_config_created_message
from nexi.config_doctor import check_command_readiness
from nexi.search import SearchResult, run_search
from nexi.tools import web_get

# Initialize MCP server
mcp = FastMCP(name="nexi")


def _config_error_message(exc: ConfigCreatedError) -> str:
    """Render the canonical config-created message for MCP clients."""
    return format_config_created_message(
        exc.config_path,
        display_path="~/.config/nexi/config.toml",
    )


def _load_ready_config(command_name: str) -> tuple[Config | None, str | None]:
    """Load config and validate readiness for one public surface."""
    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        return None, _config_error_message(exc)
    except Exception as exc:
        return None, f"Error loading NEXI config: {exc}"

    readiness_errors = check_command_readiness(config, command_name)
    if readiness_errors:
        return None, f"NEXI config is not ready: {'; '.join(readiness_errors)}"

    return config, None


def _format_agent_response(result: SearchResult) -> str:
    """Format agent search output for MCP clients."""
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


@mcp.tool
async def nexi_agent(
    query: str,
    effort: str = "m",
    verbose: bool = False,
) -> str:
    """Run the full NEXI agentic search workflow.

    Args:
        query: The search query to investigate.
        effort: Search depth/effort level: "s", "m", or "l".
        verbose: Show detailed progress including tool calls and debug info.

    Returns:
        Markdown-formatted answer with metadata and sources.
    """
    config, error = _load_ready_config("nexi")
    if error:
        return error
    assert config is not None

    search_effort = effort or config.default_effort
    search_config = (
        config
        if search_effort == config.default_effort
        else replace(config, default_effort=search_effort)
    )

    try:
        result = await run_search(
            query=query,
            config=search_config,
            effort=search_effort,
            verbose=verbose,
            progress_callback=None,
        )
        return _format_agent_response(result)
    except Exception as exc:
        return f"Search failed: {exc}"


@mcp.tool
async def nexi_search(query: str, verbose: bool = False) -> dict[str, Any]:
    """Run the direct search provider chain.

    Args:
        query: Search query to execute.
        verbose: Show provider debug output.

    Returns:
        Structured payload matching the direct search JSON shape.
    """
    config, error = _load_ready_config("nexi-search")
    if error:
        return {"error": error}
    assert config is not None

    try:
        return await run_search_chain([query], config, verbose)
    except Exception as exc:
        return {"error": f"Direct search failed: {exc}"}


@mcp.tool
async def nexi_fetch(
    urls: list[str],
    full: bool = False,
    chunks: bool = False,
    instructions: str = "",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the direct fetch provider chain.

    Args:
        urls: One or more URLs to fetch.
        full: Return full fetched content without extraction.
        chunks: Use chunk selection instead of summarization.
        instructions: Custom extraction instructions.
        verbose: Show provider debug output.

    Returns:
        Structured payload matching the direct fetch JSON shape.
    """
    if not urls:
        return {"error": "At least one URL is required"}

    config, error = _load_ready_config("nexi-fetch")
    if error:
        return {"error": error}
    assert config is not None

    try:
        return await web_get(
            urls=urls,
            config=config,
            verbose=verbose,
            instructions=instructions,
            get_full=full,
            use_chunks=chunks,
        )
    except Exception as exc:
        return {"error": f"Direct fetch failed: {exc}"}


def run(transport: str = "stdio", host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the MCP server.

    Args:
        transport: Transport type - "stdio" (default) or "http".
        host: Host address for HTTP transport.
        port: Port number for HTTP transport.
    """
    if transport == "http":
        mcp.run(transport="http", host=host, port=port)
    else:
        mcp.run()


__all__ = ["mcp", "nexi_agent", "nexi_search", "nexi_fetch", "run"]


if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

    run(transport=transport, host=host, port=port)
