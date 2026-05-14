# NEXI MCP Server

This module provides an MCP (Model Context Protocol) server for NEXI, allowing it to be used as a tool in MCP-compatible applications.

## Installation

Install with MCP support:

```bash
pip install -e ".[mcp]"
# or
uv sync --group mcp
```

## Usage

### STDIO Transport (Local Development)

For use with local MCP clients:

```bash
python -m nexi.mcp_server_cli
# or
uv run python -m nexi.mcp_server_cli
```

### HTTP Transport (Network Access)

For production deployments:

```bash
python -m nexi.mcp_server_cli http 0.0.0.0 8000
# or
uv run python -m nexi.mcp_server_cli http 0.0.0.0 8000
```

## Available Tools

### `nexi_agent`

Run the full NEXI agent. This is the MCP counterpart to `nexi`.

**Parameters:**
- `query` (required): The search query to investigate
- `effort` (optional): Search depth - "s" (quick), "m" (medium, default), "l" (deep)
- `verbose` (optional): Show detailed progress including tool calls and debug info

**Returns:**
Comprehensive answer with sources cited in markdown format, including metadata about iterations, duration, tokens, and source URLs.

**Error handling:**
- If the NEXI config file is missing or incomplete, returns an actionable message advising the user to fill in the template
- If no usable LLM or search provider is configured, returns a readiness error
- If the search fails at runtime, returns an error message

### `nexi_search`

Run the direct search backend chain. This is the MCP counterpart to `nexi-search`.

**Parameters:**
- `query` (required): Search query
- `verbose` (optional): Show provider debug output

**Returns:**
Structured search payload matching `nexi-search --json`, including `searches` and `provider_failures`.

### `nexi_fetch`

Run the direct fetch backend chain. This is the MCP counterpart to `nexi-fetch`.

**Parameters:**
- `urls` (required): One or more URLs to fetch
- `full` (optional): Return full fetched content without LLM extraction
- `chunks` (optional): Use chunk selection instead of LLM summarization
- `instructions` (optional): Custom extraction instructions for the LLM summarizer (only when `full=false` and `chunks=false`)
- `verbose` (optional): Show provider debug output

**Returns:**
Structured fetch payload matching `nexi-fetch --json`, including `pages`, `provider_failures`, and any `full_content_path` spillover markers for oversized pages.

**Error handling:**
- Returns `{"error": "..."}` when `urls` is empty, config is missing, readiness checks fail, or the fetch fails at runtime

## Example MCP Client Configuration

For use with Claude Desktop or other MCP clients, add to your MCP config:

```json
{
  "mcpServers": {
    "nexi": {
      "command": "uv",
      "args": ["run", "python", "-m", "nexi.mcp_server_cli"],
      "env": {}
    }
  }
}
```

For HTTP transport:

```json
{
  "mcpServers": {
    "nexi": {
      "url": "http://localhost:8000"
    }
  }
}
```

## Development

Run with auto-reload:

```bash
fastmcp run nexi/mcp_server.py --reload
```

## Notes

- The MCP server uses the same configuration as the CLI tool (`~/.config/nexi/config.toml`)
- If the config file is missing, NEXI creates the default template and returns a config-created message instead of launching onboarding automatically
- `nexi_agent` requires a usable LLM provider and a usable search provider
- `nexi_search` requires a usable search provider
- `nexi_fetch` requires a usable fetch provider
- All three MCP tools validate readiness via `check_command_readiness()` before executing
- `nexi_fetch` applies the same token-capping and spillover logic as the `nexi-fetch` CLI (via `post_process_direct_fetch_payload`)
- MCP tool names intentionally mirror the runtime surfaces they expose
