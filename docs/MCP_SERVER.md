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
- `max_iter` (optional): Override maximum search iterations
- `time_target` (optional): Force return after N seconds
- `verbose` (optional): Show detailed progress

**Returns:**
Comprehensive answer with sources cited in markdown format, including metadata about iterations, duration, tokens, and source URLs.

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
- `full` (optional): Return full fetched content without extraction
- `chunks` (optional): Use chunk selection instead of summarization
- `instructions` (optional): Custom extraction instructions
- `verbose` (optional): Show provider debug output

**Returns:**
Structured fetch payload matching `nexi-fetch --json`, including `pages` and `provider_failures`.

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
- MCP tool names intentionally mirror the runtime surfaces they expose
