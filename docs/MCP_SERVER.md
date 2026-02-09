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

## Available Tool

### `nexi_search`

Perform an intelligent web search using NEXI.

**Parameters:**
- `query` (required): The search query to investigate
- `effort` (optional): Search depth - "s" (quick), "m" (medium, default), "l" (deep)
- `max_iter` (optional): Override maximum search iterations
- `max_timeout` (optional): Force return after N seconds
- `verbose` (optional): Show detailed progress

**Returns:**
Comprehensive answer with sources cited in markdown format, including metadata about iterations, duration, tokens, and source URLs.

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

- The MCP server uses the same configuration as the CLI tool (`~/.local/share/nexi/config.json`)
- All searches run synchronously within the MCP tool context
- The tool returns formatted markdown with metadata for easy parsing
