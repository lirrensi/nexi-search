# NEXI 🔍

**Self-hosted perplexity in your terminal.** LLM-powered web search that stays out of your way.

![NEXI](https://raw.githubusercontent.com/lirrensi/nexi-search/main/img/nexi.jpg)

---

## Why NEXI Exists

You have questions. The web has answers. But getting from A to B shouldn't require:

- A browser tab you'll forget to close
- A $20/month subscription to a search product
- Your coding agent's precious context window filled with search noise

**NEXI is for people who live in the terminal.**

It's the search tool you pipe, script, and forget. Point your agents at it, or just type a query and get an answer with sources. No MCP dance required — just a CLI that works.

### The Problem with AI Search Today

| Pain Point | How NEXI Helps |
|------------|----------------|
| Perplexity et al. are expensive | Use any OpenAI-compatible API, including cheap models |
| Web search pollutes your agent's context | Offload searches to a dedicated tool, get summarized results |
| Browser tabs multiply like rabbits | Stay in your terminal, pipe to files, script away |
| "AI search" is a black box | `--verbose` shows every LLM call, token, and decision |

### Philosophy

- **Own your tools.** Your API keys, your models, your data.
- **Stay minimal.** Vibecoded in 2 hours. No bloat.
- **Be transparent.** Every decision is visible if you want to see it.
- **Respect the workflow.** CLI-first, pipe-friendly, script-ready.

> *Searches take 1-2 minutes for deep research. That's the cost of thoroughness. Go make tea.*

---

## Quick Start

```bash
# Install
uv tool install git+https://github.com/lirrensi/nexi-search.git

# Run (first time = interactive config)
nexi "what is the deal with rust async traits"
```

**You'll need:**
- An OpenAI-compatible API (OpenRouter, OpenAI, local, etc.)
- A Jina AI API key — **free** at [jina.ai](https://jina.ai)

---

## Configuration

Config lives at `~/.local/share/nexi/config.json`. Edit with `nexi --edit-config` or:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model": "google/gemini-2.5-flash-lite-preview-09-2025",
  "jina_key": "jina_...",
  "default_effort": "m",
  "time_target": 600,
  "max_output_tokens": 8000,
  "max_context": 128000,
  "auto_compact_thresh": 0.9,
  "compact_target_words": 5000,
  "preserve_last_n_messages": 3,
  "tokenizer_encoding": "cl100k_base"
}
```

### Context Management (Auto-Compact)

NEXI automatically manages conversation context to prevent token overflow during long searches:

- **`max_context`**: Model's context window limit (default: 128000)
- **`auto_compact_thresh`**: Trigger compaction at this fraction of context (default: 0.9 = 90%)
- **`compact_target_words`**: Target word count for summaries (default: 5000)
- **`preserve_last_n_messages`**: Number of recent assistant messages to keep un-compacted (default: 3)
- **`tokenizer_encoding`**: tiktoken encoding name (default: cl100k_base)

When approaching context limits, NEXI:
1. Extracts metadata (search queries, URLs) from conversation
2. Generates a dense summary of findings using the LLM
3. Rebuilds context with preserved messages and summary
4. Continues search without losing important information

Use `--verbose` to see token counts and compaction details.

---

## Usage

```bash
# basic search
nexi "how do rust async traits work"

# effort levels: s (quick), m (balanced), l (deep)
nexi -e l "explain quantum entanglement"
nexi -e s "what is a test"

# pipe it
echo "what is this" | nexi

# interactive REPL
nexi

# history
nexi --last 5      # show last 5 searches
nexi --prev        # show full result of latest search
nexi --show abc123 # show specific search by ID

# scripting
nexi --plain "no colors/emojis"
nexi --verbose "see all the LLM calls"
```

### CLI Options

| Flag | Description |
|------|-------------|
| `query` | Search query (positional, optional) |
| `-e, --effort s/m/l` | Search depth: small/medium/large (default: m) |
| `-v, --verbose` | Show LLM calls, tokens, tool execution |
| `--plain` | No colors/emojis, script-friendly |
| `--max-len N` | Limit output token length |
| `--max-iter N` | Max search iterations |
| `--time-target N` | Soft limit: force final answer after N seconds |
| `--last N` | Show last N searches |
| `--prev` | Show full result of latest search |
| `--show ID` | Show specific search by ID |
| `--clear-history` | Wipe search history |
| `--config` | Show config file path |
| `--edit-config` | Open config in editor |

### Effort Levels

| Level | Description | When to use |
|-------|-------------|-------------|
| `s` | Quick search (8 iterations) | Simple facts, definitions |
| `m` | Balanced (16 iterations) | Most queries |
| `l` | Deep research (32 iterations) | Complex topics |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| 401 Unauthorized | Check API keys in config |
| 429 Rate Limit | Wait or switch to different/cheaper model |
| Time target hit | Increase `time_target` or use `--time-target` |
| UTF-8 garbled (Windows) | Use `--plain` or Windows Terminal |

---

## MCP Server

NEXI can run as an MCP (Model Context Protocol) server, providing a `nexi_search` tool to MCP-compatible applications like Claude Desktop.

### Installation

```bash
# Install with MCP support
uv sync --group mcp
# or
pip install -e ".[mcp]"
```

### Running the Server

**STDIO transport (for local MCP clients):**
```bash
python -m nexi.mcp_server_cli
# or
uv run python -m nexi.mcp_server_cli
```

**HTTP transport (for network access):**
```bash
python -m nexi.mcp_server_cli http 0.0.0.0 8000
# or
uv run python -m nexi.mcp_server_cli http 0.0.0.0 8000
```

### MCP Tool: `nexi_search`

**Parameters:**
- `query` (required): The search query
- `effort` (optional): "s" (quick), "m" (medium, default), "l" (deep)
- `max_iter` (optional): Override max iterations
- `time_target` (optional): Soft limit: force final answer after N seconds
- `verbose` (optional): Show detailed progress

**Returns:** Comprehensive answer with sources in markdown format, including metadata (iterations, duration, tokens, URLs).

### Claude Desktop Configuration

Add to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, or equivalent on other platforms):

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

See [MCP_SERVER.md](docs/MCP_SERVER.md) for more details.

---

## License

MIT
