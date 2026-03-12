# NEXI — Product Specification

> **Self-hosted Perplexity in your terminal.** LLM-powered web search that stays out of your way.

---

## Overview

NEXI is an intelligent research CLI built around an agentic search loop powered by Large Language Models (LLMs). It combines provider-orchestrated web search, content fetching, and intelligent synthesis to deliver high-quality responses with proper source attribution, while keeping each backend pluggable and replaceable.

**Why NEXI exists:**

- **Quick console search** — Pipe it, script it, use it however you want
- **Offload from your coding agent** — No MCPs needed, just point your agent at a CLI tool
- **Cheaper models** — Use flash-lite instead of burning your rate limits on search
- **Self-hosted** — Your data stays on your machine

---

## Features

### Core Capabilities

- **Agentic Search Loop** — LLM autonomously decides what to search for and which pages to read
- **Multi-Query Parallel Search** — Execute 1-5 search queries in parallel for efficiency
- **Intelligent Content Extraction** — Three modes: full content, chunk-based selection, or LLM summarization
- **Pluggable Provider Chains** — Search, fetch, and LLM backends are interchangeable and ordered by preference
- **Automatic Failover** — Hard provider failures fall through to the next configured backend without aborting the workflow
- **Automatic Context Management** — Prevents token overflow via intelligent conversation compaction
- **Citation System** — Automatic source attribution with stable numbering throughout search
- **Multi-Turn Conversations** — Interactive REPL mode with conversation history

### CLI Features

- **Effort Levels** — Quick (s), balanced (m), or thorough (l) search depth
- **History Management** — Browse, view, and retrieve past searches
- **Flexible Input** — Query via argument, stdin, or interactive REPL
- **Script-Friendly** — Plain mode for piping and automation
- **Direct Tool CLIs** — `nexi-search` and `nexi-fetch` expose the backend layer without the full agent loop
- **Verbose Mode** — See all LLM calls, tool executions, and token usage

### Integration

- **MCP Server** — Expose NEXI as a Model Context Protocol tool for Claude Desktop and other MCP clients
- **Provider-Orchestrated LLM Access** — Works with multiple OpenAI-compatible providers via ordered fallback chains
- **Provider-Orchestrated Search/Fetch** — Search and content retrieval providers can be mixed, reordered, and replaced independently

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│  (nexi/cli.py)                                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Search Engine Core                           │
│                    (nexi/search.py)                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Agentic Search Loop                         │  │
│  │  1. Call LLM with tools                                  │  │
│  │  2. Execute tools (web_search, web_get, final_answer)    │  │
│  │  3. Manage context (compact if needed)                   │  │
│  │  4. Repeat until final_answer or limit reached           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Provider Orchestration Layer                  │  │
│  │  - llm_backends                                          │  │
│  │  - search_backends                                       │  │
│  │  - fetch_backends                                        │  │
│  │  - provider validation + failover                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
         ▼           ▼           ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│   Tools      │ │Citations │ │  Compaction  │
│(nexi/tools.py)│ │(nexi/    │ │(nexi/        │
│              │ │citations │ │compaction.py)│
│- web_search  │ │.py)      │ │              │
│- web_get     │ │          │ │- Extract     │
│- final_answer│ │- Track   │ │- Summarize   │
└──────────────┘ │- Format  │ │- Rebuild     │
                 └──────────┘ └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     External Services                           │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   LLM Providers      │      │ Search / Fetch Providers     │ │
│  │   (ordered chain)    │      │ (ordered chains)             │ │
│  │ - OpenRouter         │      │ - Jina                       │ │
│  │ - OpenAI-compatible  │      │ - Exa / Tavily / others      │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## User Flows

### Flow 1: Quick Search

```bash
nexi "how do rust async traits work"
```

1. User provides query via CLI argument
2. NEXI loads config, initializes search loop
3. LLM searches web, reads relevant pages, synthesizes answer
4. Answer displayed with citations
5. Search saved to history

### Flow 2: Deep Research

```bash
nexi -e l "explain quantum entanglement and its applications"
```

1. User requests thorough search (32 iterations)
2. LLM explores multiple angles in parallel
3. Context compaction triggers if needed
4. Comprehensive answer with many sources

### Flow 3: Interactive Multi-Turn

```bash
nexi
```

1. User enters interactive REPL
2. First query starts fresh search
3. Follow-up queries include conversation history
4. LLM can reference previous findings
5. Type `exit` to quit

### Flow 4: Scripted Usage

```bash
echo "what is the capital of france" | nexi --plain
```

1. Query piped via stdin
2. Plain output (no colors/emoji)
3. Suitable for scripting and automation

### Flow 5: MCP Integration

```json
{
  "mcpServers": {
    "nexi": {
      "command": "uv",
      "args": ["run", "python", "-m", "nexi.mcp_server_cli"]
    }
  }
}
```

1. Claude Desktop calls `nexi_search` tool
2. NEXI performs search
3. Returns markdown-formatted answer with metadata

### Flow 6: Direct Search Tool

```bash
nexi-search "rust async trait objects"
```

1. User or agent calls the direct search binary
2. NEXI executes the configured search backend chain
3. Failed queries retry within the active provider, then fall through to the next provider
4. Plain text or JSON results are written to stdout

### Flow 7: Direct Fetch Tool

```bash
nexi-fetch "https://example.com/spec"
```

1. User or agent calls the direct fetch binary
2. NEXI executes the configured fetch backend chain
3. Failed URLs retry within the active provider, then fall through to the next provider
4. Extracted content is written to stdout and can be piped to files or other tools

---

## CLI Commands

### Search Commands

| Command | Description |
|---------|-------------|
| `nexi "query"` | Search with default effort |
| `nexi -e s "query"` | Quick search (8 iterations) |
| `nexi -e m "query"` | Balanced search (16 iterations) |
| `nexi -e l "query"` | Thorough search (32 iterations) |
| `nexi --max-iter N "query"` | Custom max iterations |
| `nexi --time-target N "query"` | Force answer after N seconds |
| `nexi --max-len N "query"` | Limit output tokens |
| `nexi -v "query"` | Verbose mode (show all LLM calls) |
| `nexi --plain "query"` | No colors/emoji (scripting) |

### Direct Tool Commands

| Command | Description |
|---------|-------------|
| `nexi-search "query"` | Run direct search without the agentic loop |
| `nexi-search --json "query"` | Return structured search results for scripts/agents |
| `nexi-fetch "url"` | Fetch/extract content from one or more URLs |
| `nexi-fetch --json "url"` | Return structured fetch results for scripts/agents |

### History Commands

| Command | Description |
|---------|-------------|
| `nexi --last N` | Show last N searches (preview) |
| `nexi --prev` | Show full latest search result |
| `nexi --show ID` | Show specific search by ID |
| `nexi --clear-history` | Delete all search history |

### Config Commands

| Command | Description |
|---------|-------------|
| `nexi --config` | Print config file path |
| `nexi --edit-config` | Open config in $EDITOR |

### Interactive Mode

| Command | Description |
|---------|-------------|
| `nexi` | Enter interactive REPL |
| `exit`, `quit`, `q` | Exit interactive mode |
| `help`, `h` | Show available commands |

---

## Behavior Guarantees

### Search Termination

The search loop **MUST** terminate when **any** of these conditions are met:

1. `final_answer` tool called by LLM
2. Time target exceeded (if set)
3. Max iterations reached
4. Context limit exceeded (after failed compaction)
5. API error after all retries exhausted
6. User cancellation (Ctrl+C)

### Citation Behavior

- URLs are assigned stable citation numbers `[1]`, `[2]`, etc. on first fetch
- Numbers remain consistent throughout the search session
- Final answer includes formatted sources section
- Model is instructed to use `[N]` citations in responses

### Context Management

- Compaction triggers when `current_tokens + estimated_next > max_context * threshold`
- Compaction preserves: system prompt, original query, last N assistant messages
- Compaction generates dense summary of findings
- If compaction fails, search continues with original messages
- If still over limit after compaction, forced answer is returned

### Error Handling

- Individual tool failures do not abort search — errors returned in results
- Search and fetch retry failed items within the active provider before failing over
- LLM provider failures trigger immediate provider failover for the current run
- If a listed provider is missing required config, validation fails before execution
- Search and fetch provider retries use exponential backoff before failover
- Network errors are caught and reported, search continues
- Keyboard interrupt gracefully exits with code 130

### Provider Failover

- `search_backends`, `fetch_backends`, and `llm_backends` are ordered lists
- Search and fetch preserve successful items and only re-route failed queries or URLs
- LLM failover happens on the first hard provider failure or model-not-found response
- Provider failures are surfaced in logs/output so dead or unpaid providers are visible

---

## Configuration

### Required Fields

| Field | Description |
|-------|-------------|
| `llm_backends` | Ordered LLM provider chain for the full `nexi` workflow |
| `search_backends` | Ordered search provider chain |
| `fetch_backends` | Ordered fetch provider chain |
| `providers` | Provider config objects keyed by provider instance name |
| `default_effort` | Default effort level: `s`, `m`, or `l` |
| `max_output_tokens` | Maximum tokens in final answer |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `time_target` | `null` | Soft time limit in seconds |
| `max_context` | `128000` | Model's context window limit |
| `auto_compact_thresh` | `0.9` | Trigger compaction at this fraction |
| `compact_target_words` | `5000` | Target word count for summaries |
| `preserve_last_n_messages` | `3` | Recent messages to keep un-compacted |
| `tokenizer_encoding` | `cl100k_base` | tiktoken encoding name |
| `provider_timeout` | `30` | Default timeout for provider API calls |
| `search_provider_retries` | `2` | Retry attempts per search provider before failover |
| `fetch_provider_retries` | `2` | Retry attempts per fetch provider before failover |

### Provider Configuration Shape

```json
{
  "llm_backends": ["openrouter", "openai"],
  "search_backends": ["jina"],
  "fetch_backends": ["jina"],
  "providers": {
    "openrouter": {
      "type": "openai_compatible",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "<api_key>",
      "model": "google/gemini-2.5-flash-lite"
    },
    "jina": {
      "type": "jina",
      "api_key": "<api_key>"
    },
    "openai": {
      "type": "openai_compatible",
      "base_url": "https://api.openai.com/v1",
      "api_key": "<api_key>",
      "model": "gpt-4.1-mini"
    }
  }
}
```

- Each listed backend name MUST have a matching entry in `providers`
- Each provider config object MUST declare its provider `type`
- Each provider class validates its own config requirements
- Providers MAY define additional provider-specific settings inside their config object

### Config File Location

- **Path**: `~/.local/share/nexi/config.json`
- **Edit**: `nexi --edit-config` or open directly

---

## Effort Levels

| Level | Iterations | Description | When to Use |
|-------|------------|-------------|-------------|
| `s` | 8 | Quick search | Simple facts, definitions |
| `m` | 16 | Balanced | Most queries (default) |
| `l` | 32 | Thorough research | Complex topics, deep dives |

---

## Tool Definitions

### web_search

Search the web for information.

**Parameters:**
- `queries` (required): List of 1-5 search queries to execute in parallel

**Returns:** Search results with titles, URLs, and snippets

### web_get

Fetch and process content from URLs.

**Parameters:**
- `urls` (required): List of 1-8 URLs to fetch
- `instructions` (optional): Custom prompt for LLM extraction
- `get_full` (optional): Return raw content without processing (default: false)
- `use_chunks` (optional): Use chunk-based selection instead of summarization (default: false)

**Modes:**
1. `get_full=true`: Raw page content
2. `use_chunks=true`: Split into chunks, LLM picks relevant ones
3. Default: LLM summarizes based on instructions

**Returns:** Extracted/processed content with citation markers

### final_answer

Provide the final answer and terminate search.

**Parameters:**
- `answer` (required): The complete answer in markdown

**Behavior:** Immediately terminates the search loop

---

## Non-Goals

NEXI deliberately does **NOT** aim to be:

- **A general-purpose chatbot** — Focused on web search, not open-ended conversation
- **A web browser** — No JavaScript rendering, no interactive browsing
- **A search engine** — Orchestrates third-party providers, doesn't crawl the web itself
- **A citation manager** — Basic citations only, no BibTeX or academic formatting
- **A research paper writer** — Provides answers, doesn't generate papers
- **A replacement for human research** — Tool for assistance, not autonomous research

---

## Unresolved Questions

1. **Streaming responses** — Should NEXI stream answers as they're generated? Would improve perceived latency but complicates citation handling.

2. **Persistent URL cache** — Currently cache is per-session. Should we persist across sessions? Trade-off: freshness vs. speed.

3. **Multi-model support** — Should NEXI support using different models for search vs. extraction vs. compaction? Would add complexity.

4. **Custom tool definitions** — Should users be able to define custom tools? Would require security considerations.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Check provider config and API keys |
| 429 Rate Limit | Wait or rely on the next configured provider |
| Time target hit | Increase `time_target` or use `--time-target` |
| Context limit exceeded | Increase `max_context` or lower `auto_compact_thresh` |
| UTF-8 garbled (Windows) | Use `--plain` or Windows Terminal |
| MCP Server not found | Install with `uv sync --group mcp` |

---

## References

- **Architecture**: [arch.md](arch.md) — Complete technical specification
- **MCP Server**: [MCP_SERVER.md](MCP_SERVER.md) — MCP integration details
- **OpenAI API**: https://platform.openai.com/docs/api-reference — LLM API reference
