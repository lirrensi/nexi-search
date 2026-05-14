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
- **Multi-Key Provider Reliability** — Each provider instance supports multiple API keys with configurable fallback or round-robin strategies
- **Custom Python Providers** — Drop a Python file next to the config to add custom search, fetch, or LLM backends without modifying the NEXI codebase
- **Automatic Failover** — Hard provider failures and empty search responses fall through to the next configured backend without aborting the workflow
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

- **MCP Server** — Expose NEXI as Model Context Protocol tools for the full agent, direct search, and direct fetch surfaces
- **Provider-Orchestrated LLM Access** — Works with multiple OpenAI-compatible providers via ordered fallback chains
- **Provider-Orchestrated Search/Fetch** — Search and content retrieval providers can be mixed, reordered, and replaced independently
- **Multi-Key Support** — Each provider config can specify a list of API keys with `fallback` or `round_robin` strategy for higher reliability
- **Custom Providers** — Define search, fetch, or LLM backends as local Python files referenced via `provider-<name>` type strings

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
│  │ - OpenAI-compatible  │      │ - Jina / SearXNG             │ │
│  │ e.g. OpenRouter,     │      │ - Exa / Tavily / Firecrawl   │ │
│  │   OpenAI, local LLM  │      │ - Brave / SerpAPI / Serper   │ │
│  │                      │      │ - Linkup / Perplexity        │ │
│  │                      │      │ - snitchmd / Crawl4AI        │ │
│  │                      │      │ - markdown_new / Trafilatura │ │
│  │                      │      │ - Playwright / custom Python │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## User Flows

### Flow 1: First Run Bootstrap

```bash
nexi "what is the deal with rust async traits"
```

1. User runs NEXI without an existing config
2. NEXI creates `~/.config/nexi/config.toml` from the default commented template
3. NEXI prints the config path and a prominent warning that the config is still incomplete
4. NEXI exits immediately without starting a search
5. User opens the file with `nexi config` or runs `nexi onboard`

### Flow 2: Quick Search

```bash
nexi "how do rust async traits work"
```

1. User provides query via CLI argument
2. NEXI loads config, initializes search loop
3. LLM searches web, reads relevant pages, synthesizes answer
4. Answer displayed with citations
5. Search saved to history

### Flow 3: Optional Onboarding

```bash
nexi onboard
```

1. User already has the generated TOML template
2. NEXI runs a small wizard focused on the basics
3. Wizard helps activate one LLM provider and one search provider from the shipped options
4. Existing zero-config fetch defaults (`snitchmd`, `special_trafilatura`, `special_playwright`, `markdown_new`) stay enabled unless the user opts out
5. Crawl4AI is offered as a fetch option alongside other providers if the user customizes the fetch chain
6. Advanced settings remain in the file for manual editing

### Flow 4: Deep Research

```bash
nexi -e l "explain quantum entanglement and its applications"
```

1. User requests thorough search (32 iterations)
2. LLM explores multiple angles in parallel
3. Context compaction triggers if needed
4. Comprehensive answer with many sources

### Flow 5: Interactive Multi-Turn

```bash
nexi
```

1. User enters interactive REPL
2. First query starts fresh search
3. Follow-up queries include conversation history
4. LLM can reference previous findings
5. Type `exit` to quit

### Flow 6: Scripted Usage

```bash
echo "what is the capital of france" | nexi --plain
```

1. Query piped via stdin
2. Plain output (no colors/emoji)
3. Suitable for scripting and automation

### Flow 7: Recover Last Finished Result

```bash
nexi --prev
```

1. User asks for the latest completed search result
2. NEXI reads `~/.config/nexi/history.jsonl`
3. The most recent finished entry is printed in full
4. This recovers completed backgrounded runs, but not in-progress searches

### Flow 8: MCP Integration

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

1. Claude Desktop calls one of `nexi_agent`, `nexi_search`, or `nexi_fetch`
2. NEXI routes the request to the matching runtime surface
3. Agent calls return synthesized answers with metadata; direct tools return search or fetch payloads

### Flow 9: Direct Search Tool

```bash
nexi-search "rust async trait objects"
nexi-search --provider jina "rust async trait objects"
nexi-search --json "rust async trait objects"
```

1. User or agent calls the direct search binary
2. By default, NEXI executes the configured search backend chain
3. `--provider NAME` narrows execution to one named provider and bypasses the fallback chain
4. `--json` returns structured JSON output suitable for scripts and agents
5. Failed queries retry within the active provider, and empty-result queries fall through to the next provider immediately unless overridden
6. Plain text or JSON results are written to stdout

### Flow 10: Direct Fetch Tool

```bash
nexi-fetch "https://example.com/spec"
nexi-fetch --provider special_trafilatura "https://example.com/spec"
nexi-fetch --json "https://example.com/spec"
```

1. User or agent calls the direct fetch binary
2. By default, NEXI executes the configured fetch backend chain
3. `--provider NAME` narrows execution to one named provider and bypasses the fallback chain
4. `--json` returns structured JSON output with provider failure metadata
5. Failed URLs retry within the active provider, then fall through to the next provider unless overridden
6. Direct fetch output is capped at 8000 tokens per page outside agent mode
7. Oversized pages spill the full content to a temp file and print the absolute path alongside the truncated output
8. Extracted content is written to stdout and can be piped to files or other tools

---

## CLI Commands

### Search Commands

| Command | Description |
|---------|-------------|
| `nexi "query"` | Search with default effort |
| `nexi -e s "query"` | Quick search |
| `nexi -e m "query"` | Balanced search |
| `nexi -e l "query"` | Thorough search |
| `nexi -v "query"` | Verbose mode (show all LLM calls) |
| `nexi --plain "query"` | No colors/emoji (scripting) |

### Direct Tool Commands

| Command | Description |
|---------|-------------|
| `nexi-search "query"` | Run direct search without the agentic loop |
| `nexi-search --json "query"` | Return structured search results for scripts/agents |
| `nexi-search --provider NAME "query"` | Run one named search provider only |
| `nexi-fetch "url"` | Fetch/extract content from one or more URLs |
| `nexi-fetch --json "url"` | Return structured fetch results for scripts/agents |
| `nexi-fetch --provider NAME "url"` | Run one named fetch provider only |

### MCP Tools

| Tool | Description |
|------|-------------|
| `nexi_agent` | Full agentic workflow exposed over MCP; mirrors `nexi` |
| `nexi_search` | Direct backend search tool exposed over MCP; mirrors `nexi-search` |
| `nexi_fetch` | Direct backend fetch tool exposed over MCP; mirrors `nexi-fetch` |

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
| `nexi config` | Open the current config file in the user's editor |
| `nexi init` | Create the default config template if no config exists |
| `nexi onboard` | Run the small guided setup for basic provider activation |
| `nexi doctor` | Check whether the current config is valid and usable |
| `nexi clean` | Reset local config/history and recreate a fresh template |

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
2. Effort budget exhausted, then NEXI requests the best final answer from gathered information
3. Internal context guard rails stop further expansion and NEXI returns the best final answer available
4. API error after all retries exhausted
5. User cancellation (Ctrl+C)

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
- If context still cannot be recovered after compaction, NEXI stops expanding the search and returns the best final answer available

### Error Handling

- Missing config does not start a search — NEXI creates the template, prints the path, warns, and exits
- `nexi doctor` runs config readiness checks for all three public surfaces (`nexi`, `nexi-search`, `nexi-fetch`)
- Individual tool failures do not abort search — errors returned in results
- Search and fetch retry failed items within the active provider before failing over
- LLM provider failures trigger immediate provider failover for the current run
- If a listed provider is missing required config, validation fails before execution
- Search and fetch provider retries use exponential backoff before failover
- Multi-key APIs try each key in configured order (`fallback` or `round_robin`) before declaring provider failure
- Network errors are caught and reported, search continues
- Keyboard interrupt gracefully exits with code 130

### Provider Failover

- `search_backends`, `fetch_backends`, and `llm_backends` are ordered lists
- Search and fetch preserve successful items and only re-route failed queries or URLs
- LLM failover happens on the first hard provider failure or model-not-found response
- Provider failures are surfaced in logs/output so dead or unpaid providers are visible

---

## Configuration

### Format and Location

- **Config path**: `~/.config/nexi/config.toml`
- **History path**: `~/.config/nexi/history.jsonl`
- **Format**: TOML with comments in the generated template
- **Primary editing path**: `nexi config`

### Bootstrap Rules

- If `config.toml` is missing, NEXI MUST generate the default template and exit immediately
- The generated template MUST show all shipped providers, with inactive providers commented out
- The generated template SHOULD keep zero-config fetch providers enabled by default
- The generated template SHOULD leave LLM and search activation for the user to choose
- `nexi onboard` MAY modify the active provider chains, but advanced knobs remain file-driven

### Required Fields For A Usable Search Config

| Field | Description |
|-------|-------------|
| `llm_backends` | Ordered LLM provider chain for the full `nexi` workflow |
| `search_backends` | Ordered search provider chain |
| `fetch_backends` | Ordered fetch provider chain |
| `providers` | Provider config objects keyed by provider instance name |
| `default_effort` | Default effort level: `s`, `m`, or `l` |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `max_context` | `128000` | Model's context window limit |
| `auto_compact_thresh` | `0.9` | Trigger compaction at this fraction |
| `compact_target_words` | `5000` | Target word count for summaries |
| `preserve_last_n_messages` | `3` | Recent messages to keep un-compacted |
| `tokenizer_encoding` | `cl100k_base` | tiktoken encoding name |
| `provider_timeout` | `30` | Default timeout for provider API calls |
| `direct_fetch_max_tokens` | `8000` | Max emitted tokens per page for direct fetch output |
| `search_provider_retries` | `2` | Retry attempts per search provider before failover |
| `fetch_provider_retries` | `2` | Retry attempts per fetch provider before failover |

### Provider Config Fields

Each provider config object under `[providers.<name>]` may include:

| Field | Description |
|-------|-------------|
| `type` | **Required.** Provider type string (e.g. `openai_compatible`, `jina`, `searxng`, `provider-<file>`) |
| `api_key` | API key as a string or a **list of strings** for multi-key support |
| `api_key_strategy` | `"fallback"` (default) — try keys in order; `"round_robin"` — rotate starting key per request |

Provider-specific fields (like `model`, `base_url`, `max_results`, etc.) are documented per provider type.

### Default Template Shape

The generated default template keeps quiet zero-config fetch providers active by default and shows commented examples for LLM, search, and additional fetch providers.

```toml
# Activate at least one LLM provider and one search provider before running `nexi`.
# The default fetch chain uses the quiet providers only.
# Provider instances are shared across chains.
# api_key may be either a single string or a list of strings (for multiple keys).
# api_key_strategy controls per-provider key behaviour:
#   "fallback"    - try keys in order until one succeeds (default)
#   "round_robin" - rotate the starting key across requests in the same process
# Define each [providers.<name>] table only once, then reuse that name in
# search_backends and fetch_backends.
# If you need different settings for search and fetch, use different names
# like "jina_search" and "jina_fetch".

llm_backends = []
search_backends = []
fetch_backends = ["snitchmd", "special_trafilatura", "special_playwright", "markdown_new"]

default_effort = "m"
max_context = 128000
auto_compact_thresh = 0.9
compact_target_words = 5000
preserve_last_n_messages = 3
tokenizer_encoding = "cl100k_base"
provider_timeout = 30
direct_fetch_max_tokens = 8000
search_provider_retries = 2
fetch_provider_retries = 2

# Active provider configs
[providers.snitchmd]
type = "snitchmd"
mode = "precision"

[providers.special_trafilatura]
type = "special_trafilatura"

[providers.special_playwright]
type = "special_playwright"

[providers.markdown_new]
type = "markdown_new"
method = "auto"
retain_images = false
```

Commenting out the active LLM/search lines and their provider configs keeps them ready to activate by uncommenting. The full generated template includes commented examples for all shipped providers.

- Each listed backend name MUST have a matching entry in `providers`
- Each provider config object MUST declare its provider `type`
- Each provider class validates its own config requirements
- Providers MAY define additional provider-specific settings inside their config object
- Shipped and planned provider families are defined in [provider-matrix.md](provider-matrix.md)

### Result Recovery

- `nexi --prev` returns the latest completed history entry
- `nexi --last N` returns recent completed entries
- `nexi --show ID` returns one completed entry by ID
- History does not contain in-progress searches

---

## Effort Levels

| Level | Description | When to Use |
|-------|-------------|-------------|
| `s` | Quick search | Simple facts, definitions |
| `m` | Balanced | Most queries (default) |
| `l` | Thorough research | Complex topics, deep dives |

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
| 401 Unauthorized | Check provider config and API keys (use `nexi doctor`) |
| 429 Rate Limit | Add more API keys to the provider config for multi-key fallback |
| All providers failed | Use `nexi doctor` to check which providers are misconfigured |
| Search finalized early under context guard rails | Increase `max_context` or tune compaction settings |
| UTF-8 garbled (Windows) | Use `--plain` or Windows Terminal |
| MCP Server not found | Install with `uv sync --group mcp` |
| `nexi` says config not ready | Run `nexi doctor` to see specific readiness issues |

---

## References

- **Architecture**: [arch.md](arch.md) — Complete technical specification
- **Provider Matrix**: [provider-matrix.md](provider-matrix.md) — Supported and planned provider families
- **MCP Server**: [MCP_SERVER.md](MCP_SERVER.md) — MCP integration details
- **OpenAI API**: https://platform.openai.com/docs/api-reference — LLM API reference
