# NEXI Architecture Documentation

> **Complete technical specification.** Machine-readable, exhaustive. You could throw away the code and rebuild it entirely from this document.

---

## Overview

NEXI is an intelligent research CLI that uses an agentic search loop powered by Large Language Models (LLMs) to provide comprehensive, well-researched answers to user queries. It combines provider-orchestrated web search, content fetching, and intelligent synthesis to deliver high-quality responses with proper source attribution.

### Core Philosophy

- **Agentic Search**: The LLM autonomously decides what to search for and which pages to read
- **Provider-Orchestrated**: LLM, search, and fetch capabilities route through ordered provider chains
- **Context-Aware**: Automatically manages conversation context to prevent token overflow
- **Tool-Based**: Uses function calling to give the LLM structured access to web capabilities
- **Failure-Resistant**: Search and fetch retry within a provider, then fail over; LLM fails over immediately on hard failure
- **Efficient**: Parallel searches and intelligent caching minimize latency
- **Transparent**: Verbose mode shows all decisions, tool calls, provider failures, and token usage

### Scope Boundary

**This system owns**:
- Agentic search loop orchestration
- Provider orchestration for LLM, web search, and web fetch
- Conversation context management and compaction
- Citation tracking and formatting
- CLI interface and history management
- Direct `nexi-search` and `nexi-fetch` binaries
- MCP server for tool integration

**This system does NOT own**:
- LLM inference (delegates to OpenAI-compatible APIs)
- Search quality or ranking quality of external providers
- Web crawling (delegates to external providers)
- Content rendering (outputs plain text/markdown)

**Boundary interfaces**:
- Receives queries from CLI, stdin, or MCP
- Calls external LLM providers via OpenAI-compatible or provider-specific adapters
- Calls search and fetch providers through internal provider interfaces

---

## Dependencies

### External Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `click` | ^8.x | CLI framework |
| `httpx` | ^0.27+ | Async HTTP client |
| `openai` | ^1.x | OpenAI-compatible API client |
| `questionary` | ^2.x | Interactive prompts |
| `rich` | ^13.x | Terminal output formatting |
| `tiktoken` | ^0.5+ | Token counting |

### External Services

| Service | Endpoint | Purpose |
|---------|----------|---------|
| LLM Providers | Configurable per provider | Chat completions with function calling |
| Search Providers | Configurable per provider | Web search |
| Fetch Providers | Configurable per provider | Content extraction from URLs |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│  (nexi/cli.py, direct search/fetch entrypoints, main.py)        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─────────────────────────────────────────────┐
                     │                                             │
                     ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────────┐
│   Configuration Management   │              │      History Management          │
│   (nexi/config.py)           │              │      (nexi/history.py)           │
│                              │              │                                  │
│  - Load/Save Config          │              │  - JSONL Storage                 │
│  - Provider Validation       │              │  - Entry Creation/Retrieval      │
│  - Prompt Templates          │              │  - Formatting                    │
│  - Ordered Chains            │              │                                  │
└──────────────────────────────┘              └──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Search Engine Core                           │
│                    (nexi/search.py)                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Agentic Search Loop                         │  │
│  │                                                           │  │
│  │  1. Call LLM through llm_backends                        │  │
│  │  2. Parse tool_calls                                     │  │
│  │  3. Execute tools (web_search, web_get, final_answer)    │  │
│  │  4. Add results to conversation                          │  │
│  │  5. Check internal guard rails (effort, context)         │  │
│  │  6. Compact if needed                                    │  │
│  │  7. Repeat until final_answer or graceful finalization   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┬───────────────┐
         │           │           │               │
         ▼           ▼           ▼               ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
│   Tools      │ │Citations │ │  Compaction  │ │ Token Counter│
│(nexi/tools.py)│ │(nexi/    │ │(nexi/        │ │(nexi/token_  │
│              │ │citations │ │compaction.py)│ │counter.py)   │
│- web_search  │ │.py)      │ │              │ │              │
│- web_get     │ │          │ │- Extract     │ │- count_tokens│
│- final_answer│ │- Track   │ │- Summarize   │ │- count_msgs  │
│              │ │- Format  │ │- Rebuild     │ │- estimate    │
│- Orchestrators││- Detect  │ │              │ │              │
└──────────────┘ └──────────┘ └──────────────┘ └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     External Services                           │
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   LLM Providers      │      │   Search / Fetch Providers   │ │
│  │   (ordered chain)    │      │   (ordered chains)           │ │
│  │                      │      │                              │ │
│  │  - Chat Completions  │      │  - Search APIs               │ │
│  │  - Function Calling  │      │  - Reader / Extraction APIs  │ │
│  │  - Provider Failover │      │  - Provider Failover         │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. CLI Interface (`nexi/cli.py`, `main.py`, `nexi/__main__.py`, direct search/fetch entrypoints)

**Purpose**: Entry points for command-line interaction

**Runtime Surfaces**:
- `nexi`: Full agentic workflow with search loop, citations, history, and MCP-facing behavior
- `nexi-search`: Direct search CLI that exposes backend-orchestrated search without the agent loop
- `nexi-fetch`: Direct fetch CLI that exposes backend-orchestrated fetch/extraction without the agent loop

**Key Functions**:
- `main()`: Click-based CLI command handler for `nexi`
- `_run_search_command()`: Orchestrates full search execution
- `_interactive_mode()`: REPL for continuous queries with multi-turn support
- `_show_last_n()`, `_show_prev()`, `_show_by_id()`: History viewing
- Direct search/fetch entrypoints: Parse args, invoke orchestration layer, print plain text or JSON

**Features**:
- Argument parsing (query, effort, verbose, plain)
- stdin support for piping queries
- Interactive REPL mode with conversation history
- History management commands
- Config lifecycle commands: `config`, `init`, `onboard`, `doctor`, `clean`
- `--json` output mode for `nexi-search` and `nexi-fetch`
- stdout-first design so shell redirection/piping handles file output

**Multi-Turn Support**:
- Interactive mode tracks `conversation_history: list[dict[str, Any]]`
- First turn: fresh search with default system prompt
- Subsequent turns: uses `CONTINUATION_SYSTEM_PROMPT` with history
- History includes user queries and assistant responses

**Windows Compatibility**:
- UTF-8 handling in `main.py` and `nexi/__main__.py`
- Platform-specific editor selection (notepad on Windows)

---

### 2. Configuration Management (`nexi/config.py`)

**Purpose**: Load, validate, and manage user configuration

**Storage Layout**:
- Config directory: `~/.config/nexi/`
- Config file: `~/.config/nexi/config.toml`
- History file: `~/.config/nexi/history.jsonl`
- Config and history live together so reset, backup, and inspection are local to one directory

**Config Structure**:
```python
@dataclass
class Config:
    llm_backends: list[str]              # Ordered LLM provider chain
    search_backends: list[str]           # Ordered search provider chain
    fetch_backends: list[str]            # Ordered fetch provider chain
    providers: dict[str, dict[str, Any]] # Provider config objects by instance name
    default_effort: str                  # s/m/l
    max_context: int                     # Model's context window limit
    auto_compact_thresh: float           # Trigger compaction at this fraction
    compact_target_words: int            # Target word count for summaries
    preserve_last_n_messages: int        # Recent messages to keep un-compacted
    tokenizer_encoding: str              # tiktoken encoding name
    provider_timeout: int                # Default timeout for provider API calls
    search_provider_retries: int         # Retries per search provider before failover
    fetch_provider_retries: int          # Retries per fetch provider before failover
```

**Provider Configuration Model**:
- `providers` is the shared config registry for all provider instances
- Each top-level chain (`llm_backends`, `search_backends`, `fetch_backends`) references provider instance names defined in `providers`
- Each provider config object MUST include `type` so the registry can resolve the correct adapter class
- A provider config object MAY include `api_key`, `base_url`, `model`, headers, rate limits, or provider-specific tuning knobs
- A listed provider instance MUST exist in `providers`
- Each provider class MUST validate its own config before execution
- Supported and planned provider families are defined canonically in `docs/provider-matrix.md`

**Key Functions**:
- `load_config()`: Load from `~/.config/nexi/config.toml`
- `save_config()`: Persist configuration
- `validate_config()`: Validate TOML structure, provider references, and global settings
- `ensure_config()`: Load config or create the default template and stop the run
- `write_default_template()`: Create the default commented TOML template when config is missing
- `run_onboarding()`: Optional guided activation of a basic LLM + search setup
- `run_doctor()`: Check config syntax, active chains, required secrets, and command usability
- `get_system_prompt()`: Generate system prompt with effort level
- `get_compaction_prompt()`: Generate prompt for conversation summarization

**Bootstrap Rules**:
- Missing config MUST create the default TOML template and exit immediately
- The generated template MUST contain all shipped providers as visible config blocks
- Inactive providers SHOULD remain commented out in the generated template
- Zero-config fetch providers SHOULD be enabled by default in the template
- LLM and search providers SHOULD require explicit activation by the user
- `nexi onboard` is optional and MUST NOT run automatically on first use

**Doctor Rules**:
- `nexi doctor` MUST report both parse/shape errors and practical readiness errors
- For `nexi`, doctor MUST require at least one usable LLM provider and one usable search provider
- For `nexi-search`, doctor MUST require at least one usable search provider
- For `nexi-fetch`, doctor MUST require at least one usable fetch provider
- Readiness checks MUST verify required provider fields for the active chains, not just TOML syntax

**Prompt Templates**:
- `DEFAULT_SYSTEM_PROMPT_TEMPLATE`: Instructions for the search agent
- `EXTRACTOR_PROMPT_TEMPLATE`: Instructions for content extraction
- `COMPACTION_PROMPT_TEMPLATE`: Instructions for conversation summarization
- `CONTINUATION_SYSTEM_PROMPT`: Instructions for multi-turn conversations
- `CHUNK_SELECTOR_PROMPT`: Instructions for chunk-based content selection

**Effort Levels**:
```python
EFFORT_LEVELS = {
    "s": {"max_iter": 8, "description": "Quick search"},
    "m": {"max_iter": 16, "description": "Balanced"},
    "l": {"max_iter": 32, "description": "Thorough research"},
}
```

---

### 3. Search Engine Core (`nexi/search.py`)

**Purpose**: Orchestrate the agentic search loop

**Main Function**: `run_search()`

**SearchResult Data Structure**:
```python
@dataclass
class SearchResult:
    answer: str
    urls: list[str]
    url_citations: dict[str, int]      # URL -> citation number
    url_to_title: dict[str, str]       # URL -> title
    iterations: int
    duration_s: float
    tokens: int
```

**Search Loop Algorithm**:

```
1. Initialize:
    - Clear URL cache
    - Derive an internal iteration budget from effort level
    - Create conversation with system prompt + user query
    - Initialize LLM provider orchestrator from llm_backends
    - Initialize citation tracking (url_to_number, url_to_title)

2. For each iteration within the internal effort budget:
    a. Call LLM with tools and current conversation through the LLM provider chain
    b. Track token usage
    c. Parse response.message.tool_calls

    d. If tool_call is "final_answer":
       - Extract answer
       - Break loop

    e. If tool_call is "web_search":
       - Execute provider-orchestrated searches over pending queries
       - Retry failed queries inside the active provider
       - Fail over only remaining failed queries to the next provider
       - Capture titles from results for citations
       - Add assistant message (tool_calls) to conversation
       - Add tool message (results) to conversation

    f. If tool_call is "web_get":
       - Assign stable citation numbers to URLs
       - Execute provider-orchestrated fetches over pending URLs
       - Retry failed URLs inside the active provider
       - Fail over only remaining failed URLs to the next provider
       - Process content (full, chunks, or extraction)
       - Estimate tokens for result
       - Check if compaction needed:
           - If current + estimated > threshold:
             - Compact conversation
             - If still over limit:
               - Stop expanding search
               - Return the best final answer available from gathered information
       - Track URLs fetched
       - Add assistant message (tool_calls) to conversation
       - Add tool message (results with sources list) to conversation

    g. If no tool_call:
       - Force answer from message.content
       - Break loop

3. If the internal effort budget is exhausted:
   - Request final answer via _request_final_answer()

4. Return SearchResult with:
    - answer, urls, url_citations, url_to_title
    - iterations, duration_s, tokens
```

**LLM Provider Failover Rules**:
- `llm_backends` is evaluated in order
- On the first hard provider failure (auth, payment, rate-limit hard stop, network/provider unreachable, model-not-found), the active provider is marked unhealthy for the current run
- The same iteration is retried against the next configured LLM provider
- If all LLM providers fail, the search returns a forced error answer and terminates

**Citation Tracking**:
- `url_to_number: dict[str, int]`: Maps URLs to stable citation numbers [1], [2], etc.
- `url_to_title: dict[str, str]`: Maps URLs to titles (from web_search results)
- Citation numbers assigned on first URL fetch, remain stable throughout search
- Sources list appended to web_get results for model visibility

**Tool Schemas**:

Three tools are exposed to the LLM via function calling:

1. **web_search**
   - Parameters: `queries` (list[str], 1-5 items)
   - Returns: Search results with snippets and URLs
   - Used for: Initial information gathering

2. **web_get**
   - Parameters: `urls` (list[str], 1-8 items), `instructions` (str), `get_full` (bool), `use_chunks` (bool)
   - Returns: Extracted/summarized page content with citation markers
   - Used for: Deep reading of specific pages
   - Modes:
     - `get_full=true`: Raw content
     - `use_chunks=true`: Chunk-based selection
     - Default: LLM summarization

3. **final_answer**
   - Parameters: `answer` (str)
   - Returns: Terminates search loop
   - Used for: Providing the final synthesized answer

**Helper Functions**:
- `_get_tool_schemas()`: Return tool definitions for OpenAI API
- `_force_answer()`: Generate answer when limits reached
- `_request_final_answer()`: Request final answer from LLM when the effort budget is exhausted
- `run_search_sync()`: Synchronous wrapper using new event loop

**Retry Logic**:
- Search providers retry failed items up to `config.search_provider_retries` times (default: 2)
- Fetch providers retry failed items up to `config.fetch_provider_retries` times (default: 2)
- Search/fetch retries use exponential backoff before failover
- LLM providers do NOT use same-provider retry on hard failure; they fail over immediately
- On total provider chain failure, the search returns a forced error answer and terminates

---

### 4. Tool Implementations and Provider Orchestration (`nexi/tools.py`, `nexi/backends/`)

**Purpose**: Execute web search and content retrieval through provider interfaces and orchestration rules

**URL Caching**:
- In-memory cache `_url_cache: dict[str, str]`
- Cleared at start of each search session
- Prevents duplicate fetches of same URL

**HTTP Client**:
- Shared `httpx.AsyncClient` with connection pooling
- Limits: max_connections=10, max_keepalive_connections=5
- Closed via `close_http_client()`

#### Provider Protocols

```python
class Provider(Protocol):
    name: str

    def validate_config(self, config: dict[str, Any]) -> None:
        """Raise ValueError if required config is missing or invalid."""


class SearchProvider(Provider, Protocol):
    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Returns {"searches": [{"query": str, "results": [...], "error": str}]}"""


class FetchProvider(Provider, Protocol):
    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Returns {"pages": [{"url": str, "content": str, "error": str}]}"""


class LLMProvider(Provider, Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        verbose: bool,
    ) -> Any:
        """Returns provider response compatible with NEXI's chat loop."""
```

#### Provider Layout

- Provider classes live under `nexi/backends/`
- Providers are grouped by capability or shared module, but referenced by provider name in config
- A provider MAY implement one capability or several capabilities
- Adding a new provider SHOULD primarily require:
  1. adding the provider class,
  2. registering its capability mapping,
  3. documenting its config contract

#### Search / Fetch Orchestrators

**Search orchestration process**:
1. Start with the full query set as pending
2. Select the first provider instance from `search_backends`
3. Read `providers[provider_name]`
4. Resolve the provider adapter from `providers[provider_name]["type"]`
5. Validate provider config
6. Run the provider against pending queries
7. Keep successful query results
8. Retry only failed queries within the same provider up to `search_provider_retries`
9. Move any still-failed queries to the next provider in the chain
10. Return the combined result set plus provider failure metadata

**Fetch orchestration process**:
1. Start with the full URL set as pending
2. Select the first provider instance from `fetch_backends`
3. Read `providers[provider_name]`
4. Resolve the provider adapter from `providers[provider_name]["type"]`
5. Validate provider config
6. Run the provider against pending URLs
7. Keep successful URL fetches
8. Retry only failed URLs within the same provider up to `fetch_provider_retries`
9. Move any still-failed URLs to the next provider in the chain
10. Return the combined page set plus provider failure metadata

#### Direct Commands

- `nexi-search` calls the search orchestrator directly and prints either plain text or JSON
- `nexi-fetch` calls the fetch orchestrator directly and prints either extracted text or JSON
- Direct commands do not create citations/history unless explicitly requested by their own contract in a future revision
- Direct commands MUST exit non-zero if every configured provider fails for every requested item

**Response Format**:
```python
{
    "searches": [
        {
            "query": "search query",
            "results": [
                {
                    "title": "Page title",
                    "url": "https://example.com",
                    "description": "Snippet",
                    "published_time": "2024-01-01"  # optional
                }
            ],
            "error": "..."  # if failed
        }
    ]
}
```

#### Fetch Processing Modes

Raw page content returned by a fetch provider is processed in three modes:

1. Cache check for previously fetched URLs
2. Fetch raw content through the provider chain
3. Process raw content in one of the three `web_get` modes below
4. Store successful raw content in cache

#### web_get Processing Modes

**1. Full Content (`get_full=True`)**:
- Return raw markdown from the active fetch provider
- No LLM processing

**2. Chunk-Based Selection (`use_chunks=True`)**:
- Split content into logical chunks via `create_logical_chunks()`
- Ask LLM which chunks are relevant via `CHUNK_SELECTOR_PROMPT`
- Return only selected chunks
- Preserves original text, cheaper than summarization

**3. LLM Extraction (default)**:
- Use `EXTRACTOR_PROMPT_TEMPLATE` with custom instructions
- LLM summarizes/extracts relevant content
- Max tokens: 16384

**Chunk Creation**:
```python
def create_logical_chunks(md: str, target_chars: int = 480, max_chars: int = 720) -> list[Chunk]:
    """Heading-aware logical chunking for clean provider markdown.
     
    Respects headings, merges small paragraphs, splits oversized chunks.
    """
```

**Response Format**:
```python
{
    "pages": [
        {
            "url": "https://example.com",
            "content": "[1] https://example.com\n---\nExtracted content"
        }
    ]
}
```

**Error Handling**:
- HTTP errors captured and returned in result
- Exceptions caught and returned as error messages
- Individual query/URL failures don't abort entire operation
- Provider failure metadata is attached so users can see which provider failed and why

---

### 5. Citation System (`nexi/citations.py`)

**Purpose**: Track, format, and auto-add citations to answers

**Key Functions**:

1. **`extract_citation_markers(text)`**
   - Extract all `[N]` markers from text
   - Returns set of citation numbers

2. **`map_markers_to_urls(markers, number_to_url)`**
   - Map citation numbers to URLs
   - Returns dict of number -> URL

3. **`format_citations_section(marker_url_map, url_to_title, plain)`**
   - Format sources section for answer
   - Format: `[N] Title - URL` or `[N] URL`

4. **`add_citations_to_answer(answer, number_to_url, url_to_title, plain)`**
   - Add citations section if not already present
   - Auto-detect referenced URLs if no markers found

5. **`process_answer_with_citations(answer, url_citations, url_to_title, include_citations_section, plain)`**
   - Main entry point for citation post-processing

6. **`extract_citations_from_tool_result(result)`**
   - Extract URL citations from web_get tool result
   - Parses `[N] URL` format from content header

**Auto-Detection Heuristics**:
- Full URL matching
- Domain + path fragment matching
- Domain-only matching (excludes common words like "com", "github")

---

### 6. Conversation Compaction (`nexi/compaction.py`)

**Purpose**: Prevent token overflow by summarizing conversation history

**CompactionMetadata Data Structure**:
```python
@dataclass
class CompactionMetadata:
    search_queries: list[str]
    urls_fetched: list[str]
    original_query: str
```

**Trigger Condition**:
```python
def should_compact(current_tokens: int, estimated_next: int, config: Config) -> bool:
    threshold = int(config.max_context * config.auto_compact_thresh)
    return (current_tokens + estimated_next) > threshold
```

**Compaction Process**:

```
1. Extract Metadata:
   - Original query from first user message
   - All search queries from web_search tool_calls
   - All URLs from web_get tool_calls

2. Collect Content to Summarize:
   - All tool messages (web_get results)
   - All assistant messages with content

3. Generate Summary:
   - Call LLM with COMPACTION_PROMPT_TEMPLATE
   - Target: config.compact_target_words (default: 5000)
   - Preserve: numbers, dates, quotes, technical terms
   - Merge: duplicate information
   - Exclude: URLs (tracked separately)

4. Rebuild Context:
   - Keep: system prompt (index 0)
   - Keep: original user message (index 1)
   - Add: summary message with metadata + findings
   - Keep: last N assistant messages (config.preserve_last_n_messages)

5. Return new compacted message list
```

**Summary Message Format**:
```
Original query: {query}

Search queries performed:
- query 1
- query 2

Links navigated:
- url 1
- url 2

Findings:
{LLM-generated summary}
```

**Error Handling**:
- If summary generation fails → return original messages
- If still over limit after compaction → force answer and break

---

### 7. Token Counter (`nexi/token_counter.py`)

**Purpose**: Count tokens for context management

**Functions**:

1. **`get_encoding(encoding_name)`**
   - Get tiktoken encoding with caching
   - Default: `cl100k_base` (GPT-4)
   - Raises `ValueError` if encoding fails to load

2. **`count_tokens(text, encoding)`**
   - Count tokens in a text string
   - Returns 0 for empty text

3. **`count_messages_tokens(messages, encoding)`**
   - Count total tokens in OpenAI-format message list
   - Counts: role, content, tool_calls, tool_call_id
   - Adds 3 tokens per message for formatting (OpenAI convention)

4. **`estimate_page_tokens(content, encoding)`**
   - Estimate tokens for page content (tool result)
   - Wrapper around `count_tokens()`

---

### 8. History Management (`nexi/history.py`)

**Purpose**: Persist and retrieve search history

**Storage**: `~/.config/nexi/history.jsonl` (JSON Lines format)

**HistoryEntry Data Structure**:
```python
@dataclass
class HistoryEntry:
    id: str                    # Random hex ID (6 chars)
    ts: str                    # ISO 8601 timestamp
    query: str                 # User's query
    answer: str                # Final answer
    urls: list[str]            # URLs fetched
    effort: str                # s/m/l
    iterations: int            # Number of iterations
    duration_s: float          # Total duration
    tokens: int                # Total tokens used
```

**Key Functions**:
- `add_history_entry()`: Append entry to JSONL file
- `get_last_n_entries()`: Retrieve last N entries (most recent first)
- `get_entry_by_id()`: Find specific entry by ID
- `get_latest_entry()`: Get most recent entry
- `clear_history()`: Delete all history
- `create_entry()`: Create new HistoryEntry with timestamp
- `format_time_ago()`: Format timestamp as relative time
- `format_entry_preview()`: Format for --last display (truncated)
- `format_entry_full()`: Format for --prev/--show display (full)

**Error Handling**:
- Corrupted JSON lines are skipped
- Missing history file returns empty list

---

### 9. Output Formatting (`nexi/output.py`)

**Purpose**: Format and display search results

**CompactProgressTracker**:
- Tracks iterations and errors in non-TTY mode
- Generates compact summary: `[iteration 1 2 3] [error: ...]`

**Key Functions**:
- `print_answer()`: Display final answer with citations
- `print_search_start()`: Show search initiation message
- `print_result_summary()`: Display metadata in verbose mode
- `print_progress()`: Show iteration progress with context info
- `print_success()`, `print_error()`, `print_warning()`: Colored messages
- `create_progress_callback()`: Generate progress callback for search loop
- `is_tty()`: Check if output is a terminal
- `set_plain_mode()`: Disable colors/emojis for scripting
- `format_context_size()`: Format tokens as "12k/64k"

**Features**:
- Rich library for colored/formatted output (when available)
- Fallback to plain text for non-TTY or --plain mode
- Progress indicators during search
- Windows UTF-8 handling

---

### 10. Error Handling (`nexi/errors.py`)

**Purpose**: Graceful error handling without stack traces

**Key Functions**:
- `handle_error(message, exit_code, verbose)`: Print error and exit
- `safe_exit(message, exit_code)`: Exit cleanly with optional message

**Behavior**:
- In verbose mode: print full traceback
- In normal mode: print short error message to stderr
- Exit with specified code (default: 1)

---

### 11. MCP Server (`nexi/mcp_server.py`, `nexi/mcp_server_cli.py`)

**Purpose**: Expose NEXI as a Model Context Protocol server

**Tools**:
- `nexi_agent`: Full agent surface. Mirrors `nexi` and runs the LLM-driven search loop.
- `nexi_search`: Direct search surface. Mirrors `nexi-search` and runs the configured search provider chain without the agent loop.
- `nexi_fetch`: Direct fetch surface. Mirrors `nexi-fetch` and runs the configured fetch provider chain without the agent loop.

**Tool Parameters**:
- `nexi_agent`
  - `query` (required): Search query
  - `effort` (optional): "s", "m", or "l" (default: "m")
  - `verbose` (optional): Show detailed progress
- `nexi_search`
  - `query` (required): Search query
  - `verbose` (optional): Show provider debug output
- `nexi_fetch`
  - `urls` (required): One or more URLs to fetch
  - `full` (optional): Return full fetched content without extraction
  - `chunks` (optional): Use chunk selection instead of summarization
  - `instructions` (optional): Custom extraction instructions
  - `verbose` (optional): Show provider debug output

**Returns**:
- `nexi_agent`: Markdown-formatted answer with metadata and sources
- `nexi_search`: Structured direct-search payload matching the `nexi-search --json` shape
- `nexi_fetch`: Structured direct-fetch payload matching the `nexi-fetch --json` shape

**Contracts / Invariants**:
- MCP tool naming MUST match the runtime surface it exposes
- `nexi_agent` MUST be the only MCP tool that runs the agentic LLM search loop
- `nexi_search` MUST follow the same readiness rules as `nexi-search`
- `nexi_fetch` MUST follow the same readiness rules as `nexi-fetch`
- `nexi/mcp_server_cli.py` MUST be a first-class CLI entrypoint with explicit transport validation

**Transport Options**:
- **STDIO** (default): For local MCP clients (Claude Desktop)
- **HTTP**: For network access (host:port configurable)

**Usage**:
```bash
# STDIO transport
python -m nexi.mcp_server_cli

# HTTP transport
python -m nexi.mcp_server_cli http 0.0.0.0 8000
```

---

## Data Flow

### Search Execution Flow

```
User Input (CLI/MCP)
    │
    ▼
Load Config
    │
    ▼
Initialize Search Loop
    │
    ├─► Clear URL Cache
    ├─► Derive internal effort budget
    ├─► Create conversation (system + user)
    └─► Initialize LLM provider orchestrator
    │
    ▼
┌─────────────────────────────────────┐
│         Iteration Loop              │
│                                     │
│  1. Call LLM with provider fallback │
│  2. Parse tool_calls                │
│  3. Execute tool                    │
│     ├─► web_search                  │
│     │   └─► Search provider chain   │
│     ├─► web_get                     │
│     │   ├─► Check cache             │
│     │   ├─► Fetch provider chain    │
│     │   ├─► Process (full/chunks/extract) │
│     │   └─► Update cache            │
│     └─► final_answer                │
│         └─► Break loop              │
│  4. Add results to conversation     │
│  5. Check context limit             │
│     └─► Compact if needed           │
│  6. Repeat until done               │
└─────────────────────────────────────┘
    │
    ▼
Return SearchResult
    │
    ├─► Process citations
    ├─► Display answer
    ├─► Save to history
    └─► Return to caller
```

### Compaction Flow

```
Context Limit Approached
    │
    ▼
Extract Metadata
    ├─► Original query
    ├─► Search queries
    └─► URLs fetched
    │
    ▼
Collect Content
    ├─► Tool messages (web_get results)
    └─► Assistant messages
    │
    ▼
Generate Summary
    ├─► Call LLM with compaction prompt
    ├─► Target: compact_target_words
    └─► Preserve: numbers, quotes, terms
    │
    ▼
Rebuild Context
    ├─► Keep: system prompt
    ├─► Keep: original user message
    ├─► Add: summary message
    └─► Keep: last N assistant messages
    │
    ▼
Continue Search
```

### Multi-Turn Conversation Flow

```
Interactive Mode Started
    │
    ▼
┌─────────────────────────────────────┐
│         REPL Loop                   │
│                                     │
│  1. Get user input                  │
│  2. If first turn:                  │
│     └─► Fresh search (no history)   │
│  3. If subsequent turn:             │
│     ├─► Use CONTINUATION_SYSTEM_PROMPT │
│     └─► Include conversation_history │
│  4. Run search                      │
│  5. Update conversation_history:    │
│     ├─► Append user query           │
│     └─► Append assistant response   │
│  6. Repeat until exit               │
└─────────────────────────────────────┘
```

---

## Configuration Rules

### File Format and Lifecycle

- The canonical user config file MUST be TOML, not JSON
- The canonical path MUST be `~/.config/nexi/config.toml`
- If the file is missing, NEXI MUST create the default commented template and stop the current command
- The generated template MAY be structurally incomplete for search until the user activates one LLM provider and one search provider

### Required Fields

All saved configurations must include:
- `llm_backends`: Ordered list of provider names for LLM execution
- `search_backends`: Ordered list of provider names for search execution
- `fetch_backends`: Ordered list of provider names for fetch execution
- `providers`: Mapping of provider name to provider config object
- `default_effort`: One of "s", "m", "l"

### Optional Fields

- `max_context`: Positive integer (default: 128000)
- `auto_compact_thresh`: Float between 0.0 and 1.0 (default: 0.9)
- `compact_target_words`: Positive integer (default: 5000)
- `preserve_last_n_messages`: Non-negative integer (default: 3)
- `tokenizer_encoding`: Non-empty string (default: "cl100k_base")
- `provider_timeout`: Positive integer (default: 30)
- `search_provider_retries`: Positive integer (default: 2)
- `fetch_provider_retries`: Positive integer (default: 2)

### Validation Rules

1. **TOML parse**: The config file MUST parse as valid TOML
2. **providers**: Must be a table keyed by provider instance name
3. **listed providers**: Every name in an active backend chain MUST exist in `providers`
4. **provider type**: Every provider config object MUST include a supported `type`
5. **provider config**: Each provider class MUST validate its own required fields before execution
6. **default_effort**: Must be in `["s", "m", "l"]`
7. **max_context**: Must be positive integer
8. **auto_compact_thresh**: Must be between 0.0 and 1.0
9. **compact_target_words**: Must be positive integer
10. **preserve_last_n_messages**: Must be non-negative integer
11. **tokenizer_encoding**: Must be non-empty string
12. **provider_timeout**: Must be positive integer
13. **search_provider_retries / fetch_provider_retries**: Must be positive integers
14. **usable `nexi` config**: MUST include at least one usable LLM provider and one usable search provider
15. **usable `nexi-search` config**: MUST include at least one usable search provider
16. **usable `nexi-fetch` config**: MUST include at least one usable fetch provider
17. **template state**: Empty `llm_backends` or `search_backends` MAY exist in the generated template, but MUST fail readiness checks until activated

---

## Search Loop Rules

### Termination Conditions

The search loop terminates when **any** of these conditions are met:

1. **final_answer tool called**: LLM provides final answer
2. **Effort budget exhausted**: NEXI requests and returns the best final answer available
3. **Context guard rail finalization**: If compaction still cannot recover enough room, NEXI stops expanding and returns the best final answer available
4. **Provider chain exhaustion**: All configured LLM providers fail for the current request
5. **User cancellation**: KeyboardInterrupt

### Tool Call Rules

1. **web_search**:
    - Must provide 1-5 queries
    - Returns search results with snippets
    - Does NOT trigger compaction (results are small)
    - Captures titles for citation formatting
    - Retries failed queries inside the current provider before failover
    - Fails over only remaining failed queries, not successful ones

2. **web_get**:
    - Must provide 1-8 URLs
    - Returns extracted/summarized content
   - TRIGGERS compaction check (results can be large)
    - Supports `get_full` flag for raw content
    - Supports `use_chunks` flag for chunk-based selection
    - Supports custom `instructions` for extraction
    - Assigns stable citation numbers to URLs
    - Retries failed URLs inside the current provider before failover
    - Fails over only remaining failed URLs, not successful ones

3. **final_answer**:
    - Must provide answer string
    - Terminates search loop immediately
    - No further iterations

### Provider Chain Rules

1. **LLM Providers**:
   - Evaluated in `llm_backends` order
   - Hard provider failure marks that provider unhealthy for the current run
   - Model-not-found is a hard provider failure
   - The same iteration is retried against the next provider immediately

2. **Search Providers**:
   - Evaluated in `search_backends` order
   - Per-provider retries apply before failover
   - Successful query results are retained; only failed queries are re-routed

3. **Fetch Providers**:
   - Evaluated in `fetch_backends` order
   - Per-provider retries apply before failover
   - Successful URL fetches are retained; only failed URLs are re-routed

### Context Management Rules

1. **Compaction Trigger**:
   ```
   current_tokens + estimated_next > max_context * auto_compact_thresh
   ```

2. **Compaction Process**:
   - Extract metadata (queries, URLs)
   - Summarize tool results and assistant messages
   - Preserve system prompt and original query
   - Preserve last N assistant messages
   - Rebuild conversation with summary

3. **Failure Handling**:
   - If summary generation fails → continue with original messages
   - If still over limit → force answer and break

### Token Counting Rules

1. **Message Token Count**:
   - Count role string
   - Count content string (if present)
   - Count tool_calls (id, function name, arguments)
   - Count tool_call_id (for tool messages)
   - Add 3 tokens per message for formatting

2. **Estimation**:
   - Use tiktoken with configured encoding
   - Cache encodings for performance
   - Estimate page tokens before adding to conversation

### Citation Rules

1. **Number Assignment**:
   - URLs assigned citation numbers on first web_get
   - Numbers are stable throughout search session
   - Format: `[1]`, `[2]`, etc.

2. **Title Capture**:
   - Titles captured from web_search results
   - Used in sources list formatting

3. **Output Formatting**:
   - Sources section appended to answer
   - Format: `[N] Title - URL` or `[N] URL`

---

## Error Handling

### Configuration Errors

- **Missing config**: Write `~/.config/nexi/config.toml`, print path, warn that configuration is incomplete, exit immediately
- **Invalid config**: Show detailed errors and exit
- **Doctor failure**: Report actionable readiness issues without mutating the file unless explicitly asked

### Search Errors

- **Time target**: Return partial answer with timeout warning
- **Max Iterations**: Request final answer from LLM
- **Context Limit**: Return partial answer with context limit warning
- **Provider chain exhaustion**: Return error message and exit
- **KeyboardInterrupt**: Cancel search and exit with code 130

### Tool Execution Errors

- **HTTP Errors**: Return error in result, continue search
- **Network Errors**: Return error in result, continue search
- **Parsing Errors**: Return error in result, continue search
- **LLM Extraction Errors**: Fallback to raw content[:2000]
- **Chunk Selection Errors**: Fallback to first 5 chunks

### History Errors

- **Corrupted JSON Lines**: Skip corrupted entries
- **Missing File**: Return empty list
- **Write Errors**: Raise exception

---

## Performance Considerations

### Parallel Execution

1. **web_search**: Executes 1-5 queries in parallel via `asyncio.gather()`
2. **web_get**: Executes 1-8 URL fetches in parallel via `asyncio.gather()`
3. **LLM extraction**: Parallel extraction for multiple URLs
4. **Chunk selection**: Parallel chunk selection for multiple URLs
5. **Timeouts**: Configurable per provider via `provider_timeout` and provider-specific config

### Caching

1. **URL Cache**: In-memory cache prevents duplicate fetches
   - Cleared at start of each search session
   - Key: URL, Value: Raw content

2. **Encoding Cache**: tiktoken encodings cached at module level
   - Key: encoding name, Value: Encoding object

3. **HTTP Transport Reuse**: Connection reuse SHOULD preserve pooling benefits without freezing timeout behavior from first use
   - Timeout changes in long-lived processes MUST apply to subsequent provider calls
   - Transport reuse MUST NOT create hidden cross-provider coupling for incompatible timeout settings

### Token Optimization

1. **Compaction**: Reduces conversation size when approaching limits
2. **Extraction**: LLM extracts only relevant content from pages
3. **Chunk Selection**: Preserves original text, only selects relevant chunks
4. **Targeted Summaries**: Custom instructions guide extraction

### Network Efficiency

1. **Parallel Requests**: Minimize latency for multiple queries/URLs
2. **Connection Pooling**: httpx.AsyncClient reuses connections
3. **Timeouts**: Prevent hanging on slow responses

---

## Security Considerations

### API Keys

- Stored in `~/.config/nexi/config.toml`
- File permissions: User-readable only (OS-dependent)
- Provider credentials live inside `providers[provider_name]`
- Never logged or printed in verbose mode

### URL Handling

- URLs are handled by configured fetch providers
- No arbitrary code execution
- No file system access via URLs

### LLM Interactions

- System prompts guide behavior
- Tool calls are structured and validated
- No arbitrary code execution via LLM

### History

- Stored locally in JSONL format
- Contains queries and answers
- Only completed searches are persisted
- No sensitive data unless user includes it

---

## Extensibility

### Adding New Tools

1. Define tool schema in `TOOLS` list
2. Implement tool function in `tools.py`
3. Add execution logic in `execute_tool()`
4. Handle tool call in search loop

### Pluggable Providers

NEXI supports swappable provider chains for LLM, search, and fetch capabilities.

1. **Provider Implementation**:
   - Implement one or more provider protocols in `nexi/backends/`
   - Provide `name`
   - Provide `validate_config(config)`
   - Provide capability methods such as `search()`, `fetch()`, or `complete()`

2. **Provider Registration**:
   - Register the provider under its `type` for each supported capability
   - Keep provider selection declarative through config, not hardcoded conditionals

3. **Provider Configuration**:
    - Add a config object under `providers[provider_name]`
    - Set `providers[provider_name]["type"]` to the registered provider type
    - Add the provider name to one or more ordered chains: `llm_backends`, `search_backends`, `fetch_backends`

Example configuration:
```toml
llm_backends = ["openrouter"]
search_backends = ["jina"]
fetch_backends = ["crawl4ai_local", "markdown_new"]

[providers.openrouter]
type = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
api_key = "key123"
model = "google/gemini-2.5-flash-lite"

[providers.jina]
type = "jina"
api_key = "key456"
```

### Custom Prompts

1. Edit `DEFAULT_SYSTEM_PROMPT_TEMPLATE` in `config.py`
2. Edit `EXTRACTOR_PROMPT_TEMPLATE` for extraction behavior
3. Edit `COMPACTION_PROMPT_TEMPLATE` for summarization
4. Edit `CHUNK_SELECTOR_PROMPT` for chunk selection
5. Edit `CONTINUATION_SYSTEM_PROMPT` for multi-turn behavior

### Custom Compaction Strategy

1. Modify `should_compact()` for trigger logic
2. Modify `extract_metadata()` for metadata extraction
3. Modify `generate_summary()` for summarization
4. Modify `rebuild_context()` for context rebuilding

### Custom Output Formatting

1. Modify `print_answer()` for answer display
2. Modify `print_result_summary()` for metadata display
3. Add new formatting functions in `output.py`

---

## Testing Strategy

### Unit Tests

- **Config**: Test loading, validation, prompt generation
- **Token Counter**: Test token counting accuracy
- **Compaction**: Test metadata extraction, summarization, rebuilding
- **History**: Test entry creation, retrieval, formatting
- **Citations**: Test marker extraction, formatting, auto-detection
- **Tools**: Mock external APIs, test error handling

### Integration Tests

- **Search Loop**: Test full search with mocked LLM and Jina APIs
- **CLI**: Test command parsing, execution, output
- **MCP Server**: Test tool execution via MCP protocol
- **Multi-Turn**: Test conversation history in interactive mode

### Manual Testing

- **Real Searches**: Test with actual LLM and Jina APIs
- **Edge Cases**: Test timeouts, limits, errors
- **Performance**: Measure latency, token usage

---

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check API keys in config
   - Verify API key validity

2. **429 Rate Limit**
   - Wait or switch to different model
   - Increase timeout between searches

3. **Search Finalized Early Under Context Guard Rails**
   - Increase `max_context` in config
   - Lower `auto_compact_thresh` to compact earlier
   - Reduce `compact_target_words` for smaller summaries

4. **UTF-8 Garbled (Windows)**
   - Use `--plain` flag
   - Use Windows Terminal instead of cmd.exe

6. **MCP Server Not Found**
   - Install with `uv sync --group mcp`
   - Check Claude Desktop config

---

## Glossary

- **Agentic Search**: Search process where LLM autonomously decides actions
- **Compaction**: Process of summarizing conversation to reduce token count
- **Context Window**: Maximum tokens a model can process
- **Effort Level**: Search depth (s=quick, m=balanced, l=deep)
- **Function Calling**: OpenAI API feature for structured tool use
- **Provider Chain**: Ordered list of providers used for one capability with failover behavior
- **Provider**: Named backend configuration and adapter that implements one or more capabilities
- **JSONL**: JSON Lines format (one JSON object per line)
- **LLM**: Large Language Model
- **MCP**: Model Context Protocol
- **REPL**: Read-Eval-Print Loop (interactive mode)
- **tiktoken**: OpenAI's token counting library
- **Tool Call**: Structured request from LLM to execute a function
- **URL Cache**: In-memory storage of fetched page content
- **Citation Number**: Stable identifier [N] assigned to URLs for source tracking
- **Chunk Selection**: Process of splitting content and selecting relevant portions

---

## References

- **OpenAI API**: https://platform.openai.com/docs/api-reference
- **tiktoken**: https://github.com/openai/tiktoken
- **MCP Protocol**: https://modelcontextprotocol.io/
- **Click**: https://click.palletsprojects.com/
- **httpx**: https://www.python-httpx.org/
- **Rich**: https://rich.readthedocs.io/

---

## Implementation Pointers

- **Entry points**: `main.py`, `nexi/__main__.py`, `nexi/cli.py`
- **Core modules**: `nexi/search.py`, `nexi/tools.py`, `nexi/config.py`
- **Support modules**: `nexi/compaction.py`, `nexi/citations.py`, `nexi/token_counter.py`
- **Output**: `nexi/output.py`, `nexi/history.py`, `nexi/errors.py`
- **MCP**: `nexi/mcp_server.py`, `nexi/mcp_server_cli.py`
- **Provider catalog**: `docs/provider-matrix.md`
- **Config location**: `~/.config/nexi/config.toml`
- **History location**: `~/.config/nexi/history.jsonl`

---

## License

MIT License - See LICENSE file for details.
