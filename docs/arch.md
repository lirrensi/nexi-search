# NEXI Architecture Documentation

> **Complete technical specification.** Machine-readable, exhaustive. You could throw away the code and rebuild it entirely from this document.

---

## Overview

NEXI is an intelligent web search CLI tool that uses an agentic search loop powered by Large Language Models (LLMs) to provide comprehensive, well-researched answers to user queries. It combines web search, content extraction, and intelligent synthesis to deliver high-quality responses with proper source attribution.

### Core Philosophy

- **Agentic Search**: The LLM autonomously decides what to search for and which pages to read
- **Context-Aware**: Automatically manages conversation context to prevent token overflow
- **Tool-Based**: Uses function calling to give the LLM structured access to web capabilities
- **Efficient**: Parallel searches and intelligent caching minimize latency
- **Transparent**: Verbose mode shows all decisions, tool calls, and token usage

### Scope Boundary

**This system owns**:
- Agentic search loop orchestration
- Web search and content extraction via Jina AI
- Conversation context management and compaction
- Citation tracking and formatting
- CLI interface and history management
- MCP server for tool integration

**This system does NOT own**:
- LLM inference (delegates to OpenAI-compatible APIs)
- Web crawling (delegates to Jina AI)
- Content rendering (outputs plain text/markdown)

**Boundary interfaces**:
- Receives queries from CLI, stdin, or MCP
- Calls external LLM API via OpenAI-compatible protocol
- Calls Jina AI for search and content fetching

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
| LLM API | Configurable (OpenAI-compatible) | Chat completions with function calling |
| Jina Search | `https://s.jina.ai/` | Web search |
| Jina Reader | `https://r.jina.ai/` | Content extraction from URLs |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│  (nexi/cli.py, main.py, nexi/__main__.py)                       │
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
│  - Validation                │              │  - Entry Creation/Retrieval      │
│  - Prompt Templates          │              │  - Formatting                    │
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
│  │  1. Call LLM with tools                                  │  │
│  │  2. Parse tool_calls                                     │  │
│  │  3. Execute tools (web_search, web_get, final_answer)    │  │
│  │  4. Add results to conversation                          │  │
│  │  5. Check limits (time_target, max_iter, context)        │  │
│  │  6. Compact if needed                                    │  │
│  │  7. Repeat until final_answer or limit reached           │  │
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
│- Backends    │ │- Detect  │ │              │ │              │
└──────────────┘ └──────────┘ └──────────────┘ └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     External Services                           │
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   LLM API            │      │   Jina AI Services           │ │
│  │   (OpenAI-compatible)│      │                              │ │
│  │                      │      │  - s.jina.ai (Search)        │ │
│  │  - Chat Completions  │      │  - r.jina.ai (Reader)        │ │
│  │  - Function Calling  │      │                              │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. CLI Interface (`nexi/cli.py`, `main.py`, `nexi/__main__.py`)

**Purpose**: Entry points for command-line interaction

**Key Functions**:
- `main()`: Click-based CLI command handler
- `_run_search_command()`: Orchestrates search execution
- `_interactive_mode()`: REPL for continuous queries with multi-turn support
- `_show_last_n()`, `_show_prev()`, `_show_by_id()`: History viewing

**Features**:
- Argument parsing (query, effort, max_iter, time_target, verbose, plain)
- stdin support for piping queries
- Interactive REPL mode with conversation history
- History management commands
- Config editing/viewing

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

**Config Structure**:
```python
@dataclass
class Config:
    base_url: str                    # LLM API endpoint
    api_key: str                     # LLM API key
    model: str                       # Model name
    jina_key: str                    # Jina AI API key
    default_effort: str              # s/m/l
    max_output_tokens: int           # Max tokens in final answer
    time_target: int | None          # Soft time limit (seconds)
    max_context: int                 # Model's context window limit
    auto_compact_thresh: float       # Trigger compaction at this fraction
    compact_target_words: int        # Target word count for summaries
    preserve_last_n_messages: int    # Recent messages to keep un-compacted
    tokenizer_encoding: str          # tiktoken encoding name
    jina_timeout: int                # Timeout for Jina API calls
    llm_max_retries: int             # Max retry attempts for LLM API
    search_backend: str              # Backend name (default: "jina")
    content_fetcher: str             # Fetcher name (default: "jina")
    api_keys: dict[str, str]         # Additional API keys by backend name
```

**Key Functions**:
- `load_config()`: Load from `~/.local/share/nexi/config.json`
- `save_config()`: Persist configuration
- `validate_config()`: Validate all fields with detailed error messages
- `ensure_config()`: Load or run first-time setup
- `run_first_time_setup()`: Interactive configuration wizard
- `get_system_prompt()`: Generate system prompt with effort level
- `get_compaction_prompt()`: Generate prompt for conversation summarization

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
    reached_max_iter: bool
```

**Search Loop Algorithm**:

```
1. Initialize:
   - Clear URL cache
   - Set max_iter from effort level
   - Set time_target from config or CLI override
   - Create conversation with system prompt + user query
   - Initialize OpenAI client
   - Initialize citation tracking (url_to_number, url_to_title)

2. For each iteration (1 to max_iter):
   a. Check time_target → break if exceeded
   b. Call LLM with tools and current conversation (with retry logic)
   c. Track token usage
   d. Parse response.message.tool_calls

   e. If tool_call is "final_answer":
      - Extract answer
      - Break loop

   f. If tool_call is "web_search":
      - Execute parallel searches via Jina AI
      - Capture titles from results for citations
      - Add assistant message (tool_calls) to conversation
      - Add tool message (results) to conversation

   g. If tool_call is "web_get":
      - Assign stable citation numbers to URLs
      - Execute parallel fetches via Jina Reader
      - Process content (full, chunks, or extraction)
      - Estimate tokens for result
      - Check if compaction needed:
         - If current + estimated > threshold:
           - Compact conversation
           - If still over limit → force answer and break
      - Track URLs fetched
      - Add assistant message (tool_calls) to conversation
      - Add tool message (results with sources list) to conversation

   h. If no tool_call:
      - Force answer from message.content
      - Break loop

3. If max_iter reached:
   - Request final answer via _request_final_answer()

4. Return SearchResult with:
   - answer, urls, url_citations, url_to_title
   - iterations, duration_s, tokens, reached_max_iter
```

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
- `_request_final_answer()`: Request final answer from LLM when max_iter reached
- `run_search_sync()`: Synchronous wrapper using new event loop

**Retry Logic**:
- LLM API calls retry up to `config.llm_max_retries` times (default: 3)
- Exponential backoff: 1s, 2s, 4s
- On final failure, returns forced answer with error message

---

### 4. Tool Implementations (`nexi/tools.py`)

**Purpose**: Execute web search and content retrieval

**URL Caching**:
- In-memory cache `_url_cache: dict[str, str]`
- Cleared at start of each search session
- Prevents duplicate fetches of same URL

**HTTP Client**:
- Shared `httpx.AsyncClient` with connection pooling
- Limits: max_connections=10, max_keepalive_connections=5
- Closed via `close_http_client()`

#### Backend Protocols

```python
class SearchBackend(Protocol):
    async def search(
        self,
        queries: list[str],
        api_key: str,
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Returns {"searches": [{"query": str, "results": [...], "error": str}]}"""

class ContentFetcher(Protocol):
    async def fetch(
        self,
        urls: list[str],
        api_key: str,
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Returns {"pages": [{"url": str, "content": str, "error": str}]}"""
```

#### JinaSearchBackend

**Process**:
1. Create parallel tasks for each query (1-5 queries)
2. Execute via `asyncio.gather()`
3. Call Jina AI Search API: `https://s.jina.ai/?q={query}`
4. Parse response (JSON or text format)
5. Return structured results

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

#### JinaContentFetcher

**Process**:
1. Check URL cache → skip if already fetched
2. Create parallel tasks for uncached URLs (1-8 URLs)
3. Call Jina Reader API: `https://r.jina.ai/{url}`
4. Headers:
   - `X-Retain-Images: none`
   - `X-Retain-Links: gpt-oss`
   - `X-Timeout: 40`
   - `Authorization: Bearer {jina_key}` (if provided)
5. Store raw content in cache

#### web_get Processing Modes

**1. Full Content (`get_full=True`)**:
- Return raw markdown from Jina Reader
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
    """Heading-aware logical chunking for clean Jina Markdown.
    
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

**Storage**: `~/.local/share/nexi/history.jsonl` (JSON Lines format)

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

**Tool**: `nexi_search`

**Parameters**:
- `query` (required): Search query
- `effort` (optional): "s", "m", or "l" (default: "m")
- `max_iter` (optional): Override max iterations
- `time_target` (optional): Force return after N seconds
- `verbose` (optional): Show detailed progress

**Returns**: Markdown-formatted answer with metadata and sources

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
    ├─► Set max_iter, time_target
    ├─► Create conversation (system + user)
    └─► Initialize OpenAI client
    │
    ▼
┌─────────────────────────────────────┐
│         Iteration Loop              │
│                                     │
│  1. Check time_target               │
│  2. Call LLM with tools (retry)     │
│  3. Parse tool_calls                │
│  4. Execute tool                    │
│     ├─► web_search                  │
│     │   └─► Jina AI Search API      │
│     ├─► web_get                     │
│     │   ├─► Check cache             │
│     │   ├─► Jina Reader API         │
│     │   ├─► Process (full/chunks/extract) │
│     │   └─► Update cache            │
│     └─► final_answer                │
│         └─► Break loop              │
│  5. Add results to conversation     │
│  6. Check context limit             │
│     └─► Compact if needed           │
│  7. Repeat until done               │
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

### Required Fields

All configurations must include:
- `base_url`: Valid HTTP(S) URL for LLM API
- `api_key`: Non-empty string
- `model`: Non-empty string
- `default_effort`: One of "s", "m", "l"
- `max_output_tokens`: Positive integer

### Optional Fields

- `jina_key`: String or empty (free tier available)
- `time_target`: Positive integer or null (no limit)
- `max_context`: Positive integer (default: 128000)
- `auto_compact_thresh`: Float between 0.0 and 1.0 (default: 0.9)
- `compact_target_words`: Positive integer (default: 5000)
- `preserve_last_n_messages`: Non-negative integer (default: 3)
- `tokenizer_encoding`: Non-empty string (default: "cl100k_base")
- `jina_timeout`: Positive integer (default: 30)
- `llm_max_retries`: Positive integer (default: 3)

### Validation Rules

1. **base_url**: Must start with `http://` or `https://`
2. **api_key**: Must be non-empty string
3. **model**: Must be non-empty string
4. **jina_key**: Must be string (can be empty)
5. **default_effort**: Must be in `["s", "m", "l"]`
6. **max_output_tokens**: Must be positive integer
7. **time_target**: Must be positive integer or null
8. **max_context**: Must be positive integer
9. **auto_compact_thresh**: Must be between 0.0 and 1.0
10. **compact_target_words**: Must be positive integer
11. **preserve_last_n_messages**: Must be non-negative integer
12. **tokenizer_encoding**: Must be non-empty string
13. **jina_timeout**: Must be positive integer
14. **llm_max_retries**: Must be positive integer

---

## Search Loop Rules

### Termination Conditions

The search loop terminates when **any** of these conditions are met:

1. **final_answer tool called**: LLM provides final answer
2. **Time target exceeded**: `elapsed >= time_target`
3. **Max iterations reached**: `current_iteration >= max_iter`
4. **Context limit exceeded**: After failed compaction
5. **API error**: LLM API call fails after all retries
6. **User cancellation**: KeyboardInterrupt

### Tool Call Rules

1. **web_search**:
   - Must provide 1-5 queries
   - Returns search results with snippets
   - Does NOT trigger compaction (results are small)
   - Captures titles for citation formatting

2. **web_get**:
   - Must provide 1-8 URLs
   - Returns extracted/summarized content
   - TRIGGERS compaction check (results can be large)
   - Supports `get_full` flag for raw content
   - Supports `use_chunks` flag for chunk-based selection
   - Supports custom `instructions` for extraction
   - Assigns stable citation numbers to URLs

3. **final_answer**:
   - Must provide answer string
   - Terminates search loop immediately
   - No further iterations

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

- **FileNotFoundError**: Config file doesn't exist → run first-time setup
- **ValueError**: Invalid config → show detailed errors and exit

### Search Errors

- **Time target**: Return partial answer with timeout warning
- **Max Iterations**: Request final answer from LLM
- **Context Limit**: Return partial answer with context limit warning
- **API Error (after retries)**: Return error message and exit
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
5. **Timeouts**: Configurable for Jina API (default: 30s), Reader (40s)

### Caching

1. **URL Cache**: In-memory cache prevents duplicate fetches
   - Cleared at start of each search session
   - Key: URL, Value: Raw content

2. **Encoding Cache**: tiktoken encodings cached at module level
   - Key: encoding name, Value: Encoding object

3. **HTTP Client**: Shared client with connection pooling
   - max_connections=10, max_keepalive_connections=5

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

- Stored in `~/.local/share/nexi/config.json`
- File permissions: User-readable only (OS-dependent)
- Jina key is optional (free tier available)
- Never logged or printed in verbose mode

### URL Handling

- URLs are validated by Jina AI services
- No arbitrary code execution
- No file system access via URLs

### LLM Interactions

- System prompts guide behavior
- Tool calls are structured and validated
- No arbitrary code execution via LLM

### History

- Stored locally in JSONL format
- Contains queries and answers
- No sensitive data unless user includes it

---

## Extensibility

### Adding New Tools

1. Define tool schema in `TOOLS` list
2. Implement tool function in `tools.py`
3. Add execution logic in `execute_tool()`
4. Handle tool call in search loop

### Pluggable Backends

NEXI supports swappable search backends and content fetchers:

1. **SearchBackend Protocol** (`nexi/tools.py`):
   - Implement `search(queries, api_key, timeout, verbose)` method
   - Return `{"searches": [{"query": str, "results": [...], "error": str}]}`

2. **ContentFetcher Protocol** (`nexi/tools.py`):
   - Implement `fetch(urls, api_key, timeout, verbose)` method  
   - Return `{"pages": [{"url": str, "content": str, "error": str}]}`

3. **Backend Selection**:
   - Configure via `config.search_backend` and `config.content_fetcher`
   - Currently defaults to "jina" (JinaSearchBackend + JinaContentFetcher)

Example future configuration:
```python
config = Config(
    search_backend="tavily",
    content_fetcher="jina",  # Keep Jina for fetch, swap search
    api_keys={"tavily": "key123", "jina": "key456"},
    ...
)
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

3. **Time Target Hit**
   - Increase `time_target` in config
   - Use `--time-target` CLI flag

4. **Context Limit Exceeded**
   - Increase `max_context` in config
   - Lower `auto_compact_thresh` to compact earlier
   - Reduce `compact_target_words` for smaller summaries

5. **UTF-8 Garbled (Windows)**
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
- **Jina AI**: External service for web search and content extraction
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
- **Jina AI**: https://jina.ai/
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
- **Config location**: `~/.local/share/nexi/config.json`
- **History location**: `~/.local/share/nexi/history.jsonl`

---

## License

MIT License - See LICENSE file for details.
