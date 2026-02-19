# NEXI Architecture Documentation

## Overview

NEXI is an intelligent web search CLI tool that uses an agentic search loop powered by Large Language Models (LLMs) to provide comprehensive, well-researched answers to user queries. It combines web search, content extraction, and intelligent synthesis to deliver high-quality responses with proper source attribution.

### Core Philosophy

- **Agentic Search**: The LLM autonomously decides what to search for and which pages to read
- **Context-Aware**: Automatically manages conversation context to prevent token overflow
- **Tool-Based**: Uses function calling to give the LLM structured access to web capabilities
- **Efficient**: Parallel searches and intelligent caching minimize latency
- **Transparent**: Verbose mode shows all decisions, tool calls, and token usage

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
│  │  5. Check limits (timeout, max_iter, context)            │  │
│  │  6. Compact if needed                                    │  │
│  │  7. Repeat until final_answer or limit reached           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────┐
│   Tools      │ │Compaction│ │  Token Counter   │
│(nexi/tools.py)│ │(nexi/    │ │(nexi/token_      │
│              │ │compaction│ │counter.py)       │
│- web_search  │ │.py)      │ │                  │
│- web_get     │ │          │ │- count_tokens    │
│- final_answer│ │- Extract │ │- count_messages  │
│              │ │- Summarize│ │- estimate_pages  │
└──────────────┘ │- Rebuild │ └──────────────────┘
                 └──────────┘
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
│  │  - Function Calling  │      │  - LLM Summarizer            │ │
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
- `_interactive_mode()`: REPL for continuous queries
- `_show_last_n()`, `_show_prev()`, `_show_by_id()`: History viewing

**Features**:
- Argument parsing (query, effort, max_iter, max_timeout, verbose, plain)
- stdin support for piping queries
- Interactive REPL mode
- History management commands
- Config editing/viewing

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
    max_timeout: int                 # Max search duration (seconds)
    max_output_tokens: int           # Max tokens in final answer
    max_context: int                 # Model's context window limit
    auto_compact_thresh: float       # Trigger compaction at this fraction
    compact_target_words: int        # Target word count for summaries
    preserve_last_n_messages: int     # Recent messages to keep un-compacted
    tokenizer_encoding: str          # tiktoken encoding name
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

**Search Loop Algorithm**:

```
1. Initialize:
   - Clear URL cache
   - Set max_iter from effort level
   - Set timeout from config or CLI override
   - Create conversation with system prompt + user query
   - Initialize OpenAI client

2. For each iteration (1 to max_iter):
   a. Check timeout → break if exceeded
   b. Call LLM with tools and current conversation
   c. Track token usage
   d. Parse response.message.tool_calls

   e. If tool_call is "final_answer":
      - Extract answer
      - Break loop

   f. If tool_call is "web_search":
      - Execute parallel searches via Jina AI
      - Add assistant message (tool_calls) to conversation
      - Add tool message (results) to conversation

   g. If tool_call is "web_get":
      - Execute parallel fetches via Jina Reader
      - Estimate tokens for result
      - Check if compaction needed:
         - If current + estimated > threshold:
           - Compact conversation
           - If still over limit → force answer and break
      - Track URLs fetched
      - Add assistant message (tool_calls) to conversation
      - Add tool message (results) to conversation

   h. If no tool_call:
      - Force answer from message.content
      - Break loop

3. If max_iter reached:
   - Force answer with _force_answer()

4. Return SearchResult with:
   - answer, urls, iterations, duration_s, tokens, reached_max_iter
```

**Tool Schemas**:

Three tools are exposed to the LLM via function calling:

1. **web_search**
   - Parameters: `queries` (list[str], 1-5 items)
   - Returns: Search results with snippets and URLs
   - Used for: Initial information gathering

2. **web_get**
   - Parameters: `urls` (list[str], 1-8 items), `instructions` (str), `get_full` (bool)
   - Returns: Extracted/summarized page content
   - Used for: Deep reading of specific pages

3. **final_answer**
   - Parameters: `answer` (str)
   - Returns: Terminates search loop
   - Used for: Providing the final synthesized answer

**Helper Functions**:
- `_get_tool_schemas()`: Return tool definitions for OpenAI API
- `_force_answer()`: Generate answer when limits reached
- `run_search_sync()`: Synchronous wrapper using new event loop

---

### 4. Tool Implementations (`nexi/tools.py`)

**Purpose**: Execute web search and content retrieval

**URL Caching**:
- In-memory cache `_url_cache: dict[str, str]`
- Cleared at start of each search session
- Prevents duplicate fetches of same URL

#### `web_search()`

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
            ]
        }
    ]
}
```

**Error Handling**:
- HTTP errors captured and returned in result
- Exceptions caught and returned as error messages
- Individual query failures don't abort entire search

#### `web_get()`

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
6. If `get_full=False` (default):
   - Call `_extract_with_llm()` to extract relevant content
   - Use `EXTRACTOR_PROMPT_TEMPLATE` with custom instructions
7. Format as: `{url}\n---\n{content}`

**Response Format**:
```python
{
    "pages": [
        {
            "url": "https://example.com",
            "content": "https://example.com\n---\nExtracted content"
        }
    ]
}
```

#### `_extract_with_llm()`

**Purpose**: Use LLM to extract relevant information from page content

**Process**:
1. Build extraction prompt with query and instructions
2. Call LLM with system prompt + page content
3. Max tokens: 2000
4. Return extracted content
5. Fallback to raw content[:2000] on error

---

### 5. Conversation Compaction (`nexi/compaction.py`)

**Purpose**: Prevent token overflow by summarizing conversation history

**Trigger Condition**:
```python
should_compact(current_tokens, estimated_next, config):
    threshold = config.max_context * config.auto_compact_thresh
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

### 6. Token Counter (`nexi/token_counter.py`)

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

### 7. History Management (`nexi/history.py`)

**Purpose**: Persist and retrieve search history

**Storage**: `~/.local/share/nexi/history.jsonl` (JSON Lines format)

**Data Structure**:
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

### 8. Output Formatting (`nexi/output.py`)

**Purpose**: Format and display search results

**Key Functions**:
- `print_answer()`: Display final answer with Rich or plain text
- `print_search_start()`: Show search initiation message
- `print_result_summary()`: Display metadata in verbose mode
- `print_success()`, `print_error()`, `print_warning()`: Colored messages
- `create_progress_callback()`: Generate progress callback for search loop
- `is_tty()`: Check if output is a terminal
- `set_plain_mode()`: Disable colors/emojis for scripting

**Features**:
- Rich library for colored/formatted output (when available)
- Fallback to plain text for non-TTY or --plain mode
- Progress indicators during search
- Windows UTF-8 handling

---

### 9. MCP Server (`nexi/mcp_server.py`, `nexi/mcp_server_cli.py`)

**Purpose**: Expose NEXI as a Model Context Protocol server

**Tool**: `nexi_search`

**Parameters**:
- `query` (required): Search query
- `effort` (optional): "s", "m", or "l" (default: "m")
- `max_iter` (optional): Override max iterations
- `max_timeout` (optional): Force return after N seconds
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
    ├─► Set max_iter, timeout
    ├─► Create conversation (system + user)
    └─► Initialize OpenAI client
    │
    ▼
┌─────────────────────────────────────┐
│         Iteration Loop              │
│                                     │
│  1. Check timeout                   │
│  2. Call LLM with tools             │
│  3. Parse tool_calls                │
│  4. Execute tool                    │
│     ├─► web_search                  │
│     │   └─► Jina AI Search API      │
│     ├─► web_get                     │
│     │   ├─► Check cache             │
│     │   ├─► Jina Reader API         │
│     │   ├─► LLM Extractor (optional)│
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

---

## Configuration Rules

### Required Fields

All configurations must include:
- `base_url`: Valid HTTP(S) URL for LLM API
- `api_key`: Non-empty string
- `model`: Non-empty string
- `default_effort`: One of "s", "m", "l"
- `max_timeout`: Positive integer (seconds)
- `max_output_tokens`: Positive integer

### Optional Fields

- `jina_key`: String or null (free tier available)
- `max_context`: Positive integer (default: 128000)
- `auto_compact_thresh`: Float between 0.0 and 1.0 (default: 0.9)
- `compact_target_words`: Positive integer (default: 5000)
- `preserve_last_n_messages`: Non-negative integer (default: 3)
- `tokenizer_encoding`: Non-empty string (default: "cl100k_base")

### Validation Rules

1. **base_url**: Must start with `http://` or `https://`
2. **api_key**: Must be non-empty string
3. **model**: Must be non-empty string
4. **jina_key**: Must be string or null
5. **default_effort**: Must be in `["s", "m", "l"]`
6. **max_timeout**: Must be positive integer
7. **max_output_tokens**: Must be positive integer
8. **max_context**: Must be positive integer
9. **auto_compact_thresh**: Must be between 0.0 and 1.0
10. **compact_target_words**: Must be positive integer
11. **preserve_last_n_messages**: Must be non-negative integer
12. **tokenizer_encoding**: Must be non-empty string

---

## Search Loop Rules

### Termination Conditions

The search loop terminates when **any** of these conditions are met:

1. **final_answer tool called**: LLM provides final answer
2. **Timeout exceeded**: `elapsed >= max_timeout`
3. **Max iterations reached**: `current_iteration >= max_iter`
4. **Context limit exceeded**: After failed compaction
5. **API error**: LLM API call fails
6. **User cancellation**: KeyboardInterrupt

### Tool Call Rules

1. **web_search**:
   - Must provide 1-5 queries
   - Returns search results with snippets
   - Does NOT trigger compaction (results are small)

2. **web_get**:
   - Must provide 1-8 URLs
   - Returns extracted/summarized content
   - TRIGGERS compaction check (results can be large)
   - Supports `get_full` flag for raw content
   - Supports custom `instructions` for extraction

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

---

## Error Handling

### Configuration Errors

- **FileNotFoundError**: Config file doesn't exist → run first-time setup
- **ValueError**: Invalid config → show detailed errors and exit

### Search Errors

- **Timeout**: Return partial answer with timeout warning
- **Max Iterations**: Return best answer with iteration limit warning
- **Context Limit**: Return partial answer with context limit warning
- **API Error**: Return error message and exit
- **KeyboardInterrupt**: Cancel search and exit with code 130

### Tool Execution Errors

- **HTTP Errors**: Return error in result, continue search
- **Network Errors**: Return error in result, continue search
- **Parsing Errors**: Return error in result, continue search
- **LLM Extraction Errors**: Fallback to raw content[:2000]

### History Errors

- **Corrupted JSON Lines**: Skip corrupted entries
- **Missing File**: Return empty list
- **Write Errors**: Raise exception

---

## Performance Considerations

### Parallel Execution

1. **web_search**: Executes 1-5 queries in parallel via `asyncio.gather()`
2. **web_get**: Executes 1-8 URL fetches in parallel via `asyncio.gather()`
3. **Timeouts**: 30s for search, 40s for reader

### Caching

1. **URL Cache**: In-memory cache prevents duplicate fetches
   - Cleared at start of each search session
   - Key: URL, Value: Raw content

2. **Encoding Cache**: tiktoken encodings cached at module level
   - Key: encoding name, Value: Encoding object

### Token Optimization

1. **Compaction**: Reduces conversation size when approaching limits
2. **Extraction**: LLM extracts only relevant content from pages
3. **Targeted Summaries**: Custom instructions guide extraction

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

1. Define tool schema in `_get_tool_schemas()`
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

3. **Backend Selection** (future):
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
- **Tools**: Mock external APIs, test error handling

### Integration Tests

- **Search Loop**: Test full search with mocked LLM and Jina APIs
- **CLI**: Test command parsing, execution, output
- **MCP Server**: Test tool execution via MCP protocol

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

3. **Timeout**
   - Increase `max_timeout` in config
   - Use `--max-timeout` CLI flag

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

## Future Enhancements

### Potential Features

1. **Multi-turn Conversations**: Allow follow-up questions
2. **Source Ranking**: Rank sources by relevance
3. **Citation Format**: Support different citation styles
4. **Export Options**: Export to PDF, Markdown, JSON
5. **Search History Search**: Search within history
6. **Custom Tool Definitions**: Allow user-defined tools
7. **Streaming Responses**: Stream answers as they're generated
8. **Parallel LLM Calls**: Use multiple LLMs for consensus
9. **Local LLM Support**: Support local models via Ollama
10. **Webhooks**: Notify external systems on search completion

### Performance Improvements

1. **Persistent URL Cache**: Cache across search sessions
2. **Incremental Compaction**: Compact more frequently
3. **Smart Token Estimation**: Better token estimation
4. **Connection Pooling**: Reuse HTTP connections
5. **Compression**: Compress cached content

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

## License

MIT License - See LICENSE file for details.
