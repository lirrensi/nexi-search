# NEXI Installation & Configuration

## Install Methods

### Option 1: UV (Recommended)

```bash
uv tool install git+https://github.com/lirrensi/nexi-search.git
```

### Option 2: Pipx

```bash
pipx install git+https://github.com/lirrensi/nexi-search.git
```

### Option 3: Pip

```bash
pip install --user --upgrade git+https://github.com/lirrensi/nexi-search.git
```

## Verify Installation

```bash
nexi --version
nexi --config
```

## First-Time Configuration

Run `nexi` without arguments to start the interactive config wizard:

```bash
$ nexi
Welcome to NEXI! 🔍
Let's set up your configuration...

LLM Base URL: [https://openrouter.ai/api/v1]
API Key: [your-openai-compatible-key]
Model: [google/gemini-2.5-flash-lite]
Jina API Key (optional): [jina_...]
Default effort level: [m]
...
```

### Required Credentials

- **OpenAI-compatible API**: Endpoint URL, API key, and model name
  - Default: `https://openrouter.ai/api/v1`
  - Recommended models: `google/gemini-2.5-flash-lite` (cheap, fast)
- **Jina AI API Key**: Free at https://jina.ai (required for web search)

## Manual Config File

Config location:
- Linux/macOS: `~/.local/share/nexi/config.json`
- Windows: `%LOCALAPPDATA%\nexi\config.json`

Create manually:

```bash
mkdir -p ~/.local/share/nexi
cat > ~/.local/share/nexi/config.json << 'EOF'
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model": "google/gemini-2.5-flash-lite",
  "jina_key": "jina_...",
  "default_effort": "m",
  "max_output_tokens": 8192,
  "time_target": 600,
  "max_context": 128000,
  "auto_compact_thresh": 0.9,
  "compact_target_words": 5000,
  "preserve_last_n_messages": 3,
  "tokenizer_encoding": "cl100k_base",
  "jina_timeout": 30,
  "llm_max_retries": 3
}
EOF
```

## Context Management Settings

For long research sessions, NEXI automatically compacts conversation:

```json
{
  "max_context": 128000,
  "auto_compact_thresh": 0.9,
  "compact_target_words": 5000,
  "preserve_last_n_messages": 3
}
```

- `max_context`: Model's context window (default: 128k tokens)
- `auto_compact_thresh`: Trigger at 90% of context
- `compact_target_words`: Target summary word count
- `preserve_last_n_messages`: Recent messages to keep uncompressed

## Validation

Test your setup:

```bash
# Quick test
nexi --plain "hello"

# Verbose test (see all API calls)
nexi -v "test query"

# Check history for errors
nexi --last 1
```

## MCP Server Setup (Optional)

Install with MCP support:

```bash
uv sync --group mcp
# or
pip install -e ".[mcp]"
```

Run as MCP server:

```bash
# STDIO transport
python -m nexi.mcp_server_cli

# HTTP transport
python -m nexi.mcp_server_cli http 0.0.0.0 8000
```

Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

## Troubleshooting Install Issues

| Issue | Solution |
|-------|----------|
| Command not found | Ensure `~/.local/bin` (or pipx/uv bin dir) is in PATH |
| Config not found | Run `nexi` to trigger wizard, or create manually |
| Permission denied | Use `pip install --user` or check directory permissions |
| JSON parse error | Validate with `python -m json.tool config.json` |
