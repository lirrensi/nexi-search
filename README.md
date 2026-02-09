# NEXI 🔍

**Intelligent web search CLI tool powered by LLMs and Jina AI**

![NEXI](https://raw.githubusercontent.com/lirrensi/nexi-search/main/img/nexi.jpg)

NEXI is an agentic search tool that uses large language models to understand your query, perform web searches using Jina AI, and synthesize comprehensive answers from multiple sources.

## ✨ Features

- **Agentic Search**: LLM-powered search that understands context and performs multi-step research
- **Smart Synthesis**: Combines information from multiple sources into coherent answers
- **Flexible Effort Levels**: Choose between quick (s), medium (m), or deep (l) search depth
- **Interactive Mode**: REPL-style interface for continuous searching
- **Search History**: Automatically saves and retrieves past searches
- **Verbose Logging**: See exactly what's happening under the hood
- **Cross-Platform**: Works on Windows, macOS, and Linux with proper UTF-8 support
- **Plain Mode**: Script-friendly output without colors or emojis

## 📦 Installation

### Option 1: Using uv (Recommended)

```bash
# Install directly from GitHub
uv tool install git+https://github.com/lirrensi/nexi-search.git

# Or clone and install
git clone https://github.com/lirrensi/nexi-search.git
cd nexi-search
uv tool install .
```

### Option 2: Using pipx

```bash
# Install directly from GitHub
pipx install git+https://github.com/lirrensi/nexi-search.git

# Or clone and install
git clone https://github.com/lirrensi/nexi-search.git
cd nexi-search
pipx install .
```

### Option 3: Using pip (Development)

```bash
# Clone the repository
git clone https://github.com/lirrensi/nexi-search.git
cd nexi-search

# Install in editable mode
pip install -e .
```

## ⚙️ Configuration

NEXI requires API keys for both the LLM provider and Jina AI.

### First-Time Setup

Run NEXI without arguments to start the interactive setup:

```bash
python -m nexi
```

You'll be prompted to enter:

1. **OpenAI-compatible API endpoint** (e.g., `https://openrouter.ai/api/v1`)
2. **API key** for your LLM provider
3. **Model name** (e.g., `google/gemini-2.5-flash-lite-preview-09-2025`)
4. **Jina AI API key** (get one at https://jina.ai)

### Manual Configuration

Edit the config file directly:

```bash
nexi --edit-config
```

Or find it at:
- **Windows**: `%LOCALAPPDATA%\nexi\config.json`
- **macOS/Linux**: `~/.local/share/nexi/config.json`

Example config:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model": "google/gemini-2.5-flash-lite-preview-09-2025",
  "jina_key": "jina_...",
  "default_effort": "m",
  "max_timeout": 300,
  "max_output_tokens": 8000
}
```

## 🚀 Usage

### Basic Search

```bash
# Simple search
nexi "how to use rust async traits"

# With effort level
nexi -e l "explain quantum entanglement"

# Quick search
nexi -e s "what is a test"
```

### Advanced Options

```bash
# Limit output length
nexi --max-len 40000 "detailed topic"

# Limit iterations
nexi --max-iter 5 "complex question"

# Force timeout
nexi --max-timeout 60 "quick answer"

# Verbose mode (see all the details)
nexi --verbose "how does photosynthesis work"

# Plain mode (no colors/emojis)
nexi --plain "script-friendly output"
```

### Interactive Mode

```bash
# Start REPL
nexi

# Or just run without arguments
python -m nexi
```

In interactive mode:
- Type your query and press Enter
- Type `exit`, `quit`, or `q` to quit
- Type `help` or `h` for commands

### Search History

```bash
# Show last 5 searches
nexi --last 5

# Show full result of latest search
nexi --prev

# Show full result by ID
nexi --show abc123

# Clear all history
nexi --clear-history
```

### Configuration Commands

```bash
# Show config file path
nexi --config

# Open config in editor
nexi --edit-config
```

### Piping from stdin

```bash
echo "your query here" | nexi
```

## 🎯 Effort Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `s` (small) | Quick, single search | Simple questions, definitions |
| `m` (medium) | Balanced search | Most queries, moderate depth |
| `l` (large) | Deep, multi-step research | Complex topics, comprehensive answers |

## 📊 Verbose Mode

Use `--verbose` to see detailed information:

```
[LLM] Calling google/gemini-2.5-flash-lite-preview-09-2025...
[LLM] Messages count: 2
[LLM] Response received
[LLM] Tokens: 429 (prompt: 407, completion: 22)
[Jina Search] Starting 1 parallel searches...
  [Jina Search] Query: what is a test
  [Jina Search] URL: https://s.jina.ai/?q=what is a test
  [Jina Search] Status: 200
  [Jina Search] Parsed 10 results from text format
[Tool Result] web_search returned 1 results
  ✓ 'what is a test' returned 10 results

Answer ready!

A **test** is generally a procedure intended to establish the quality...

Search completed in 48.3s (2 iterations, 3259 tokens)
```

## 🔧 Troubleshooting

### Windows UTF-8 Issues

If you see garbled characters or emojis on Windows:

1. **Set console code page** (NEXI does this automatically, but you can verify):
   ```cmd
   chcp 65001
   ```

2. **Use Windows Terminal** instead of Command Prompt for better Unicode support

3. **If issues persist**, use `--plain` mode:
   ```bash
   nexi --plain "your query"
   ```

### API Errors

- **401 Unauthorized**: Check your API keys in the config file
- **429 Rate Limit**: Wait a moment and try again, or switch to a different model
- **Timeout**: Increase `max_timeout` in config or use `--max-timeout`

### Jina AI Errors

- **401 Unauthorized**: Verify your Jina AI API key
- **No results**: Try rephrasing your query or using a different effort level

### Connection Issues

- **Timeout**: Check your internet connection
- **Proxy**: Set `HTTP_PROXY` and `HTTPS_PROXY` environment variables if needed

## 📁 Project Structure

```
nexi-search/
├── nexi/
│   ├── __init__.py       # Public API exports
│   ├── __main__.py       # Entry point with Windows encoding fix
│   ├── cli.py            # Click CLI interface
│   ├── config.py         # Configuration management
│   ├── history.py        # Search history (JSONL)
│   ├── output.py         # Output formatting (Rich)
│   ├── search.py         # Agentic search loop
│   ├── tools.py          # Jina AI tools (async)
│   └── prompts/          # System prompts (s/m/l)
├── main.py               # Alternative entry point
├── pyproject.toml        # Dependencies
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request at https://github.com/lirrensi/nexi-search/pulls.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Jina AI** for providing the search API
- **OpenAI** for the API specification
- **Rich** for beautiful terminal output
- **Click** for the CLI framework

## 📞 Support

For issues, questions, or suggestions, please open an issue at https://github.com/lirrensi/nexi-search/issues.

---

Made with 💕 by the NEXI team
