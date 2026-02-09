# NEXI 🔍

**Self-hosted perplexity in your terminal.** LLM-powered web search that stays out of your way.

![NEXI](https://raw.githubusercontent.com/lirrensi/nexi-search/main/img/nexi.jpg)

## Why?

- **Quick console search** - pipe it, script it, whatever
- **Offload from your coding agent** - no MCPs needed, just point your agent at a CLI tool. Manually type queries to get context without polluting it with bullshit
- **Cheaper models** - use flash-lite instead of burning your rate limits on search

Searches can take 1-2 minutes for deep research. It's a smol CLI tool vibecoded in 2hrs.

## Install

```bash
# uv (recommended)
uv tool install git+https://github.com/lirrensi/nexi-search.git

# pipx
pipx install git+https://github.com/lirrensi/nexi-search.git
```

## Setup

```bash
nexi  # interactive config on first run
```

Needs:
- OpenAI-compatible API endpoint + key + model
- Jina AI API key (free at https://jina.ai)

### Config File

Located at:
- Linux/macOS: `~/.local/share/nexi/config.json`
- Windows: `%LOCALAPPDATA%\nexi\config.json`

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

Edit with `nexi --edit-config` or just open the file.

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
| `--max-timeout N` | Timeout in seconds |
| `--last N` | Show last N searches |
| `--prev` | Show full result of latest search |
| `--show ID` | Show specific search by ID |
| `--clear-history` | Wipe search history |
| `--config` | Show config file path |
| `--edit-config` | Open config in editor |

### Effort Levels

| Level | Description | When to use |
|-------|-------------|-------------|
| `s` | Single search, quick answer | Simple facts, definitions |
| `m` | Balanced, multi-source | Most queries |
| `l` | Deep research, many iterations | Complex topics |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| 401 Unauthorized | Check API keys in config |
| 429 Rate Limit | Wait or switch to different/cheaper model |
| Timeout | Increase `max_timeout` or use `--max-timeout` |
| UTF-8 garbled (Windows) | Use `--plain` or Windows Terminal |

## License

MIT
