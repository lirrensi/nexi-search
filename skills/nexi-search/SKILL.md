---
name: nexi-search
description: Guide for using NEXI - the self-hosted LLM-powered web search CLI. Use when users need help running searches, understanding effort levels, using history commands, piping output, or troubleshooting Nexi. Assumes Nexi is already installed; see references/installation.md for install details.
---

# NEXI Usage Guide

NEXI is a terminal-based web search tool powered by LLMs. It searches the web, reads pages, and synthesizes answers with citations.

## Quick Install (if needed)

```bash
uv tool install git+https://github.com/lirrensi/nexi-search.git
# Then run: nexi (triggers config wizard)
```

See [references/installation.md](references/installation.md) for detailed install/config.
Note that first it required to configure with config.json file! Ask user to manually setup first OR write a config as described in reference.

## Basic Usage

### Simple Search

```bash
nexi "how do rust async traits work"
```

### Effort Levels

Control search depth with `-e`:

```bash
nexi -e s "quick fact"      # 8 iterations - fast
nexi -e m "typical query"   # 16 iterations - balanced (default)
nexi -e l "deep research"   # 32 iterations - thorough
```

### Verbose & Plain Modes

```bash
nexi -v "query"        # See LLM calls and tool execution
nexi --plain "query"   # No colors/emojis - good for scripts
```

## Piping and Scripting

```bash
# Pipe input to Nexi
echo "what is this" | nexi
cat question.txt | nexi

# Save output
nexi --plain "topic" > result.txt

# Use in scripts
ANSWER=$(nexi --plain "weather in Tokyo")
```

## Interactive Mode (REPL)

Start with just `nexi`:

```
$ nexi
nexi> how do I use docker compose
[answer appears]

nexi> what about volumes
[continues with context]

nexi> exit
```

Commands: `exit`, `quit`, `q` to leave.

## History Commands

```bash
nexi --last 5          # Show last 5 searches
nexi --prev            # Show latest result
nexi --show abc123     # Show specific by ID
nexi --clear-history   # Delete all history
```

## Configuration

```bash
nexi --config          # Show config path
nexi --edit-config     # Open in $EDITOR
```

Config lives at `~/.local/share/nexi/config.json`.

## CLI Options

| Flag | Description |
|------|-------------|
| `-e, --effort s/m/l` | Search depth |
| `-v, --verbose` | Show LLM calls |
| `--plain` | No colors/emojis |
| `--max-len N` | Limit output tokens |
| `--max-iter N` | Max iterations |
| `--time-target N` | Force answer after N seconds |
| `--last N` | Show last N searches |
| `--prev` | Show latest result |
| `--show ID` | Show specific search |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Check API keys in config |
| 429 Rate Limit | Wait or switch model |
| Slow searches | Use `-e s` or increase `time_target` |
| UTF-8 issues | Use `--plain` |

## References

- Detailed install/config: [references/installation.md](references/installation.md)
