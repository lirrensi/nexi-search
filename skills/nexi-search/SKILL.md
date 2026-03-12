---
name: nexi-search
description: NEXI provides three useful CLI surfaces - `nexi` for full agent answers with citations, `nexi-search` for direct search-provider results, and `nexi-fetch` for direct page fetching or extraction. Use it when you need current information from the web or want a pipe-friendly search and fetch toolchain.
---

# NEXI for Agents

NEXI has three main command surfaces for agents:

- `nexi` - full agentic search with synthesized answers and citations
- `nexi-search` - direct search-provider execution without the agent loop
- `nexi-fetch` - direct fetch or extraction without the agent loop

## Invocation

Pick the command that matches the job:

```bash
nexi --plain "your search query"
nexi-search --json "your search query"
nexi-fetch --json "https://example.com"
```

- Use `nexi --plain` when you want a final synthesized answer with citations.
- Use `nexi-search --json` when you want raw search results for scripts or agent post-processing.
- Use `nexi-fetch --json` when you want fetched page content or extraction payloads.

`nexi --plain` output is plain text with citations in `[1]`, `[2]` format and a sources section at the end.

## Effort Levels

Control agent search depth with `-e`:

| Level | Iterations | Use When |
|-------|------------|----------|
| `-e s` | 8 | Quick facts, simple lookups |
| `-e m` | 16 | Default, most queries |
| `-e l` | 32 | Deep research, complex topics |

Example:
```bash
nexi --plain -e l "compare React vs Svelte performance 2024"
```

## Output Format

```text
[Answer text with inline citations like [1] and [2]...]

Sources:
[1] https://example.com/page1 - "Page Title"
[2] https://example.com/page2 - "Another Title"
```

Parse the sources section to extract URLs for further processing if needed.

## Additional Flags

| Flag | Purpose |
|------|---------|
| `--max-len N` | Limit output tokens (default 8192) |
| `--max-iter N` | Override max iterations |
| `--time-target N` | Force answer after N seconds |
| `-v` | Verbose: show tool calls (debugging) |
| `--json` | Structured output for `nexi-search` and `nexi-fetch` |

## Config Lifecycle

If the config file does not exist yet, NEXI creates `~/.config/nexi/config.toml`, tells you it is incomplete, and exits.

Useful commands:

```bash
nexi config
nexi onboard
nexi doctor
```

- `nexi config` opens the current config file
- `nexi onboard` guides a basic LLM plus search setup
- `nexi doctor` checks whether `nexi`, `nexi-search`, and `nexi-fetch` are ready

## Prerequisites

NEXI requires configuration before use. See [references/installation.md](references/installation.md) for:
- Install commands
- Required API keys and default fetch setup
- Config file location and format

If NEXI fails with config errors, read the installation reference to help set it up.
