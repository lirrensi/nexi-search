---
name: nexi-search
description: NEXI provides three useful CLI surfaces - `nexi` for full agent answers with citations, `nexi-search` for direct search-provider results, and `nexi-fetch` for direct page fetching or extraction. Use it when you need current information from the web or want a pipe-friendly search and fetch toolchain.
---

# NEXI for Agents

Use the command that matches the job:

- `nexi` - full agentic search with synthesized answers and citations
- `nexi-search` - direct search-provider execution without the agent loop
- `nexi-fetch` - direct fetch or extraction without the agent loop

## Quick use

```bash
nexi --plain "your search query"
nexi-search --json "your search query"
nexi-fetch --json "https://example.com"
```

- `nexi --plain` = synthesized answer + citations.
- `nexi-search --json` = raw search results for scripts or post-processing.
- `nexi-fetch --json` = fetched content or extraction payloads.

`nexi --plain` outputs plain text with `[1]`, `[2]` citations and a sources section.

## Effort levels

Control search depth with `-e`:

| Level | Use When |
|-------|----------|
| `-e s` | Quick facts, simple lookups |
| `-e m` | Default, most queries |
| `-e l` | Deep research, complex topics |

Example:
```bash
nexi --plain -e l "compare React vs Svelte performance 2024"
```

## Common flags

| Flag | Purpose |
|------|---------|
| `--plain` | Plain output for scripts and agents |
| `-v` | Verbose: show tool calls and debug output |
| `--query-text TEXT` | Pass the query explicitly instead of positionally |
| `--last N` | Show the last N saved search previews |
| `--prev` | Show the latest saved full result |
| `--show ID` | Show one saved result by ID |
| `--json` | Structured output for `nexi-search` and `nexi-fetch` |

## Multi-key provider support

Every credentialed provider accepts `api_key` as either a single string or a list of strings. This means a provider can try multiple API keys before falling through to the next provider in the chain.

Configure per-provider key strategy in `~/.config/nexi/config.toml`:

```toml
[providers.tavily]
type = "tavily"
api_key = ["<key_one>", "<key_two>"]
api_key_strategy = "fallback"    # try in order (default)
# api_key_strategy = "round_robin"  # rotate starting key per request
```

- `"fallback"` (default) — keys are tried in order until one succeeds; only then does NEXI move to the next provider
- `"round_robin"` — the starting key advances on each call, distributing load across keys in long-lived processes

Zero-key providers (`special_trafilatura`, `markdown_new`, `snitchmd`, etc.) are unaffected.

If all keys for a provider fail, failure metadata includes `failure_kind: "api_key_exhausted"` without exposing real key values.

## Direct provider override

Use a specific provider when the fallback chain keeps failing or the user explicitly wants one provider.

- `nexi-search --provider NAME` targets one configured search provider only.
- `nexi-fetch --provider NAME` targets one configured fetch provider only.
- This bypasses fallback chaining, so missing or mismatched providers fail fast.

## Configuration commands

```bash
nexi config
nexi onboard
nexi doctor
```

- `nexi config` = open the current config file
- `nexi onboard` = guided LLM + search setup
- `nexi doctor` = check `nexi`, `nexi-search`, and `nexi-fetch`

## When to read refs

- New install or runtime setup issues: [references/installation.md](references/installation.md)
- Config file lifecycle and shape: [references/configuration.md](references/configuration.md)
- Provider-specific setup (Tavily, Exa, SearXNG, Playwright, etc.): [references/providers.md](references/providers.md)
