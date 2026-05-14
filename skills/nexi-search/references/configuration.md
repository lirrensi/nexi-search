# NEXI Configuration

## Config location

- Linux/macOS: `~/.config/nexi/config.toml`
- Windows: `%USERPROFILE%\\.config\\nexi\\config.toml`

## First run

On first run, NEXI creates the commented config template automatically and exits so you can fill it in:

```bash
nexi "test query"
```

Then open it or run guided onboarding:

```bash
nexi config
nexi onboard
nexi doctor
```

## What belongs here

This file is for the config lifecycle and the base TOML shape.

- file location
- first-run bootstrap behavior
- `nexi config`, `nexi onboard`, `nexi doctor`
- the shared config skeleton

For provider-specific setup, use [references/providers.md](providers.md).

## Multi-Key Support

Every credentialed provider accepts `api_key` as either a single string or a **list of strings**:

```toml
# Single key (backward compatible)
api_key = "<your_api_key>"

# Multiple keys — tried in order with fallback, or rotated with round_robin
api_key = ["<key_one>", "<key_two>"]
api_key_strategy = "fallback"
```

The `api_key_strategy` field controls per-provider key behaviour:
- `"fallback"` (default) — tries keys in order until one succeeds before moving to the next provider
- `"round_robin"` — rotates the starting key across requests in the same process

See the README `Multi-Key Provider Support` section for full details.

## Base template

```toml
# api_key may be either a single string or a list of strings (for multiple keys).
# api_key_strategy controls per-provider key behaviour:
#   "fallback"    - try keys in order until one succeeds (default)
#   "round_robin" - rotate the starting key across requests in the same process

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

# [providers.crawl4ai_local]
# type = "crawl4ai"
# headless = true

# Uncomment one LLM and one search provider.
# llm_backends = ["openrouter"]
# search_backends = ["tavily"]
#
# [providers.openrouter]
# type = "openai_compatible"
# base_url = "https://openrouter.ai/api/v1"
# api_key = "<your_api_key>"
# api_key_strategy = "fallback"
# model = "google/gemini-2.5-flash-lite"
#
# [providers.tavily]
# type = "tavily"
# api_key = ["<your_first_key>", "<your_second_key>"]
# api_key_strategy = "fallback"
# search_depth = "basic"
# topic = "general"
# max_results = 5
```

## Minimal requirements

1. one LLM provider for `nexi`
2. one search provider for `nexi` and `nexi-search`
3. fetch works immediately with the default zero-config backends

## Test it

```bash
nexi --plain "hello world"
nexi-search --json "hello world"
nexi-fetch --json "https://example.com"
nexi-search --provider jina "hello world"
nexi-fetch --provider special_trafilatura "https://example.com"
```

The public search-depth control is `-e s|m|l`; output-token and iteration budgets are internal.

## Common issues

| Error | Fix |
|-------|-----|
| Command not found | Add uv tool dir to PATH |
| 401 Unauthorized | Check `api_key` in config |
| Config created then exit | Open `config.toml`, enable one LLM and one search provider |
| `nexi-search` not ready | Run `nexi doctor` and enable one search provider |
| `nexi-fetch` not ready | Run `nexi doctor` and enable one fetch provider |
| No search results | Check the active search provider key is valid |
| `--provider` mismatch | Use a provider family that matches the command (`search` or `fetch`) |
