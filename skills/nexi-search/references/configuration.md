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

## Base template

```toml
llm_backends = []
search_backends = []
fetch_backends = ["special_trafilatura", "special_playwright", "markdown_new"]

default_effort = "m"

[providers.markdown_new]
type = "markdown_new"
method = "auto"
retain_images = false

# [providers.crawl4ai_local]
# type = "crawl4ai"
# headless = true

# Uncomment one LLM and one search provider.
# llm_backends = ["openrouter"]
# search_backends = ["jina"]
#
# [providers.openrouter]
# type = "openai_compatible"
# base_url = "https://openrouter.ai/api/v1"
# api_key = "<your_api_key>"
# model = "google/gemini-2.5-flash-lite"
#
# [providers.jina]
# type = "jina"
# api_key = "<your_api_key>"
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
