# NEXI Installation & Configuration

## Install

```bash
uv tool install --upgrade git+https://github.com/lirrensi/nexi-search.git
```

Verify:
```bash
nexi --version
```

## Configuration

Config location:
- Linux/macOS: `~/.config/nexi/config.toml`
- Windows: `%USERPROFILE%\\.config\\nexi\\config.toml`

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

```toml
llm_backends = []
search_backends = []
fetch_backends = ["crawl4ai_local", "markdown_new"]

default_effort = "m"

[providers.markdown_new]
type = "markdown_new"
method = "auto"
retain_images = false

[providers.crawl4ai_local]
type = "crawl4ai"
headless = true

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

## Required Credentials

1. **One LLM provider**
   - OpenRouter is the default OpenAI-compatible example
   - Any OpenAI-compatible endpoint works if you provide `base_url`, `api_key`, and `model`

2. **One search provider**
   - Jina is the default commented example
   - Search requires a configured provider before `nexi` can run end-to-end

3. **Fetch providers**
   - `crawl4ai` and `markdown_new` are the default zero-config fetch backends in the template
   - They can work without an API key, though `crawl4ai` still depends on the local runtime being available

## Test

```bash
nexi --plain "hello world"
nexi-search --json "hello world"
nexi-fetch --json "https://example.com"
```

The public search-depth control is `-e s|m|l`; output-token and iteration budgets are internal now.

## Common Issues

| Error | Fix |
|-------|-----|
| Command not found | Add uv tool dir to PATH |
| 401 Unauthorized | Check `api_key` in config |
| Config created then exit | Open `config.toml`, enable one LLM and one search provider |
| `nexi-search` not ready | Run `nexi doctor` and enable one search provider |
| `nexi-fetch` not ready | Run `nexi doctor` and enable one fetch provider |
| No search results | Check the active search provider key is valid |
