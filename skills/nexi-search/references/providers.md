# NEXI Provider Setup Guide

Use this when the user asks things like:

- “What is Tavily?”
- “How do I set up Exa?”
- “Which ones are cloud vs local?”
- “How do I make Playwright work?”

For the canonical support matrix, see `docs/provider-matrix.md`.

## Categories

| Category | What it means |
| --- | --- |
| Cloud/API | Needs an API key and a vendor dashboard |
| Self-hosted/local service | You run the backend yourself |
| Local/runtime | Uses your machine’s browser/runtime/Docker |
| Zero-config | Usually works without a key |
| Custom | A `provider-<file>.py` file in the config dir |

## Shared rules

1. Define each provider instance once, then reuse it in `llm_backends`, `search_backends`, or `fetch_backends`.
2. If search and fetch need different settings, use two instance names, like `jina_search` and `jina_fetch`.

### Multi-Key Support

Every credentialed provider accepts `api_key` as either a **single string** or a **list of strings**:

```toml
# One key (backward compatible)
api_key = "<your_api_key>"

# Multiple keys — tried in order with fallback, or rotated
api_key = ["<key_one>", "<key_two>"]
api_key_strategy = "fallback"
```

The `api_key_strategy` field controls per-provider key behaviour:
- `"fallback"` (default) — tries keys in order until one succeeds, then moves to the next provider if all fail
- `"round_robin"` — rotates the starting key across requests in the same process

All credentialed provider examples below include `api_key_strategy`. Zero-key providers (trafilatura, playwright, markdown_new, snitchmd, crawl4ai) do not use this field.

---

## LLM providers

### `openrouter` — Cloud/API

What it is: OpenAI-compatible hosted LLM access.

Setup:
1. Create an account in the OpenRouter dashboard.
2. Create an API key.
3. Pick a model.
4. Paste the key into config.

```toml
llm_backends = ["openrouter"]

[providers.openrouter]
type = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
model = "google/gemini-2.5-flash-lite"
```

Good for: fast hosted models with simple setup.

### `openai` — Cloud/API

What it is: Direct OpenAI API.

Setup:
1. Create an OpenAI account.
2. Create an API key.
3. Choose a model.

```toml
[providers.openai]
type = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
model = "gpt-4.1-mini"
```

Good for: standard OpenAI-compatible usage.

### `local_openai` — Local/runtime

What it is: Any local OpenAI-compatible server.

Setup:
1. Start your local server.
2. Make sure it exposes an OpenAI-compatible `/v1` endpoint.
3. Point NEXI at it.

```toml
[providers.local_openai]
type = "openai_compatible"
base_url = "http://localhost:11434/v1"
api_key = "local-key"
api_key_strategy = "fallback"
model = "your-model"
```

Good for: local models and offline-ish workflows.

### `custom_llm` — Custom

What it is: A local Python provider file.

Setup:
1. Put `custom_llm.py` in `~/.config/nexi/`.
2. Implement the provider hooks expected by NEXI.
3. Use `type = "provider-custom_llm"`.

```toml
[providers.custom_llm]
type = "provider-custom_llm"
```

---

## Search providers

### `jina` — Cloud/API

What it is: Search API with fetch support too.

Setup:
1. Create a Jina account.
2. Generate an API key.
3. Add it to config.

```toml
search_backends = ["jina"]

[providers.jina]
type = "jina"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
```

Good for: simple hosted search.

### `searxng` — Self-hosted/local service

What it is: self-hosted metasearch.

Setup:
1. Run SearXNG locally.
2. Point NEXI at its base URL.

```bash
docker run -d --name searxng -p 8999:8080 searxng/searxng:latest
```

```toml
[providers.searxng]
type = "searxng"
base_url = "http://localhost:8999"
```

Good for: local control and self-hosting.

### `tavily` — Cloud/API

What it is: search-first API.

Setup:
1. Create a Tavily account.
2. Copy the API key.
3. Paste it into config.

```toml
[providers.tavily]
type = "tavily"
api_key = ["<your_first_key>", "<your_second_key>"]
api_key_strategy = "fallback"
search_depth = "basic"
topic = "general"
max_results = 5
```

Good for: quick search with source grounding. Supports multiple keys — list two or more keys for automatic fallback.

### `exa` — Cloud/API

What it is: semantic search plus page retrieval.

Setup:
1. Create an Exa account.
2. Get the API key.
3. Add it to config.

```toml
[providers.exa]
type = "exa"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
num_results = 5
text = true
```

Good for: semantic discovery and content pulls.

### `firecrawl` — Cloud/API

What it is: crawl/search/scrape provider.

Setup:
1. Create a Firecrawl account.
2. Copy the API key.
3. Paste it into config.

```toml
[providers.firecrawl]
type = "firecrawl"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
only_main_content = true
formats = ["markdown"]
limit = 5
```

Good for: crawling and page extraction.

### `linkup` — Cloud/API

What it is: search-oriented provider with fetch support.

Setup:
1. Create a Linkup account.
2. Get the API key.
3. Add it to config.

```toml
[providers.linkup]
type = "linkup"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
depth = "standard"
output_type = "searchResults"
```

### `brave` — Cloud/API

What it is: search-only provider.

```toml
[providers.brave]
type = "brave"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
count = 5
```

### `serpapi` — Cloud/API

What it is: search-only SERP API.

```toml
[providers.serpapi]
type = "serpapi"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
engine = "google"
```

### `serper` — Cloud/API

What it is: search-only SERP API.

```toml
[providers.serper]
type = "serper"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
num = 5
```

### `perplexity` — Cloud/API

What it is: search-only provider.

```toml
[providers.perplexity]
type = "perplexity_search"
api_key = "<your_api_key>"
api_key_strategy = "fallback"
max_results = 5
```

### `custom_search` — Custom

What it is: a local Python search provider.

```toml
[providers.custom_search]
type = "provider-custom_search"
```

---

## Fetch providers

### `markdown_new` — Zero-config

What it is: remote markdown fetch fallback.

```toml
[providers.markdown_new]
type = "markdown_new"
method = "auto"
retain_images = false
```

Good for: quick markdown extraction with no extra key.

### `special_trafilatura` — Zero-config

What it is: resilient HTTP fetch with Trafilatura-first extraction.

Setup: none.

Good for: simple, low-friction extraction.

### `special_playwright` — Local/runtime

What it is: rendered-page fetch using a browser runtime.

Setup:
1. Install Playwright in your environment.
2. Install a browser, usually Chromium.

```bash
python -m playwright install chromium
```

If you use `uv`, run it through your project environment.

Good for: JavaScript-heavy pages.

### `snitchmd` — Local/runtime

What it is: Docker-backed rendered-page-to-markdown fetch.

Setup:
1. Install Docker Desktop.
2. Make sure Docker is running.
3. Keep the default `snitchmd` provider active.

Good for: robust rendered markdown with explicit Docker errors.

### `crawl4ai_local` — Local/runtime

What it is: opt-in local browser/runtime fetch provider.

Setup:
1. Start a browser/runtime with remote debugging on port `9222`.
2. Enable the provider in config.

```toml
# [providers.crawl4ai_local]
# type = "crawl4ai"
# headless = true
# cdp_url = "http://localhost:9222"
```

Good for: advanced local crawling setups.

### `jina`, `tavily`, `exa`, `firecrawl`, `linkup` — Cloud/API

These also work as fetch providers when configured in `fetch_backends`.

Use the same API key and instance settings as the search setup, but make sure the instance name is listed under `fetch_backends` too.

### `custom_fetch` — Custom

What it is: a local Python fetch provider.

```toml
[providers.custom_fetch]
type = "provider-custom_fetch"
```

---

## Quick chooser

- Want local search? `searxng`
- Want hosted search? `jina`, `tavily`, or `exa`
- Want browser-heavy fetch? `special_playwright` or `crawl4ai_local`
- Want no setup? `special_trafilatura` or `markdown_new`
- Want Docker-backed fetch? `snitchmd`

## After setup

Run:

```bash
nexi doctor
```

Then test the specific path you configured.
