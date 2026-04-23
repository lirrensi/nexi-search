# NEXI

**A resilient research and retrieval suite for the terminal.**

NEXI is not just another search wrapper with a cute prompt and one hardcoded backend pretending to be a product. It is a configurable CLI and MCP toolset for agentic research, direct search, and direct fetching, built around provider chains, failover, and script-friendly surfaces.

![NEXI](https://raw.githubusercontent.com/lirrensi/nexi-search/main/img/nexi.jpg)

---

## Version 2.1.0

NEXI 2.1.0 keeps the 2.0 cleanup release intact and sands down more of the rough edges.

- ✨ provider chains are now the core model, with ordered fallbacks across LLM, search, and fetch
- 🧾 config is now TOML at `~/.config/nexi/config.toml`
- 🚪 first run creates a commented template and exits cleanly instead of forcing a wizard
- 🛠️ `nexi`, `nexi-search`, and `nexi-fetch` are now clear first-class surfaces
- 🔌 MCP now mirrors those same surfaces with `nexi_agent`, `nexi_search`, and `nexi_fetch`
- ⚡ default fetch works out of the box with zero-config backends
- 🎚️ public search-depth control is just `--effort`, while iteration and token budgets stay internal
- ♻️ HTTP-backed providers reuse shared clients safely in long-lived CLI and MCP sessions

If you are upgrading from an older release, treat this as a config migration. The old config shape is not the v2 contract anymore.

---

## Why This Slaps

- 🧠 agentic research when you want the full answer
- 🔎 direct search when you want raw search results
- 📄 direct fetch when you want page content or extraction
- 🪜 fallback chains when a provider decides to be dramatic
- 🤖 MCP surfaces that mirror the CLI instead of making up weird names
- 🧩 one stable interface over many vendors

---

## What NEXI Is

NEXI gives you one suite with three practical runtime surfaces:

- `nexi` - full agentic research loop with citations and synthesis
- `nexi-search` - direct search-provider execution without the agent loop
- `nexi-fetch` - direct page fetching or extraction without the agent loop

You choose providers. You choose order. You choose fallbacks. NEXI handles the fixed interface on top.

That means it works well as:

- 🧑‍💻 a terminal-native research tool
- 🔁 a pipeable backend for agents and scripts
- 🧱 a stable abstraction over changing provider APIs
- 🛡️ a failure-resistant search stack instead of a single brittle vendor binding

## What NEXI Is Not

- not a hosted search subscription
- not a browser replacement with a giant UI
- not a one-provider toy CLI
- not a magical black box that hides how it got the answer

---

## Why It Exists

Most AI search tools are either expensive, opaque, locked to one provider, or annoying to automate.

NEXI exists for people who want:

- terminal-first workflows
- direct ownership of models, keys, and providers
- predictable command-line interfaces for agents
- transparent behavior with verbose debugging when needed
- graceful degradation when one provider fails

With NEXI, the interface stays stable while your backend stack evolves, providers wobble, and the internet continues being the internet.

---

## Supported Providers

Here is the practical short version of what NEXI supports today:

### LLM 🧠

- `openai_compatible` - OpenRouter, OpenAI, local OpenAI-compatible servers, and similar APIs
- custom Python LLM providers via `provider-<file>`

### Search 🔎

- `jina`
- `searxng` (self-hosted)
- `tavily`
- `exa`
- `firecrawl`
- `linkup`
- `brave`
- `serpapi`
- `serper`
- `perplexity_search`
- custom Python search providers via `provider-<file>`

### Fetch 📄

- `markdown_new`
- `crawl4ai`
- `special_trafilatura`
- `special_playwright`
- `jina`
- `tavily`
- `exa`
- `firecrawl`
- `linkup`
- custom Python fetch providers via `provider-<file>`

For the full canonical matrix, see `docs/provider-matrix.md`.

SearXNG is fully supported for local self-hosting.

---

## Why 2.0 Is Better

| Problem | NEXI 2.0 answer |
|---------|------------------|
| One provider goes down | Ordered fallbacks continue through the next configured provider |
| Search and fetch are coupled together awkwardly | Search, fetch, and LLM chains are configured independently |
| CLI surfaces are muddy | `nexi`, `nexi-search`, and `nexi-fetch` each have a clear role |
| MCP naming is confusing | MCP now mirrors the same runtime model exactly |
| First-run setup is messy | Missing config creates a readable TOML template and exits cleanly |
| Agents need structured payloads | Direct search and fetch support `--json` |

---

## Quick Start

```bash
# Install or upgrade
uv tool install --upgrade git+https://github.com/lirrensi/nexi-search.git

# First run creates the config template and exits
nexi "what is the deal with rust async traits"

# Open the config, use guided onboarding, or run readiness checks
nexi config
nexi onboard
nexi doctor
```

You will typically need:

- one LLM provider for `nexi`
- one search provider for `nexi` and `nexi-search`
- fetch can work immediately with the default zero-config fetch backends

### SearXNG locally

If you want a local search backend, SearXNG works well with Docker:

```bash
docker run -d --name searxng -p 8999:8080 searxng/searxng:latest
```

Then point NEXI at it in `~/.config/nexi/config.toml`:

```toml
search_backends = ["searxng"]

[providers.searxng]
type = "searxng"
base_url = "http://localhost:8999"
```

---

## Breaking Change for Upgrades

NEXI 2.0 changes local configuration.

- config now lives at `~/.config/nexi/config.toml`
- history now lives at `~/.config/nexi/history.jsonl`
- first run creates the template and exits
- `nexi config`, `nexi init`, `nexi onboard`, `nexi doctor`, and `nexi clean` are the supported config lifecycle commands

If you are upgrading from pre-2.0 behavior, recreate or review your config instead of assuming the old file layout still applies.

---

## Configuration

Config lives at `~/.config/nexi/config.toml`.

NEXI generates a commented template automatically if it is missing. The template enables zero-config fetch defaults and shows commented examples for provider families that require activation.

Useful commands:

- `nexi config` - open the current config file
- `nexi onboard` - guide a minimal working setup
- `nexi doctor` - check whether `nexi`, `nexi-search`, and `nexi-fetch` are ready
- `nexi clean` - reset config and history, then recreate the template

Example shape:

```toml
llm_backends = []
search_backends = []
fetch_backends = ["crawl4ai_local", "special_trafilatura", "special_playwright", "markdown_new"]

default_effort = "m"
max_context = 128000
provider_timeout = 30
direct_fetch_max_tokens = 8000
search_provider_retries = 2
fetch_provider_retries = 2

[providers.markdown_new]
type = "markdown_new"
method = "auto"
retain_images = false

[providers.crawl4ai_local]
type = "crawl4ai"
headless = true

# Uncomment one LLM and one search provider.
# llm_backends = ["openrouter"]
# search_backends = ["searxng"]
#
# [providers.openrouter]
# type = "openai_compatible"
# base_url = "https://openrouter.ai/api/v1"
# api_key = "<your_api_key>"
# model = "google/gemini-2.5-flash-lite"
#
# [providers.searxng]
# type = "searxng"
# base_url = "http://localhost:8999"
```

---

## Runtime Surfaces

### `nexi` 🧠

Use the full agent when you want a final answer with citations.

```bash
nexi "how do rust async traits work"
nexi -e l "explain quantum entanglement"
nexi --plain "script-friendly output"
nexi --verbose "show all tool calls"
```

### `nexi-search` 🔎

Use direct search when you want raw search-provider results without the agent loop.

```bash
nexi-search "rust async trait objects"
nexi-search --json "rust async trait objects"
nexi-search --provider jina "rust async trait objects"
```

### `nexi-fetch` 📄

Use direct fetch when you want page content or extraction without the full agent.

```bash
nexi-fetch "https://example.com/spec"
nexi-fetch --json "https://example.com/spec"
nexi-fetch --full "https://example.com/spec"
nexi-fetch --provider special_trafilatura "https://example.com/spec"
```

---

## Common Commands

### Main CLI ✨

| Command | Description |
|---------|-------------|
| `nexi "query"` | Run the full agentic workflow |
| `nexi -e s|m|l "query"` | Set quick, balanced, or deep effort |
| `nexi --plain "query"` | Plain output for scripts |
| `nexi --query-text "query"` | Pass the query explicitly instead of positionally |
| `nexi --last N` | Show the last N search previews |
| `nexi --prev` | Show the latest full result |
| `nexi --show ID` | Show one saved result by ID |
| `nexi --version` | Show the installed CLI version |

### Direct Tools 🔧

| Command | Description |
|---------|-------------|
| `nexi-search "query"` | Run direct search without the agent loop |
| `nexi-search --json "query"` | Return structured search output |
| `nexi-search --provider NAME "query"` | Use only one search provider instance |
| `nexi-fetch "url"` | Run direct fetch or extraction |
| `nexi-fetch --json "url"` | Return structured fetch output |
| `nexi-fetch --full "url"` | Return full fetched content |
| `nexi-fetch --provider NAME "url"` | Use only one fetch provider instance |

### Config Lifecycle 🧾

| Command | Description |
|---------|-------------|
| `nexi config` | Open the current config file |
| `nexi init` | Create the default config template if missing |
| `nexi onboard` | Guided activation of a basic provider setup |
| `nexi doctor` | Validate config and readiness |
| `nexi clean` | Reset config/history and recreate the template |

---

## MCP Server

NEXI can run as an MCP server and now mirrors the same three runtime surfaces.

### MCP Tools 🔌

| Tool | Mirrors | Purpose |
|------|---------|---------|
| `nexi_agent` | `nexi` | Full agentic research workflow |
| `nexi_search` | `nexi-search` | Direct search-provider execution |
| `nexi_fetch` | `nexi-fetch` | Direct fetch or extraction |

### Installation 📦

```bash
uv sync --group mcp
# or
pip install -e ".[mcp]"
```

### Running the Server 🚀

```bash
# STDIO transport
python -m nexi.mcp_server_cli

# HTTP transport
python -m nexi.mcp_server_cli http 0.0.0.0 8000
```

### Claude Desktop Configuration 🤖

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

See `docs/MCP_SERVER.md` for more details.

---

## Troubleshooting

When something breaks, the answer is usually: run `nexi doctor`, check which chain is active, and see which provider is being a little goblin today.

| Issue | Fix |
|-------|-----|
| Config created then exit | Open `~/.config/nexi/config.toml`, activate providers, or run `nexi onboard` |
| `nexi` not ready | Run `nexi doctor`; you need one usable LLM provider and one usable search provider |
| `nexi-search` not ready | Run `nexi doctor`; you need one usable search provider |
| `nexi-fetch` not ready | Run `nexi doctor`; you need one usable fetch provider |
| 401 Unauthorized | Check the active provider `api_key` |
| 429 Rate Limit | Wait, reduce load, or switch providers |
| UTF-8 garbled on Windows | Use Windows Terminal or `--plain` |

---

## License

MIT
