# NEXI Provider Matrix

This document is the canonical matrix of provider families that NEXI supports today and provider families that are planned but not yet supported.

If a provider family is not listed here, it is not part of the product.

## Status Model

| Status | Meaning |
| --- | --- |
| Supported | Implemented in code and valid in user config today |
| Planned | Intentionally tracked for future support but not valid in user config today |

## Template Rules

- The generated `~/.config/nexi/config.toml` template MUST show the shipped provider families that matter to end users.
- Zero-config fetch providers SHOULD be active in the generated template.
- Providers that need secrets or explicit activation SHOULD appear as commented examples in the template.
- `nexi doctor` MUST treat only `Supported` provider families as valid.

## Supported Provider Families

| Type string | Capabilities | Suggested instance names | Template treatment | Required fields | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `openai_compatible` | LLM | `openrouter`, `openai`, `local_openai` | Commented example | `base_url`, `api_key`, `model` | Supported | Covers OpenRouter, OpenAI, local OpenAI-compatible servers, and similar APIs. |
| `jina` | Search, Fetch | `jina` | Commented example | `api_key` | Supported | Search requires user activation; fetch is available but not zero-config. |
| `searxng` | Search | `searxng` | Commented example | `base_url` | Supported | Self-hosted search backend; optional query scoping via `engines`, `categories`, `language`, and `safesearch`. |
| `special_trafilatura` | Fetch | `special_trafilatura` | Active by default | none | Supported | Zero-config resilient fetch fallback: HTTPX + Trafilatura + best-effort text extraction. |
| `special_playwright` | Fetch | `special_playwright` | Active by default | none | Supported | Headed Playwright fetch fallback that extracts rendered page text instead of raw HTML. |
| `markdown_new` | Fetch | `markdown_new` | Active by default | none | Supported | Zero-key remote markdown fetch fallback. |
| `crawl4ai` | Fetch | `crawl4ai_local` | Commented example | none | Supported | Local/runtime-backed fetch provider example. Optional runtime dependency remains a user environment concern. |
| `tavily` | Search, Fetch | `tavily` | Commented example | `api_key` | Supported | Source-first search and extraction provider family. |
| `exa` | Search, Fetch | `exa` | Commented example | `api_key` | Supported | Semantic search plus page-content retrieval. |
| `firecrawl` | Search, Fetch | `firecrawl` | Commented example | `api_key` | Supported | Search and scrape-style fetch provider family. |
| `linkup` | Search, Fetch | `linkup` | Commented example | `api_key` | Supported | Search-oriented provider with fetch support. |
| `brave` | Search | `brave` | Commented example | `api_key` | Supported | Search-only provider. |
| `serpapi` | Search | `serpapi` | Commented example | `api_key` | Supported | Search-only provider. |
| `serper` | Search | `serper` | Commented example | `api_key` | Supported | Search-only provider. |
| `perplexity_search` | Search | `perplexity` | Commented example | `api_key` | Supported | Search-only provider. |
| `provider-<file>` | LLM, Search, Fetch | `custom_llm`, `custom_search`, `custom_fetch` | Commented example | local Python file in config directory | Supported | Custom provider type that loads capability functions from `~/.config/nexi/<file>.py`. |

## Planned Provider Families

These provider families are tracked intentionally, but they are not valid shipped provider types until they move into the Supported table.

| Type string | Expected capabilities | Status | Notes |
| --- | --- | --- | --- |
| `cloudflare_markdown` | Fetch | Planned | Strong future fetch candidate for rendered markdown output. |
| `you` | Search, Fetch | Planned | Possible answer-grounded search/fetch provider family. |
| `brightdata_serp` | Search | Planned | Enterprise SERP option, not first-class yet. |
| `dataforseo` | Search | Planned | SEO-heavy search source, tracked but not enabled. |
| `scrapingdog` | Search | Planned | Commodity SERP option, tracked but not enabled. |
| `oxylabs` | Search | Planned | Enterprise SERP provider, tracked but not enabled. |
| `diffbot` | Fetch | Planned | Structured extraction option, not implemented. |
| `apify` | Crawl, Fetch | Planned | Better fit for future crawl/job workflows than current one-shot fetch. |

## Provider Contracts

- Every configured provider instance MUST declare one `type` from the Supported table.
- A provider family MAY support one capability or several capabilities.
- Provider families that need credentials MUST fail readiness checks when active but incomplete.
- Supported provider-specific field requirements live in code-level validation and MUST remain consistent with this matrix.
- Planned provider families MUST NOT be accepted by config validation or doctor until implemented.

## Default Template Expectations

The generated template SHOULD look like this at a high level:

- `fetch_backends = ["special_trafilatura", "special_playwright", "markdown_new"]`
- `llm_backends = []` until the user activates one
- `search_backends = []` until the user activates one
- visible commented examples for common LLM/search providers and the `crawl4ai_local` opt-in example

## Notes On Discovery And Planning

- This file is the stable product-facing matrix.
- Deeper research notes, API comparisons, or vendor scouting belong in planning documents, not in the canonical support matrix.

## Direct Provider Overrides

- `nexi-search --provider NAME` and `nexi-fetch --provider NAME` may target any Supported provider instance that matches the command capability.
- The override bypasses the configured fallback chain and uses only the named provider instance.
- If the named provider is missing or does not support the command capability, the CLI fails immediately.
