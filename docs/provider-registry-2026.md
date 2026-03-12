# NEXI Provider Registry 2026

Researched on 2026-03-12.

This document expands the provider shortlist into a fuller registry of realistic candidates for NEXI search and fetch backends.

## Scope and method

- Goal: identify providers that can be added to NEXI as `search`, `fetch`, or future `crawl` backends.
- Bias: plain HTTP APIs, JSON or Markdown output, minimal SDK lock-in, and clean mapping to NEXI's existing contracts.
- Verification standard:
  - `Verified`: official docs or changelog page found and endpoint/product shape is clear enough to implement.
  - `Partial`: official product page exists, but exact API surface or auth model was not fully confirmable from accessible docs.
  - `Unverified`: no stable official API docs found, docs are gated, or results mainly point to wrappers, blog posts, or aggregators.
- Important limitation: not every candidate from the brainstorm has equally accessible official docs. Those gaps are called out explicitly instead of being hand-waved.

## What fits NEXI best

- Best fit: one-shot HTTP APIs that return JSON search results or Markdown page content.
- Good fit: providers with separate search and extraction endpoints under one API key.
- Medium fit: answer-first APIs, crawl-job APIs, or scraper platforms that need normalization or polling.
- Hard fit: SDK-first local stacks, browser-install runtimes, or crawl platforms that return many pages per job rather than one page per URL.

## Current backend shape

- Search adapters map to `search(queries, config, timeout, verbose) -> {"searches": [...]}`.
- Fetch adapters map to `fetch(urls, config, timeout, verbose) -> {"pages": [...]}`.
- Job-based site crawls are useful, but they fit a future `web_crawl` contract better than the current one-URL fetch shape.
- Pure HTTP providers remain the best first implementations because they only need `httpx`.

## Already in NEXI

| Provider | Capability | Type string | Status | Notes |
| --- | --- | --- | --- | --- |
| Jina Search (`s.jina.ai`) | Search | `jina` | Shipped | Lightweight baseline and easy default fallback. |
| Jina Reader (`r.jina.ai`) | Fetch | `jina` | Shipped | Simple URL-to-Markdown baseline with low implementation cost. |

## Executive recommendation

### Best first-wave additions

| Provider | Capability | Type string | Verification | Why it is first-wave |
| --- | --- | --- | --- | --- |
| Tavily Search | Search | `tavily` | Verified | Strong source-first search, good for agent workflows, clean JSON. |
| Tavily Extract | Fetch | `tavily` | Verified | Same vendor, direct multi-URL extraction, easy pair with search. |
| Exa Search | Search | `exa` | Verified | Very strong semantic/discovery fit and clean response model. |
| Exa Contents | Fetch | `exa` | Verified | High-quality page content endpoint, good mapping to `web_get`. |
| Firecrawl Search | Search | `firecrawl` | Verified | Hybrid-friendly API that already thinks in AI-agent terms. |
| Firecrawl Scrape | Fetch | `firecrawl` | Verified | Strong Markdown output and JS-page handling. |
| Linkup Search | Search | `linkup` | Verified | Good answer-plus-sources shape with direct search API. |
| Linkup Fetch | Fetch | `linkup` | Partial | Attractive hybrid fit, but fetch docs are less public than search docs. |
| Perplexity Search | Search | `perplexity_search` | Partial | Useful search candidate if raw search endpoint stays stable. |
| Cloudflare `/markdown` | Fetch | `cloudflare_markdown` | Verified | Very promising rendered Markdown endpoint with low adapter complexity. |

### Good second-wave additions

| Provider | Capability | Type string | Verification | Why it is second-wave |
| --- | --- | --- | --- | --- |
| SerpAPI | Search | `serpapi` | Verified | Reliable commodity SERP fallback, but less differentiated. |
| Brave Search API | Search | `brave` | Verified | Independent index and privacy angle, but lower urgency than Tavily or Exa. |
| You.com Search / Research | Search | `you` | Partial | Interesting answer-grounded option, but overlaps first-wave hybrids. |
| You.com Contents | Fetch | `you` | Partial | Viable fetch candidate if content endpoint remains supported. |
| markdown.new | Fetch | `markdown_new` | Verified | Very easy fallback or default fetch backend. |
| Serper | Search | `serper` | Partial | Popular and cheap, but docs are less canonical than first-wave options. |
| Crawl4AI | Fetch | `crawl4ai` | Verified | Best self-hosted option, but requires optional runtime setup. |

### Keep for enterprise or future crawl work

| Provider | Capability | Verification | Why not early |
| --- | --- | --- | --- |
| Bright Data SERP API | Search | Partial | Powerful, but heavier platform surface and enterprise positioning. |
| Bright Data Web Extraction | Fetch | Partial | Strong, but more unblocker/platform than content-native fetch. |
| DataForSEO | Search | Verified | Rich API, but more SEO-centric than agent-centric. |
| Oxylabs | Search | Partial | Enterprise-scale SERP coverage, but not the cleanest first adapter. |
| Apify Website Content Crawler | Crawl/fetch | Verified | Better as future `web_crawl` than one-shot `web_get`. |
| Diffbot | Fetch | Verified | Smart structured extraction, but less Markdown-native. |
| Zyte API | Fetch | Verified | Strong extraction platform, but broad and enterprise-heavy. |
| Scrapfly / ZenRows / ScraperAPI | Fetch | Partial | Better as hard-target unblockers than first-choice content adapters. |

## Search candidates

### Tavily

- Capability: search, extract.
- Verification: `Verified`.
- Why it fits: source-first results, relevance-oriented shape, and a fetch/extract pair under the same auth model.
- NEXI fit: excellent for both `web_search` and `web_get`.
- Suggested type strings: `tavily`.
- Priority: `P1`.

### Exa

- Capability: search, contents.
- Verification: `Verified`.
- Why it fits: semantic retrieval is differentiated from commodity SERPs, and contents retrieval maps cleanly to page fetch.
- NEXI fit: excellent.
- Suggested type strings: `exa`.
- Priority: `P1`.

### Firecrawl

- Capability: search, scrape, crawl, map, agent-style browsing.
- Verification: `Verified`.
- Why it fits: unusually close to NEXI's use case, with direct Markdown output and strong hybrid ergonomics.
- NEXI fit: excellent for search/fetch, future fit for crawl.
- Suggested type strings: `firecrawl`.
- Priority: `P1`.

### Linkup

- Capability: search and likely fetch/content retrieval.
- Verification: `Verified` for search, `Partial` for fetch.
- Why it fits: AI-search-oriented output with sources and a modern API surface.
- NEXI fit: good to excellent.
- Suggested type strings: `linkup`.
- Priority: `P1`.

### Perplexity Sonar / search

- Capability: answer-first research plus search-style retrieval.
- Verification: `Partial`.
- Why it fits: strong live-web grounding and citation behavior.
- NEXI fit: better for search than fetch. Sonar chat should not be forced into a raw SERP contract unless NEXI wants answer-first blending.
- Suggested type strings: `perplexity_search`.
- Priority: `P1.5`.

### SerpAPI

- Capability: multi-engine SERP API.
- Verification: `Verified`.
- Why it fits: stable classic fallback and broad engine support.
- NEXI fit: good, though less differentiated than semantic or hybrid options.
- Suggested type strings: `serpapi`.
- Priority: `P1.5`.

### Brave Search API

- Capability: web search.
- Verification: `Verified`.
- Why it fits: independent index and simple HTTP API.
- NEXI fit: good search-only candidate.
- Suggested type strings: `brave`.
- Priority: `P2`.

### Serper

- Capability: Google-style SERP API.
- Verification: `Partial`.
- Why it fits: simple, cheap, popular.
- NEXI fit: good if a low-cost commodity SERP option is wanted.
- Suggested type strings: `serper`.
- Priority: `P2`.

### You.com Research / Search

- Capability: answer-grounded search and content surfaces.
- Verification: `Partial`.
- Why it fits: conversational search plus sources.
- NEXI fit: useful, but overlaps with Tavily, Linkup, and Perplexity.
- Suggested type strings: `you`.
- Priority: `P2`.

### Bright Data SERP API

- Capability: search at scale.
- Verification: `Partial`.
- Why it fits: enterprise reliability and scale.
- NEXI fit: medium. More relevant if NEXI wants high-volume or anti-bot infrastructure.
- Suggested type strings: `brightdata_serp`.
- Priority: `P3`.

### ScrapingDog

- Capability: Google-style SERP scraping API.
- Verification: `Partial`.
- Why it fits: price/performance for commodity SERPs.
- NEXI fit: low to medium because it overlaps heavily with Serper and SerpAPI.
- Suggested type strings: `scrapingdog`.
- Priority: `P3`.

### Jina Search (`s.jina.ai`)

- Capability: search.
- Verification: `Partial` for public docs, but product behavior is already known from current implementation.
- Why it fits: tiny integration cost and good fallback ergonomics.
- NEXI fit: already shipped.
- Suggested type strings: `jina`.
- Priority: existing baseline.

### DataForSEO

- Capability: search and SEO-oriented SERP data.
- Verification: `Verified`.
- Why it fits: mature and wide-featured.
- NEXI fit: medium. Useful when bulk SEO or SERP detail matters more than simple agent search.
- Suggested type strings: `dataforseo`.
- Priority: `P3`.

### Oxylabs

- Capability: SERP APIs and scraping infrastructure.
- Verification: `Partial`.
- Why it fits: strong enterprise scale.
- NEXI fit: medium. Similar role to Bright Data.
- Suggested type strings: `oxylabs`.
- Priority: `P3`.

### Bing Web Search API

- Capability: web search.
- Verification: `Partial`.
- Why it fits: official Microsoft lineage.
- NEXI fit: low. It is less compelling than newer independent or hybrid APIs unless an Azure-centric deployment requires it.
- Suggested type strings: `bing`.
- Priority: `P4`.

### Kagi API

- Capability: privacy-focused search.
- Verification: `Partial`.
- Why it fits: high-quality search brand and no-ads positioning.
- NEXI fit: low to medium unless access is broadly available and docs stabilize.
- Suggested type strings: `kagi`.
- Priority: `P4`.

### DuckDuckGo wrappers / DDGS / "dgg api"

- Capability: unofficial SERP access through wrappers or scraping.
- Verification: `Unverified` as a stable official API candidate.
- Why it does not fit: no durable official general SERP API surfaced cleanly in accessible docs.
- NEXI fit: poor for a core registry entry.
- Suggested action: filter out as a first-class provider type.

## Content and fetch candidates

### markdown.new

- Capability: fetch URL to Markdown.
- Verification: `Verified`.
- Why it fits: very low-friction HTTP interface and no-login appeal.
- NEXI fit: very good fallback fetch backend.
- Suggested type strings: `markdown_new`.
- Priority: `P1.5`.

### Cloudflare Browser Rendering `/markdown`

- Capability: render page and return Markdown.
- Verification: `Verified`.
- Why it fits: modern rendered-content endpoint with low integration complexity.
- NEXI fit: excellent.
- Suggested type strings: `cloudflare_markdown`.
- Priority: `P1`.

### Cloudflare Browser Rendering `/crawl`

- Capability: site crawl returning ordered content.
- Verification: `Verified`.
- Why it fits: strong future crawl backend.
- NEXI fit: not a direct `web_get` fit because it is a crawl contract, not one page per URL.
- Suggested action: hold for future `web_crawl`.

### Firecrawl Scrape

- Capability: fetch/extract/crawl.
- Verification: `Verified`.
- Why it fits: excellent Markdown output, JS handling, and AI-oriented ergonomics.
- NEXI fit: excellent.
- Suggested type strings: `firecrawl`.
- Priority: `P1`.

### Jina Reader (`r.jina.ai`)

- Capability: fetch URL to LLM-friendly text/Markdown.
- Verification: `Partial` for accessible docs, but runtime behavior is already validated by current use.
- Why it fits: dead-simple fallback and proven baseline.
- NEXI fit: already shipped.
- Suggested type strings: `jina`.

### Tavily Extract

- Capability: URL extraction.
- Verification: `Verified`.
- Why it fits: strong pairing with Tavily Search and direct multi-URL support.
- NEXI fit: excellent.
- Suggested type strings: `tavily`.
- Priority: `P1`.

### Exa Contents

- Capability: page contents retrieval.
- Verification: `Verified`.
- Why it fits: clean fetch endpoint that aligns well with one page per URL.
- NEXI fit: excellent.
- Suggested type strings: `exa`.
- Priority: `P1`.

### Linkup Fetch

- Capability: content retrieval.
- Verification: `Partial`.
- Why it fits: good hybrid story if the API surface stays simple and public.
- NEXI fit: good.
- Suggested type strings: `linkup`.
- Priority: `P1.5`.

### You.com Contents

- Capability: page content fetch.
- Verification: `Partial`.
- Why it fits: easy hybrid pairing if the endpoint remains supported.
- NEXI fit: good.
- Suggested type strings: `you`.
- Priority: `P2`.

### Crawl4AI

- Capability: open-source local or self-hosted crawling and extraction.
- Verification: `Verified`.
- Why it fits: best self-hosted privacy-first candidate and produces LLM-friendly output.
- NEXI fit: good, but requires optional dependency and runtime setup.
- Suggested type strings: `crawl4ai`.
- Priority: `P2`.

### Apify Website Content Crawler / LLM scraper actors

- Capability: crawl and extraction workflows.
- Verification: `Verified`.
- Why it fits: broad actor ecosystem and cloud scheduling.
- NEXI fit: medium. Stronger as future crawl integration than direct one-shot fetch.
- Suggested type strings: `apify_content`, `apify_crawl`.
- Priority: `P3`.

### ScrapeGraphAI

- Capability: LLM-driven extraction.
- Verification: `Partial`.
- Why it fits: flexible extraction intent.
- NEXI fit: low to medium because it is less deterministic and more LLM-coupled than preferred fetch backends.
- Suggested type strings: `scrapegraphai`.
- Priority: `P4`.

### Bright Data Web Extraction

- Capability: extraction with JS rendering and anti-bot tooling.
- Verification: `Partial`.
- Why it fits: strong for hard sites at enterprise scale.
- NEXI fit: medium. Better as a specialist fallback than a default provider.
- Suggested type strings: `brightdata_extract`.
- Priority: `P3`.

### Skrape.ai

- Capability: AI scraping and Markdown/JSON extraction.
- Verification: `Unverified`.
- Why it is risky: public docs were not cleanly confirmable from accessible official sources during this pass.
- NEXI fit: low until docs and maturity are clearer.
- Suggested action: do not add yet.

### Scrapfly / ZenRows / ScraperAPI

- Capability: scraping infrastructure and extraction.
- Verification: `Partial`.
- Why they fit: solid hard-target tooling.
- NEXI fit: medium as specialist fetch fallbacks, not first-choice content providers.
- Suggested type strings: `scrapfly`, `zenrows`, `scraperapi`.
- Priority: `P3`.

### Diffbot / Zyte API

- Capability: structured extraction.
- Verification: `Verified`.
- Why they fit: strong structured data products.
- NEXI fit: medium to low for current NEXI because the fetch contract prefers Markdown-like article content over graph-heavy structured output.
- Suggested type strings: `diffbot`, `zyte`.
- Priority: `P3`.

### Microlink / SimpleScraper ExtractMarkdown

- Capability: quick URL normalization or Markdown extraction.
- Verification: `Partial`.
- Why they fit: useful for testing and lightweight fallback experiments.
- NEXI fit: low as core providers.
- Suggested type strings: `microlink`, `simplescraper`.
- Priority: `P4`.

### Workers AI `toMarkdown()`

- Capability: conversion helper.
- Verification: `Partial`.
- Why it fits: useful building block.
- NEXI fit: helper only, not a standalone provider registry entry.

### Self-hosted stack: Playwright + Readability/Turndown, Trafilatura, BeautifulSoup + LLM

- Capability: local extraction.
- Verification: conceptually clear, not a single provider.
- Why it fits: full local control.
- NEXI fit: future optional-local backend family, not a registry vendor entry.

## Hybrid candidates

These are the strongest candidates if NEXI wants one vendor that can do both search and content extraction.

| Provider | Search | Fetch | Verification | Recommendation |
| --- | --- | --- | --- | --- |
| Firecrawl | Yes | Yes | Verified | Best hybrid candidate overall. |
| Tavily | Yes | Yes | Verified | Strongest pure HTTP agent-search pair. |
| Exa | Yes | Yes | Verified | Best semantic/discovery-oriented hybrid. |
| Linkup | Yes | Yes | Partial on fetch | Good hybrid bet if fetch docs stay accessible. |
| Jina | Yes | Yes | Partial docs, proven behavior | Keep as current baseline. |
| You.com | Yes | Yes | Partial | Optional later hybrid add. |
| Apify | Yes, indirectly | Yes | Verified | Better as crawl platform than simple hybrid backend. |

## Implementation notes per provider family

### Pure HTTP and easy to add

- `tavily`
- `exa`
- `firecrawl`
- `linkup`
- `serpapi`
- `brave`
- `cloudflare_markdown`
- `markdown_new`

These are the cleanest fits for `httpx`-only adapters.

### Search-first only

- `perplexity_search`
- `serper`
- `scrapingdog`
- `dataforseo`
- `oxylabs`
- `bing`
- `kagi`

These mostly help on the search side and would still need a separate fetch backend.

### Fetch-first only

- `cloudflare_markdown`
- `markdown_new`
- `crawl4ai`
- `diffbot`
- `zyte`
- `scrapfly`
- `zenrows`
- `scraperapi`

### Better as future crawl/job providers

- Cloudflare `/crawl`
- Apify crawlers
- Firecrawl crawl/map

## Suggested type strings and config fields

| Provider type | Required fields | Optional fields |
| --- | --- | --- |
| `tavily` | `api_key` | `search_depth`, `topic` |
| `exa` | `api_key` | `text`, `highlights`, `summary` |
| `firecrawl` | `api_key` | `formats`, `only_main_content` |
| `linkup` | `api_key` | provider-specific request options |
| `perplexity_search` | `api_key` | search mode knobs |
| `serpapi` | `api_key` | `engine`, `gl`, `hl` |
| `serper` | `api_key` | `gl`, `hl`, `location` |
| `brave` | `api_key` | `country`, `search_lang` |
| `cloudflare_markdown` | `api_token`, `account_id` | rendering options |
| `markdown_new` | none | `method`, `retain_images` |
| `you` | `api_key` | request tuning fields |
| `crawl4ai` | local runtime config | `browser_type`, `headless`, `cache_mode` |

## Best implementation order

1. `tavily`, `exa`, `firecrawl`
2. `linkup`, `cloudflare_markdown`, `markdown_new`
3. `serpapi`, `brave`, `perplexity_search`
4. `serper`, `you`, `crawl4ai`
5. Enterprise/specialist providers only if a concrete use case appears

## Docs verification notes

These candidates had enough public official material to keep moving:

- Tavily
- Exa
- Firecrawl
- Linkup search
- Cloudflare `/markdown` and `/crawl`
- SerpAPI
- Brave Search API
- Crawl4AI
- Apify
- DataForSEO
- Diffbot
- Zyte

These candidates were only partially confirmable in a clean docs pass:

- Perplexity raw search endpoint
- Linkup fetch/content endpoint
- You.com search/content endpoints
- Serper
- Bright Data SERP and extraction products
- Oxylabs
- Bing Web Search API current posture
- Kagi API availability/details
- Jina public docs for `s.jina.ai` and `r.jina.ai`
- Scrapfly / ZenRows / ScraperAPI
- Microlink / SimpleScraper

These candidates should be treated as not-docs-verified for now:

- DuckDuckGo wrapper ecosystem / "dgg api"
- Skrape.ai

## Bottom line

If the goal is to make the registry materially stronger without bloating implementation cost, the best additions are:

- Search: `tavily`, `exa`, `firecrawl`, `linkup`, `serpapi`
- Fetch: `tavily`, `exa`, `firecrawl`, `cloudflare_markdown`, `markdown_new`
- Optional local: `crawl4ai`

If the goal is broad completeness rather than immediate implementation, keep the enterprise and crawl-platform entries in the registry, but mark them as later-stage additions rather than near-term adapter work.
