# Plan: Provider-Orchestrated Search, Fetch, and LLM Backends
_Done looks like this: `nexi` uses ordered provider chains for LLM, search, and fetch, and the repo ships `nexi-search` and `nexi-fetch` as direct CLIs backed by the same orchestration layer._

---

# Checklist
- [x] Step 1: Expand packaging and script registration
- [x] Step 2: Create backend package skeleton
- [x] Step 3: Reshape config schema and legacy migration
- [x] Step 4: Implement provider protocols and registry
- [x] Step 5: Port Jina and OpenAI-compatible adapters
- [x] Step 6: Implement search and fetch orchestrators
- [x] Step 7: Implement LLM orchestrator
- [x] Step 8: Refactor `nexi/tools.py` to use orchestrators
- [x] Step 9: Refactor `nexi/search.py` to use LLM provider fallback
- [x] Step 10: Add `nexi-search` direct CLI
- [x] Step 11: Add `nexi-fetch` direct CLI
- [x] Step 12: Update shared callers and exports
- [x] Step 13: Rewrite and add tests
- [x] Step 14: Run full verification

---

## Context
- Canon lives in `docs/product.md` and `docs/arch.md`. Code must follow those two files exactly.
- Current hardcoded backend paths:
  - `nexi/tools.py` instantiates `JinaSearchBackend()` and `JinaContentFetcher()` directly.
  - `nexi/search.py` constructs `AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)` directly and retries with `config.llm_max_retries`.
  - `nexi/config.py` still uses legacy top-level fields (`base_url`, `api_key`, `model`, `jina_key`).
  - `nexi/cli.py` and `nexi/mcp_server.py` reconstruct temporary `Config(...)` objects with those legacy fields.
- Packaging landmine: `pyproject.toml` currently has `[tool.setuptools] packages = ["nexi"]`. Adding `nexi/backends/` without changing packaging will omit the new subpackage from builds.
- Existing tests already cover config, tools, MCP, citations, compaction, history, and token counting. Reuse those files instead of creating a parallel test strategy.

## Prerequisites
- Work from `C:\Users\rx\001_Code\100_M\Nexi_Search`.
- Run `uv sync` before touching code. If `uv sync` fails, stop and report the failure output.
- Do not use live provider credentials in tests. Mock every network call.
- Keep the shipped provider set for this pass to:
  - `jina` for search and fetch
  - `openai_compatible` for LLM
- Keep provider examples in code and tests aligned with the shipped provider set above.

## Scope Boundaries
- Do not edit `docs/product.md` or `docs/arch.md` in this plan. The docs are already updated.
- Do not change citation numbering behavior in `nexi/citations.py`.
- Do not change history storage format in `nexi/history.py`.
- Do not change compaction behavior in `nexi/compaction.py` except for import or signature compatibility required by the new config object.
- Do not add provider adapters for `exa`, `duckduckgo`, `tavily`, or `firecrawl` in this pass.

---

## Steps

### Step 1: Expand packaging and script registration
Open `pyproject.toml`. In `[project.scripts]`, keep `nexi = "nexi.cli:main"` and add `nexi-search = "nexi.search_cli:main"` and `nexi-fetch = "nexi.fetch_cli:main"`. Replace `[tool.setuptools] packages = ["nexi"]` with package discovery that includes `nexi` and every `nexi.*` subpackage.

✅ Success: `pyproject.toml` contains three script entries and setuptools package discovery includes `nexi.*`.
❌ If failed: restore `pyproject.toml` to the pre-edit state and stop. Do not create `nexi/backends/` until packaging is corrected.

### Step 2: Create backend package skeleton
Create these files with module docstrings, `from __future__ import annotations`, and minimal public exports only:
- `nexi/backends/__init__.py`
- `nexi/backends/base.py`
- `nexi/backends/registry.py`
- `nexi/backends/orchestrators.py`
- `nexi/backends/jina.py`
- `nexi/backends/openai_compatible.py`

Do not move logic in this step. Create only the package structure and import-safe module skeletons.

✅ Success: `uv run python -c "import nexi.backends"` runs without `ImportError`.
❌ If failed: remove the new files created in `nexi/backends/` and stop.

### Step 3: Reshape config schema and legacy migration
Open `nexi/config.py`. Replace the `Config` dataclass fields with the canon from `docs/arch.md`: `llm_backends`, `search_backends`, `fetch_backends`, `providers`, `default_effort`, `max_output_tokens`, `time_target`, `max_context`, `auto_compact_thresh`, `compact_target_words`, `preserve_last_n_messages`, `tokenizer_encoding`, `provider_timeout`, `search_provider_retries`, and `fetch_provider_retries`. Remove legacy top-level fields from the dataclass.

Add a legacy migration path inside `load_config()` or a dedicated helper in `nexi/config.py` that converts existing config files with `base_url`, `api_key`, `model`, and `jina_key` into the new shape:
- create provider instance `openai_default` with `type = "openai_compatible"`
- create provider instance `jina` with `type = "jina"`
- set `llm_backends = ["openai_default"]`
- set `search_backends = ["jina"]`
- set `fetch_backends = ["jina"]`

Update `DEFAULT_CONFIG`, `validate_config()`, `run_first_time_setup()`, and `save_config()` to read and write only the new canonical shape.

✅ Success: `uv run python -c "from nexi.config import Config, load_config; print(Config.__annotations__.keys())"` shows the new fields, and loading an old-format config produces a `Config` object with populated provider chains.
❌ If failed: revert `nexi/config.py` to the last working state in the working tree and stop.

### Step 4: Implement provider protocols and registry
Open `nexi/backends/base.py` and define protocol or dataclass contracts for `Provider`, `SearchProvider`, `FetchProvider`, and `LLMProvider`. Each provider contract must expose `name`, `validate_config(config: dict[str, Any])`, and the capability method required by the provider type.

Open `nexi/backends/registry.py` and implement explicit registries keyed by provider `type`, not provider instance name. Add resolver functions that:
- accept a provider instance name and the top-level `providers` mapping
- read `providers[provider_name]["type"]`
- return the registered adapter class for the requested capability
- raise `ValueError` if the provider instance is missing, the `type` field is missing, or the type is unsupported for the requested capability

✅ Success: `uv run python -c "from nexi.backends.registry import resolve_search_provider"` imports successfully and registry lookups fail with `ValueError`, not `KeyError`.
❌ If failed: keep `nexi/backends/base.py` and `nexi/backends/registry.py` import-safe, remove the broken resolver logic, and stop.

### Step 5: Port Jina and OpenAI-compatible adapters
Move the current Jina HTTP logic out of `nexi/tools.py` into `nexi/backends/jina.py`.

In `nexi/backends/jina.py`:
- move the current single-query search HTTP flow into a `JinaSearchProvider`
- move the current single-URL fetch HTTP flow into a `JinaFetchProvider`
- keep shared `httpx.AsyncClient` reuse and `_url_cache` behavior compatible with current code
- add `validate_config()` that accepts the new provider config shape and requires only fields that Jina truly needs

In `nexi/backends/openai_compatible.py`:
- create `OpenAICompatibleLLMProvider`
- move the current chat completion call shape from `nexi/search.py` into `complete(...)`
- require `base_url`, `api_key`, and `model` in `validate_config()`

Register `jina` and `openai_compatible` in `nexi/backends/registry.py`.

✅ Success: `uv run python -c "from nexi.backends.jina import JinaSearchProvider, JinaFetchProvider; from nexi.backends.openai_compatible import OpenAICompatibleLLMProvider"` imports successfully.
❌ If failed: move the copied logic back into the source module that still works, keep the new module imports clean, and stop.

### Step 6: Implement search and fetch orchestrators
Open `nexi/backends/orchestrators.py`. Add two async functions with narrow, testable signatures:
- `run_search_chain(queries, config, verbose)`
- `run_fetch_chain(urls, config, verbose)`

Implement the behavior exactly from `docs/arch.md`:
- read pending items from the input list
- iterate through `config.search_backends` or `config.fetch_backends` in order
- resolve the provider adapter from `providers[provider_name]["type"]`
- validate provider config before every provider run
- retry failed items only within the current provider using `config.search_provider_retries` or `config.fetch_provider_retries`
- preserve successful items immediately
- move only remaining failed items to the next provider
- return the existing tool payload shape plus `provider_failures`

Use exponential backoff for same-provider retries. Keep provider failure metadata structured and non-secret.

✅ Success: a direct unit test can prove that one successful query stays in the final result while a failed query moves to the next provider.
❌ If failed: leave the orchestrator functions in place, return only explicit error payloads, and stop before refactoring callers.

### Step 7: Implement LLM orchestrator
In `nexi/backends/orchestrators.py`, add an async function named `run_llm_chain(messages, tools, config, verbose, max_tokens)`.

Implement the behavior exactly from `docs/arch.md`:
- iterate through `config.llm_backends` in order
- resolve the provider adapter from `providers[provider_name]["type"]`
- validate provider config before the call
- call `complete(...)` once per provider
- on hard provider failure, mark the provider unhealthy for the current run and immediately try the next provider
- treat model-not-found as a hard provider failure
- raise a final structured error only after every configured LLM provider fails

Do not add same-provider retry for hard LLM failures.

✅ Success: a direct unit test can prove that one failing LLM provider yields a successful completion from the next provider in the same iteration.
❌ If failed: keep `run_llm_chain(...)` raising a structured terminal error and stop before changing `nexi/search.py`.

### Step 8: Refactor `nexi/tools.py` to use orchestrators
Open `nexi/tools.py`. Keep the public tool schema names (`web_search`, `web_get`, `final_answer`) unchanged. Remove direct backend instantiation from `web_search()` and `web_get()`. Make both functions accept the full `Config` object instead of `jina_key`, and route through `run_search_chain(...)` and `run_fetch_chain(...)`.

Keep these behaviors unchanged:
- URL cache clearing API surface
- `web_get` processing modes (`get_full`, `use_chunks`, default extraction)
- response shapes used by `nexi/search.py` and `nexi/citations.py`

Expose provider failure metadata in returned payloads without breaking existing `searches` and `pages` keys.

✅ Success: `execute_tool("web_search", ...)` and `execute_tool("web_get", ...)` still return payloads with `searches` and `pages` keys, and now also include provider failure metadata when relevant.
❌ If failed: restore the last working payload format in `nexi/tools.py` and stop.

### Step 9: Refactor `nexi/search.py` to use LLM provider fallback
Open `nexi/search.py`. Remove direct `AsyncOpenAI` client construction and the in-function `for attempt in range(config.llm_max_retries)` loop. Replace that block with calls to `run_llm_chain(...)` from `nexi/backends/orchestrators.py`.

Keep these behaviors unchanged:
- iteration accounting
- time-target handling
- compaction triggers
- stable citation numbering
- final answer flow when `final_answer` is called or max iterations are reached

Pass the full `Config` object into tool execution calls instead of `config.jina_key`.

✅ Success: `run_search_sync(...)` still returns `SearchResult`, and an LLM provider failure can be surfaced in verbose mode without crashing the loop immediately if another provider exists.
❌ If failed: restore the last working `run_search()` control flow in `nexi/search.py` and stop.

### Step 10: Add `nexi-search` direct CLI
Create `nexi/search_cli.py`. Use Click with `context_settings={"help_option_names": ["-h", "--help"]}`. Add a positional `query`, `--json`, and `-v/--verbose`. Load config with `ensure_config()`. Call the search orchestrator directly, not `run_search_sync()`.

Human-readable stdout mode must print plain text results only. JSON mode must print the exact structured payload returned by the search orchestrator.

✅ Success: `uv run nexi-search --help` shows the new command and `uv run nexi-search --json "test query"` reaches the search orchestrator path in tests.
❌ If failed: remove `nexi/search_cli.py`, remove the `nexi-search` script entry from `pyproject.toml`, and stop.

### Step 11: Add `nexi-fetch` direct CLI
Create `nexi/fetch_cli.py`. Use Click with `context_settings={"help_option_names": ["-h", "--help"]}`. Add positional `urls` with `nargs=-1`, `--json`, `--full`, `--chunks`, `--instructions`, and `-v/--verbose`. Load config with `ensure_config()`. Call the fetch orchestrator and existing extraction helpers directly, not the full search loop.

Human-readable stdout mode must print fetched/extracted content exactly once per requested URL. JSON mode must print the exact structured payload returned by the fetch path.

✅ Success: `uv run nexi-fetch --help` shows the new command and `uv run nexi-fetch --json "https://example.com"` reaches the fetch orchestrator path in tests.
❌ If failed: remove `nexi/fetch_cli.py`, remove the `nexi-fetch` script entry from `pyproject.toml`, and stop.

### Step 12: Update shared callers and exports
Open `nexi/cli.py`, `nexi/mcp_server.py`, and `nexi/__init__.py`.

In `nexi/cli.py` and `nexi/mcp_server.py`:
- stop reconstructing legacy `Config(...)` objects with removed fields
- clone the loaded `Config` object using the new fields only
- override only `default_effort`, `time_target`, and `max_output_tokens` where needed

In `nexi/__init__.py`:
- keep existing public exports working where possible
- export the new direct CLI helpers only if they belong in the public API

✅ Success: `uv run python -c "from nexi import Config, run_search_sync"` imports successfully, and `tests/test_mcp_server.py` can still call `nexi_search(...)`.
❌ If failed: restore import compatibility in `nexi/__init__.py` and stop.

### Step 13: Rewrite and add tests
Edit these existing test files to match the new config and orchestration model:
- `tests/test_config.py`
- `tests/test_tools.py`
- `tests/test_mcp_server.py`

Add these new test files:
- `tests/test_backends_registry.py`
- `tests/test_backends_orchestrators.py`
- `tests/test_direct_cli.py`

Required coverage:
- legacy config migration to the new provider shape
- provider config validation failure
- search partial failover across two mocked providers
- fetch partial failover across two mocked providers
- immediate LLM failover across two mocked providers
- `nexi-search --json`
- `nexi-fetch --json`

Mock every network call. Do not hit real provider endpoints.

✅ Success: all six scenarios above have explicit tests and all touched test files pass individually.
❌ If failed: keep failing tests in place, capture the exact failing command output, and stop before running the full suite.

### Step 14: Run full verification
Run these commands from repo root in this order:

1. `ruff check .`
2. `pytest tests/test_config.py tests/test_tools.py tests/test_mcp_server.py tests/test_backends_registry.py tests/test_backends_orchestrators.py tests/test_direct_cli.py`
3. `pytest tests/`
4. `uv run nexi-search --help`
5. `uv run nexi-fetch --help`

Do not skip any command. If a command fails, stop at the first failing command and keep the full output.

✅ Success: every command above exits with status code 0.
❌ If failed: do not continue to the next command. Record the failing command and output in the handoff note.

---

## Verification
- `pyproject.toml` exposes `nexi`, `nexi-search`, and `nexi-fetch`.
- `nexi/config.py` loads both new-format configs and legacy configs.
- `nexi/backends/` is importable as a packaged submodule.
- `nexi/tools.py` no longer instantiates hardcoded backends directly.
- `nexi/search.py` no longer constructs `AsyncOpenAI` directly from top-level config fields.
- `nexi-search --json` prints structured search results.
- `nexi-fetch --json` prints structured fetch results.
- All tests and lint commands from Step 14 pass.

## Rollback
- If a critical step breaks the repo and cannot be recovered, revert only the files touched by this plan with `git checkout -- pyproject.toml nexi/config.py nexi/tools.py nexi/search.py nexi/cli.py nexi/mcp_server.py nexi/__init__.py tests/test_config.py tests/test_tools.py tests/test_mcp_server.py tests/test_backends_registry.py tests/test_backends_orchestrators.py tests/test_direct_cli.py`.
- Remove newly added modules with `git clean -fd -- nexi/backends nexi/search_cli.py nexi/fetch_cli.py` only if those files were created by this plan and remain untracked.
- After rollback, run `pytest tests/test_config.py tests/test_tools.py tests/test_mcp_server.py` to confirm the repo returned to the pre-plan baseline.

Plan complete. Handing off to Executor.
