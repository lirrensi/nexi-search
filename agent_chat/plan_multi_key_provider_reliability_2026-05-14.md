# Plan: Multi-Key Provider Reliability
_Done looks like this: every credentialed provider instance accepts `api_key` as either a string or a list of strings, the runtime retries alternate keys inside the same provider before falling through to the next provider, and long-running processes can opt into process-local round robin without breaking existing configs._

---

# Checklist
- [x] Step 1: Create central API-key resolution module
- [x] Step 2: Add provider-level API-key strategy examples and config wording
- [x] Step 3: Route search providers through per-key execution
- [x] Step 4: Route fetch providers through per-key execution
- [x] Step 5: Route LLM providers through per-key execution
- [x] Step 6: Make every credentialed provider validator accept string-or-list API keys
- [x] Step 7: Add regression and strategy tests
- [x] Step 8: Run verification commands

---

## Context
- Canon for current behavior lives in `docs/product.md`, `docs/arch.md`, and `docs/provider-matrix.md`.
- The current runtime already has provider-instance failover in `nexi/backends/orchestrators.py`, but it does not have per-provider API-key fallback.
- `nexi/config.py` and `nexi/config_template.py` already preserve arbitrary provider fields under `providers[...]`, so adding one new optional provider field does not require a schema rewrite.
- `nexi/config_doctor.py` calls each provider adapter's `validate_config()` against the raw provider config from disk. Because of that, every credentialed provider validator must accept both `api_key = "value"` and `api_key = ["value1", "value2"]` before doctor can pass.
- The runtime methods inside these files currently assume a single resolved key string at execution time:
  - `nexi/backends/openai_compatible.py`
  - `nexi/backends/jina.py`
  - `nexi/backends/perplexity_search.py`
  - `nexi/backends/serper.py`
  - `nexi/backends/serpapi.py`
  - `nexi/backends/brave.py`
  - `nexi/backends/tavily.py`
  - `nexi/backends/linkup.py`
  - `nexi/backends/firecrawl.py`
  - `nexi/backends/exa.py`
- The least risky design for this pass is: keep provider runtime methods receiving a single resolved key string, and move all list handling, fallback ordering, and round-robin state into one central helper module plus the orchestrators.
- `nexi-search`, `nexi-fetch`, REPL mode, and MCP all create fresh processes or long-lived processes on top of the same orchestrators, so process-local state inside one central module is enough for v1.

## Prerequisites
- Work from `C:\Users\rx\001_Code\100_M\Nexi_Search`.
- Run `uv sync` before editing code. If `uv sync` fails, stop and record the full failure output.
- Do not use real provider credentials in tests. Keep all tests mocked.
- Keep `api_key` fully backward compatible. Existing configs with a single string must continue to work without modification.
- Keep round-robin state process-local only. Do not write key health or rotation state to disk.

## Scope Boundaries
- Do not edit `docs/product.md`, `docs/arch.md`, or `docs/provider-matrix.md` in this plan.
- Do not add persistent health tracking to `~/.config/nexi/`.
- Do not add adaptive scoring, latency weighting, or error-rate ranking in this pass.
- Do not change CLI flags or add new top-level config sections outside provider config fields.
- Do not change zero-key providers such as `special_trafilatura`, `special_playwright`, `markdown_new`, `snitchmd`, or `crawl4ai` except for import compatibility if needed.
- Do not change search loop logic in `nexi/search.py`, tool payload shapes in `nexi/tools.py`, citation behavior in `nexi/citations.py`, or history behavior in `nexi/history.py`.

---

## Steps

### Step 1: Create central API-key resolution module
Create a new module at `nexi/backends/api_keys.py`.

In `nexi/backends/api_keys.py`, implement only the API-key responsibilities listed below:
- a normalization helper that accepts a provider config dictionary and returns a cleaned list of non-empty key strings from `api_key`
- support `api_key` as either a single string or a list of strings
- preserve Jina's optional-key behavior by allowing an empty list when the provider config omits `api_key`
- a validation helper that raises `ValueError` with provider-specific messages when `api_key` is the wrong type or the list contains blank or non-string items
- a process-local round-robin state store keyed by provider instance name
- a helper that reads `api_key_strategy` from the provider config and supports exactly two values: `"fallback"` and `"round_robin"`
- default `api_key_strategy` to `"fallback"` when the field is missing
- a helper that returns an ordered list of per-attempt provider config copies where each returned config has a single resolved string in `api_key`

Use pure helpers where possible. Do not mutate the original provider config dictionary. Return copied dictionaries for per-attempt configs.

✅ Success: `uv run python -c "from nexi.backends.api_keys import build_api_key_attempt_configs"` imports successfully.
❌ If failed: delete `nexi/backends/api_keys.py`, remove any imports added for that module, and stop.

### Step 2: Add provider-level API-key strategy examples and config wording
Open `nexi/config_template.py`.

Make these changes only:
- In `PROVIDER_EXAMPLES`, keep every existing single-string `api_key` example valid.
- Add `api_key_strategy = "fallback"` to the credentialed provider examples that already define `api_key`.
- Add at least one visible example that shows list syntax for `api_key`, using an existing provider example block rather than inventing a new provider family.
- Add one top-of-file comment line near the existing provider-instance comments explaining that `api_key` may be either a string or a list of strings and that `api_key_strategy` supports `fallback` and `round_robin`.

Do not add new top-level config keys. Keep the change scoped to per-provider example fields and template comments.

✅ Success: `render_config_toml()` output contains the words `api_key_strategy`, `round_robin`, and at least one TOML list-form `api_key = [ ... ]` example.
❌ If failed: restore `nexi/config_template.py` to the pre-edit state and stop.

### Step 3: Route search providers through per-key execution
Open `nexi/backends/orchestrators.py`.

Refactor `run_search_chain(...)` so each provider instance can make multiple intra-provider attempts before the orchestrator moves to the next provider instance.

Implement the behavior exactly as follows:
- Keep the outer iteration over `config.search_backends` unchanged.
- For each `provider_name`, call the new helper from `nexi/backends/api_keys.py` to build the ordered list of per-key provider config copies.
- If the helper returns no alternate configs because the provider has no key requirement, run the provider once with the original config.
- For each per-key config copy, call `provider.validate_config(attempt_config)` before executing that attempt.
- For each per-key config copy, keep the existing same-provider query retry loop controlled by `config.search_provider_retries`.
- Preserve successful queries immediately.
- Move unresolved queries from one key attempt to the next key attempt inside the same provider instance.
- Move only still-unresolved queries to the next provider instance in `config.search_backends`.
- Expand `provider_failures` metadata so failures can name both `provider` and the key-attempt position without exposing the key value itself.
- Add a distinct failure kind for exhausted same-provider key attempts. Use a non-secret string such as `"api_key_exhausted"`.

Do not print or store real key values in verbose output or failure metadata.

✅ Success: a unit test can prove that one search provider instance with `api_key = ["bad", "good"]` resolves a failed query by retrying the same provider instance before the orchestrator moves to the next provider instance.
❌ If failed: restore `run_search_chain(...)` to provider-instance-only failover, keep `nexi/backends/api_keys.py` import-safe, and stop.

### Step 4: Route fetch providers through per-key execution
Open `nexi/backends/orchestrators.py` again.

Refactor `run_fetch_chain(...)` to match the same per-key execution rules used for search:
- build per-key provider config copies from `nexi/backends/api_keys.py`
- validate each per-key config copy before executing that attempt
- keep the existing same-provider URL retry loop controlled by `config.fetch_provider_retries`
- preserve successful pages immediately
- move unresolved URLs from one key attempt to the next key attempt inside the same provider instance
- move only still-unresolved URLs to the next provider instance in `config.fetch_backends`
- record non-secret failure metadata for exhausted key attempts

Keep zero-key fetch providers working exactly as they do today.

✅ Success: a unit test can prove that one fetch provider instance with `api_key = ["bad", "good"]` resolves a failed URL by retrying the same provider instance before the orchestrator moves to the next provider instance.
❌ If failed: restore `run_fetch_chain(...)` to provider-instance-only failover, keep other orchestrator functions importable, and stop.

### Step 5: Route LLM providers through per-key execution
Open `nexi/backends/orchestrators.py` again.

Refactor `run_llm_chain(...)` so one provider instance can retry alternate API keys before falling through to the next provider instance.

Implement the behavior exactly as follows:
- Keep the outer iteration over `config.llm_backends` unchanged.
- For each `provider_name`, build the ordered list of per-key provider config copies from `nexi/backends/api_keys.py`.
- For each per-key config copy, call `provider.validate_config(attempt_config)` before executing the completion attempt.
- Call `provider.complete(...)` once per per-key config copy.
- If one key attempt raises an exception, record a non-secret failure entry and continue to the next key attempt for the same provider instance.
- Only after all key attempts for that provider instance fail should `run_llm_chain(...)` continue to the next provider instance.
- Preserve the final `ProviderChainError` behavior when every configured provider instance and every configured key attempt fails.

Do not add same-key retry loops for LLMs. The only intra-provider LLM behavior in this pass is alternate-key fallback.

✅ Success: a unit test can prove that one LLM provider instance with `api_key = ["bad", "good"]` succeeds on the second key without calling the next provider instance.
❌ If failed: restore `run_llm_chain(...)` to provider-instance-only failover, keep `ProviderChainError` behavior intact, and stop.

### Step 6: Make every credentialed provider validator accept string-or-list API keys
Edit these files:
- `nexi/backends/openai_compatible.py`
- `nexi/backends/jina.py`
- `nexi/backends/perplexity_search.py`
- `nexi/backends/serper.py`
- `nexi/backends/serpapi.py`
- `nexi/backends/brave.py`
- `nexi/backends/tavily.py`
- `nexi/backends/linkup.py`
- `nexi/backends/firecrawl.py`
- `nexi/backends/exa.py`

In each credentialed provider class:
- replace direct `api_key` type checks inside `validate_config()` with the shared helper from `nexi/backends/api_keys.py`
- keep existing provider-specific required/optional behavior unchanged after normalization
- keep provider runtime methods using a single resolved string key at execution time

For `JinaSearchProvider` and `JinaFetchProvider`, preserve current optional-key semantics. For every other credentialed provider, keep `api_key` required.

Do not change request payload semantics, endpoint URLs, or HTTP header names in this step.

✅ Success: doctor-style validation passes when a provider config uses `api_key = ["k1", "k2"]`, and existing unit tests that use `api_key = "key"` still pass unchanged.
❌ If failed: restore the last working validator implementation in the specific provider file that failed and stop before changing more provider files.

### Step 7: Add regression and strategy tests
Edit `tests/test_config.py`, `tests/test_backends_orchestrators.py`, and `tests/test_backends_http_providers.py`. Add one new test file at `tests/test_backends_api_keys.py`.

Required coverage:
- `api_key` string form still validates for a credentialed provider
- `api_key` list form validates for a credentialed provider
- blank list items and non-string list items fail validation with `ValueError`
- omitted `api_key` still validates for Jina
- `api_key_strategy = "fallback"` keeps key order stable across repeated helper calls in the same process
- `api_key_strategy = "round_robin"` advances the starting key across repeated helper calls in the same process
- search orchestration retries the same provider instance with the second key before trying the next provider instance
- fetch orchestration retries the same provider instance with the second key before trying the next provider instance
- LLM orchestration retries the same provider instance with the second key before trying the next provider instance
- provider failure metadata never contains a raw API key value
- `Config.from_dict(...)` and `render_config_toml(...)` both preserve list-form `api_key` values without rewriting them to strings

Mock every network call. Do not hit real provider endpoints.

✅ Success: all new scenarios above have explicit tests, and no existing config or provider test loses backward-compatibility coverage.
❌ If failed: keep the failing tests in place, capture the exact failing command output, and stop before the verification step.

### Step 8: Run verification commands
Run these commands from `C:\Users\rx\001_Code\100_M\Nexi_Search` in this exact order:

1. `uv sync`
2. `uv run ruff check .`
3. `uv run pytest tests/test_backends_api_keys.py tests/test_backends_http_providers.py tests/test_backends_orchestrators.py tests/test_config.py`
4. `uv run pytest tests/`

Do not skip any command. Stop at the first failing command.

✅ Success: every command above exits with status code 0.
❌ If failed: do not continue to the next command. Record the failing command and the full output in the handoff note.

---

## Verification
- Every credentialed provider instance accepts `api_key` as either a string or a list of strings.
- Every existing single-string config still works without changes.
- Search, fetch, and LLM orchestration all retry alternate keys inside the same provider instance before moving to the next provider instance.
- `api_key_strategy = "fallback"` preserves original key order.
- `api_key_strategy = "round_robin"` changes the starting key across repeated calls in the same process.
- No failure metadata or verbose log path exposes real API key values.
- `nexi/config_doctor.py` accepts list-form API keys through provider `validate_config()`.

## Rollback
- If a critical step cannot be completed, run `git diff -- agent_chat/plan_multi_key_provider_reliability_2026-05-14.md` to keep the plan file, then revert only the implementation files changed for this plan with `git checkout -- nexi/backends/api_keys.py nexi/backends/orchestrators.py nexi/backends/openai_compatible.py nexi/backends/jina.py nexi/backends/perplexity_search.py nexi/backends/serper.py nexi/backends/serpapi.py nexi/backends/brave.py nexi/backends/tavily.py nexi/backends/linkup.py nexi/backends/firecrawl.py nexi/backends/exa.py nexi/config_template.py tests/test_backends_api_keys.py tests/test_backends_http_providers.py tests/test_backends_orchestrators.py tests/test_config.py`.
- If `git checkout -- ...` fails because a listed file does not exist yet, remove only the newly created file with a normal filesystem delete and stop.
