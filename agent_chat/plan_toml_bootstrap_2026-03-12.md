# Plan: TOML Bootstrap and Config Commands
_Done means NEXI creates a commented TOML config template on first run, exits cleanly when config is missing or unusable, and exposes `config`, `init`, `onboard`, `doctor`, and `clean` without breaking `nexi`, `nexi-search`, `nexi-fetch`, or MCP._

---

# Checklist
- [x] Step 1: Add the TOML template renderer and move config/history paths to `~/.config/nexi`
- [x] Step 2: Replace JSON config loading, saving, and bootstrap behavior in `nexi/config.py`
- [x] Step 3: Add command-readiness and doctor helpers for `nexi`, `nexi-search`, and `nexi-fetch`
- [x] Step 4: Convert `nexi/cli.py` into a group with `config`, `init`, `onboard`, `doctor`, and `clean`
- [x] Step 5: Implement the small onboarding wizard that writes back through the TOML renderer
- [x] Step 6: Wire `nexi-search`, `nexi-fetch`, `nexi/mcp_server.py`, and history to the new config lifecycle
- [x] Step 7: Rewrite and add automated tests for TOML bootstrap, subcommands, readiness, and paths
- [x] Step 8: Run lint, run the targeted test suite, and capture the final verification output

---

## Context

- Canon is already updated in `docs/product.md`, `docs/arch.md`, and `docs/provider-matrix.md`.
- Current code still uses JSON config and the old path in `nexi/config.py:17-18` and `nexi/history.py:13`.
- Current bootstrap still auto-runs `run_first_time_setup()` from `nexi/config.py:511-655`.
- Current `nexi` CLI still uses `--config` and `--edit-config` in `nexi/cli.py:67-109`.
- `nexi-search` and `nexi-fetch` currently call `ensure_config()` directly in `nexi/search_cli.py:61-72` and `nexi/fetch_cli.py:41-68`.
- `nexi/mcp_server.py:50-102` currently assumes config loading either works or returns a generic error string.
- `nexi/backends/custom_python.py:29-44` derives custom provider file paths from `CONFIG_DIR`, so moving `CONFIG_DIR` changes custom-provider location automatically.

## Prerequisites

- Run `git status --short` from repo root before editing. Current worktree is already dirty. Preserve every pre-existing change.
- Do not overwrite local changes already present in `nexi/config.py`, `nexi/backends/registry.py`, `nexi/backends/custom_python.py`, `tests/test_config.py`, `tests/test_backends_custom_python.py`, `README.md`, `docs/arch.md`, `docs/product.md`, `docs/MCP_SERVER.md`, `AGENTS.md`, and `skills/nexi-search/references/installation.md`.
- Use Python 3.12 stdlib `tomllib` for TOML parsing. Do not add a YAML parser. Do not reintroduce JSONC.
- Keep all file I/O UTF-8 encoded.
- If any targeted file has new user edits beyond the list above by the time execution starts, stop and report instead of guessing how to merge them.

## Scope Boundaries

- Do not change provider adapter behavior in `nexi/backends/brave.py`, `nexi/backends/exa.py`, `nexi/backends/firecrawl.py`, `nexi/backends/jina.py`, `nexi/backends/linkup.py`, `nexi/backends/markdown_new.py`, `nexi/backends/perplexity_search.py`, `nexi/backends/serpapi.py`, `nexi/backends/serper.py`, or `nexi/backends/tavily.py`.
- Do not change the search loop in `nexi/search.py` or tool execution in `nexi/tools.py` except for config/readiness call sites required by this plan.
- Do not edit documentation files during execution. Docs are already the canon for this change.
- Do not add legacy JSON migration behavior. This change ships the new TOML model only.
- Do not add YAML support, JSONC support, streaming output, or in-progress history persistence.

---

## Steps

### Step 1: Add the TOML template renderer and move config/history paths to `~/.config/nexi`

Create a new module `nexi/config_template.py`.

In `nexi/config_template.py`, define the canonical rendered template for the exact provider examples documented in `docs/provider-matrix.md`:

- active-by-default fetch instances: `crawl4ai_local`, `special_trafilatura`, `special_playwright`, `markdown_new`
- commented LLM examples: `openrouter`, `openai`, `local_openai`, `custom_llm`
- commented search examples: `jina`, `tavily`, `exa`, `firecrawl`, `linkup`, `brave`, `serpapi`, `serper`, `perplexity`, `custom_search`
- commented fetch examples: `jina`, `tavily`, `exa`, `firecrawl`, `linkup`, `special_trafilatura`, `special_playwright`, `custom_fetch`

Implement these functions in `nexi/config_template.py`:

1. `render_config_toml(active_config: dict[str, Any] | None = None) -> str`
2. `write_default_template(config_path: Path, force: bool = False) -> bool`

`render_config_toml()` must emit a complete commented TOML file with:

- `llm_backends = []`
- `search_backends = []`
- `fetch_backends = ["crawl4ai_local", "special_trafilatura", "special_playwright", "markdown_new"]`
- scalar defaults from `DEFAULT_CONFIG`
- uncommented provider tables for every provider present in `active_config["providers"]`
- commented example tables for inactive shipped provider examples
- comments that tell the user to activate one LLM provider and one search provider before running `nexi`

Then open `nexi/config.py` and change:

- `CONFIG_DIR` to `Path.home() / ".config" / "nexi"`
- `CONFIG_FILE` to `CONFIG_DIR / "config.toml"`

Do not change `nexi/history.py` in this step; `nexi/history.py` already imports `CONFIG_DIR`.

✅ Success: `nexi/config_template.py` exists; `render_config_toml()` returns TOML text that contains `[providers.markdown_new]`, `[providers.crawl4ai_local]`, `# [providers.openrouter]`, and `# [providers.jina]`; `nexi/config.py` points at `~/.config/nexi/config.toml`.
❌ If failed: delete `nexi/config_template.py`, restore only the edits made in `nexi/config.py` during this step, and stop. Do not continue with a partial renderer.

### Step 2: Replace JSON config loading, saving, and bootstrap behavior in `nexi/config.py`

Open `nexi/config.py` and make these exact changes:

1. Remove `import json` and add `import tomllib`.
2. Remove `run_first_time_setup()` entirely.
3. Remove `_is_legacy_config()` and `_migrate_legacy_config()` entirely.
4. Change `DEFAULT_CONFIG` so the runtime defaults match the new spec:
   - `llm_backends = []`
   - `search_backends = []`
   - `fetch_backends = ["crawl4ai_local", "special_trafilatura", "special_playwright", "markdown_new"]`
   - keep scalar defaults unchanged
   - keep provider-specific default values only for active fetch providers and any metadata still needed for onboarding
5. Rewrite `_normalize_config_data()` so it fills missing scalar defaults but does not inject inactive provider tables into parsed configs.
6. Rewrite `validate_config()` so `llm_backends`, `search_backends`, and `fetch_backends` only need to be lists of non-empty strings; do not require non-empty chains in `validate_config()`.
7. Add a custom exception class named `ConfigCreatedError` with a `config_path: Path` attribute and a fixed user-facing message stating that the template was created and must be filled in.
8. Rewrite `load_config()` to parse TOML from `CONFIG_FILE` with `tomllib.load()`.
9. Rewrite `save_config()` to call `render_config_toml(config.to_dict())` from `nexi/config_template.py` and write UTF-8 text to disk.
10. Rewrite `ensure_config()` so missing config calls `write_default_template(CONFIG_FILE, force=False)` and raises `ConfigCreatedError` instead of returning a `Config`.
11. Keep `Config.to_dict()` and `Config.from_dict()`.
12. Export `ConfigCreatedError` in `__all__`.

Make TOML parse failures raise `ValueError` with the original parser message included. Do not fall back to onboarding on parse failure.

✅ Success: `load_config()` reads TOML, `save_config()` writes TOML, `ensure_config()` raises `ConfigCreatedError` when `config.toml` is missing, and no code path in `nexi/config.py` writes JSON anymore.
❌ If failed: undo only the edits made in `nexi/config.py` during this step and stop. Do not leave `ensure_config()` half-switched between JSON and TOML.

### Step 3: Add command-readiness and doctor helpers for `nexi`, `nexi-search`, and `nexi-fetch`

Create a new module `nexi/config_doctor.py`.

In `nexi/config_doctor.py`, implement pure helpers that do not print:

1. `check_command_readiness(config: Config, command_name: str) -> list[str]`
2. `build_doctor_report(config: Config) -> dict[str, list[str]]`

`check_command_readiness()` must behave exactly like this:

- for `command_name == "nexi"`: require at least one usable LLM provider and at least one usable search provider
- for `command_name == "nexi-search"`: require at least one usable search provider
- for `command_name == "nexi-fetch"`: require at least one usable fetch provider

For each active provider in the relevant chain, resolve the provider class through `resolve_llm_provider()`, `resolve_search_provider()`, or `resolve_fetch_provider()` from `nexi/backends/registry.py`, instantiate the provider, and call `validate_config()`.

Return friendly, user-facing errors. Use exact messages that point at the fix, for example:

- `Activate at least one LLM provider in llm_backends`
- `Activate at least one search provider in search_backends`
- `Active provider 'jina' is not ready: Jina api_key must be a string`

`build_doctor_report()` must return a mapping for all three commands: `nexi`, `nexi-search`, and `nexi-fetch`.

✅ Success: `nexi/config_doctor.py` exists; calling `build_doctor_report()` on a template config returns readiness errors for `nexi` and `nexi-search` but not for `nexi-fetch`.
❌ If failed: delete `nexi/config_doctor.py` and stop. Do not continue without a dedicated readiness layer.

### Step 4: Convert `nexi/cli.py` into a group with `config`, `init`, `onboard`, `doctor`, and `clean`

Open `nexi/cli.py` and replace the single-command structure with `@click.group(invoke_without_command=True)`.

Keep the current default search/history behavior in the group callback when `ctx.invoked_subcommand is None`.

Make these exact CLI changes:

1. Remove the `--config` and `--edit-config` options.
2. Keep `-q/--query-text`, `--last`, `--prev`, `--show`, and `--clear-history` in the group callback.
3. Keep positional `query` support for normal searches.
4. Leave `-q/--query-text` in place so users can still search for strings like `config` or `doctor`.
5. Replace the hardcoded version string in `@click.version_option` with `importlib.metadata.version("nexi-search")`.

Add these subcommands to `nexi/cli.py`:

- `config`: if `CONFIG_FILE` is missing, call `write_default_template(CONFIG_FILE, force=False)` first; then open the config file in `$EDITOR`, defaulting to `notepad` on Windows and `nano` elsewhere
- `init`: create the default template only when the file is missing; print whether the template was created or already existed
- `doctor`: load the current config, run `build_doctor_report()`, and print one line per command (`PASS` or `FAIL` plus errors)
- `clean`: delete `CONFIG_FILE` and `get_history_path()` if present, recreate the template with `write_default_template(CONFIG_FILE, force=True)`, and print the recreated path
- `onboard`: delegate to the onboarding helper created in Step 5

Update `_run_search_command()` and `_interactive_mode()` so they catch `ConfigCreatedError`, print the config path plus the bootstrap warning, and exit with code `1` without opening any wizard automatically.

✅ Success: `nexi config`, `nexi init`, `nexi doctor`, `nexi clean`, and `nexi onboard` all appear in `nexi --help`; `nexi --prev` still works; `nexi "query"` still routes to the search path.
❌ If failed: restore only the edits made in `nexi/cli.py` during this step and stop. Do not continue with a partially converted Click entrypoint.

### Step 5: Implement the small onboarding wizard that writes back through the TOML renderer

Create a new module `nexi/onboard.py`.

In `nexi/onboard.py`, implement `run_onboarding() -> None` with `questionary`.

`run_onboarding()` must do exactly this:

1. If `CONFIG_FILE` is missing, call `write_default_template(CONFIG_FILE, force=False)`.
2. Load the current config with `load_config()`.
3. Ask the user to choose one LLM setup from `openrouter`, `openai`, `local_openai`, or `custom_llm`.
4. Ask only for the fields required by the chosen LLM setup:
   - `openrouter`: prompt for `api_key` and `model`; hardcode `base_url = "https://openrouter.ai/api/v1"`
   - `openai`: prompt for `api_key` and `model`; hardcode `base_url = "https://api.openai.com/v1"`
   - `local_openai`: prompt for `base_url`, `api_key`, and `model`
   - `custom_llm`: prompt for `provider_name` and `provider_type`
5. Ask the user to choose one search setup from `jina`, `tavily`, `exa`, `firecrawl`, `linkup`, `brave`, `serpapi`, `serper`, `perplexity`, or `custom_search`.
6. Ask only for the fields required by the chosen search setup.
7. Ask whether to keep the default fetch chain `["crawl4ai_local", "special_trafilatura", "special_playwright", "markdown_new"]`.
8. If the user answers no to the previous question, show a checkbox with `crawl4ai_local`, `special_trafilatura`, `special_playwright`, `markdown_new`, `jina`, `tavily`, `exa`, `firecrawl`, `linkup`, and `custom_fetch`, then ask only for the required fields of any newly selected fetch providers.
9. Write the resulting config with `save_config()`.

Do not ask about `default_effort`, `time_target`, compaction, retries, or other advanced settings.

If any `questionary` prompt returns `None`, print `Onboarding cancelled` and return without writing partial changes.

✅ Success: running `run_onboarding()` on the template file activates exactly one LLM chain entry, exactly one search chain entry, preserves commented provider examples in the saved TOML, and leaves advanced scalars unchanged.
❌ If failed: delete `nexi/onboard.py`, revert the onboarding hook in `nexi/cli.py`, and stop. Do not continue with a wizard that rewrites the config without preserving the template shape.

### Step 6: Wire `nexi-search`, `nexi-fetch`, `nexi/mcp_server.py`, and history to the new config lifecycle

Open `nexi/search_cli.py`, `nexi/fetch_cli.py`, `nexi/mcp_server.py`, and `nexi/history.py`.

Make these exact changes:

1. In `nexi/history.py`, keep the JSONL format but confirm `HISTORY_FILE = CONFIG_DIR / "history.jsonl"` now resolves into `~/.config/nexi/history.jsonl` through the new `CONFIG_DIR`.
2. In `nexi/search_cli.py`, catch `ConfigCreatedError` from `ensure_config()` and raise `click.ClickException` with the bootstrap message instead of printing a traceback.
3. In `nexi/search_cli.py`, call `check_command_readiness(config, "nexi-search")` before `run_search_chain()` and abort with `click.ClickException` when the list is non-empty.
4. In `nexi/fetch_cli.py`, catch `ConfigCreatedError` from `ensure_config()` and raise `click.ClickException` with the bootstrap message.
5. In `nexi/fetch_cli.py`, call `check_command_readiness(config, "nexi-fetch")` before `web_get()` and abort with `click.ClickException` when the list is non-empty.
6. In `nexi/mcp_server.py`, catch `ConfigCreatedError` and return a user-facing error string that includes `~/.config/nexi/config.toml`.
7. In `nexi/mcp_server.py`, run `check_command_readiness(config, "nexi")` and return a user-facing error string when readiness fails.
8. Do not add in-progress history persistence. `nexi --prev` must continue to mean latest completed run only.

✅ Success: `nexi-search`, `nexi-fetch`, and MCP all fail cleanly when config is missing or not ready, and `nexi/history.py` still stores completed results in JSONL under `~/.config/nexi/history.jsonl`.
❌ If failed: restore only the edits made in `nexi/search_cli.py`, `nexi/fetch_cli.py`, `nexi/mcp_server.py`, and `nexi/history.py` during this step and stop.

### Step 7: Rewrite and add automated tests for TOML bootstrap, subcommands, readiness, and paths

Edit `tests/test_config.py`, `tests/test_direct_cli.py`, `tests/test_mcp_server.py`, and `tests/test_history.py`. Create `tests/test_cli_main.py`.

Make these exact test changes:

1. Replace JSON fixture files in `tests/test_config.py` with TOML fixture files.
2. In `tests/test_config.py`, add tests for:
   - `write_default_template()` writing `config.toml`
   - `load_config()` parsing TOML
   - `validate_config()` allowing empty `llm_backends` and `search_backends`
   - `ensure_config()` raising `ConfigCreatedError` after writing the template
   - `save_config()` writing commented examples such as `# [providers.openrouter]`
   - custom provider file lookup under the new `CONFIG_DIR`
3. In `tests/test_cli_main.py`, add tests for:
   - missing-config search bootstraps `config.toml` and exits non-zero
   - `nexi init` creates the file once and leaves an existing file untouched
   - `nexi doctor` reports readiness failures on the template
   - `nexi clean` deletes history and rewrites the template
   - `nexi config` opens the file path through a mocked editor command
   - default search still works when `ensure_config()` and `run_search_sync()` are mocked
4. In `tests/test_direct_cli.py`, add missing-config and readiness-failure tests for both `nexi-search` and `nexi-fetch`.
5. In `tests/test_mcp_server.py`, add tests for missing config and readiness failure messages.
6. In `tests/test_history.py`, assert that `get_history_path()` points to `.config/nexi/history.jsonl` when `CONFIG_DIR` is monkeypatched.
7. Delete or rewrite any assertion that still expects `config.json`, JSON disk serialization, or `run_first_time_setup()`.

✅ Success: all updated tests compile, there are no remaining test assertions for `config.json`, and `tests/test_cli_main.py` exists.
❌ If failed: restore only the tests edited in this step and stop. Do not proceed to the final verification with a half-updated test suite.

### Step 8: Run lint, run the targeted test suite, and capture the final verification output

From repo root, run these commands in order:

1. `ruff check nexi tests`
2. `pytest tests/test_config.py tests/test_cli_main.py tests/test_direct_cli.py tests/test_history.py tests/test_mcp_server.py tests/test_backends_custom_python.py`

If either command fails, stop immediately. Do not edit unrelated modules to force a green run.

After both commands pass, run `git diff --stat` and save the output in the execution report so Ma'at can review the final footprint.

✅ Success: `ruff check nexi tests` exits `0`; the targeted `pytest` command exits `0`; `git diff --stat` shows only the files touched by this plan.
❌ If failed: copy the full failing command output into the execution report, leave the failing files in place for review, and stop. Do not run `git restore` on dirty files that existed before this plan.

---

## Verification

- `nexi/config.py` reads TOML, writes TOML, and raises `ConfigCreatedError` instead of starting a wizard.
- `nexi/config_template.py` renders a commented template with shipped provider examples visible.
- `nexi/cli.py` exposes `config`, `init`, `onboard`, `doctor`, and `clean` while preserving `nexi "query"`, `nexi --prev`, `nexi --last`, and `nexi --show`.
- `nexi/onboard.py` activates one LLM provider and one search provider without asking advanced config questions.
- `nexi-search`, `nexi-fetch`, and `nexi/mcp_server.py` all fail cleanly when config is missing or not ready.
- `nexi/history.py` still stores only completed runs in `~/.config/nexi/history.jsonl`.
- `ruff check nexi tests` passes.
- `pytest tests/test_config.py tests/test_cli_main.py tests/test_direct_cli.py tests/test_history.py tests/test_mcp_server.py tests/test_backends_custom_python.py` passes.

## Rollback

The worktree already contains pre-existing local edits in files this plan touches. Do not use `git restore .`, `git reset --hard`, or any destructive rollback command.

If execution becomes unrecoverable:

1. Run `git diff -- nexi/config.py nexi/cli.py nexi/search_cli.py nexi/fetch_cli.py nexi/mcp_server.py nexi/history.py tests/test_config.py tests/test_direct_cli.py tests/test_history.py tests/test_mcp_server.py tests/test_cli_main.py > agent_chat/toml_bootstrap_failed_changes.patch`
2. Delete only new files created by this plan if they are clearly isolated:
   - `nexi/config_template.py`
   - `nexi/config_doctor.py`
   - `nexi/onboard.py`
   - `tests/test_cli_main.py`
3. Stop and report that rollback on pre-existing dirty files must be handled manually.

Plan complete. Handing off to Executor.
