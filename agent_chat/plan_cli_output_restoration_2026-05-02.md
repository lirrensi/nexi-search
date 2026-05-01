# Plan: Restore CLI output while keeping runtime noise suppression
_Done means `nexi`, `nexi-search`, and `nexi-fetch` print normal user-facing output in non-verbose mode, while noisy runtime chatter remains suppressed._

---

# Checklist
- [x] Step 1: Replace broad stdio redirection with warning/unraisable-only suppression
- [x] Step 2: Add focused tests for runtime noise behavior and direct CLI output
- [x] Step 3: Run targeted pytest suite and confirm pass

---

## Context
- The current regression is in `nexi/runtime_noise.py`, where `suppress_runtime_chatter()` redirects both stdout and stderr to in-memory buffers in non-verbose mode.
- `nexi/cli.py`, `nexi/search_cli.py`, and `nexi/fetch_cli.py` all execute user-visible code inside `with suppress_runtime_chatter(verbose):`.
- This currently suppresses final answer and normal CLI output, making runs appear to do nothing.
- Existing failing evidence: `tests/test_cli_main.py::test_default_search_uses_search_path_when_dependencies_are_mocked` expects output text but receives empty output.

## Prerequisites
- Repository root is `C:\Users\rx\001_Code\100_M\Nexi_Search`.
- Python environment is managed by uv; commands must use `uv run ...`.
- No new dependencies are required.

## Scope Boundaries
- Do not modify provider implementations under `nexi/backends/`.
- Do not modify search loop behavior in `nexi/search.py`.
- Do not modify history persistence behavior in `nexi/history.py`.
- Do not modify docs in this plan.

---

## Steps

### Step 1: Replace broad stdio redirection with warning/unraisable-only suppression
Open `nexi/runtime_noise.py` and modify `suppress_runtime_chatter(verbose: bool)` so non-verbose mode suppresses Python warnings only and does not redirect `sys.stdout` or `sys.stderr`.

Required edits in `nexi/runtime_noise.py`:
1. Remove unused imports `io` and `contextlib.redirect_stdout`/`contextlib.redirect_stderr` usage.
2. Keep `warnings.catch_warnings()` and `warnings.simplefilter("ignore")` in non-verbose mode.
3. Ensure the context manager simply yields inside the warnings suppression block.
4. Preserve existing verbose-mode behavior (immediate yield, no suppression).
5. Keep `configure_runtime_noise(verbose)` behavior unchanged.

✅ Success: `suppress_runtime_chatter(False)` no longer captures normal `print()` output from CLI commands.
❌ If failed: Re-open `nexi/runtime_noise.py`, remove any remaining stdout/stderr redirection code paths, and re-run Step 3.

### Step 2: Add focused tests for runtime noise behavior and direct CLI output
Edit existing tests and add new tests.

Required edits:
1. In `tests/test_cli_main.py`, keep `test_default_search_uses_search_path_when_dependencies_are_mocked` assertions expecting visible output (`"Mock answer"`).
2. Add new test file `tests/test_runtime_noise.py` with tests that validate:
   - In non-verbose mode, `suppress_runtime_chatter(False)` does not suppress stdout output from a `print()` call.
   - In non-verbose mode, `warnings.warn(...)` output is suppressed inside the context manager.
   - In verbose mode, `suppress_runtime_chatter(True)` does not alter stdout behavior.
3. Add test coverage for direct CLI output in non-verbose mode by editing `tests/test_direct_cli.py`:
   - Add one `nexi-search` invocation test that patches config/readiness/orchestrator to return one result and asserts human-readable output is present.
   - Add one `nexi-fetch` invocation test that patches config/readiness/fetch function to return one page and asserts content output is present.

✅ Success: Test files assert that non-verbose CLI output is visible and warning suppression still works.
❌ If failed: Fix test patch targets to the exact imported symbols in each CLI module (`nexi.search_cli.*`, `nexi.fetch_cli.*`) and re-run Step 3.

### Step 3: Run targeted pytest suite and confirm pass
From repository root, run the following commands:
1. `uv run pytest tests/test_cli_main.py::test_default_search_uses_search_path_when_dependencies_are_mocked -q`
2. `uv run pytest tests/test_runtime_noise.py -q`
3. `uv run pytest tests/test_direct_cli.py -q`

If any test fails, fix code/tests and repeat Step 3 until all three commands pass.

✅ Success: All three commands return passing status with no failures.
❌ If failed: Do not proceed. Capture failing traceback details and apply minimal fixes limited to in-scope files, then re-run Step 3.

---

## Verification
- Confirm the previously failing CLI test now passes and includes visible output.
- Confirm runtime noise tests prove warning suppression still works without hiding stdout.
- Confirm direct CLI tests prove `nexi-search` and `nexi-fetch` still print user-facing payloads in non-verbose mode.

## Rollback
If the change introduces unexpected regressions:
1. Revert only modified files from this plan using git restore:
   - `git restore nexi/runtime_noise.py tests/test_runtime_noise.py tests/test_direct_cli.py tests/test_cli_main.py`
2. Re-run:
   - `uv run pytest tests/test_cli_main.py::test_default_search_uses_search_path_when_dependencies_are_mocked -q`
3. Report rollback status and failure details.
