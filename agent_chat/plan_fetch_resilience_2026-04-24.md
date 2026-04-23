# Plan: Resilient direct fetch providers and token-capped output
_Done looks like this: `nexi-fetch` and `nexi_fetch` prefer `special_trafilatura` and `special_playwright`, never hard-fail on a single bad provider, and truncate oversized direct-fetch output to 8000 tokens while saving the full page to a temp file._

---

# Checklist
- [x] Step 1: Update canon docs for the new fetch flow
- [x] Step 2: Add/adjust config defaults and provider registry entries
- [x] Step 3: Implement `special_trafilatura` and `special_playwright`
- [x] Step 4: Add direct-fetch token truncation + temp-file spillover
- [x] Step 5: Keep agent-mode `web_get` untouched except for shared plumbing
- [x] Step 6: Update tests for provider chain and truncation behavior
- [x] Step 7: Run verification and fix regressions

---

## Scope
- Add two new fetch provider families:
  - `special_trafilatura`
  - `special_playwright`
- Make the direct fetch surfaces use a stronger default chain that keeps fallback behavior intact.
- For direct fetch only, cap emitted content at 8000 tokens and spill the full page to a temp file per page.
- Preserve current agent-tool behavior unless shared helpers need a tiny refactor.

## Notes
- The new providers should be registered by `type` and exposed in `docs/provider-matrix.md`.
- `special_trafilatura` should try HTTPX first, then Trafilatura extraction, with fallback content handling.
- `special_playwright` should use headed Playwright, extract page value (not raw HTML), and remain non-intrusive.
- Direct fetch output should include the absolute temp-file path for any truncated page.
