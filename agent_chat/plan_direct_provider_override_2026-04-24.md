# Plan: Direct provider override for search and fetch
_Done looks like this: `nexi-search --provider special_playwright` and `nexi-fetch --provider special_trafilatura` bypass the configured fallback chain, hit only the named provider, and fail fast if the provider does not match the command capability._

## Checklist
- [ ] Add `--provider` flag to `nexi-search` and `nexi-fetch`
- [ ] Route direct commands through a single provider when the flag is set
- [ ] Hard-fail on capability mismatch or missing provider name
- [ ] Keep normal chain behavior unchanged when the flag is absent
- [ ] Update docs/canon and usage examples
- [ ] Add tests for direct-provider success and mismatch failure
- [ ] Run targeted tests and fix regressions

## Notes
- This is a direct-command feature only; the agentic `nexi` loop stays unchanged.
- The override should still honor the provider’s own retries.
- The override should bypass fallback to any other provider in the chain.
