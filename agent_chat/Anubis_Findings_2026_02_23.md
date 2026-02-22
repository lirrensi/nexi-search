# NEXI Code Review — Page Fetching & Parsing Edge Cases

**Date**: 2026-02-23  
**Scope**: HTTP client handling, page fetching, content parsing, chunking, error paths

---

## CRITICAL

### CRITICAL — HTTP Client Never Gets Timeout Update

**Location**: `nexi/tools.py:243-256` (`get_http_client`)  
**Problem**: The HTTP client is created once and cached globally. The `timeout` parameter is only used during initial creation. If different calls pass different timeouts, all subsequent calls use the first timeout value.  
**Impact**: Long-running searches with different timeout needs will silently use the wrong timeout. A 30s timeout set on first call persists even if later calls request 60s.  
**Fix**: Either remove timeout from client-level and use per-request timeouts, or invalidate/recreate the client when timeout changes.

---

### CRITICAL — JSON Parse Failure in Tool Arguments Causes Silent Data Loss

**Location**: `nexi/search.py:338` and `nexi/compaction.py:83-86`  
**Problem**: `json.loads(tool_call.function.arguments)` will raise `JSONDecodeError` on malformed LLM output. In `search.py` this crashes the iteration. In `compaction.py` it silently continues without extracting metadata.  
**Impact**: Malformed LLM tool call arguments (common with cheaper models) crash the search loop entirely. No retry, no fallback.  
**Fix**: Wrap in try/except with specific fallback behavior — re-prompt LLM or use empty dict default.

---

## HIGH

### HIGH — Search Result Parsing Fragile on Format Variations

**Location**: `nexi/tools.py:474-518` (text format parsing)  
**Problem**: The parser assumes rigid Jina text format: `[N] Title:`, `[N] URL Source:`, `[N] Description:`. Any deviation — extra whitespace, missing colon, different marker format — produces empty results.  
**Impact**: Valid search results are silently dropped. No error is raised, no fallback parsing. User gets "no results found" when results exist.  
**Fix**: Add validation logging, try multiple regex patterns, or fall back to raw text when structured parsing fails.

---

### HIGH — Empty Content Pages Cached Indefinitely

**Location**: `nexi/tools.py:204` (cache write) and `nexi/tools.py:177-179` (cache read)  
**Problem**: Empty or error content is cached. Once a URL fails, it's cached with empty content. Subsequent requests return the cached empty string without re-fetching.  
**Impact**: Transient failures become permanent. A 503 error on first attempt means the URL is "poisoned" for the entire search session.  
**Fix**: Only cache successful fetches with non-empty content. Exclude error results from cache.

---

### HIGH — Chunk Selection Fallback Returns Wrong Data

**Location**: `nexi/tools.py:873-881` (`_select_chunks_with_llm`)  
**Problem**: On LLM failure, the fallback returns the **first 5 chunks** regardless of query relevance. This silently injects irrelevant content.  
**Impact**: When chunk selection API fails, users get completely irrelevant content masquerading as "relevant chunks." No indication that fallback occurred.  
**Fix**: Return empty string with error marker, or return raw content with truncation flag, or log fallback prominently.

---

### HIGH — LLM Extraction Truncates to 2000 Chars on Error

**Location**: `nexi/tools.py:935-939` (`_extract_with_llm`)  
**Problem**: On any exception, returns `content[:2000]`. No indication to caller that truncation/fallback occurred. The 2000 char limit is arbitrary and may cut mid-sentence.  
**Impact**: Failed extraction silently returns truncated garbage. User has no idea the LLM failed. Content may be cut at critical points.  
**Fix**: Return full content with an error prefix, or pass error info back to caller, or retry before truncating.

---

## MEDIUM

### MEDIUM — URL Cache Not Cleared Between Multi-Turn Conversations

**Location**: `nexi/search.py:136` (`clear_url_cache`)  
**Problem**: Cache is cleared at start of `run_search()` but multi-turn REPL sessions call `run_search()` for each query, clearing the cache between turns. URLs fetched in previous turns are re-fetched.  
**Impact**: Unnecessary API calls in multi-turn sessions. Wastes time and API quota.  
**Fix**: Move cache clearing to REPL initialization, not per-search. Or use TTL-based cache invalidation.

---

### MEDIUM — Citation Number Parsing Regex Over-Matches

**Location**: `nexi/tools.py:826-828` (chunk number extraction)  
**Problem**: Regex `r"\d+"` extracts ALL numbers from LLM response. If LLM says "I found 3 relevant chunks: 1, 4, and 7", the code extracts `[3, 1, 4, 7]` — including the "3" which is not a chunk number.  
**Impact**: Wrong chunks selected. The word count in LLM response becomes a fake chunk number.  
**Fix**: Use more specific regex that matches only standalone numbers or numbers prefixed with "chunk".

---

### MEDIUM — Sources List Appended to Every Page in Batch

**Location**: `nexi/search.py:224-236` (`update_tool_result_with_sources`)  
**Problem**: Sources list is appended to **every page's content** in a batch web_get result. If 8 URLs are fetched, the sources list appears 8 times in the conversation.  
**Impact**: Token bloat. Same sources list duplicated across all pages, wasting context window.  
**Fix**: Append sources list once, to the last page, or as a separate message.

---

### MEDIUM — Heading Regex Splits on Inline Headings

**Location**: `nexi/tools.py:51` (`create_logical_chunks`)  
**Problem**: Regex `r"(^#{1,6}\s+.+$)"` with `MULTILINE` flag matches headings anywhere they appear at line start. But it captures the heading **without** the following content, then tries to recombine.  
**Impact**: Pages with many headings produce fragmented chunks. Heading text may be separated from its body.  
**Fix**: Use lookahead/lookbehind or capture heading with following paragraphs in one pass.

---

### MEDIUM — HTTP Client Never Closed on Normal Exit

**Location**: `nexi/tools.py:259-264` (`close_http_client`)  
**Problem**: `close_http_client()` exists but is never called in the normal execution path. The `run_search_sync()` function creates a loop and runs search, but never calls `close_http_client()`.  
**Impact**: Resource leak. HTTP connections left open. On Windows, this can cause "address in use" errors on rapid successive searches.  
**Fix**: Call `close_http_client()` in `run_search_sync()` finally block or in CLI exit handler.

---

## LOW

### LOW — Duplicate Code in Manual Chunk Test

**Location**: `tests/manual_chunk_test.py:135-170`  
**Problem**: The `create_logical_chunks` function body is duplicated (lines 51-133, then again 135-170). The second copy is unreachable but exists in source.  
**Impact**: Maintenance burden. Confusion about which version is used.  
**Fix**: Remove dead code.

---

### LOW — Verbose Print Lacks Context for Chunk Selection

**Location**: `nexi/tools.py:847-850`  
**Problem**: When rejecting out-of-range chunk numbers, verbose output says "Ignoring out-of-range numbers" but doesn't show what the LLM actually responded.  
**Impact**: Debugging chunk selection issues requires more context than provided.  
**Fix**: Include full LLM response in verbose output.

---

## Coverage

- **Analyzed**: HTTP client lifecycle, cache handling, error paths, JSON parsing, chunking logic, LLM extraction fallbacks, citation handling, source list generation
- **Not analyzed**: Actual network behavior under load, Jina API rate limiting handling, Windows-specific encoding issues beyond the existing fixes
- **Confidence**: High — the code is well-structured and issues are clear from static analysis
