"""Tool implementations for NEXI search loop."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from openai import AsyncOpenAI

from nexi.config import EXTRACTOR_PROMPT_TEMPLATE

# In-memory cache for fetched URLs: {url: raw_content}
_url_cache: dict[str, str] = {}


def clear_url_cache() -> None:
    """Clear the URL cache (call at start of each search session)."""
    global _url_cache
    _url_cache.clear()


__all__ = [
    "clear_url_cache",
    "execute_tool",
    "web_search",
    "web_get",
]


# Tool schemas for LLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Jina AI. Supports multiple parallel queries for efficient research. Returns snippets and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search queries to execute in parallel",
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_get",
            "description": "Fetch and process content from URLs using Jina Reader. Automatically converts PDFs, HTML to clean markdown. Supports multiple URLs in parallel. When get_full is false (default), content is processed via LLM summarizer using the provided instructions to extract/summarize specific information. When get_full is true, returns the raw page content without any filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"},
                        "description": "List of URLs to fetch content from",
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Custom prompt guiding the LLM summarizer on what to look for. Make instructions explicit, self-contained, and dense—general for broad overviews or specific for targeted details. This helps chain crawls: If the summary lists next URLs, you can browse those next. Always keep requests focused to avoid vague outputs. Only used when get_full is false.",
                    },
                    "get_full": {
                        "type": "boolean",
                        "description": "If true, returns the full page content without LLM processing. If false (default), content is summarized/extracted based on instructions.",
                        "default": False,
                    },
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer to the user's query. This terminates the search loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer to the user's query, formatted in markdown",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


async def web_search(
    queries: list[str], jina_key: str, verbose: bool = False
) -> dict[str, Any]:
    """Search the web using Jina AI.

    Args:
        queries: List of search queries (1-5)
        jina_key: Jina AI API key
        verbose: Show detailed logs

    Returns:
        Dictionary with search results
    """
    results = []

    if verbose:
        print(f"[Jina Search] Starting {len(queries)} parallel searches...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create tasks for parallel execution
        tasks = []
        for query in queries:
            task = _search_single(client, query, jina_key, verbose)
            tasks.append(task)

        # Execute all searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        for query, result in zip(queries, search_results):
            if isinstance(result, Exception):
                error_msg = str(result)
                if verbose:
                    print(f"  [Jina Search] ❌ Exception: {error_msg}")
                results.append(
                    {
                        "query": query,
                        "error": error_msg,
                        "results": [],
                    }
                )
            else:
                results.append(result)

    if verbose:
        print(f"[Jina Search] Completed {len(results)} searches")

    return {"searches": results}


async def _search_single(
    client: httpx.AsyncClient, query: str, jina_key: str, verbose: bool = False
) -> dict[str, Any]:
    """Execute a single search query."""
    headers = {}
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    if verbose:
        print(f"  [Jina Search] Query: {query}")
        print(f"  [Jina Search] URL: https://s.jina.ai/?q={query}")
        print(f"  [Jina Search] Headers: {headers}")

    response = None
    try:
        response = await client.get(
            "https://s.jina.ai/",
            params={"q": query},
            headers=headers,
        )
        response.raise_for_status()

        # Parse response - Jina returns text format, not JSON
        raw_text = response.text
        if verbose:
            print(f"  [Jina Search] Status: {response.status_code}")
            print(f"  [Jina Search] Raw response (first 500 chars): {raw_text[:500]}")

        # Try to parse as JSON first (for backward compatibility)
        try:
            data = response.json()
            if verbose:
                print(f"  [Jina Search] Parsed as JSON")
            return {
                "query": query,
                "results": data if isinstance(data, list) else [data],
            }
        except Exception:
            # If JSON parsing fails, parse the text format
            if verbose:
                print(f"  [Jina Search] Parsing as text format")

            # Parse text format: [1] Title: ... [1] URL Source: ... [1] Description: ...
            results = []
            lines = raw_text.split("\n")
            current_result = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for new result marker
                if line.startswith("[") and "] Title:" in line:
                    # Save previous result if exists
                    if current_result:
                        results.append(current_result)
                    # Start new result
                    current_result = {}
                    # Extract title
                    title_part = line.split("] Title: ", 1)
                    if len(title_part) > 1:
                        current_result["title"] = title_part[1].strip()
                elif "] URL Source:" in line:
                    url_part = line.split("] URL Source: ", 1)
                    if len(url_part) > 1:
                        current_result["url"] = url_part[1].strip()
                elif "] Description:" in line:
                    desc_part = line.split("] Description: ", 1)
                    if len(desc_part) > 1:
                        current_result["description"] = desc_part[1].strip()
                elif "] Published Time:" in line:
                    time_part = line.split("] Published Time: ", 1)
                    if len(time_part) > 1:
                        current_result["published_time"] = time_part[1].strip()

            # Add last result
            if current_result:
                results.append(current_result)

            if verbose:
                print(f"  [Jina Search] Parsed {len(results)} results from text format")

            return {
                "query": query,
                "results": results,
            }

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
        if verbose:
            print(f"  [Jina Search] ❌ ERROR: {error_msg}")
        return {
            "query": query,
            "error": error_msg,
            "results": [],
        }
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"  [Jina Search] ❌ ERROR: {error_msg}")
            # Try to show raw response if available
            if response:
                try:
                    print(f"  [Jina Search] Raw response: {response.text[:500]}")
                except:
                    pass
        return {
            "query": query,
            "error": error_msg,
            "results": [],
        }


async def web_get(
    urls: list[str],
    jina_key: str,
    verbose: bool = False,
    instructions: str = "",
    get_full: bool = False,
    llm_client: AsyncOpenAI | None = None,
    llm_model: str | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Fetch and extract content from URLs using Jina Reader + LLM summarizer.

    Args:
        urls: List of URLs to fetch (1-8)
        jina_key: Jina AI API key
        verbose: Show detailed logs
        instructions: Custom prompt for LLM extraction (used when get_full=False)
        get_full: If True, return raw content without LLM processing
        llm_client: OpenAI client for LLM extraction (required when get_full=False)
        llm_model: Model name for LLM extraction (required when get_full=False)
        query: Original search query for extraction context

    Returns:
        Dictionary with page contents, formatted as [link]\n---\n[content]
    """
    contents = []

    if verbose:
        print(f"[Jina Reader] Starting {len(urls)} parallel fetches...")
        if not get_full:
            print(f"[Jina Reader] LLM extraction enabled (model: {llm_model})")

    async with httpx.AsyncClient(timeout=40.0) as client:
        # Create tasks for parallel execution (only for URLs not in cache)
        tasks = []
        urls_to_fetch = []
        for url in urls:
            if url in _url_cache:
                if verbose:
                    print(f"  [Jina Reader] Cache hit for: {url}")
                # Will process cached content later
            else:
                urls_to_fetch.append(url)
                task = _get_single(client, url, jina_key, verbose)
                tasks.append(task)

        # Execute all fetches in parallel
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process newly fetched URLs and update cache
        for url, result in zip(urls_to_fetch, fetch_results):
            if isinstance(result, Exception):
                error_msg = str(result)
                if verbose:
                    print(f"  [Jina Reader] ❌ Exception: {error_msg}")
                contents.append(
                    {
                        "url": url,
                        "error": error_msg,
                        "content": "",
                    }
                )
            else:
                # Store raw content in cache
                raw_content = result.get("content", "")
                _url_cache[url] = raw_content

                # Process content based on get_full flag
                if get_full:
                    # Return raw content as-is
                    formatted_content = f"{url}\n---\n{raw_content}"
                else:
                    # Use LLM to extract relevant content
                    if not llm_client or not llm_model:
                        formatted_content = f"{url}\n---\n{raw_content}"
                    else:
                        extracted = await _extract_with_llm(
                            llm_client,
                            llm_model,
                            raw_content,
                            query,
                            instructions,
                            verbose,
                        )
                        formatted_content = f"{url}\n---\n{extracted}"

                contents.append(
                    {
                        "url": url,
                        "content": formatted_content,
                    }
                )

        # Process cached URLs
        for url in urls:
            if url in _url_cache and url not in urls_to_fetch:
                raw_content = _url_cache[url]

                # Process content based on get_full flag
                if get_full:
                    # Return raw content as-is
                    formatted_content = f"{url}\n---\n{raw_content}"
                else:
                    # Use LLM to extract relevant content
                    if not llm_client or not llm_model:
                        formatted_content = f"{url}\n---\n{raw_content}"
                    else:
                        extracted = await _extract_with_llm(
                            llm_client,
                            llm_model,
                            raw_content,
                            query,
                            instructions,
                            verbose,
                        )
                        formatted_content = f"{url}\n---\n{extracted}"

                contents.append(
                    {
                        "url": url,
                        "content": formatted_content,
                    }
                )

    if verbose:
        print(f"[Jina Reader] Completed {len(contents)} fetches")

    return {"pages": contents}


async def _extract_with_llm(
    client: AsyncOpenAI,
    model: str,
    content: str,
    query: str,
    instructions: str,
    verbose: bool = False,
) -> str:
    """Extract relevant content using LLM.

    Args:
        client: OpenAI client
        model: Model name
        content: Raw page content
        query: Original search query
        instructions: Extraction instructions
        verbose: Show detailed logs

    Returns:
        Extracted/summarized content
    """
    # Build extraction prompt
    prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
        query=query,
        instructions=instructions
        if instructions
        else "Extract all relevant information from this page.",
    )

    if verbose:
        print(f"  [LLM Extractor] Processing content ({len(content)} chars)")
        print(f"  [LLM Extractor] Query: {query}")
        if instructions:
            print(f"  [LLM Extractor] Instructions: {instructions[:100]}...")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=2000,
        )

        extracted = response.choices[0].message.content or "No content extracted"

        if verbose:
            print(f"  [LLM Extractor] Extracted {len(extracted)} chars")

        return extracted
    except Exception as e:
        if verbose:
            print(f"  [LLM Extractor] ❌ ERROR: {e}")
        # Fallback to raw content on error
        return content[:2000]


async def _get_single(
    client: httpx.AsyncClient, url: str, jina_key: str, verbose: bool = False
) -> dict[str, Any]:
    """Fetch content from a single URL."""
    headers = {
        "X-Retain-Images": "none",
        "X-Retain-Links": "gpt-oss",
        "X-Timeout": "40",
    }
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    if verbose:
        print(f"  [Jina Reader] URL: {url}")
        print(f"  [Jina Reader] Headers: {headers}")

    response = None
    try:
        # URL encode the URL for the Jina API
        encoded_url = httpx.URL(url).raw_path.decode("utf-8")
        response = await client.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
        )
        response.raise_for_status()

        if verbose:
            print(f"  [Jina Reader] Status: {response.status_code}")
            print(f"  [Jina Reader] Content length: {len(response.text)} chars")

        return {
            "url": url,
            "content": response.text,
        }
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
        if verbose:
            print(f"  [Jina Reader] ❌ ERROR: {error_msg}")
        return {
            "url": url,
            "error": error_msg,
            "content": "",
        }
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"  [Jina Reader] ❌ ERROR: {error_msg}")
            if response:
                try:
                    print(f"  [Jina Reader] Raw response: {response.text[:500]}")
                except:
                    pass
        return {
            "url": url,
            "error": error_msg,
            "content": "",
        }


async def execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    jina_key: str,
    verbose: bool = False,
    llm_client: AsyncOpenAI | None = None,
    llm_model: str | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Execute a tool by name.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        jina_key: Jina AI API key
        verbose: Show detailed logs
        llm_client: OpenAI client for LLM extraction (for web_get)
        llm_model: Model name for LLM extraction (for web_get)
        query: Original search query (for web_get extraction)

    Returns:
        Tool execution result
    """
    if tool_name == "web_search":
        queries = tool_args.get("queries", [])
        return await web_search(queries, jina_key, verbose)
    elif tool_name == "web_get":
        urls = tool_args.get("urls", [])
        instructions = tool_args.get("instructions", "")
        get_full = tool_args.get("get_full", False)
        return await web_get(
            urls,
            jina_key,
            verbose,
            instructions,
            get_full,
            llm_client,
            llm_model,
            query,
        )
    elif tool_name == "final_answer":
        return {"answer": tool_args.get("answer", "")}
    else:
        return {"error": f"Unknown tool: {tool_name}"}
