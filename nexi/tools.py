"""Tool implementations for NEXI search loop."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

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
            "description": "Fetch full content from URLs using Jina Reader. Automatically converts PDFs, HTML to clean markdown. Supports multiple URLs in parallel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"},
                        "description": "List of URLs to fetch content from",
                        "minItems": 1,
                        "maxItems": 8,
                    }
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
    urls: list[str], jina_key: str, verbose: bool = False
) -> dict[str, Any]:
    """Fetch full content from URLs using Jina Reader.

    Args:
        urls: List of URLs to fetch (1-8)
        jina_key: Jina AI API key
        verbose: Show detailed logs

    Returns:
        Dictionary with page contents
    """
    contents = []

    if verbose:
        print(f"[Jina Reader] Starting {len(urls)} parallel fetches...")

    async with httpx.AsyncClient(timeout=40.0) as client:
        # Create tasks for parallel execution
        tasks = []
        for url in urls:
            task = _get_single(client, url, jina_key, verbose)
            tasks.append(task)

        # Execute all fetches in parallel
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(urls, fetch_results):
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
                contents.append(result)

    if verbose:
        print(f"[Jina Reader] Completed {len(contents)} fetches")

    return {"pages": contents}


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
    tool_name: str, tool_args: dict[str, Any], jina_key: str, verbose: bool = False
) -> dict[str, Any]:
    """Execute a tool by name.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        jina_key: Jina AI API key
        verbose: Show detailed logs

    Returns:
        Tool execution result
    """
    if tool_name == "web_search":
        queries = tool_args.get("queries", [])
        return await web_search(queries, jina_key, verbose)
    elif tool_name == "web_get":
        urls = tool_args.get("urls", [])
        return await web_get(urls, jina_key, verbose)
    elif tool_name == "final_answer":
        return {"answer": tool_args.get("answer", "")}
    else:
        return {"error": f"Unknown tool: {tool_name}"}
