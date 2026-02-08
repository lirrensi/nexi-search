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


async def web_search(queries: list[str], jina_key: str) -> dict[str, Any]:
    """Search the web using Jina AI.

    Args:
        queries: List of search queries (1-5)
        jina_key: Jina AI API key

    Returns:
        Dictionary with search results
    """
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create tasks for parallel execution
        tasks = []
        for query in queries:
            task = _search_single(client, query, jina_key)
            tasks.append(task)

        # Execute all searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        for query, result in zip(queries, search_results):
            if isinstance(result, Exception):
                results.append(
                    {
                        "query": query,
                        "error": str(result),
                        "results": [],
                    }
                )
            else:
                results.append(result)

    return {"searches": results}


async def _search_single(
    client: httpx.AsyncClient, query: str, jina_key: str
) -> dict[str, Any]:
    """Execute a single search query."""
    headers = {}
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        response = await client.get(
            "https://s.jina.ai/",
            params={"q": query},
            headers=headers,
        )
        response.raise_for_status()

        # Parse response - Jina returns JSON array of results
        data = response.json()

        return {
            "query": query,
            "results": data if isinstance(data, list) else [data],
        }
    except httpx.HTTPStatusError as e:
        return {
            "query": query,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "results": [],
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "results": [],
        }


async def web_get(urls: list[str], jina_key: str) -> dict[str, Any]:
    """Fetch full content from URLs using Jina Reader.

    Args:
        urls: List of URLs to fetch (1-8)
        jina_key: Jina AI API key

    Returns:
        Dictionary with page contents
    """
    contents = []

    async with httpx.AsyncClient(timeout=40.0) as client:
        # Create tasks for parallel execution
        tasks = []
        for url in urls:
            task = _get_single(client, url, jina_key)
            tasks.append(task)

        # Execute all fetches in parallel
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(urls, fetch_results):
            if isinstance(result, Exception):
                contents.append(
                    {
                        "url": url,
                        "error": str(result),
                        "content": "",
                    }
                )
            else:
                contents.append(result)

    return {"pages": contents}


async def _get_single(
    client: httpx.AsyncClient, url: str, jina_key: str
) -> dict[str, Any]:
    """Fetch content from a single URL."""
    headers = {
        "X-Retain-Images": "none",
        "X-Retain-Links": "gpt-oss",
        "X-Timeout": "40",
    }
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    try:
        # URL encode the URL for the Jina API
        encoded_url = httpx.URL(url).raw_path.decode("utf-8")
        response = await client.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
        )
        response.raise_for_status()

        return {
            "url": url,
            "content": response.text,
        }
    except httpx.HTTPStatusError as e:
        return {
            "url": url,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "content": "",
        }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "content": "",
        }


def execute_tool(
    tool_name: str, tool_args: dict[str, Any], jina_key: str
) -> dict[str, Any]:
    """Execute a tool by name.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        jina_key: Jina AI API key

    Returns:
        Tool execution result
    """
    if tool_name == "web_search":
        queries = tool_args.get("queries", [])
        return asyncio.run(web_search(queries, jina_key))
    elif tool_name == "web_get":
        urls = tool_args.get("urls", [])
        return asyncio.run(web_get(urls, jina_key))
    elif tool_name == "final_answer":
        return {"answer": tool_args.get("answer", "")}
    else:
        return {"error": f"Unknown tool: {tool_name}"}
