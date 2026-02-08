"""Agentic search loop for NEXI."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from openai import AsyncOpenAI

from nexi.config import Config, EFFORT_LEVELS, get_prompt_for_effort
from nexi.tools import execute_tool


@dataclass
class SearchResult:
    """Result of a search operation."""

    answer: str
    urls: list[str] = field(default_factory=list)
    iterations: int = 0
    duration_s: float = 0.0
    tokens: int = 0
    reached_max_iter: bool = False


ProgressCallback = Callable[[str, int, int], None]


async def run_search(
    query: str,
    config: Config,
    effort: str = "m",
    max_iter: int | None = None,
    max_timeout: int | None = None,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> SearchResult:
    """Run agentic search loop.

    Args:
        query: User's search query
        config: NEXI configuration
        effort: Effort level (s/m/l)
        max_iter: Override max iterations
        max_timeout: Override timeout
        verbose: Show detailed progress
        progress_callback: Called with (message, iteration, total)

    Returns:
        SearchResult with answer and metadata
    """
    start_time = time.time()

    # Determine max iterations
    if max_iter is None:
        max_iter = EFFORT_LEVELS.get(effort, EFFORT_LEVELS["m"])["max_iter"]

    # Determine timeout
    timeout = max_timeout if max_timeout is not None else config.max_timeout

    # Load system prompt
    system_prompt = get_prompt_for_effort(effort)

    # Initialize OpenAI client
    client = AsyncOpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )

    # Initialize conversation
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Track state
    urls_fetched: set[str] = set()
    total_tokens = 0
    final_answer = ""
    current_iteration = 0

    def report_progress(message: str, iteration: int = 0) -> None:
        """Report progress via callback."""
        if progress_callback:
            progress_callback(message, iteration, max_iter)

    report_progress("Starting search...", 0)

    try:
        for current_iteration in range(1, max_iter + 1):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                report_progress(
                    f"Timeout reached after {elapsed:.1f}s", current_iteration
                )
                final_answer = _force_answer(messages, "Search timeout reached")
                break

            report_progress(
                f"Iteration {current_iteration}/{max_iter}", current_iteration
            )

            # Call LLM with tools
            try:
                if verbose:
                    print(f"[LLM] Calling {config.model}...")
                    print(f"[LLM] Messages count: {len(messages)}")

                response = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    tools=_get_tool_schemas(),
                    tool_choice="auto",
                    max_tokens=config.max_output_tokens,
                )

                if verbose:
                    print(f"[LLM] Response received")
                    if response.usage:
                        print(
                            f"[LLM] Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                        )
            except Exception as e:
                error_msg = f"LLM API error: {e}"
                report_progress(error_msg, current_iteration)
                if verbose:
                    print(f"[LLM] ❌ ERROR: {error_msg}")
                final_answer = _force_answer(messages, f"API error: {e}")
                break

            # Track tokens
            if response.usage:
                total_tokens += response.usage.total_tokens or 0

            message = response.choices[0].message

            # Check for tool calls
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    report_progress(
                        f"Tool: {tool_name} | Args: {tool_args}", current_iteration
                    )

                # Execute tool
                if tool_name == "final_answer":
                    final_answer = tool_args.get("answer", "")
                    report_progress("Answer ready!", current_iteration)
                    break

                elif tool_name == "web_search":
                    queries = tool_args.get("queries", [])
                    report_progress(
                        f"Searching for: {', '.join(queries)}",
                        current_iteration,
                    )
                    result = await execute_tool(
                        tool_name, tool_args, config.jina_key, verbose
                    )

                    if verbose:
                        print(
                            f"[Tool Result] web_search returned {len(result.get('searches', []))} results"
                        )
                        for search in result.get("searches", []):
                            if "error" in search:
                                print(
                                    f"  ❌ Error for '{search.get('query')}': {search['error']}"
                                )
                            else:
                                print(
                                    f"  ✓ '{search.get('query')}' returned {len(search.get('results', []))} results"
                                )

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
                    result = await execute_tool(
                        tool_name, tool_args, config.jina_key, verbose
                    )

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

                elif tool_name == "web_get":
                    urls = tool_args.get("urls", [])
                    report_progress(f"Reading {len(urls)} pages...", current_iteration)
                    result = await execute_tool(
                        tool_name, tool_args, config.jina_key, verbose
                    )

                    if verbose:
                        print(
                            f"[Tool Result] web_get returned {len(result.get('pages', []))} pages"
                        )
                        for page in result.get("pages", []):
                            if "error" in page:
                                print(
                                    f"  ❌ Error for '{page.get('url')}': {page['error']}"
                                )
                            else:
                                content_len = len(page.get("content", ""))
                                print(
                                    f"  ✓ '{page.get('url')}' fetched {content_len} chars"
                                )

                    # Track URLs
                    urls_fetched.update(urls)

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

                    # Track URLs
                    urls_fetched.update(urls)

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

            else:
                # No tool call - model should have used final_answer
                # Force an answer
                final_answer = message.content or "No answer provided"
                report_progress("Answer ready!", current_iteration)
                break

        else:
            # Max iterations reached
            report_progress(f"Max iterations ({max_iter}) reached", max_iter)
            final_answer = _force_answer(
                messages, f"Reached maximum iterations ({max_iter})"
            )

    except asyncio.CancelledError:
        report_progress("Search cancelled", 0)
        final_answer = "Search cancelled by user"
        raise

    duration = time.time() - start_time

    return SearchResult(
        answer=final_answer,
        urls=list(urls_fetched),
        iterations=current_iteration,
        duration_s=duration,
        tokens=total_tokens,
        reached_max_iter=current_iteration >= max_iter,
    )


def _get_tool_schemas() -> list[dict[str, Any]]:
    """Get tool schemas for OpenAI function calling."""
    return [
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


def _force_answer(messages: list[dict[str, Any]], reason: str) -> str:
    """Force a final answer when limits reached.

    Args:
        messages: Conversation history
        reason: Why we're forcing the answer

    Returns:
        Forced answer string
    """
    last_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            last_content = msg.get("content", "")
            break

    return f"""⚠️ {reason}

Based on the research conducted, here is the best answer available:

[The search was interrupted. The information gathered so far suggests:]

{last_content[:1000] if last_content else "No information gathered yet."}

---
*Note: Results may be incomplete due to search limitations.*
"""


def run_search_sync(
    query: str,
    config: Config,
    effort: str = "m",
    max_iter: int | None = None,
    max_timeout: int | None = None,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> SearchResult:
    """Synchronous wrapper for run_search.

    Args:
        query: User's search query
        config: NEXI configuration
        effort: Effort level (s/m/l)
        max_iter: Override max iterations
        max_timeout: Override timeout
        verbose: Show detailed progress
        progress_callback: Called with (message, iteration, total)

    Returns:
        SearchResult with answer and metadata
    """
    # Use a new event loop to avoid conflicts with existing loops
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            run_search(
                query=query,
                config=config,
                effort=effort,
                max_iter=max_iter,
                max_timeout=max_timeout,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        )
    finally:
        loop.close()
