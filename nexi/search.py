"""Agentic search loop for NEXI."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nexi.backends.orchestrators import ProviderChainError, run_llm_chain
from nexi.compaction import compact_conversation, should_compact
from nexi.config import EFFORT_LEVELS, INTERNAL_LLM_MAX_TOKENS, Config, get_system_prompt
from nexi.token_counter import count_messages_tokens, estimate_page_tokens
from nexi.tools import TOOLS, clear_url_cache, close_http_client, execute_tool


@dataclass
class SearchResult:
    """Result of a search operation."""

    answer: str
    urls: list[str] = field(default_factory=list)
    url_citations: dict[str, int] = field(default_factory=dict)  # URL -> citation number
    url_to_title: dict[str, str] = field(default_factory=dict)  # URL -> title
    iterations: int = 0
    duration_s: float = 0.0
    tokens: int = 0


ProgressCallback = Callable[..., None]


async def _request_final_answer(
    messages: list[dict[str, Any]],
    config: Config,
    verbose: bool = False,
) -> str:
    """Request a final answer from the LLM using gathered information.

    Args:
        messages: Conversation history
        config: NEXI configuration
        verbose: Show detailed progress

    Returns:
        Final answer string
    """
    # Add a user message requesting final answer
    final_messages = messages.copy()
    final_messages.append(
        {
            "role": "user",
            "content": "Please provide a comprehensive final answer based on all the information gathered so far. Use the final_answer tool to submit your response.",
        }
    )

    try:
        if verbose:
            print("[Final Stage] Requesting final answer from LLM...")

        response = await run_llm_chain(
            messages=final_messages,
            tools=_get_tool_schemas(),
            config=config,
            verbose=verbose,
            max_tokens=INTERNAL_LLM_MAX_TOKENS,
        )

        if verbose:
            print("[Final Stage] Response received")

        message = response.choices[0].message

        # Check for tool calls (should be final_answer)
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_data = getattr(tool_call, "function", None)
            if function_data is None:
                return message.content or "No answer provided"

            tool_name = function_data.name
            tool_args = json.loads(function_data.arguments)

            if tool_name == "final_answer":
                return tool_args.get("answer", "No answer provided")
            else:
                # LLM called a different tool, extract content instead
                return message.content or "No answer provided"
        else:
            # No tool call, return content directly
            return message.content or "No answer provided"

    except Exception as e:
        if verbose:
            print(f"[Final Stage] Error: {e}")
        # Fallback: extract last tool content
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                last_content = msg.get("content", "")
                break
        return f"Error generating final answer: {e}\n\nLast gathered information:\n{last_content[:1000] if last_content else 'No information gathered.'}"


async def run_search(
    query: str,
    config: Config,
    effort: str = "m",
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    initial_messages: list[dict[str, Any]] | None = None,
) -> SearchResult:
    """Run agentic search loop.

    Args:
        query: User's search query
        config: NEXI configuration
        effort: Effort level (s/m/l)
        verbose: Show detailed progress
        progress_callback: Called with (message, iteration, total)

    Returns:
        SearchResult with answer and metadata
    """
    start_time = time.time()

    # Clear URL cache at start of search session
    clear_url_cache()

    try:
        max_iterations = int(EFFORT_LEVELS[effort]["max_iter"])
    except (KeyError, TypeError, ValueError):
        max_iterations = int(EFFORT_LEVELS["m"]["max_iter"])

    # Load system prompt
    system_prompt = get_system_prompt(effort)

    # Initialize conversation
    if initial_messages is not None:
        messages: list[dict[str, Any]] = initial_messages.copy()
        # Append the new user query
        messages.append({"role": "user", "content": query})
    else:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    # Track state
    urls_fetched: set[str] = set()
    total_tokens = 0
    current_tokens = 0
    final_answer = ""
    current_iteration = 0

    # URL citation tracking - stable numbering throughout the search
    url_to_number: dict[str, int] = {}  # URL -> [1], [2], etc.
    next_url_number: int = 1

    def get_url_number(url: str) -> int:
        """Get stable citation number for URL, assigning new one if needed."""
        nonlocal next_url_number
        if url not in url_to_number:
            url_to_number[url] = next_url_number
            next_url_number += 1
        return url_to_number[url]

    def get_url_title(url: str) -> str:
        """Get title for URL. Returns empty string if not set."""
        return url_to_title.get(url, "")

    url_to_title: dict[str, str] = {}  # URL -> title (from web_search results)

    def get_sources_list() -> str:
        """Generate the current sources list for the model with format: [1] Title - URL."""
        if not url_to_number:
            return ""
        lines = ["", "Sources:"]
        # Sort by citation number
        for num in sorted(url_to_number.values()):
            # Find URL for this number
            url = [u for u, n in url_to_number.items() if n == num][0]
            # Get title if available (from web_search results)
            title = get_url_title(url)
            if title:
                lines.append(f"[{num}] {title} - {url}")
            else:
                lines.append(f"[{num}] {url}")
        return "\n".join(lines)

    def update_tool_result_with_sources(result: dict[str, Any]) -> dict[str, Any]:
        """Append current sources list to tool result content (only on last page)."""
        # Only update web_get results
        if "pages" not in result:
            return result

        sources_list = get_sources_list()
        if not sources_list:
            return result

        # Update only the last page's content to include sources
        # This avoids duplicating the sources list across all pages in a batch
        updated_pages = []
        pages = result.get("pages", [])
        for i, page in enumerate(pages):
            updated_page = page.copy()
            content = page.get("content", "")
            # Remove old sources section if present
            if "Sources:" in content:
                content = content.split("Sources:")[0].rstrip()
            # Add sources list only to the last page
            if i == len(pages) - 1:
                updated_page["content"] = content + sources_list
            else:
                updated_page["content"] = content
            updated_pages.append(updated_page)

        return {
            **{key: value for key, value in result.items() if key != "pages"},
            "pages": updated_pages,
        }

    def report_progress(
        message: str,
        iteration: int = 0,
        context_size: int | None = None,
        urls: list[str] | None = None,
    ) -> None:
        """Report progress via callback."""
        if progress_callback:
            progress_callback(message, iteration, max_iterations, context_size, urls)

    report_progress("Starting search...", 0)

    try:
        for current_iteration in range(1, max_iterations + 1):
            report_progress(f"Iteration {current_iteration}/{max_iterations}", current_iteration)

            try:
                response = await run_llm_chain(
                    messages=messages,
                    tools=_get_tool_schemas(),
                    config=config,
                    verbose=verbose,
                    max_tokens=INTERNAL_LLM_MAX_TOKENS,
                )
            except ProviderChainError as exc:
                if verbose:
                    print(f"[LLM] Provider chain exhausted: {exc.provider_failures}")
                final_answer = _force_answer(messages, str(exc))
                break

            # Track tokens
            if response.usage:
                total_tokens += response.usage.total_tokens or 0
                current_tokens = count_messages_tokens(messages, config.tokenizer_encoding)

                if verbose:
                    threshold = int(config.max_context * config.auto_compact_thresh)
                    print(
                        f"[Tokens] Current: {current_tokens}, Max: {config.max_context}, Threshold: {threshold}"
                    )

            message = response.choices[0].message

            # Check for tool calls
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_data = getattr(tool_call, "function", None)
                if function_data is None:
                    final_answer = message.content or "No answer provided"
                    report_progress("Answer ready!", current_iteration)
                    break

                tool_name = function_data.name
                # Parse tool arguments with fallback for malformed JSON
                try:
                    tool_args = json.loads(function_data.arguments)
                except json.JSONDecodeError:
                    if verbose:
                        print("[Warning] Malformed tool arguments, using empty dict")
                    tool_args = {}

                if verbose:
                    report_progress(f"Tool: {tool_name} | Args: {tool_args}", current_iteration)

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
                        tool_name,
                        tool_args,
                        config,
                        verbose,
                        query=query,
                    )

                    if verbose:
                        print(
                            f"[Tool Result] web_search returned {len(result.get('searches', []))} results"
                        )
                        for search in result.get("searches", []):
                            if "error" in search:
                                print(f"  [ERROR] '{search.get('query')}': {search['error']}")
                            else:
                                print(
                                    f"  [OK] '{search.get('query')}' returned {len(search.get('results', []))} results"
                                )

                    # Capture titles from search results for citations
                    for search_result in result.get("searches", []):
                        for item in search_result.get("results", []):
                            if "url" in item:
                                url = item["url"]
                                if "title" in item:
                                    url_to_title[url] = item["title"]

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
                                        "arguments": function_data.arguments,
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

                    # Get citation numbers for URLs BEFORE fetching (stable numbering)
                    url_numbers = {url: get_url_number(url) for url in urls}

                    result = await execute_tool(
                        tool_name,
                        tool_args,
                        config,
                        verbose,
                        query=query,
                        url_numbers=url_numbers,  # Pass URL numbers for citation markers
                        url_titles=url_to_title,  # Pass URL titles for sources list
                    )

                    if verbose:
                        print(
                            f"[Tool Result] web_get returned {len(result.get('pages', []))} pages"
                        )
                        for page in result.get("pages", []):
                            if "error" in page:
                                print(f"  [ERROR] '{page.get('url')}': {page['error']}")
                            else:
                                content_len = len(page.get("content", ""))
                                print(f"  [OK] '{page.get('url')}' fetched {content_len} chars")

                    # Estimate tokens for tool result
                    estimated_tokens = estimate_page_tokens(
                        json.dumps(result), config.tokenizer_encoding
                    )

                    # Check if compaction needed
                    if should_compact(current_tokens, estimated_tokens, config):
                        threshold = int(config.max_context * config.auto_compact_thresh)
                        if verbose:
                            print(
                                f"[Compaction] Triggered: {current_tokens} + {estimated_tokens} > {threshold}"
                            )

                        messages = await compact_conversation(
                            messages=messages,
                            original_query=query,
                            config=config,
                            verbose=verbose,
                        )

                        current_tokens = count_messages_tokens(messages, config.tokenizer_encoding)

                        # Check if still over limit
                        if current_tokens > config.max_context:
                            if verbose:
                                print(
                                    f"[Compaction] ❌ Still over limit: {current_tokens} > {config.max_context}"
                                )
                            final_answer = _force_answer(messages, "Context limit exceeded")
                            break

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
                                        "arguments": function_data.arguments,
                                    },
                                }
                            ],
                        }
                    )
                    # Update result with current sources list for the model to see
                    result = update_tool_result_with_sources(result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

            else:
                # No tool call - model should have used final_answer
                # Force an answer, but filter out tool call-like content
                content = message.content or "No answer provided"
                # Filter out tool call-like content that should not be visible
                if "<tool call" in content or "<function_calls>" in content:
                    # Model misbehaved - returning tool call text instead of using tools
                    if verbose:
                        print("[Warning] Model returned tool call text instead of using tools")
                    # Try to extract actual answer content after tool call markers
                    lines_list = content.split(chr(10))
                    answer_lines = []
                    in_tool_call = False
                    for line in lines_list:
                        if "<tool call" in line or "<function_calls>" in line:
                            in_tool_call = True
                        elif in_tool_call and (">" in line or "</" in line):
                            # End of tool call block
                            if "</tool" in line or "</function_calls>" in line:
                                in_tool_call = False
                        elif not in_tool_call and line.strip():
                            answer_lines.append(line)
                    final_answer = chr(10).join(answer_lines).strip() or "No answer provided"
                else:
                    final_answer = content
                report_progress("Answer ready!", current_iteration)
                break

        else:
            # Max iterations reached - enter final stage
            report_progress(
                f"Max iterations ({max_iterations}) reached, requesting final answer",
                max_iterations,
            )
            final_answer = await _request_final_answer(
                messages=messages,
                config=config,
                verbose=verbose,
            )

    except asyncio.CancelledError:
        report_progress("Search cancelled", 0)
        final_answer = "Search cancelled by user"
        raise

    duration = time.time() - start_time

    return SearchResult(
        answer=final_answer,
        urls=list(urls_fetched),
        url_citations=url_to_number,  # Include URL citation mapping
        url_to_title=url_to_title,  # Include URL title mapping
        iterations=current_iteration,
        duration_s=duration,
        tokens=total_tokens,
    )


def _get_tool_schemas() -> list[dict[str, Any]]:
    """Get tool schemas for OpenAI function calling.

    Returns the TOOLS constant from tools.py to avoid duplication.
    """
    return TOOLS


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
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
    initial_messages: list[dict[str, Any]] | None = None,
) -> SearchResult:
    """Synchronous wrapper for run_search.

    Args:
        query: User's search query
        config: NEXI configuration
        effort: Effort level (s/m/l)
        verbose: Show detailed progress
        progress_callback: Called with (message, iteration, total)
        initial_messages: Optional conversation history for multi-turn

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
                verbose=verbose,
                progress_callback=progress_callback,
                initial_messages=initial_messages,
            )
        )
    finally:
        # Close HTTP client to release connections
        loop.run_until_complete(close_http_client())
        loop.close()
