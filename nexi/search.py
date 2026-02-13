"""Agentic search loop for NEXI."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from nexi.compaction import compact_conversation, should_compact
from nexi.config import EFFORT_LEVELS, Config, get_system_prompt
from nexi.token_counter import count_messages_tokens, estimate_page_tokens
from nexi.tools import TOOLS, clear_url_cache, execute_tool


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
    reached_max_iter: bool = False


ProgressCallback = Callable[[str, int, int], None]


async def _request_final_answer(
    messages: list[dict[str, Any]],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """Request a final answer from the LLM using gathered information.

    Args:
        messages: Conversation history
        client: OpenAI client
        model: Model name
        max_tokens: Max output tokens
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

        response = await client.chat.completions.create(
            model=model,
            messages=final_messages,
            tools=_get_tool_schemas(),
            tool_choice="auto",
            max_tokens=max_tokens,
        )

        if verbose:
            print("[Final Stage] Response received")

        message = response.choices[0].message

        # Check for tool calls (should be final_answer)
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

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
    max_iter: int | None = None,
    time_target: int | None = None,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> SearchResult:
    """Run agentic search loop.

    Args:
        query: User's search query
        config: NEXI configuration
        effort: Effort level (s/m/l)
        max_iter: Override max iterations
        time_target: Optional time limit in seconds (None = no limit)
        verbose: Show detailed progress
        progress_callback: Called with (message, iteration, total)

    Returns:
        SearchResult with answer and metadata
    """
    start_time = time.time()

    # Clear URL cache at start of search session
    clear_url_cache()

    # Determine max iterations with fallback to medium
    if max_iter is None:
        try:
            max_iter = EFFORT_LEVELS.get(effort, EFFORT_LEVELS["m"])["max_iter"]
        except (KeyError, TypeError):
            max_iter = EFFORT_LEVELS["m"]["max_iter"]

    # Ensure max_iter is an int
    if not isinstance(max_iter, int):
        max_iter = EFFORT_LEVELS["m"]["max_iter"]

    # Determine timeout (use CLI option if provided, otherwise use config)
    time_target_total = time_target if time_target is not None else config.time_target

    # Load system prompt
    system_prompt = get_system_prompt(max_iter, effort)

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
        """Append current sources list to tool result content."""
        # Only update web_get results
        if "pages" not in result:
            return result

        sources_list = get_sources_list()
        if not sources_list:
            return result

        # Update each page's content to include sources at the end
        updated_pages = []
        for page in result.get("pages", []):
            updated_page = page.copy()
            content = page.get("content", "")
            # Remove old sources section if present
            if "Sources:" in content:
                content = content.split("Sources:")[0].rstrip()
            # Add fresh sources list
            updated_page["content"] = content + sources_list
            updated_pages.append(updated_page)

        return {"pages": updated_pages}

    def report_progress(
        message: str,
        iteration: int = 0,
        context_size: int | None = None,
        urls: list[str] | None = None,
    ) -> None:
        """Report progress via callback."""
        if progress_callback:
            progress_callback(message, iteration, max_iter, context_size, urls)

    report_progress("Starting search...", 0)

    try:
        for current_iteration in range(1, max_iter + 1):
            # Check timeout (only if time_target is set)
            if time_target_total is not None:
                elapsed = time.time() - start_time
                if elapsed >= time_target_total:
                    report_progress(f"Time target reached after {elapsed:.1f}s", current_iteration)
                    final_answer = await _request_final_answer(
                        messages=messages,
                        client=client,
                        model=config.model,
                        max_tokens=config.max_output_tokens,
                        verbose=verbose,
                    )
                    break

            report_progress(f"Iteration {current_iteration}/{max_iter}", current_iteration)

            # Call LLM with tools (with retry logic)
            response = None
            last_error = None
            for attempt in range(config.llm_max_retries):
                try:
                    if verbose:
                        print(
                            f"[LLM] Calling {config.model}... (attempt {attempt + 1}/{config.llm_max_retries})"
                        )
                        print(f"[LLM] Messages count: {len(messages)}")

                    response = await client.chat.completions.create(
                        model=config.model,
                        messages=messages,
                        tools=_get_tool_schemas(),
                        tool_choice="auto",
                        max_tokens=config.max_output_tokens,
                    )

                    if verbose:
                        print("[LLM] Response received")
                        if response.usage:
                            print(
                                f"[LLM] Tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                            )
                    break  # Success, exit retry loop
                except Exception as e:
                    last_error = e
                    error_msg = (
                        f"LLM API error (attempt {attempt + 1}/{config.llm_max_retries}): {e}"
                    )
                    report_progress(error_msg, current_iteration)
                    if verbose:
                        print(f"[LLM] ❌ ERROR: {error_msg}")

                    if attempt < config.llm_max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s
                        backoff_delay = 2**attempt
                        if verbose:
                            print(f"[LLM] Retrying in {backoff_delay}s...")
                        await asyncio.sleep(backoff_delay)
                    else:
                        # All retries exhausted
                        final_answer = _force_answer(
                            messages,
                            f"API error after {config.llm_max_retries} retries: {last_error}",
                        )
                        break

            if response is None:
                # All retries failed, exit the search loop
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
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

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
                        tool_name, tool_args, config.jina_key, verbose, timeout=config.jina_timeout
                    )

                    if verbose:
                        print(
                            f"[Tool Result] web_search returned {len(result.get('searches', []))} results"
                        )
                        for search in result.get("searches", []):
                            if "error" in search:
                                print(f"  ❌ Error for '{search.get('query')}': {search['error']}")
                            else:
                                print(
                                    f"  ✓ '{search.get('query')}' returned {len(search.get('results', []))} results"
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

                    # Get citation numbers for URLs BEFORE fetching (stable numbering)
                    url_numbers = {url: get_url_number(url) for url in urls}

                    result = await execute_tool(
                        tool_name,
                        tool_args,
                        config.jina_key,
                        verbose,
                        client,
                        config.model,
                        query,
                        timeout=config.jina_timeout,
                        url_numbers=url_numbers,  # Pass URL numbers for citation markers
                        url_titles=url_to_title,  # Pass URL titles for sources list
                    )

                    if verbose:
                        print(
                            f"[Tool Result] web_get returned {len(result.get('pages', []))} pages"
                        )
                        for page in result.get("pages", []):
                            if "error" in page:
                                print(f"  ❌ Error for '{page.get('url')}': {page['error']}")
                            else:
                                content_len = len(page.get("content", ""))
                                print(f"  ✓ '{page.get('url')}' fetched {content_len} chars")

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
                            client=client,
                            model=config.model,
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
                                        "arguments": tool_call.function.arguments,
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
                        print(f"[Warning] Model returned tool call text instead of using tools")
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
                f"Max iterations ({max_iter}) reached, requesting final answer", max_iter
            )
            final_answer = await _request_final_answer(
                messages=messages,
                client=client,
                model=config.model,
                max_tokens=config.max_output_tokens,
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
        reached_max_iter=current_iteration >= max_iter,
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
    max_iter: int | None = None,
    time_target: int | None = None,
    verbose: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> SearchResult:
    """Synchronous wrapper for run_search.

    Args:
        query: User's search query
        config: NEXI configuration
        effort: Effort level (s/m/l)
        max_iter: Override max iterations
        time_target: Optional time limit in seconds (None = no limit)
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
                time_target=time_target,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        )
    finally:
        loop.close()
