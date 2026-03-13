"""Tool implementations for NEXI search loop."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from nexi.backends.http_client import close_http_client, get_http_client
from nexi.backends.jina import clear_url_cache
from nexi.backends.orchestrators import run_fetch_chain, run_llm_chain, run_search_chain
from nexi.config import CHUNK_SELECTOR_PROMPT, EXTRACTOR_PROMPT_TEMPLATE, Config


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    number: int
    content: str
    char_count: int = field(init=False)
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())

    def __str__(self) -> str:
        return (
            f"[CHUNK {self.number}] ({self.char_count} chars, {self.word_count} words)\n"
            f"{self.content}"
        )


def create_logical_chunks(md: str, target_chars: int = 480, max_chars: int = 720) -> list[Chunk]:
    """Heading-aware logical chunking for clean provider markdown."""
    if len(md) < 300:
        return [Chunk(number=1, content=md.strip())]

    segments = re.split(r"(^#{1,6}\s+.+$)", md, flags=re.MULTILINE)
    chunks: list[str] = []
    current: list[str] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        current.append(segment)

        combined = "\n\n".join(current)
        if len(combined) > max_chars and len(current) > 1:
            chunks.append("\n\n".join(current[:-1]))
            current = [current[-1]]

    if current:
        chunks.append("\n\n".join(current))

    merged: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(merged[-1]) + len(chunk) < target_chars * 1.6:
            merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)

    refined: list[str] = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            refined.append(chunk)
            continue

        paragraphs = chunk.split("\n\n")
        sub_chunk: list[str] = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            candidate = "\n\n".join(sub_chunk + [paragraph])
            if len(candidate) > max_chars and sub_chunk:
                refined.append("\n\n".join(sub_chunk))
                sub_chunk = [paragraph]
            else:
                sub_chunk.append(paragraph)
        if sub_chunk:
            refined.append("\n\n".join(sub_chunk))

    return [Chunk(number=index + 1, content=chunk) for index, chunk in enumerate(refined)]


__all__ = [
    "TOOLS",
    "clear_url_cache",
    "close_http_client",
    "create_logical_chunks",
    "execute_tool",
    "get_http_client",
    "web_get",
    "web_search",
]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using the configured provider chain. Supports multiple parallel queries and returns snippets and URLs.",
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
            "description": "Fetch and process content from URLs using the configured fetch provider chain. Supports full content, chunk selection, or LLM extraction.",
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
                        "description": "Custom prompt guiding the LLM summarizer on what to look for. Only used when get_full=false and use_chunks=false.",
                    },
                    "get_full": {
                        "type": "boolean",
                        "description": "If true, returns the full page content without LLM processing.",
                        "default": False,
                    },
                    "use_chunks": {
                        "type": "boolean",
                        "description": "If true, splits page into logical chunks, asks LLM for relevant chunk numbers, and returns only those chunks.",
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
    queries: list[str],
    config: Config,
    verbose: bool = False,
) -> dict[str, Any]:
    """Search the web using the configured provider chain."""
    return await run_search_chain(queries, config, verbose)


async def web_get(
    urls: list[str],
    config: Config,
    verbose: bool = False,
    instructions: str = "",
    get_full: bool = False,
    use_chunks: bool = False,
    query: str = "",
    url_numbers: dict[str, int] | None = None,
    url_titles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fetch and process content from URLs using configured providers."""
    raw_result = await run_fetch_chain(urls, config, verbose)
    pages_by_url = {
        page["url"]: page
        for page in raw_result.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("url"), str)
    }

    contents: list[dict[str, Any]] = []
    urls_with_content = [
        (url, pages_by_url[url].get("content", ""))
        for url in urls
        if url in pages_by_url and not pages_by_url[url].get("error")
    ]

    for url in urls:
        page = pages_by_url.get(url)
        if not page or not page.get("error"):
            continue
        contents.append({"url": url, "error": page["error"], "content": ""})

    if verbose:
        print(f"[Fetch] Starting {len(urls)} fetches...")
        if use_chunks:
            print("[Fetch] Chunk-based selection enabled")
        elif not get_full:
            print("[Fetch] LLM extraction enabled")

    if use_chunks and urls_with_content:
        chunk_results = await asyncio.gather(
            *[
                _select_chunks_with_llm(
                    config=config,
                    content=raw_content,
                    query=query,
                    verbose=verbose,
                )
                for _, raw_content in urls_with_content
            ],
            return_exceptions=True,
        )
        for (url, _), result in zip(urls_with_content, chunk_results, strict=False):
            if isinstance(result, BaseException):
                contents.append({"url": url, "error": str(result), "content": ""})
                continue
            contents.append({"url": url, "content": _format_page_content(url, result, url_numbers)})
    elif not get_full and urls_with_content:
        extraction_results = await asyncio.gather(
            *[
                _extract_with_llm(
                    config=config,
                    content=raw_content,
                    query=query,
                    instructions=instructions,
                    verbose=verbose,
                )
                for _, raw_content in urls_with_content
            ],
            return_exceptions=True,
        )
        for (url, _), result in zip(urls_with_content, extraction_results, strict=False):
            if isinstance(result, BaseException):
                contents.append({"url": url, "error": str(result), "content": ""})
                continue
            contents.append({"url": url, "content": _format_page_content(url, result, url_numbers)})
    else:
        for url, raw_content in urls_with_content:
            contents.append(
                {
                    "url": url,
                    "content": _format_page_content(url, raw_content, url_numbers),
                }
            )

    ordered_pages = {page["url"]: page for page in contents}
    result_pages = [ordered_pages[url] for url in urls if url in ordered_pages]

    if verbose:
        print(f"[Fetch] Completed {len(result_pages)} fetches")

    return {
        "pages": result_pages,
        "provider_failures": raw_result.get("provider_failures", []),
    }


def _format_page_content(
    url: str,
    content: str,
    url_numbers: dict[str, int] | None,
) -> str:
    """Format page content with optional citation number."""
    citation_num = url_numbers.get(url) if url_numbers else None
    prefix = f"[{citation_num}] {url}" if citation_num else url
    return f"{prefix}\n---\n{content}"


def _response_text(response: Any) -> str:
    """Extract assistant text from an LLM response."""
    return response.choices[0].message.content or ""


async def _select_chunks_with_llm(
    config: Config,
    content: str,
    query: str,
    verbose: bool = False,
) -> str:
    """Select relevant chunks from content using the configured LLM chain."""
    chunks = create_logical_chunks(content)
    if not chunks:
        return content

    if verbose:
        print(f"  [Chunk Selector] Split into {len(chunks)} chunks")

    system_prompt = CHUNK_SELECTOR_PROMPT.format(query=query)
    user_content = "\n\n".join(f"[CHUNK {chunk.number}] {chunk.content}" for chunk in chunks)

    try:
        response = await run_llm_chain(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=[],
            config=config,
            verbose=verbose,
            max_tokens=100,
        )
        llm_response = _response_text(response).strip().lower()
        if verbose:
            print(f"  [Chunk Selector] LLM response: {llm_response}")

        if llm_response in {"none", "n/a", "null"}:
            return ""

        number_source = llm_response.split(":", 1)[1] if ":" in llm_response else llm_response
        all_numbers = re.findall(r"\d+", number_source)
        if not all_numbers:
            return ""

        selected_numbers = {
            int(number) for number in all_numbers if 1 <= int(number) <= len(chunks)
        }
        if not selected_numbers:
            return ""

        selected_chunks = [chunks[number - 1] for number in sorted(selected_numbers)]
        return "\n\n---\n\n".join(chunk.content for chunk in selected_chunks)
    except Exception as exc:
        if verbose:
            print(f"  [Chunk Selector] ERROR: {exc}")
            print(f"  [Chunk Selector] Falling back to first {min(5, len(chunks))} chunks")
        return "\n\n---\n\n".join(chunk.content for chunk in chunks[:5])


async def _extract_with_llm(
    config: Config,
    content: str,
    query: str,
    instructions: str,
    verbose: bool = False,
) -> str:
    """Extract relevant content using the configured LLM chain."""
    prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
        query=query,
        instructions=instructions or "Extract all relevant information from this page.",
    )

    if verbose:
        print(f"  [LLM Extractor] Processing content ({len(content)} chars)")
        print(f"  [LLM Extractor] Query: {query}")
        if instructions:
            print(f"  [LLM Extractor] Instructions: {instructions[:100]}...")

    try:
        response = await run_llm_chain(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            tools=[],
            config=config,
            verbose=verbose,
            max_tokens=16384,
        )
        extracted = _response_text(response) or "No content extracted"
        if verbose:
            print(f"  [LLM Extractor] Extracted {len(extracted)} chars")
        return extracted
    except Exception as exc:
        if verbose:
            print(f"  [LLM Extractor] ERROR: {exc}")
        return content[:2000]


async def execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    config: Config,
    verbose: bool = False,
    query: str = "",
    url_numbers: dict[str, int] | None = None,
    url_titles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a tool by name."""
    if tool_name == "web_search":
        queries = tool_args.get("queries", [])
        return await web_search(queries, config, verbose)
    if tool_name == "web_get":
        urls = tool_args.get("urls", [])
        instructions = tool_args.get("instructions", "")
        get_full = tool_args.get("get_full", False)
        use_chunks = tool_args.get("use_chunks", False)
        return await web_get(
            urls=urls,
            config=config,
            verbose=verbose,
            instructions=instructions,
            get_full=get_full,
            use_chunks=use_chunks,
            query=query,
            url_numbers=url_numbers,
            url_titles=url_titles,
        )
    if tool_name == "final_answer":
        return {"answer": tool_args.get("answer", "")}
    return {"error": f"Unknown tool: {tool_name}"}
