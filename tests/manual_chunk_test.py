"""Manual test for page chunking experiments.

This tests the idea of splitting fetched pages into numbered chunks,
then asking LLM to just return relevant chunk numbers instead of summarizing.

Run with: python tests/manual_chunk_test.py
"""

from __future__ import annotations

import asyncio
import re
import sys
import textwrap
from dataclasses import dataclass, field

import httpx

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    number: int
    content: str
    char_count: int
    word_count: int

    def __str__(self) -> str:
        return f"[CHUNK {self.number}] ({self.char_count} chars, {self.word_count} words)\n{self.content}"


@dataclass
class ChunkedPage:
    """Result of chunking a page."""

    url: str
    original_length: int
    chunks: list[Chunk] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"URL: {self.url}\nOriginal: {self.original_length} chars\nChunks: {len(self.chunks)}"
        )


def create_logical_chunks(md: str, target_chars: int = 480, max_chars: int = 720) -> list[Chunk]:
    """Heading-aware logical chunking for clean Jina Markdown.

    - Respects headings automatically
    - Merges small paragraphs under the same heading
    - Splits oversized chunks at paragraph boundaries
    - Works on ANY page

    Args:
        md: Markdown content
        target_chars: Target chars per chunk
        max_chars: Max chars before forcing split

    Returns:
        List of Chunk objects
    """
    if len(md) < 300:
        return [
            Chunk(
                number=1,
                content=md.strip(),
                char_count=len(md.strip()),
                word_count=len(md.strip().split()),
            )
        ]

    # Split while keeping headings with their content
    segments = re.split(r"(^#{1,6}\s+.+$)", md, flags=re.MULTILINE)

    chunks = []
    current = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        current.append(seg)

        combined = "\n\n".join(current)
        if len(combined) > max_chars and len(current) > 1:
            chunks.append("\n\n".join(current[:-1]))
            current = [current[-1]]

    if current:
        chunks.append("\n\n".join(current))

    # Final gentle merge of any tiny leftover chunks
    final = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if final and len(final[-1]) + len(chunk) < target_chars * 1.6:
            final[-1] += "\n\n" + chunk
        else:
            final.append(chunk)

    # Split oversized chunks at paragraph boundaries
    refined = []
    for chunk in final:
        if len(chunk) <= max_chars:
            refined.append(chunk)
        else:
            # Split by paragraphs
            paras = chunk.split("\n\n")
            sub_chunk = []
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if len("\n\n".join(sub_chunk + [para])) > max_chars and sub_chunk:
                    refined.append("\n\n".join(sub_chunk))
                    sub_chunk = [para]
                else:
                    sub_chunk.append(para)
            if sub_chunk:
                refined.append("\n\n".join(sub_chunk))

    # Convert to Chunk objects
    return [
        Chunk(number=i + 1, content=c, char_count=len(c), word_count=len(c.split()))
        for i, c in enumerate(refined)
    ]

    # Split while keeping headings with their content
    segments = re.split(r"(^#{1,6}\s+.+$)", md, flags=re.MULTILINE)

    chunks = []
    current = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        current.append(seg)

        combined = "\n\n".join(current)
        if len(combined) > max_chars and len(current) > 1:
            chunks.append("\n\n".join(current[:-1]))
            current = [current[-1]]

    if current:
        chunks.append("\n\n".join(current))

    # Final gentle merge of any tiny leftover chunks
    final = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if final and len(final[-1]) + len(chunk) < target_chars * 1.6:
            final[-1] += "\n\n" + chunk
        else:
            final.append(chunk)

    # Convert to Chunk objects
    return [
        Chunk(number=i + 1, content=c, char_count=len(c), word_count=len(c.split()))
        for i, c in enumerate(final)
    ]


def split_paragraphs_with_fallback(content: str, max_chars: int = 512) -> list[Chunk]:
    """Split by paragraphs, then split long paragraphs by char limit.

    Args:
        content: The text to split
        max_chars: Maximum characters per chunk

    Returns:
        List of Chunk objects
    """
    # Split by double newlines (paragraphs)
    paragraphs = content.split("\n\n")

    chunks = []
    chunk_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            # Paragraph fits, use as-is
            chunks.append(
                Chunk(
                    number=chunk_num,
                    content=para,
                    char_count=len(para),
                    word_count=len(para.split()),
                )
            )
            chunk_num += 1
        else:
            # Paragraph too long, split by max_chars
            for start in range(0, len(para), max_chars):
                text = para[start : start + max_chars]
                chunks.append(
                    Chunk(
                        number=chunk_num,
                        content=text,
                        char_count=len(text),
                        word_count=len(text.split()),
                    )
                )
                chunk_num += 1

    return chunks


def split_by_chars(content: str, chunk_size: int = 500) -> list[Chunk]:
    """Split content into chunks by character count.

    Args:
        content: The text to split
        chunk_size: Target characters per chunk

    Returns:
        List of Chunk objects
    """
    chunks = []
    for i, start in enumerate(range(0, len(content), chunk_size)):
        text = content[start : start + chunk_size]
        chunks.append(
            Chunk(
                number=i + 1,
                content=text,
                char_count=len(text),
                word_count=len(text.split()),
            )
        )
    return chunks


def split_by_paragraphs(content: str, min_chunk_size: int = 200) -> list[Chunk]:
    """Split content by paragraphs, combining small ones.

    Args:
        content: The text to split
        min_chunk_size: Minimum characters per chunk (combine small paragraphs)

    Returns:
        List of Chunk objects
    """
    # Split by double newlines (paragraph breaks)
    paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_text = ""
    chunk_num = 1

    for para in paragraphs:
        # If adding this paragraph doesn't exceed min size much, combine
        if len(current_text) < min_chunk_size:
            current_text = f"{current_text}\n\n{para}".strip()
        else:
            # Save current chunk and start new one
            if current_text:
                chunks.append(
                    Chunk(
                        number=chunk_num,
                        content=current_text,
                        char_count=len(current_text),
                        word_count=len(current_text.split()),
                    )
                )
                chunk_num += 1
            current_text = para

    # Don't forget the last chunk
    if current_text:
        chunks.append(
            Chunk(
                number=chunk_num,
                content=current_text,
                char_count=len(current_text),
                word_count=len(current_text.split()),
            )
        )

    return chunks


def split_by_sentences(content: str, sentences_per_chunk: int = 5) -> list[Chunk]:
    """Split content by sentences.

    Args:
        content: The text to split
        sentences_per_chunk: Number of sentences per chunk

    Returns:
        List of Chunk objects
    """
    # Simple sentence splitting (not perfect but good enough for testing)
    sentences = re.split(r"(?<=[.!?])\s+", content)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        text = " ".join(chunk_sentences)
        chunks.append(
            Chunk(
                number=i // sentences_per_chunk + 1,
                content=text,
                char_count=len(text),
                word_count=len(text.split()),
            )
        )

    return chunks


def split_by_semantic_blocks(content: str, target_size: int = 500) -> list[Chunk]:
    """Split content by semantic blocks (headers, code blocks, paragraphs).

    Tries to keep related content together.

    Args:
        content: The text to split
        target_size: Target characters per chunk

    Returns:
        List of Chunk objects
    """
    # Split by markdown headers, code blocks, and paragraphs
    # This regex finds: headers (##), code blocks (```), or paragraph breaks
    pattern = r"(^#{1,6}\s+.*$|```[\s\S]*?```|\n\s*\n)"
    parts = re.split(pattern, content, flags=re.MULTILINE)

    # Reconstruct blocks
    blocks = []
    current_block = ""

    for part in parts:
        if not part:
            continue
        # Check if this is a header or code block
        is_header = re.match(r"^#{1,6}\s+", part)
        is_code = part.startswith("```")

        if is_header or is_code:
            # Save current block if exists
            if current_block.strip():
                blocks.append(current_block.strip())
            # Start new block with this element
            current_block = part
        elif part.strip() == "":
            # Paragraph break - save current block
            if current_block.strip():
                blocks.append(current_block.strip())
            current_block = ""
        else:
            # Regular content - add to current block
            current_block += part

    if current_block.strip():
        blocks.append(current_block.strip())

    # Now combine blocks into chunks of target size
    chunks = []
    current_text = ""
    chunk_num = 1

    for block in blocks:
        if len(current_text) + len(block) > target_size and current_text:
            chunks.append(
                Chunk(
                    number=chunk_num,
                    content=current_text,
                    char_count=len(current_text),
                    word_count=len(current_text.split()),
                )
            )
            chunk_num += 1
            current_text = block
        else:
            current_text = f"{current_text}\n\n{block}".strip()

    if current_text:
        chunks.append(
            Chunk(
                number=chunk_num,
                content=current_text,
                char_count=len(current_text),
                word_count=len(current_text.split()),
            )
        )

    return chunks


async def fetch_page(url: str) -> str:
    """Fetch a page using Jina Reader.

    Args:
        url: URL to fetch

    Returns:
        Markdown content of the page
    """
    print(f"Fetching: {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "X-Retain-Images": "none",
            "X-Retain-Links": "gpt-oss",
        }

        response = await client.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
        )
        response.raise_for_status()

        print(f"Fetched {len(response.text)} characters")
        return response.text


def display_chunks(chunks: list[Chunk], title: str, show_content: bool = True) -> None:
    """Display chunks in a readable format.

    Args:
        chunks: List of chunks to display
        title: Title for this chunking method
        show_content: Whether to show full content or just stats
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

    total_chars = sum(c.char_count for c in chunks)
    total_words = sum(c.word_count for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    avg_words = total_words / len(chunks) if chunks else 0

    print(f"\nStats: {len(chunks)} chunks, {total_chars} chars, {total_words} words")
    print(f"Avg per chunk: {avg_chars:.0f} chars, {avg_words:.0f} words")

    if show_content:
        print("\n" + "-" * 80)
        for chunk in chunks:
            print(f"\n{chunk}")
            print("-" * 40)


def display_compact_view(chunks: list[Chunk], title: str) -> None:
    """Display chunks in compact format (first 100 chars of each).

    Args:
        chunks: List of chunks to display
        title: Title for this chunking method
    """
    print("\n" + "=" * 80)
    print(f" {title} (Compact View)")
    print("=" * 80)

    for chunk in chunks:
        preview = chunk.content[:100].replace("\n", " ")
        if len(chunk.content) > 100:
            preview += "..."
        print(f"[{chunk.number:3d}] {chunk.char_count:5d}c {chunk.word_count:4d}w | {preview}")


async def main() -> None:
    """Run the chunking experiments."""
    # Test URL - using a smaller page for clearer output
    # test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    test_url = "https://docs.python.org/3/tutorial/controlflow.html"

    # Fetch the page
    content = await fetch_page(test_url)

    # Show original content stats
    print(f"\nOriginal content: {len(content)} chars, {len(content.split())} words")
    print(f"\n{'=' * 80}")
    print("FIRST 800 CHARS OF RAW CONTENT:")
    print("=" * 80)
    print(content[:800])
    print("...")

    # Test different chunking methods
    print("\n" + "#" * 80)
    print("# CHUNKING COMPARISON")
    print("#" * 80)

    # Old approach: naive paragraph split
    chunks_naive = split_paragraphs_with_fallback(content, max_chars=512)
    print(f"\n1. Naive paragraph split: {len(chunks_naive)} chunks")
    print(f"   Avg chunk: {sum(c.char_count for c in chunks_naive) / len(chunks_naive):.0f} chars")

    # New approach: heading-aware logical chunking
    chunks = create_logical_chunks(content, target_chars=480, max_chars=720)
    print(f"\n2. Heading-aware logical: {len(chunks)} chunks")
    print(f"   Avg chunk: {sum(c.char_count for c in chunks) / len(chunks):.0f} chars")

    # Show compact view of logical chunks
    print("\n" + "=" * 80)
    print(" LOGICAL CHUNKS (first 25)")
    print("=" * 80)
    for chunk in chunks[:25]:
        preview = chunk.content[:90].replace("\n", " ")
        if len(chunk.content) > 90:
            preview += "..."
        print(f"[{chunk.number:3d}] {chunk.char_count:4d}c | {preview}")

    if len(chunks) > 25:
        print(f"... and {len(chunks) - 25} more chunks")

    # Show the LLM prompt format
    print("\n\n" + "#" * 80)
    print("# LLM PROMPT FORMAT EXAMPLE")
    print("#" * 80)

    # Use logical chunks for the example
    sample_chunks = chunks[5:12]
    sample_prompt = f"""Given the query: "How do for loops work in Python?"

Here are chunks from a webpage. Return ONLY the chunk numbers that contain relevant information.

{chr(10).join(f"[CHUNK {c.number}] {c.content[:180]}..." if len(c.content) > 180 else f"[CHUNK {c.number}] {c.content}" for c in sample_chunks)}

Return format: Just the numbers, comma-separated. Example: 6, 8, 11"""

    print("\nSample prompt to LLM:")
    print("-" * 40)
    print(sample_prompt)

    print("\n\nExpected LLM response: 6, 8, 10")
    print("(Then we manually grab those chunks and include them in context)")

    # Show what we'd retrieve
    print("\n\n" + "#" * 80)
    print("# WHAT WE'D RETRIEVE (chunks 6, 8, 10)")
    print("#" * 80)
    for i in [6, 8, 10]:
        if i <= len(chunks):
            c = chunks[i - 1]
            print(f"\n[CHUNK {c.number}] ({c.char_count} chars, {c.word_count} words)")
            print("-" * 40)
            print(c.content)


if __name__ == "__main__":
    asyncio.run(main())
