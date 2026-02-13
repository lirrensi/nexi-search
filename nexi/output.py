"""Output formatting for NEXI."""

from __future__ import annotations

import sys
from collections.abc import Callable

from rich.console import Console
from rich.markdown import Markdown

from nexi.citations import process_answer_with_citations, format_citations_section

# Global console instance
_console: Console | None = None


class CompactProgressTracker:
    """Track progress in compact mode for non-TTY sessions."""

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.iterations: list[int] = []
        self.errors: list[str] = []

    def add_iteration(self, iteration: int) -> None:
        """Add an iteration to the tracker.

        Args:
            iteration: Iteration number
        """
        if iteration not in self.iterations:
            self.iterations.append(iteration)

    def add_error(self, error: str) -> None:
        """Add an error to the tracker.

        Args:
            error: Error message
        """
        self.errors.append(error)

    def get_summary(self) -> str:
        """Get the compact progress summary.

        Returns:
            Summary string like "[iteration 1 2 3 4 5]" or "[iteration 1 2 3] [error: ...]"
        """
        parts: list[str] = []
        if self.iterations:
            parts.append(f"[iteration {' '.join(map(str, self.iterations))}]")
        if self.errors:
            parts.append(f"[error: {' | '.join(self.errors)}]")
        return " ".join(parts) if parts else ""


def get_console() -> Console:
    """Get or create console instance."""
    global _console
    if _console is None:
        # Use UTF-8 encoding with error handling for Windows compatibility
        _console = Console(
            force_terminal=True,
            soft_wrap=True,
        )
    return _console


def set_plain_mode(plain: bool = True) -> None:
    """Set plain mode (no colors/emoji).

    Args:
        plain: Whether to use plain mode
    """
    global _console
    if plain:
        _console = Console(
            force_terminal=False,
            no_color=True,
            soft_wrap=True,
        )
    else:
        _console = Console(
            force_terminal=True,
            soft_wrap=True,
        )


def is_tty() -> bool:
    """Check if stdout is a TTY."""
    return sys.stdout.isatty()


def print_message(message: str, emoji: str = "", plain: bool = False) -> None:
    """Print a message with optional emoji.

    Args:
        message: Message to print
        emoji: Emoji prefix (ignored in plain mode)
        plain: Force plain mode
    """
    try:
        if plain or not is_tty():
            # Strip emoji in plain mode
            print(message)
        else:
            console = get_console()
            if emoji:
                console.print(f"{emoji} {message}")
            else:
                console.print(message)
    except UnicodeEncodeError:
        # Fallback for systems that can't handle Unicode
        safe_message = message.encode("ascii", "ignore").decode("ascii")
        print(safe_message)


def print_markdown(text: str, plain: bool = False) -> None:
    """Print markdown-formatted text.

    Args:
        text: Markdown text to print
        plain: Force plain mode (prints as plain text)
    """
    try:
        if plain or not is_tty():
            print(text)
        else:
            console = get_console()
            console.print(Markdown(text))
    except UnicodeEncodeError:
        # Fallback for systems that can't handle Unicode
        safe_text = text.encode("ascii", "ignore").decode("ascii")
        print(safe_text)


def print_search_start(query: str, plain: bool = False) -> None:
    """Print search start message.

    Args:
        query: Search query
        plain: Force plain mode
    """
    if plain or not is_tty():
        print(f'Searching for "{query}"...')
    else:
        console = get_console()
        console.print(f'🔍 Searching for "{query}"...')


def print_progress(
    message: str,
    plain: bool = False,
    context_size: int | None = None,
    max_context: int | None = None,
    urls: list[str] | None = None,
) -> None:
    """Print progress message with optional details.

    Args:
        message: Progress message
        plain: Force plain mode
        context_size: Current context size in tokens
        max_context: Maximum context size for formatting
        urls: URLs being processed
    """
    if plain or not is_tty():
        # Format context as current/max
        if context_size is not None and max_context is not None:
            ctx_str = f" | Context: {format_context_size(context_size, max_context)}"
        elif context_size is not None:
            ctx_str = f" | Context: {format_context_size(context_size, context_size)}"
        else:
            ctx_str = ""

        # Add URLs
        url_str = ""
        if urls:
            url_str = f" | URLs: {', '.join(urls[:3])}"
            if len(urls) > 3:
                url_str += f"... +{len(urls) - 3} more"

        print(f"[{message}{ctx_str}{url_str}]")
    else:
        console = get_console()
        # Add appropriate emoji based on message content
        if "search" in message.lower():
            console.print(f"🔍 {message}")
        elif "read" in message.lower() or "page" in message.lower():
            console.print(f"📄 {message}")
        elif "ready" in message.lower() or "complete" in message.lower():
            console.print(f"✨ {message}")
        elif "timeout" in message.lower() or "error" in message.lower():
            console.print(f"⚠️  {message}")
        else:
            console.print(message)


def format_context_size(current: int, max_: int) -> str:
    """Format context size as 'current/max' with k suffix.

    Args:
        current: Current token count
        max_: Maximum token count

    Returns:
        Formatted string like '12k/64k'
    """
    if max_ == 0:
        return f"{current}"

    # Convert to k (thousands) for readability
    current_k = current / 1000
    max_k = max_ / 1000

    # Format with 1 decimal place if needed, otherwise integer
    if current_k == int(current_k):
        current_str = f"{int(current_k)}k"
    else:
        current_str = f"{current_k:.1f}k"

    if max_k == int(max_k):
        max_str = f"{int(max_k)}k"
    else:
        max_str = f"{max_k:.1f}k"

    return f"{current_str}/{max_str}"


def print_answer(
    answer: str,
    plain: bool = False,
    url_citations: dict[str, int] | None = None,
    url_to_title: dict[str, str] | None = None,
    include_citations: bool = True,
) -> None:
    """Print the final answer with optional citations.

    Args:
        answer: Answer text (markdown)
        plain: Force plain mode
        url_citations: URL -> citation number mapping
        url_to_title: URL -> title mapping (for formatting sources list)
        include_citations: Whether to append citations section
    """
    # Process answer with citations if provided
    # NOTE: We don't add citations to answer text to avoid markdown collapsing line breaks
    # We print citations separately as plain text
    if url_citations and include_citations:
        # Extract citations from answer if model added them
        answer = process_answer_with_citations(
            answer,
            url_citations,
            url_to_title,
            include_citations_section=False,  # Don't add citations to answer text
            plain=plain,
        )

    print()  # Blank line before answer
    print_markdown(answer, plain)

    # Print citations section as plain text to preserve line breaks
    if url_citations and include_citations:
        citations_section = format_citations_section(
            {num: url for url, num in url_citations.items()},
            url_to_title,
            plain=plain,
        )
        if citations_section:
            print(citations_section)

    print()  # Blank line after answer


def print_result_summary(
    iterations: int,
    duration_s: float,
    tokens: int,
    urls: list[str],
    plain: bool = False,
) -> None:
    """Print search result summary (verbose mode).

    Args:
        iterations: Number of iterations
        duration_s: Duration in seconds
        tokens: Token count
        urls: URLs fetched
        plain: Force plain mode
    """
    if plain or not is_tty():
        print(f"Search completed in {duration_s:.1f}s ({iterations} iterations, {tokens} tokens)")
        if urls:
            print(f"URLs fetched: {len(urls)}")
    else:
        console = get_console()
        console.print(
            f"✨ Search completed in {duration_s:.1f}s ({iterations} iterations, {tokens} tokens)"
        )
        if urls:
            console.print(f"   URLs fetched: {len(urls)}")


def print_history_entry(entry_text: str, plain: bool = False) -> None:
    """Print a history entry.

    Args:
        entry_text: Formatted entry text
        plain: Force plain mode
    """
    print(entry_text)


def print_error(message: str, plain: bool = False) -> None:
    """Print an error message.

    Args:
        message: Error message
        plain: Force plain mode
    """
    if plain or not is_tty():
        print(f"Error: {message}", file=sys.stderr)
    else:
        console = get_console()
        console.print(f"❌ Error: {message}", style="red")


def print_warning(message: str, plain: bool = False) -> None:
    """Print a warning message.

    Args:
        message: Warning message
        plain: Force plain mode
    """
    if plain or not is_tty():
        print(f"Warning: {message}")
    else:
        console = get_console()
        console.print(f"⚠️  {message}", style="yellow")


def print_success(message: str, plain: bool = False) -> None:
    """Print a success message.

    Args:
        message: Success message
        plain: Force plain mode
    """
    if plain or not is_tty():
        print(message)
    else:
        console = get_console()
        console.print(f"✅ {message}", style="green")


def create_progress_callback(
    verbose: bool = False,
    plain: bool = False,
    compact: bool = False,
) -> tuple[Callable[[str, int, int], None], CompactProgressTracker | None]:
    """Create a progress callback function.

    Args:
        verbose: Show detailed progress
        plain: Force plain mode
        compact: Use compact mode (track iterations silently, print summary at end)

    Returns:
        Tuple of (callback function, tracker instance or None)
    """
    tracker = CompactProgressTracker() if compact else None

    def callback(
        message: str,
        iteration: int,
        total: int,
        context_size: int | None = None,
        urls: list[str] | None = None,
    ) -> None:
        if compact and tracker:
            # In compact mode, track iterations silently
            if iteration > 0:
                tracker.add_iteration(iteration)
            # Track errors
            if "error" in message.lower() or "timeout" in message.lower():
                tracker.add_error(message)
        elif verbose:
            print(f"[{iteration}/{total}] {message}")
        else:
            print_progress(message, plain, context_size, total, urls)

    return callback, tracker
