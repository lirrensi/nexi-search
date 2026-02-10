"""Output formatting for NEXI."""

from __future__ import annotations

import sys
from collections.abc import Callable

from rich.console import Console
from rich.markdown import Markdown

# Global console instance
_console: Console | None = None


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


def print_progress(message: str, plain: bool = False) -> None:
    """Print progress message.

    Args:
        message: Progress message
        plain: Force plain mode
    """
    if plain or not is_tty():
        print(message)
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


def print_answer(answer: str, plain: bool = False) -> None:
    """Print the final answer.

    Args:
        answer: Answer text (markdown)
        plain: Force plain mode
    """
    print()  # Blank line before answer
    print_markdown(answer, plain)
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
    verbose: bool = False, plain: bool = False
) -> Callable[[str, int, int], None]:
    """Create a progress callback function.

    Args:
        verbose: Show detailed progress
        plain: Force plain mode

    Returns:
        Callback function for search progress
    """

    def callback(message: str, iteration: int, total: int) -> None:
        if verbose:
            print(f"[{iteration}/{total}] {message}")
        else:
            print_progress(message, plain)

    return callback
