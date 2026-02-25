"""Error handling utilities for NEXI.

Provides fail-resistant error handling for CLI tools.
"""

from __future__ import annotations

import sys
from typing import Any, NoReturn


def _safe_print(text: str, file: Any = sys.stdout) -> None:
    """Print text safely, handling encoding errors on Windows.

    Args:
        text: Text to print
        file: File to print to (default: stdout)
    """
    try:
        print(text, file=file)
    except UnicodeEncodeError:
        # Fallback for systems that can't handle Unicode
        safe_text = text.encode("utf-8", errors="replace").decode("utf-8")
        try:
            print(safe_text, file=file)
        except UnicodeEncodeError:
            # Last resort: strip all non-ASCII
            safe_text = text.encode("ascii", "ignore").decode("ascii")
            print(safe_text, file=file)


def handle_error(message: str, exit_code: int = 1, verbose: bool = False) -> NoReturn:
    """Handle errors gracefully without stack traces.

    Args:
        message: Short error message to display
        exit_code: Exit code to use (default: 1)
        verbose: If True, show full error details
    """
    if verbose:
        import traceback

        traceback.print_exc()

    # Print short error message to stderr with encoding safety
    _safe_print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)


def safe_exit(message: str | None = None, exit_code: int = 0) -> NoReturn:
    """Exit cleanly with optional message.

    Args:
        message: Optional message to print before exiting
        exit_code: Exit code (default: 0 for success)
    """
    if message:
        _safe_print(message)
    sys.exit(exit_code)
