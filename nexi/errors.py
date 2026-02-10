"""Error handling utilities for NEXI.

Provides fail-resistant error handling for CLI tools.
"""

from __future__ import annotations

import sys
from typing import NoReturn


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

    # Print short error message to stderr
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)


def safe_exit(message: str | None = None, exit_code: int = 0) -> NoReturn:
    """Exit cleanly with optional message.

    Args:
        message: Optional message to print before exiting
        exit_code: Exit code (default: 0 for success)
    """
    if message:
        print(message)
    sys.exit(exit_code)
