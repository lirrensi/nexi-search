"""Runtime noise control for CLI commands.

This module keeps normal commands quiet by suppressing browser/runtime chatter,
warnings, and unraisable cleanup traces unless verbose mode is enabled.
"""

# FILE: nexi/runtime_noise.py
# PURPOSE: Centralize quiet/verbose runtime noise behavior for CLI commands.
# OWNS: Warning suppression context and process-level noise toggles.
# EXPORTS: configure_runtime_noise, suppress_runtime_chatter
# DOCS: agent_chat/plan_cli_output_restoration_2026-05-02.md

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings

_ORIGINAL_UNRAISABLE_HOOK = sys.unraisablehook


def configure_runtime_noise(verbose: bool) -> None:
    """Configure process-level noise behavior for CLI runs.

    Args:
        verbose: Whether verbose output is enabled.
    """
    if verbose:
        os.environ.pop("NODE_NO_WARNINGS", None)
        sys.unraisablehook = _ORIGINAL_UNRAISABLE_HOOK
        logging.disable(logging.NOTSET)
        return

    os.environ.setdefault("NODE_NO_WARNINGS", "1")
    sys.unraisablehook = _quiet_unraisable_hook
    logging.disable(logging.CRITICAL)


@contextlib.contextmanager
def suppress_runtime_chatter(verbose: bool):
    """Suppress Python-side chatter while a command is executing."""
    if verbose:
        yield
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _quiet_unraisable_hook(unraisable: object) -> None:
    """Silence unraisable cleanup exceptions in quiet mode."""
    return None


__all__ = ["configure_runtime_noise", "suppress_runtime_chatter"]
