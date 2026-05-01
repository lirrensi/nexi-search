"""Tests for runtime noise suppression behavior."""

# FILE: tests/test_runtime_noise.py
# PURPOSE: Verify runtime noise suppression keeps CLI stdout visible while silencing warnings.
# OWNS: Behavior tests for suppress_runtime_chatter verbose and non-verbose modes.
# EXPORTS: none
# DOCS: agent_chat/plan_cli_output_restoration_2026-05-02.md

from __future__ import annotations

import warnings

from nexi.runtime_noise import suppress_runtime_chatter


def test_suppress_runtime_chatter_non_verbose_keeps_stdout_visible(capsys) -> None:
    """Non-verbose mode should not suppress normal stdout output."""
    with suppress_runtime_chatter(False):
        print("visible output")

    captured = capsys.readouterr()
    assert "visible output" in captured.out


def test_suppress_runtime_chatter_non_verbose_suppresses_warnings(capsys) -> None:
    """Non-verbose mode should suppress warnings emitted in the block."""
    with suppress_runtime_chatter(False):
        warnings.warn("hidden warning", UserWarning)

    captured = capsys.readouterr()
    assert "hidden warning" not in captured.err


def test_suppress_runtime_chatter_verbose_keeps_stdout_visible(capsys) -> None:
    """Verbose mode should leave stdout behavior unchanged."""
    with suppress_runtime_chatter(True):
        print("verbose output")

    captured = capsys.readouterr()
    assert "verbose output" in captured.out
