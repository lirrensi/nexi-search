"""Unit tests for output module."""

from __future__ import annotations

from nexi.output import (
    create_progress_callback,
    get_console,
    is_tty,
    print_answer,
    print_error,
    print_markdown,
    print_message,
    print_progress,
    print_result_summary,
    print_search_start,
    print_success,
    print_warning,
    set_plain_mode,
)


def test_is_tty():
    """Test is_tty function."""
    # Should return False when stdout is not a tty (in test environment)
    result = is_tty()
    assert isinstance(result, bool)


def test_get_console():
    """Test get_console returns console instance."""
    console = get_console()
    assert console is not None

    # Should return same instance
    console2 = get_console()
    assert console is console2


def test_set_plain_mode():
    """Test set_plain_mode changes console."""
    set_plain_mode(True)
    console = get_console()
    # Console should be recreated

    set_plain_mode(False)
    console2 = get_console()
    # Should be different instance


def test_print_message(capsys):
    """Test print_message function."""
    print_message("Test message", emoji="🚀")
    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_print_message_plain(capsys):
    """Test print_message in plain mode."""
    print_message("Test message", emoji="🚀", plain=True)
    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_print_markdown(capsys):
    """Test print_markdown function."""
    print_markdown("# Heading\n\nSome text", plain=True)
    captured = capsys.readouterr()
    assert "Heading" in captured.out


def test_print_search_start(capsys):
    """Test print_search_start function."""
    print_search_start("test query", plain=True)
    captured = capsys.readouterr()
    assert "test query" in captured.out
    assert "Searching" in captured.out


def test_print_progress(capsys):
    """Test print_progress function."""
    print_progress("Reading page...", plain=True)
    captured = capsys.readouterr()
    assert "Reading page" in captured.out


def test_print_answer(capsys):
    """Test print_answer function."""
    print_answer("Test answer", plain=True)
    captured = capsys.readouterr()
    assert "Test answer" in captured.out


def test_print_result_summary(capsys):
    """Test print_result_summary function."""
    print_result_summary(
        iterations=5,
        duration_s=10.5,
        tokens=1000,
        urls=["http://example.com"],
        plain=True,
    )
    captured = capsys.readouterr()
    assert "5 iterations" in captured.out or "10.5" in captured.out


def test_print_error(capsys):
    """Test print_error function."""
    print_error("Test error", plain=True)
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert "Test error" in captured.err


def test_print_warning(capsys):
    """Test print_warning function."""
    print_warning("Test warning", plain=True)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "Test warning" in captured.out


def test_print_success(capsys):
    """Test print_success function."""
    print_success("Test success", plain=True)
    captured = capsys.readouterr()
    assert "Test success" in captured.out


def test_create_progress_callback_verbose(capsys):
    """Test create_progress_callback with verbose=True."""
    callback, _ = create_progress_callback(verbose=True, plain=True)
    callback("Test message", 1, 5)

    captured = capsys.readouterr()
    assert "[1/5]" in captured.out
    assert "Test message" in captured.out


def test_create_progress_callback_non_verbose(capsys):
    """Test create_progress_callback with verbose=False."""
    callback, _ = create_progress_callback(verbose=False, plain=True)
    callback("Test message", 1, 5)

    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_print_answer_with_citations_proper_formatting(capsys) -> None:
    """Test that citations section is printed with line breaks, not on single line."""
    answer = "Test answer with [1][2] citations."
    url_citations = {
        "https://example.com/page1": 1,
        "https://example.com/page2": 2,
    }
    url_to_title = {
        "https://example.com/page1": "Page 1 Title",
        "https://example.com/page2": "Page 2 Title",
    }

    print_answer(answer, plain=True, url_citations=url_citations, url_to_title=url_to_title)

    captured = capsys.readouterr()

    # Check that answer is printed
    assert "Test answer with [1][2] citations." in captured.out

    # Check that citations section is printed
    assert "Sources:" in captured.out

    # The sources should have line breaks (not all on one line)
    lines = captured.out.split("\n")

    # Find the lines containing sources
    source_lines = [
        line for line in lines if line.strip().startswith("[1]") or line.strip().startswith("[2]")
    ]

    # We should have at least 2 source lines
    assert len(source_lines) >= 2

    # Check that they're not on the same line
    for i in range(len(source_lines) - 1):
        # Each source line should be complete
        assert source_lines[i].strip()
        # Next source should be on a new line
        assert source_lines[i + 1].strip().startswith("[")  # Next source starts with [


def test_print_answer_citations_with_titles(capsys) -> None:
    """Test that citations include titles in format: [1] Title - URL."""
    answer = "Test answer."
    url_citations = {
        "https://example.com/page1": 1,
    }
    url_to_title = {
        "https://example.com/page1": "Test Title",
    }

    print_answer(answer, plain=True, url_citations=url_citations, url_to_title=url_to_title)

    captured = capsys.readouterr()

    # Check that sources include titles
    assert "Test Title" in captured.out
    assert "https://example.com/page1" in captured.out
    assert "Sources:" in captured.out
