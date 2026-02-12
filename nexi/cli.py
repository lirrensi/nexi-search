"""CLI interface for NEXI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from nexi.config import (
    Config,
    ensure_config,
    get_config_path,
)
from nexi.errors import handle_error
from nexi.history import (
    add_history_entry,
    create_entry,
    format_entry_full,
    format_entry_preview,
    get_entry_by_id,
    get_last_n_entries,
    get_latest_entry,
)
from nexi.history import (
    clear_history as clear_history_func,
)
from nexi.output import (
    create_progress_callback,
    is_tty,
    print_answer,
    print_error,
    print_result_summary,
    print_search_start,
    print_success,
    print_warning,
    set_plain_mode,
)
from nexi.search import run_search_sync


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="For shell completion: nexi --install-completion",
)
@click.argument("query", required=False)
@click.option("-q", "--query-text", help="Explicit query string")
@click.option(
    "-e",
    "--effort",
    type=click.Choice(["s", "m", "l"]),
    help="Search depth: s|m|l",
)
@click.option("--max-len", type=int, help="Max output tokens")
@click.option("--max-iter", type=int, help="Max search iterations")
@click.option("--time-target", type=int, help="Soft limit: force final answer after N seconds")
@click.option("-v", "--verbose", is_flag=True, help="Show tool calls and debug info")
@click.option("--plain", is_flag=True, help="Disable emoji/colors for scripting")
@click.option("--last", type=int, metavar="N", help="Show last N searches")
@click.option("--prev", is_flag=True, help="Show full latest search result")
@click.option("--show", type=str, metavar="ID", help="Show full result by ID")
@click.option("--clear-history", is_flag=True, help="Delete all search history")
@click.option("--config", "show_config", is_flag=True, help="Print config file path")
@click.option("--edit-config", is_flag=True, help="Open config in $EDITOR")
@click.version_option(version="0.1.0", prog_name="nexi")
def main(
    query: str | None,
    query_text: str | None,
    effort: str | None,
    max_len: int | None,
    max_iter: int | None,
    time_target: int | None,
    verbose: bool,
    plain: bool,
    last: int | None,
    prev: bool,
    show: str | None,
    clear_history: bool,
    show_config: bool,
    edit_config: bool,
) -> None:
    """NEXI - Intelligent web search CLI tool.

    Examples:
        nexi "how to use rust async traits"
        nexi --effort l --max-len 40000 -q "explain quantum entanglement"
        nexi -e s "quick python question"
        nexi --last 5
        nexi --prev
    """
    # Handle plain mode
    if plain or not is_tty():
        set_plain_mode(True)

    # Handle config commands
    if show_config:
        click.echo(get_config_path())
        return

    if edit_config:
        config_path = get_config_path()
        editor = os.environ.get("EDITOR", "notepad" if sys.platform == "win32" else "nano")
        click.echo(f"Opening {config_path} in {editor}...")
        os.system(f'{editor} "{config_path}"')
        return

    # Handle history commands
    if clear_history:
        clear_history_func()
        print_success("History cleared")
        return

    if last is not None:
        _show_last_n(last, plain)
        return

    if prev:
        _show_prev(plain)
        return

    if show:
        _show_by_id(show, plain)
        return

    # Get query from argument or option or stdin
    search_query = query or query_text

    if not search_query:
        # Check if stdin has data
        if not sys.stdin.isatty():
            search_query = sys.stdin.read().strip()

    if not search_query:
        # Enter interactive mode
        _interactive_mode()
        return

    # Run search
    _run_search_command(
        query=search_query,
        effort=effort,
        max_len=max_len,
        max_iter=max_iter,
        time_target=time_target,
        verbose=verbose,
        plain=plain,
    )


def _run_search_command(
    query: str,
    effort: str | None,
    max_len: int | None,
    max_iter: int | None,
    time_target: int | None,
    verbose: bool,
    plain: bool,
) -> None:
    """Execute a search command."""
    # Load config
    try:
        config = ensure_config()
    except Exception as e:
        handle_error(f"Failed to load config: {e}", verbose=verbose)

    # Override config with CLI options
    search_effort = effort or config.default_effort
    search_time_target = time_target if time_target is not None else config.time_target
    search_max_tokens = max_len if max_len is not None else config.max_output_tokens

    # Create temporary config with overrides
    search_config = Config(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model,
        jina_key=config.jina_key,
        default_effort=search_effort,
        time_target=search_time_target,
        max_output_tokens=search_max_tokens,
        jina_timeout=config.jina_timeout,
        llm_max_retries=config.llm_max_retries,
    )

    # Determine if we should use compact mode (non-TTY and not verbose)
    compact = not is_tty() and not verbose

    # Print search start (skip in compact mode)
    if not compact:
        print_search_start(query, plain)

    # Create progress callback
    progress_callback, tracker = create_progress_callback(verbose, plain, compact)

    try:
        # Run search
        result = run_search_sync(
            query=query,
            config=search_config,
            effort=search_effort,
            max_iter=max_iter,
            time_target=time_target,
            verbose=verbose,
            progress_callback=progress_callback,  # type: ignore[arg-type]
        )

        # Print compact summary before answer in compact mode
        if compact and tracker:
            summary = tracker.get_summary()
            if summary:
                print(summary)

        # Print answer
        print_answer(result.answer, plain)

        # Print summary in verbose mode
        if verbose:
            print_result_summary(
                iterations=result.iterations,
                duration_s=result.duration_s,
                tokens=result.tokens,
                urls=result.urls,
                plain=plain,
            )

        # Save to history
        entry = create_entry(
            query=query,
            answer=result.answer,
            urls=result.urls,
            effort=search_effort,
            iterations=result.iterations,
            duration_s=result.duration_s,
            tokens=result.tokens,
        )
        add_history_entry(entry)

    except KeyboardInterrupt:
        print_warning("\\nSearch cancelled")
        sys.exit(130)
    except Exception as e:
        handle_error(f"Search failed: {e}", verbose=verbose)


def _show_last_n(n: int, plain: bool) -> None:
    """Show last N history entries."""
    entries = get_last_n_entries(n)

    if not entries:
        print_warning("No searches yet! Try: nexi 'your query here'")
        return

    for i, entry in enumerate(entries, 1):
        click.echo(format_entry_preview(entry, i))
        click.echo()

    click.echo("[Use 'nexi --show <ID>' to see full result]")


def _show_prev(plain: bool) -> None:
    """Show previous (latest) search result."""
    entry = get_latest_entry()

    if not entry:
        print_warning("No searches yet! Try: nexi 'your query here'")
        return

    click.echo(format_entry_full(entry))


def _show_by_id(entry_id: str, plain: bool) -> None:
    """Show search result by ID."""
    entry = get_entry_by_id(entry_id)

    if not entry:
        print_error(f"No search found with ID: {entry_id}")
        sys.exit(1)

    click.echo(format_entry_full(entry))


def _interactive_mode() -> None:
    """Run interactive mode (REPL)."""
    import questionary

    # Load config
    try:
        config = ensure_config()
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)

    # Display ASCII art
    art_path = Path(__file__).parent.parent / "img" / "art.txt"
    if art_path.exists():
        try:
            with open(art_path, encoding="utf-8") as f:
                art = f.read()
            click.echo(art)
        except Exception:
            pass

    print_success("Welcome to NEXI interactive mode!")
    print("Type 'exit' or press Ctrl+C to quit\n")

    while True:
        try:
            query = questionary.text("🔍 nexi>").ask()

            if not query or query.lower() in ("exit", "quit", "q"):
                print_success("Bye bye~ 💕")
                break

            if query.lower() in ("help", "h"):
                click.echo("Commands: exit, quit, q, help, h")
                continue

            # Run search
            _run_search_command(
                query=query,
                effort=None,
                max_len=None,
                max_iter=None,
                time_target=None,
                verbose=False,
                plain=False,
            )

        except KeyboardInterrupt:
            print_success("\nBye bye~ 💕")
            break
        except Exception as e:
            print_error(f"Error: {e}")


if __name__ == "__main__":
    main()
