"""CLI interface for NEXI."""

from __future__ import annotations

import importlib.metadata
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from nexi.config import (
    CONFIG_FILE,
    CONTINUATION_SYSTEM_PROMPT,
    ConfigCreatedError,
    ensure_config,
    format_config_created_message,
    write_default_template,
)
from nexi.config_doctor import build_doctor_report, check_command_readiness
from nexi.errors import handle_error
from nexi.history import (
    add_history_entry,
    create_entry,
    format_entry_full,
    format_entry_preview,
    get_entry_by_id,
    get_history_path,
    get_last_n_entries,
    get_latest_entry,
)
from nexi.history import (
    clear_history as clear_history_func,
)
from nexi.output import (
    _safe_print_stderr,
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


class QueryCommandGroup(click.Group):
    """Click group that treats unknown subcommands as search queries."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Preserve unknown trailing args for default search handling."""
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            raise click.exceptions.NoArgsIsHelpError(ctx)

        rest = click.Command.parse_args(self, ctx, args)

        if self.chain:
            ctx._protected_args = rest
            ctx.args = []
        elif rest:
            normalized = click.utils.make_str(rest[0])
            if normalized in self.commands:
                ctx._protected_args, ctx.args = rest[:1], rest[1:]
            else:
                ctx._protected_args = []
                ctx.args = rest

        return ctx.args

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve known subcommands and leave unknown args for query handling."""
        if args:
            normalized = click.utils.make_str(args[0])
            if normalized in self.commands:
                return super().resolve_command(ctx, args)
        return None, None, args


def _get_cli_version() -> str:
    """Resolve the installed CLI version."""
    try:
        return importlib.metadata.version("nexi-search")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


@click.group(
    cls=QueryCommandGroup,
    context_settings={
        "help_option_names": ["-h", "--help"],
        "allow_extra_args": True,
    },
    invoke_without_command=True,
    epilog="For shell completion: nexi --install-completion",
)
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
@click.version_option(version=_get_cli_version(), prog_name="nexi")
@click.pass_context
def main(
    ctx: click.Context,
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
) -> None:
    """NEXI - Intelligent web search CLI tool."""
    if plain or not is_tty():
        set_plain_mode(True)

    if ctx.invoked_subcommand is not None:
        return

    if clear_history:
        clear_history_func()
        print_success("History cleared")
        return

    if last is not None:
        _show_last_n(last)
        return

    if prev:
        _show_prev()
        return

    if show:
        _show_by_id(show)
        return

    positional_query = " ".join(ctx.args).strip() or None
    search_query = positional_query or query_text
    if not search_query and not sys.stdin.isatty():
        search_query = sys.stdin.read().strip()

    if not search_query:
        _interactive_mode()
        return

    _run_search_command(
        query=search_query,
        effort=effort,
        max_len=max_len,
        max_iter=max_iter,
        time_target=time_target,
        verbose=verbose,
        plain=plain,
    )


@main.command("config")
def config_command() -> None:
    """Open the current config in the user's editor."""
    write_default_template(CONFIG_FILE, force=False)
    editor = os.environ.get("EDITOR", "notepad" if sys.platform == "win32" else "nano")
    command = [*shlex.split(editor, posix=sys.platform != "win32"), str(CONFIG_FILE)]
    subprocess.run(command, check=False)


@main.command("init")
def init_command() -> None:
    """Create the default config template if needed."""
    created = write_default_template(CONFIG_FILE, force=False)
    if created:
        click.echo(f"Created config template at {CONFIG_FILE}")
        return
    click.echo(f"Config already exists at {CONFIG_FILE}")


@main.command("doctor")
def doctor_command() -> None:
    """Check config readiness for public commands."""
    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        raise click.ClickException(format_config_created_message(exc.config_path)) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    report = build_doctor_report(config)
    failed = False
    for command_name in ("nexi", "nexi-search", "nexi-fetch"):
        errors = report[command_name]
        if errors:
            failed = True
            click.echo(f"{command_name}: FAIL - {'; '.join(errors)}")
            continue
        click.echo(f"{command_name}: PASS")

    if failed:
        raise SystemExit(1)


@main.command("clean")
def clean_command() -> None:
    """Reset local config and history, then recreate the template."""
    history_path = get_history_path()
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    if history_path.exists():
        history_path.unlink()
    write_default_template(CONFIG_FILE, force=True)
    click.echo(f"Recreated config template at {CONFIG_FILE}")


@main.command("onboard")
def onboard_command() -> None:
    """Run the guided onboarding flow."""
    from nexi.onboard import run_onboarding

    run_onboarding()


def _run_search_command(
    query: str,
    effort: str | None,
    max_len: int | None,
    max_iter: int | None,
    time_target: int | None,
    verbose: bool,
    plain: bool,
    initial_messages: list[dict[str, Any]] | None = None,
) -> str:
    """Execute a search command."""
    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        print_error(format_config_created_message(exc.config_path))
        sys.exit(1)
    except Exception as exc:
        handle_error(f"Failed to load config: {exc}", verbose=verbose)

    readiness_errors = check_command_readiness(config, "nexi")
    if readiness_errors:
        handle_error("; ".join(readiness_errors), verbose=verbose)

    search_effort = effort or config.default_effort
    search_time_target = time_target if time_target is not None else config.time_target
    search_max_tokens = max_len if max_len is not None else config.max_output_tokens

    search_config = replace(
        config,
        default_effort=search_effort,
        time_target=search_time_target,
        max_output_tokens=search_max_tokens,
    )

    compact = not is_tty() and not verbose
    if not compact:
        print_search_start(query, plain)

    progress_callback, _tracker = create_progress_callback(verbose, plain, compact)

    try:
        result = run_search_sync(
            query=query,
            config=search_config,
            effort=search_effort,
            max_iter=max_iter,
            time_target=time_target,
            verbose=verbose,
            progress_callback=progress_callback,  # type: ignore[arg-type]
            initial_messages=initial_messages,
        )

        if compact:
            _safe_print_stderr("")

        print_answer(result.answer, plain, result.url_citations, result.url_to_title)

        if verbose:
            print_result_summary(
                iterations=result.iterations,
                duration_s=result.duration_s,
                tokens=result.tokens,
                urls=result.urls,
                plain=plain,
            )

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

        return result.answer

    except KeyboardInterrupt:
        print_warning("\nSearch cancelled")
        sys.exit(130)
    except Exception as exc:
        handle_error(f"Search failed: {exc}", verbose=verbose)
        return ""


def _show_last_n(n: int) -> None:
    """Show last N history entries."""
    entries = get_last_n_entries(n)

    if not entries:
        print_warning("No searches yet! Try: nexi 'your query here'")
        return

    for index, entry in enumerate(entries, 1):
        click.echo(format_entry_preview(entry, index))
        click.echo()

    click.echo("[Use 'nexi --show <ID>' to see full result]")


def _show_prev() -> None:
    """Show previous (latest) search result."""
    entry = get_latest_entry()
    if not entry:
        print_warning("No searches yet! Try: nexi 'your query here'")
        return
    click.echo(format_entry_full(entry))


def _show_by_id(entry_id: str) -> None:
    """Show search result by ID."""
    entry = get_entry_by_id(entry_id)
    if not entry:
        print_error(f"No search found with ID: {entry_id}")
        sys.exit(1)
    click.echo(format_entry_full(entry))


def _interactive_mode() -> None:
    """Run interactive mode (REPL) with multi-turn support."""
    import questionary

    try:
        config = ensure_config()
    except ConfigCreatedError as exc:
        print_error(format_config_created_message(exc.config_path))
        sys.exit(1)
    except Exception as exc:
        print_error(f"Failed to load config: {exc}")
        sys.exit(1)

    readiness_errors = check_command_readiness(config, "nexi")
    if readiness_errors:
        print_error("; ".join(readiness_errors))
        sys.exit(1)

    art_path = Path(__file__).parent.parent / "img" / "art.txt"
    if art_path.exists():
        try:
            with open(art_path, encoding="utf-8") as file_obj:
                click.echo(file_obj.read())
        except Exception:
            pass

    print_success("Welcome to NEXI interactive mode!")
    print("Type 'exit' or press Ctrl+C to quit\n")

    conversation_history: list[dict[str, Any]] = []
    is_first_turn = True

    while True:
        try:
            query = questionary.text("🔍 nexi>").ask()

            if not query or query.lower() in ("exit", "quit", "q"):
                print_success("Bye bye~ 💕")
                break

            if query.lower() in ("help", "h"):
                click.echo("Commands: exit, quit, q, help, h")
                continue

            if is_first_turn:
                initial_messages = None
            else:
                initial_messages = [
                    {"role": "system", "content": CONTINUATION_SYSTEM_PROMPT},
                    *conversation_history.copy(),
                ]

            answer = _run_search_command(
                query=query,
                effort=None,
                max_len=None,
                max_iter=None,
                time_target=None,
                verbose=True,
                plain=False,
                initial_messages=initial_messages,
            )

            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})
            is_first_turn = False

        except KeyboardInterrupt:
            print_success("\nBye bye~ 💕")
            break
        except Exception as exc:
            print_error(f"Error: {exc}")


if __name__ == "__main__":
    main()
