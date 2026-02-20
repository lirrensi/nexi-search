"""NEXI - Intelligent web search CLI tool."""

__version__ = "0.4.1"

from nexi.config import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    EFFORT_LEVELS,
    EXTRACTOR_PROMPT_TEMPLATE,
    Config,
    ensure_config,
    get_config_path,
    get_system_prompt,
    load_config,
    run_first_time_setup,
    save_config,
)
from nexi.history import (
    HistoryEntry,
    add_history_entry,
    clear_history,
    create_entry,
    format_entry_full,
    format_entry_preview,
    get_entry_by_id,
    get_history_path,
    get_last_n_entries,
    get_latest_entry,
)
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
from nexi.search import SearchResult, run_search, run_search_sync
from nexi.tools import (
    TOOLS,
    clear_url_cache,
    execute_tool,
    web_get,
    web_search,
)

__all__ = [
    "__version__",
    "Config",
    "ensure_config",
    "get_config_path",
    "load_config",
    "save_config",
    "run_first_time_setup",
    "get_system_prompt",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE",
    "EXTRACTOR_PROMPT_TEMPLATE",
    "EFFORT_LEVELS",
    "TOOLS",
    "clear_url_cache",
    "execute_tool",
    "web_search",
    "web_get",
    "HistoryEntry",
    "get_history_path",
    "add_history_entry",
    "get_last_n_entries",
    "get_entry_by_id",
    "get_latest_entry",
    "clear_history",
    "create_entry",
    "format_entry_preview",
    "format_entry_full",
    "SearchResult",
    "run_search",
    "run_search_sync",
    "get_console",
    "set_plain_mode",
    "is_tty",
    "print_message",
    "print_markdown",
    "print_search_start",
    "print_progress",
    "print_answer",
    "print_result_summary",
    "print_error",
    "print_warning",
    "print_success",
    "create_progress_callback",
]

# MCP server (optional import)
try:
    from nexi.mcp_server import mcp, nexi_search  # noqa: F401
    from nexi.mcp_server import run as run_mcp_server  # noqa: F401

    __all__.extend(["mcp", "nexi_search", "run_mcp_server"])
except ImportError:
    pass
