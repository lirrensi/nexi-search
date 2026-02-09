# AGENTS.md

This file contains guidelines for agentic coding agents working on the NEXI codebase.

## Build, Lint, and Test Commands

```bash
# Install dependencies
uv sync

# Run the application
python -m nexi
# or
uv run python -m nexi

# Run with a query
python -m nexi "your search query"

# Linting (ruff is used - evidenced by .ruff_cache)
ruff check .
ruff check --fix .

# Format code
ruff format .

# Type checking (if mypy is added)
mypy nexi/

# Run tests (pytest - add test files to tests/ directory)
pytest tests/                    # Run all tests
pytest tests/test_config.py      # Run single test file
pytest tests/test_config.py::test_load_config  # Run single test
pytest -v                        # Verbose output
pytest -x                        # Stop on first failure
pytest --cov=nexi               # With coverage
```

## Code Style Guidelines

### Python Version and Imports

- Python 3.12+ required
- Always include `from __future__ import annotations` at the top of every module
- Use modern type hint syntax with `|` union operator: `str | None`, `int | None`
- Import order: standard library → third-party → local imports (with blank lines between groups)

```python
"""Module docstring."""

from __future__ import annotations

import json
from pathlib import Path

import click
import httpx
from openai import AsyncOpenAI

from nexi.config import Config, load_config
from nexi.tools import execute_tool
```

### Type Hints and Annotations

- Use type hints for all function parameters and return values
- Use `dict[str, Any]` instead of `Dict[str, Any]`
- Use `list[str]` instead of `List[str]`
- Use `Callable[[str, int], None]` for function types
- Annotate class attributes with types

```python
def search(query: str, config: Config, verbose: bool = False) -> SearchResult:
    """Search function with full type annotations."""
    ...

@dataclass
class SearchResult:
    answer: str
    urls: list[str] = field(default_factory=list)
    iterations: int = 0
```

### Naming Conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private functions: `_leading_underscore`
- Module-level constants: `UPPER_CASE`

```python
MAX_TIMEOUT = 300
CONFIG_FILE = Path.home() / ".nexi" / "config.json"

class ConfigManager:
    def load_config(self) -> Config:
        ...

def _helper_function(value: str) -> str:
    ...
```

### Error Handling

- Use specific exceptions in try/except blocks
- Provide meaningful error messages
- Use `raise from` to preserve exception chains
- Handle `KeyboardInterrupt` gracefully in CLI applications
- Return error information in data structures for async operations

```python
try:
    config = load_config()
except FileNotFoundError as e:
    print_error(f"Config not found: {e}")
    sys.exit(1)
except ValueError as e:
    print_error(f"Invalid config: {e}")
    sys.exit(1)
except Exception as e:
    print_error(f"Unexpected error: {e}")
    sys.exit(1)
```

### Async/Await Patterns

- Use `asyncio.new_event_loop()` for synchronous wrappers around async code
- Use `async with` for context managers
- Use `asyncio.gather()` for parallel operations
- Always close event loops in finally blocks

```python
def run_sync():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_function())
    finally:
        loop.close()
```

### File I/O and Encoding

- Always specify `encoding="utf-8"` when opening files
- Use `Path` objects from `pathlib` instead of string paths
- Use `ensure_ascii=False` when writing JSON with Unicode

```python
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

### Dataclasses

- Use `@dataclass` for configuration and data structures
- Use `field(default_factory=list)` for mutable defaults
- Implement `to_dict()` and `from_dict()` class methods for serialization

```python
@dataclass
class HistoryEntry:
    id: str
    query: str
    urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryEntry:
        return cls(**data)
```

### CLI Patterns (Click)

- Use `@click.command()` with `context_settings=dict(help_option_names=["-h", "--help"])`
- Use `@click.argument()` for required positional arguments
- Use `@click.option()` for optional flags and parameters
- Type hints automatically convert command-line arguments

```python
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("query", required=False)
@click.option("-v", "--verbose", is_flag=True, help="Show verbose output")
def main(query: str | None, verbose: bool) -> None:
    ...
```

### Windows Compatibility

- The project has special Windows UTF-8 handling in `main.py`
- Always test on Windows when making changes to output/encoding
- Use `sys.platform == "win32"` checks for Windows-specific code

### Documentation

- All modules must have docstrings
- All public functions must have docstrings with Args/Returns sections
- Use Google-style docstrings or consistent style throughout

```python
def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool

    Returns:
        Tool execution result as dictionary
    """
    ...
```

### Module Structure

- Keep modules focused and cohesive
- Use `__all__` to explicitly define public API
- Import from `nexi.__init__.py` for external usage

```python
__all__ = [
    "Config",
    "load_config",
    "save_config",
    "SearchResult",
]
```

## Testing Guidelines

- Tests should be placed in `tests/` directory
- Use `pytest` for testing framework
- Test files should be named `test_*.py`
- Use fixtures for common test setup
- Mock external API calls (OpenAI, Jina AI) in tests
- Test both success and error paths

```python
# tests/test_config.py
import pytest
from nexi.config import Config, load_config

def test_load_config(tmp_path):
    """Test config loading from file."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"base_url": "http://test.com"}')
    # ... test implementation
```

## Project-Specific Notes

- **Entry Points**: `main.py` and `nexi/__main__.py` are entry points with Windows UTF-8 fixes
- **Configuration**: Stored in `~/.local/share/nexi/config.json`
- **History**: Stored as JSONL in `~/.local/share/nexi/history.jsonl`
- **Prompts**: Stored in `~/.local/share/nexi/prompts/` directory
- **Async Operations**: All HTTP operations use `httpx.AsyncClient`
- **LLM Integration**: Uses OpenAI-compatible API via `AsyncOpenAI` client
- **Search Tools**: `web_search` and `web_get` use Jina AI APIs
- **Output**: Uses Rich library for terminal formatting with fallback to plain text

## Dependencies

Key dependencies to understand:
- `click` - CLI framework
- `httpx` - Async HTTP client
- `openai` - OpenAI API client (used for compatible APIs)
- `questionary` - Interactive prompts
- `rich` - Terminal output formatting
