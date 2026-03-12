"""Simple tests to verify MCP server module structure."""

from __future__ import annotations

import sys
from pathlib import Path


def test_mcp_server_files_exist() -> None:
    """Test that MCP server files exist."""
    project_root = Path(__file__).parent.parent
    mcp_server_py = project_root / "nexi" / "mcp_server.py"
    mcp_server_cli_py = project_root / "nexi" / "mcp_server_cli.py"

    assert mcp_server_py.exists(), "mcp_server.py not found"
    assert mcp_server_cli_py.exists(), "mcp_server_cli.py not found"


def test_mcp_server_import_structure() -> None:
    """Test that MCP server exposes the expected tool surface."""
    project_root = Path(__file__).parent.parent
    mcp_server_py = project_root / "nexi" / "mcp_server.py"

    content = mcp_server_py.read_text(encoding="utf-8")

    assert "from fastmcp import FastMCP" in content
    assert "from nexi.search import SearchResult, run_search" in content
    assert "from nexi.backends.orchestrators import run_search_chain" in content
    assert "from nexi.tools import web_get" in content
    assert content.count("@mcp.tool") == 3
    assert "def nexi_agent(" in content
    assert "def nexi_search(" in content
    assert "def nexi_fetch(" in content


def test_mcp_server_cli_uses_click() -> None:
    """Test that the MCP server CLI is a Click entrypoint."""
    project_root = Path(__file__).parent.parent
    mcp_server_cli_py = project_root / "nexi" / "mcp_server_cli.py"

    content = mcp_server_cli_py.read_text(encoding="utf-8")

    assert "import click" in content
    assert "@click.command" in content
    assert 'click.Choice(["stdio", "http"])' in content
    assert "def main(" in content
    assert "run(transport=transport, host=host, port=port)" in content


def test_pyproject_mcp_dependencies() -> None:
    """Test that pyproject.toml has MCP dependencies."""
    project_root = Path(__file__).parent.parent
    pyproject = project_root / "pyproject.toml"

    content = pyproject.read_text(encoding="utf-8")

    assert "[dependency-groups]" in content
    assert "mcp" in content
    assert "fastmcp" in content


if __name__ == "__main__":
    try:
        test_mcp_server_files_exist()
        test_mcp_server_import_structure()
        test_mcp_server_cli_uses_click()
        test_pyproject_mcp_dependencies()
        sys.exit(0)
    except AssertionError as exc:
        print(f"Test failed: {exc}")
        sys.exit(1)
