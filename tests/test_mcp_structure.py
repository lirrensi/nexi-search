"""Simple test to verify MCP server module structure (without fastmcp)."""

from __future__ import annotations

import sys
from pathlib import Path


def test_mcp_server_files_exist():
    """Test that MCP server files exist."""
    project_root = Path(__file__).parent.parent
    mcp_server_py = project_root / "nexi" / "mcp_server.py"
    mcp_server_cli_py = project_root / "nexi" / "mcp_server_cli.py"

    assert mcp_server_py.exists(), "mcp_server.py not found"
    assert mcp_server_cli_py.exists(), "mcp_server_cli.py not found"

    print("OK: MCP server files exist")


def test_mcp_server_import_structure():
    """Test that MCP server has correct import structure."""
    project_root = Path(__file__).parent.parent
    mcp_server_py = project_root / "nexi" / "mcp_server.py"

    content = mcp_server_py.read_text()

    # Check for required imports
    assert "from nexi.config import Config, ensure_config" in content
    assert "from nexi.search import run_search_sync" in content
    assert "from fastmcp import FastMCP" in content

    # Check for tool function
    assert "@mcp.tool" in content
    assert "def nexi_search(" in content

    # Check for parameters
    assert "query: str" in content
    assert "effort: str" in content
    assert "max_iter: int | None" in content
    assert "time_target: int | None" in content
    assert "verbose: bool" in content

    print("OK: MCP server has correct import structure")


def test_mcp_server_cli_exists():
    """Test that MCP server CLI exists and has correct structure."""
    project_root = Path(__file__).parent.parent
    mcp_server_cli_py = project_root / "nexi" / "mcp_server_cli.py"

    content = mcp_server_cli_py.read_text()

    # Check for main function
    assert "def main()" in content
    assert "if __name__ ==" in content

    # Check for transport handling
    assert "transport =" in content
    assert "stdio" in content

    print("OK: MCP server CLI has correct structure")


def test_pyproject_mcp_dependencies():
    """Test that pyproject.toml has MCP dependencies."""
    project_root = Path(__file__).parent.parent
    pyproject = project_root / "pyproject.toml"

    content = pyproject.read_text()

    # Check for MCP dependency group
    assert "[dependency-groups]" in content
    assert "mcp" in content
    assert "fastmcp" in content

    print("OK: pyproject.toml has MCP dependencies")


if __name__ == "__main__":
    print("Testing MCP server implementation...")
    print()

    try:
        test_mcp_server_files_exist()
        test_mcp_server_import_structure()
        test_mcp_server_cli_exists()
        test_pyproject_mcp_dependencies()

        print()
        print("=" * 60)
        print("All tests passed! OK")
        print("=" * 60)
        print()
        print("To use the MCP server:")
        print("  1. Install fastmcp: uv sync --group mcp")
        print("  2. Run server: uv run python -m nexi.mcp_server_cli")
        print("  3. See MCP_SERVER.md for details")
        sys.exit(0)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
