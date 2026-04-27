"""SnitchMD fetch provider for NEXI."""

# FILE: nexi/backends/snitchmd.py
# PURPOSE: Adapt SnitchMD (Docker-based) into NEXI fetch payloads.
# OWNS: SnitchMD-backed fetch configuration and execution.
# EXPORTS: SnitchFetchProvider

from __future__ import annotations

import json
import os
import subprocess
from html import unescape
from typing import Any

from bs4 import BeautifulSoup

from nexi.backends.http_client import get_http_client


class SnitchFetchProvider:
    """Docker-based fetch provider backed by SnitchMD."""

    name = "snitchmd"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate SnitchMD provider config."""
        wait = config.get("wait")
        if wait is not None and not isinstance(wait, (int, float, str)):
            raise ValueError("snitchmd wait must be a number or string")

        wait_for_selector = config.get("wait_for_selector")
        if wait_for_selector is not None and not isinstance(wait_for_selector, str):
            raise ValueError("snitchmd wait_for_selector must be a string")

        mode = config.get("mode")
        if mode is not None and mode not in ("precision", "recall"):
            raise ValueError("snitchmd mode must be 'precision' or 'recall'")

        no_cache = config.get("no_cache")
        if no_cache is not None and not isinstance(no_cache, bool):
            raise ValueError("snitchmd no_cache must be a boolean")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content from SnitchMD."""
        pages = []
        for url in urls:
            if verbose:
                print(f"  [SnitchMD] URL: {url}")
            try:
                page = await _fetch_with_snitchmd(url, config, timeout, verbose)
            except Exception as exc:
                page = await _fetch_via_http_fallback(url, timeout, verbose)
                if not page.get("content"):
                    page = {"url": url, "content": "", "error": str(exc)}
            pages.append(page)

        return {"pages": pages}


async def _fetch_with_snitchmd(
    url: str,
    config: dict[str, Any],
    timeout: float,
    verbose: bool,
) -> dict[str, Any]:
    """Fetch a URL using SnitchMD Docker container."""
    # Check Docker availability first
    if not _is_docker_available():
        return {
            "url": url,
            "content": "",
            "error": "Docker is not available. Install Docker and try again, "
            "or use a different fetch provider (e.g., crawl4ai, jina, firecrawl).",
        }

    cmd = _build_command(url, config, verbose)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input="",  # stdin
        )
    except subprocess.TimeoutExpired:
        return {"url": url, "content": "", "error": "SnitchMD timeout"}

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            return {"url": url, "content": "", "error": f"SnitchMD error: {stderr}"}
        return {"url": url, "content": "", "error": f"SnitchMD exit code: {result.returncode}"}

    output = result.stdout.strip()
    if not output:
        return {"url": url, "content": "", "error": "SnitchMD returned empty output"}

    # Try JSON output first for better metadata
    if config.get("json_output", True):
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                content = data.get("markdown") or data.get("content") or ""
                if content:
                    final_url = data.get("url", url)
                    return {"url": final_url, "content": content}
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback to plain text output
    return {"url": url, "content": output}


def _build_command(url: str, config: dict[str, Any], verbose: bool) -> list[str]:
    """Build the SnitchMD docker command."""
    cmd = ["docker", "run", "--rm", "-i"]

    # Mount cache directory
    cache_dir = os.path.expanduser("~/.cache/snitchmd")
    os.makedirs(cache_dir, exist_ok=True)
    cmd.extend(["-v", f"{cache_dir}:/cache"])

    # Image
    cmd.append("syabro/snitchmd")

    # Optional flags
    if config.get("json_output", True):
        cmd.append("--json")

    if config.get("no_cache"):
        cmd.append("--no-cache")

    wait = config.get("wait")
    if wait is not None:
        cmd.extend(["--wait", str(wait)])

    wait_for_selector = config.get("wait_for_selector")
    if wait_for_selector:
        cmd.extend(["--wait-for-selector", wait_for_selector])

    mode = config.get("mode")
    if mode == "precision":
        cmd.append("--favor-precision")
    elif mode == "recall":
        cmd.append("--favor-recall")

    # URL must come last
    cmd.append(url)

    return cmd


def _is_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


async def _fetch_via_http_fallback(
    url: str,
    timeout: float,
    verbose: bool,
) -> dict[str, Any]:
    """Fallback to direct HTTP fetch when SnitchMD fails."""
    client = get_http_client(timeout=timeout)

    if verbose:
        print(f"  [SnitchMD HTTP Fallback] URL: {url}")

    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        return {"url": url, "content": "", "error": str(exc)}

    # Simple text extraction as fallback
    try:
        soup = BeautifulSoup(unescape(response.text), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        content = soup.get_text("\n")
        lines = [line.strip() for line in content.splitlines()]
        content = "\n".join(line for line in lines if line)
    except Exception:
        content = response.text

    if not content.strip():
        return {"url": url, "content": "", "error": "No content returned"}

    return {"url": url, "content": content}


__all__ = ["SnitchFetchProvider"]
