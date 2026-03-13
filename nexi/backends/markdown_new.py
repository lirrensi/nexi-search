"""markdown.new fetch provider for NEXI."""

from __future__ import annotations

from typing import Any

import httpx

from nexi.backends.http_client import get_http_client


class MarkdownNewFetchProvider:
    """Fetch provider backed by markdown.new."""

    name = "markdown_new"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate markdown.new provider config."""
        method = config.get("method")
        if method is not None and not isinstance(method, str):
            raise ValueError("markdown.new method must be a string")

        retain_images = config.get("retain_images")
        if retain_images is not None and not isinstance(retain_images, bool):
            raise ValueError("markdown.new retain_images must be a boolean")

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Fetch content from markdown.new."""
        client = get_http_client(timeout=timeout)
        pages = []
        for url in urls:
            pages.append(await _fetch_single(client, url, config, verbose))
        return {"pages": pages}


async def _fetch_single(
    client: httpx.AsyncClient,
    url: str,
    config: dict[str, Any],
    verbose: bool,
) -> dict[str, Any]:
    """Fetch a single page via markdown.new."""
    payload = {"url": url}
    method = config.get("method")
    if isinstance(method, str) and method.strip():
        payload["method"] = method

    retain_images = config.get("retain_images")
    if isinstance(retain_images, bool):
        payload["images"] = "true" if retain_images else "false"

    headers = {"Accept": "text/markdown"}

    if verbose:
        print(f"  [markdown.new] URL: {url}")

    try:
        response = await client.post("https://markdown.new/", json=payload, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "url": url,
            "content": "",
            "error": f"HTTP {exc.response.status_code}: {exc.response.text}",
        }
    except Exception as exc:
        return {"url": url, "content": "", "error": str(exc)}

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
        content = data.get("markdown") or data.get("content") or ""
    else:
        content = response.text

    if not isinstance(content, str) or not content.strip():
        return {"url": url, "content": "", "error": "No content returned"}

    return {"url": url, "content": content}


__all__ = ["MarkdownNewFetchProvider"]
