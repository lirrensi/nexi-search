"""Tests for direct fetch output helpers."""

from __future__ import annotations

from pathlib import Path

from nexi.direct_fetch import post_process_direct_fetch_payload


class FakeEncoding:
    """Character-based fake encoding for deterministic truncation tests."""

    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


def test_direct_fetch_truncation_spills_full_content_to_tempfile(monkeypatch) -> None:
    """Oversized pages are truncated and spilled to a temp file."""
    monkeypatch.setattr("nexi.direct_fetch.get_encoding", lambda encoding_name: FakeEncoding())

    payload = {
        "pages": [
            {
                "url": "https://example.com",
                "content": "abcdefghijklmnopqrstuvwxyz",
            }
        ],
        "provider_failures": [],
    }

    processed = post_process_direct_fetch_payload(
        payload,
        max_tokens=8,
        encoding_name="fake",
    )

    page = processed["pages"][0]
    full_content_path = Path(page["full_content_path"])

    assert page["content"] == "abcdefgh"
    assert full_content_path.is_absolute()
    assert full_content_path.read_text(encoding="utf-8") == "abcdefghijklmnopqrstuvwxyz"


def test_direct_fetch_helper_leaves_small_pages_unchanged(monkeypatch) -> None:
    """Pages under the cap are not spilled to disk."""
    monkeypatch.setattr("nexi.direct_fetch.get_encoding", lambda encoding_name: FakeEncoding())

    payload = {"pages": [{"url": "https://example.com", "content": "abc"}]}

    processed = post_process_direct_fetch_payload(payload, max_tokens=8, encoding_name="fake")

    assert processed["pages"][0]["content"] == "abc"
    assert "full_content_path" not in processed["pages"][0]
