"""OpenAI-compatible provider adapters for NEXI."""

from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI

from nexi.backends.api_keys import normalize_api_keys, validate_api_keys


class OpenAICompatibleLLMProvider:
    """OpenAI-compatible LLM provider."""

    name = "openai_compatible"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate OpenAI-compatible config."""
        base_url = config.get("base_url", "")
        model = config.get("model", "")

        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("OpenAI-compatible base_url must be a valid HTTP(S) URL")
        validate_api_keys(config, "OpenAI-compatible")
        if not normalize_api_keys(config):
            raise ValueError("OpenAI-compatible api_key must be a non-empty string or list")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI-compatible model must be a non-empty string")

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        verbose: bool,
        max_tokens: int,
    ) -> Any:
        """Execute a chat completion request."""
        self.validate_config(config)

        client = AsyncOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
        request: dict[str, Any] = {
            "model": config["model"],
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        if verbose:
            print(f"[LLM] Calling {config['model']} via {self.name}")
            print(f"[LLM] Messages count: {len(messages)}")

        response = await client.chat.completions.create(**request)

        if verbose:
            print("[LLM] Response received")
            if response.usage:
                print(
                    "[LLM] Tokens: "
                    f"{response.usage.total_tokens} "
                    f"(prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                )

        return response


__all__ = ["OpenAICompatibleLLMProvider"]
