"""API-key resolution and multi-key strategy for NEXI provider backends."""

# FILE: nexi/backends/api_keys.py
# PURPOSE: Normalize, validate, and build per-attempt provider configs with resolved API keys.
# OWNS: API-key normalization, validation, round-robin state, and per-attempt config construction.
# EXPORTS: normalize_api_keys, validate_api_keys, get_api_key_strategy, build_api_key_attempt_configs, reset_round_robin_state
# DOCS: docs/arch.md; agent_chat/plan_multi_key_provider_reliability_2026-05-14.md

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any

# Process-local round-robin state: provider_name -> next_start_index
_round_robin_state: dict[str, int] = {}
_round_robin_lock = threading.Lock()


def normalize_api_keys(config: dict[str, Any]) -> list[str]:
    """Extract and normalize API keys from a provider config.

    Returns a list of non-empty key strings.
    Returns an empty list when api_key is absent or explicitly empty.

    This is a pure helper: it does not mutate *config*.
    """
    api_key = config.get("api_key")
    if api_key is None:
        return []
    if isinstance(api_key, str):
        stripped = api_key.strip()
        return [stripped] if stripped else []
    if isinstance(api_key, list):
        return [k for k in api_key if isinstance(k, str) and k.strip()]
    return []


def validate_api_keys(config: dict[str, Any], provider_name: str = "Provider") -> None:
    """Validate that api_key (if present) is a string or a list of strings.

    Raises ``ValueError`` with provider-specific messages on invalid types,
    empty strings, blank items, or non-string list items.

    *api_key* is optional: omitting it or setting it to ``None`` is valid
    (preserves Jina's optional-key semantics).  An explicitly empty list is
    also valid.
    """
    api_key = config.get("api_key")
    if api_key is None:
        return
    if isinstance(api_key, str):
        if not api_key.strip():
            raise ValueError(f"{provider_name} api_key must not be an empty string")
        return
    if isinstance(api_key, list):
        for i, k in enumerate(api_key):
            if not isinstance(k, str):
                raise ValueError(
                    f"{provider_name} api_key list item {i} must be a string, "
                    f"got {type(k).__name__}"
                )
            if not k.strip():
                raise ValueError(
                    f"{provider_name} api_key list item {i} must not be blank"
                )
        return
    raise ValueError(
        f"{provider_name} api_key must be a string or list of strings, "
        f"got {type(api_key).__name__}"
    )


def get_api_key_strategy(config: dict[str, Any]) -> str:
    """Read ``api_key_strategy`` from *config*, defaulting to ``"fallback"``.

    Returns exactly ``"fallback"`` or ``"round_robin"``.
    """
    strategy = config.get("api_key_strategy", "fallback")
    if strategy not in ("fallback", "round_robin"):
        strategy = "fallback"
    return strategy


def build_api_key_attempt_configs(
    provider_config: dict[str, Any],
    provider_name: str = "",
) -> list[dict[str, Any]]:
    """Build ordered per-attempt provider-config copies with resolved API keys.

    Each returned dict is a **deep copy** of *provider_config* with a single
    resolved non-empty string in ``api_key``.

    Returns an empty list when the provider has no key requirement (i.e.
    no ``api_key`` field or all keys are empty).

    Does **not** mutate *provider_config*.

    Strategy behaviour
    ------------------
    * ``"fallback"`` (default) – keys appear in their original order every call.
    * ``"round_robin"`` – the starting key advances by one position on each
      call for the same *provider_name* within the current process.
    """
    keys = normalize_api_keys(provider_config)
    if not keys:
        return []

    strategy = get_api_key_strategy(provider_config)

    if strategy == "round_robin":
        with _round_robin_lock:
            start = _round_robin_state.get(provider_name, 0) % len(keys)
            ordered = keys[start:] + keys[:start]
            _round_robin_state[provider_name] = (start + 1) % len(keys)
    else:
        ordered = list(keys)

    return [{**deepcopy(provider_config), "api_key": key} for key in ordered]


def reset_round_robin_state() -> None:
    """Clear process-local round-robin state.  Intended for testing."""
    with _round_robin_lock:
        _round_robin_state.clear()


__all__ = [
    "build_api_key_attempt_configs",
    "get_api_key_strategy",
    "normalize_api_keys",
    "reset_round_robin_state",
    "validate_api_keys",
]
