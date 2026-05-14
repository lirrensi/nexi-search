"""Tests for the central API-key resolution module."""

# FILE: tests/test_backends_api_keys.py
# PURPOSE: Verify API-key normalization, validation, strategy, and round-robin state.
# OWNS: Unit tests for nexi/backends/api_keys.py
# DOCS: agent_chat/plan_multi_key_provider_reliability_2026-05-14.md

from __future__ import annotations

from typing import Any

import pytest

from nexi.backends.api_keys import (
    build_api_key_attempt_configs,
    get_api_key_strategy,
    normalize_api_keys,
    reset_round_robin_state,
    validate_api_keys,
)

# ---------------------------------------------------------------------------
# normalize_api_keys
# ---------------------------------------------------------------------------


def test_normalize_string_key() -> None:
    """Single string key is returned as a one-element list."""
    assert normalize_api_keys({"api_key": "sk-abc"}) == ["sk-abc"]


def test_normalize_list_key() -> None:
    """List of strings is returned as-is."""
    assert normalize_api_keys({"api_key": ["k1", "k2"]}) == ["k1", "k2"]


def test_normalize_empty_list_returns_empty() -> None:
    """Empty list key returns empty list."""
    assert normalize_api_keys({"api_key": []}) == []


def test_normalize_missing_key_returns_empty() -> None:
    """Missing api_key returns empty list."""
    assert normalize_api_keys({"type": "jina"}) == []


def test_normalize_none_key_returns_empty() -> None:
    """Explicit None api_key returns empty list."""
    assert normalize_api_keys({"api_key": None}) == []


def test_normalize_blank_string_returns_empty() -> None:
    """Blank string api_key returns empty list."""
    assert normalize_api_keys({"api_key": "  "}) == []


def test_normalize_filters_blank_items() -> None:
    """List with blank items filters them out."""
    assert normalize_api_keys({"api_key": ["k1", "", "k3"]}) == ["k1", "k3"]


def test_normalize_wrong_type_returns_empty() -> None:
    """Non-str, non-list api_key returns empty list (validation catches this)."""
    assert normalize_api_keys({"api_key": 42}) == []


# ---------------------------------------------------------------------------
# validate_api_keys
# ---------------------------------------------------------------------------


def test_validate_string_key_passes() -> None:
    """Single string api_key passes validation."""
    validate_api_keys({"api_key": "sk-valid"}, "TestProvider")  # no raise


def test_validate_list_key_passes() -> None:
    """List of strings api_key passes validation."""
    validate_api_keys({"api_key": ["k1", "k2"]}, "TestProvider")  # no raise


def test_validate_missing_key_passes() -> None:
    """Missing api_key passes validation (Jina optional-key semantics)."""
    validate_api_keys({"type": "jina"}, "Jina")  # no raise


def test_validate_none_key_passes() -> None:
    """Explicit None api_key passes validation."""
    validate_api_keys({"api_key": None}, "Jina")  # no raise


def test_validate_empty_list_passes() -> None:
    """Empty list api_key passes validation (will be caught by required check later)."""
    validate_api_keys({"api_key": []}, "TestProvider")  # no raise


def test_validate_blank_string_raises() -> None:
    """Blank string api_key raises ValueError."""
    with pytest.raises(ValueError, match="must not be an empty string"):
        validate_api_keys({"api_key": "  "}, "TestProvider")


def test_validate_non_string_list_item_raises() -> None:
    """List with non-string item raises ValueError."""
    with pytest.raises(ValueError, match="must be a string"):
        validate_api_keys({"api_key": ["k1", 42]}, "TestProvider")


def test_validate_blank_list_item_raises() -> None:
    """List with blank item raises ValueError."""
    with pytest.raises(ValueError, match="must not be blank"):
        validate_api_keys({"api_key": ["k1", ""]}, "TestProvider")


def test_validate_wrong_type_raises() -> None:
    """Non-str, non-list type raises ValueError."""
    with pytest.raises(ValueError, match="must be a string or list"):
        validate_api_keys({"api_key": 42}, "TestProvider")


# ---------------------------------------------------------------------------
# get_api_key_strategy
# ---------------------------------------------------------------------------


def test_strategy_defaults_to_fallback() -> None:
    """Missing api_key_strategy defaults to 'fallback'."""
    assert get_api_key_strategy({"api_key": "k1"}) == "fallback"


def test_strategy_fallback() -> None:
    """Explicit 'fallback' is returned as-is."""
    assert get_api_key_strategy({"api_key_strategy": "fallback"}) == "fallback"


def test_strategy_round_robin() -> None:
    """Explicit 'round_robin' is returned as-is."""
    assert get_api_key_strategy({"api_key_strategy": "round_robin"}) == "round_robin"


def test_strategy_unknown_falls_back() -> None:
    """Unknown strategy value falls back to 'fallback'."""
    assert get_api_key_strategy({"api_key_strategy": "unknown"}) == "fallback"


# ---------------------------------------------------------------------------
# build_api_key_attempt_configs  –  fallback strategy
# ---------------------------------------------------------------------------


def test_fallback_returns_copies() -> None:
    """Each per-attempt config is a dict copy with one resolved api_key."""
    original: dict[str, Any] = {"type": "test", "api_key": ["k1", "k2"]}
    configs = build_api_key_attempt_configs(original, "test_provider")

    assert len(configs) == 2
    assert configs[0]["api_key"] == "k1"
    assert configs[1]["api_key"] == "k2"
    # Original is not mutated
    assert original["api_key"] == ["k1", "k2"]


def test_fallback_stable_order() -> None:
    """Fallback strategy keeps key order stable across repeated calls."""
    cfg = {"type": "test", "api_key": ["a", "b", "c"]}
    first = [c["api_key"] for c in build_api_key_attempt_configs(cfg, "test_stable")]
    second = [c["api_key"] for c in build_api_key_attempt_configs(cfg, "test_stable")]
    assert first == ["a", "b", "c"]
    assert second == ["a", "b", "c"]


def test_no_keys_returns_empty() -> None:
    """Provider with no keys returns empty list."""
    assert build_api_key_attempt_configs({"type": "test"}, "no_keys") == []


def test_single_string_key_returns_one_config() -> None:
    """Single string key returns one config copy with that key."""
    configs = build_api_key_attempt_configs({"api_key": "single"}, "single_provider")
    assert len(configs) == 1
    assert configs[0]["api_key"] == "single"


# ---------------------------------------------------------------------------
# build_api_key_attempt_configs  –  round_robin strategy
# ---------------------------------------------------------------------------


def test_round_robin_advances_start() -> None:
    """Round-robin advances the starting key across repeated calls."""
    reset_round_robin_state()
    cfg: dict[str, Any] = {"type": "rr", "api_key": ["a", "b", "c"], "api_key_strategy": "round_robin"}

    call1 = [c["api_key"] for c in build_api_key_attempt_configs(cfg, "rr_test")]
    call2 = [c["api_key"] for c in build_api_key_attempt_configs(cfg, "rr_test")]
    call3 = [c["api_key"] for c in build_api_key_attempt_configs(cfg, "rr_test")]

    assert call1 == ["a", "b", "c"]
    assert call2 == ["b", "c", "a"]
    assert call3 == ["c", "a", "b"]


def test_round_robin_is_per_provider() -> None:
    """Round-robin state is independent per provider name."""
    reset_round_robin_state()
    cfg_a: dict[str, Any] = {"type": "a", "api_key": ["k1", "k2"], "api_key_strategy": "round_robin"}
    cfg_b: dict[str, Any] = {"type": "b", "api_key": ["x1", "x2"], "api_key_strategy": "round_robin"}

    call_a1 = [c["api_key"] for c in build_api_key_attempt_configs(cfg_a, "provider_a")]
    call_b1 = [c["api_key"] for c in build_api_key_attempt_configs(cfg_b, "provider_b")]
    call_a2 = [c["api_key"] for c in build_api_key_attempt_configs(cfg_a, "provider_a")]

    assert call_a1 == ["k1", "k2"]
    assert call_b1 == ["x1", "x2"]
    assert call_a2 == ["k2", "k1"]


def test_reset_round_robin_state() -> None:
    """Reset clears state so next call starts at first key."""
    reset_round_robin_state()
    cfg: dict[str, Any] = {"type": "test", "api_key": ["a", "b"], "api_key_strategy": "round_robin"}

    build_api_key_attempt_configs(cfg, "reset_test")  # advances state
    reset_round_robin_state()
    call = [c["api_key"] for c in build_api_key_attempt_configs(cfg, "reset_test")]
    assert call == ["a", "b"]


# ---------------------------------------------------------------------------
# Metadata never contains raw key values
# ---------------------------------------------------------------------------


def test_metadata_no_key_values_in_orchestrator() -> None:
    """Failure metadata built by _provider_failure never contains an api_key value.

    This tests the contract: the failure metadata shape has no 'api_key' field.
    """
    from nexi.backends.orchestrators import _provider_failure  # type: ignore[attr-defined]

    entry = _provider_failure(
        capability="search",
        provider_name="test",
        provider_type="test_type",
        items=["q1"],
        error="boom",
        stage="execute",
        attempts=2,
        failure_kind="provider_error",
        attempt_key="key_1",
    )
    assert "api_key" not in entry
    assert entry["attempt_key"] == "key_1"
    assert entry["failure_kind"] == "provider_error"
