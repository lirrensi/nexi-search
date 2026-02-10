"""Unit tests for config module."""

from __future__ import annotations

from nexi.config import (
    DEFAULT_CONFIG,
    Config,
    get_compaction_prompt,
    validate_config,
)


def test_config_with_new_fields() -> None:
    """Test Config with new context management fields."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        max_timeout=240,
        max_output_tokens=8192,
        max_context=128000,
        auto_compact_thresh=0.9,
        compact_target_words=5000,
        preserve_last_n_messages=3,
        tokenizer_encoding="cl100k_base",
    )
    assert config.max_context == 128000
    assert config.auto_compact_thresh == 0.9
    assert config.compact_target_words == 5000
    assert config.preserve_last_n_messages == 3
    assert config.tokenizer_encoding == "cl100k_base"


def test_config_defaults() -> None:
    """Test Config with default values for new fields."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        max_timeout=240,
        max_output_tokens=8192,
    )
    assert config.max_context == 128000
    assert config.auto_compact_thresh == 0.9
    assert config.compact_target_words == 5000
    assert config.preserve_last_n_messages == 3
    assert config.tokenizer_encoding == "cl100k_base"


def test_config_to_dict() -> None:
    """Test converting config to dictionary."""
    config = Config(
        base_url="https://api.test.com",
        api_key="test_key",
        model="test_model",
        jina_key="test_jina",
        default_effort="m",
        max_timeout=240,
        max_output_tokens=8192,
        max_context=128000,
    )
    config_dict = config.to_dict()
    assert config_dict["max_context"] == 128000
    assert config_dict["auto_compact_thresh"] == 0.9


def test_config_from_dict() -> None:
    """Test creating config from dictionary."""
    config_dict = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "jina_key": "test_jina",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "max_context": 128000,
        "auto_compact_thresh": 0.9,
        "compact_target_words": 5000,
        "preserve_last_n_messages": 3,
        "tokenizer_encoding": "cl100k_base",
    }
    config = Config.from_dict(config_dict)
    assert config.max_context == 128000
    assert config.auto_compact_thresh == 0.9


def test_validate_config_new_fields_valid() -> None:
    """Test validation of new config fields with valid values."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "max_context": 128000,
        "auto_compact_thresh": 0.9,
        "compact_target_words": 5000,
        "preserve_last_n_messages": 3,
        "tokenizer_encoding": "cl100k_base",
    }
    is_valid, errors = validate_config(config)
    assert is_valid
    assert len(errors) == 0


def test_validate_config_invalid_max_context() -> None:
    """Test validation with invalid max_context."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "max_context": -1,
    }
    is_valid, errors = validate_config(config)
    assert not is_valid
    assert any("max_context" in e for e in errors)


def test_validate_config_invalid_auto_compact_thresh() -> None:
    """Test validation with invalid auto_compact_thresh."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "auto_compact_thresh": 1.5,
    }
    is_valid, errors = validate_config(config)
    assert not is_valid
    assert any("auto_compact_thresh" in e for e in errors)


def test_validate_config_invalid_compact_target_words() -> None:
    """Test validation with invalid compact_target_words."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "compact_target_words": -100,
    }
    is_valid, errors = validate_config(config)
    assert not is_valid
    assert any("compact_target_words" in e for e in errors)


def test_validate_config_invalid_preserve_last_n_messages() -> None:
    """Test validation with invalid preserve_last_n_messages."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "preserve_last_n_messages": -1,
    }
    is_valid, errors = validate_config(config)
    assert not is_valid
    assert any("preserve_last_n_messages" in e for e in errors)


def test_validate_config_invalid_tokenizer_encoding() -> None:
    """Test validation with invalid tokenizer_encoding."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "tokenizer_encoding": "",
    }
    is_valid, errors = validate_config(config)
    assert not is_valid
    assert any("tokenizer_encoding" in e for e in errors)


def test_config_backward_compatibility() -> None:
    """Test that configs without new fields work."""
    config_dict = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "jina_key": "test_jina",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
    }
    config = Config.from_dict(config_dict)
    assert config.max_context == 128000  # Default
    assert config.auto_compact_thresh == 0.9  # Default
    assert config.compact_target_words == 5000  # Default
    assert config.preserve_last_n_messages == 3  # Default
    assert config.tokenizer_encoding == "cl100k_base"  # Default


def test_get_compaction_prompt() -> None:
    """Test compaction prompt generation."""
    prompt = get_compaction_prompt(
        original_query="test query",
        content="test content",
        target_words=1000,
    )
    assert "test query" in prompt
    assert "test content" in prompt
    assert "1000" in prompt
    assert "PRESERVE EXACTLY" in prompt
    assert "MERGE" in prompt


def test_get_compaction_prompt_defaults() -> None:
    """Test compaction prompt with default target_words."""
    prompt = get_compaction_prompt(
        original_query="test query",
        content="test content",
    )
    assert "test query" in prompt
    assert "test content" in prompt
    assert "5000" in prompt  # Default target_words


def test_default_config_has_new_fields() -> None:
    """Test that DEFAULT_CONFIG includes new fields."""
    assert "max_context" in DEFAULT_CONFIG
    assert "auto_compact_thresh" in DEFAULT_CONFIG
    assert "compact_target_words" in DEFAULT_CONFIG
    assert "preserve_last_n_messages" in DEFAULT_CONFIG
    assert "tokenizer_encoding" in DEFAULT_CONFIG


def test_default_config_values() -> None:
    """Test DEFAULT_CONFIG has correct default values."""
    assert DEFAULT_CONFIG["max_context"] == 128000
    assert DEFAULT_CONFIG["auto_compact_thresh"] == 0.9
    assert DEFAULT_CONFIG["compact_target_words"] == 5000
    assert DEFAULT_CONFIG["preserve_last_n_messages"] == 3
    assert DEFAULT_CONFIG["tokenizer_encoding"] == "cl100k_base"


def test_validate_config_optional_fields() -> None:
    """Test that new fields are optional."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
    }
    is_valid, errors = validate_config(config)
    assert is_valid
    assert len(errors) == 0


def test_validate_config_auto_compact_thresh_zero() -> None:
    """Test auto_compact_thresh can be 0.0 (disable compaction)."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "auto_compact_thresh": 0.0,
    }
    is_valid, errors = validate_config(config)
    assert is_valid
    assert len(errors) == 0


def test_validate_config_auto_compact_thresh_one() -> None:
    """Test auto_compact_thresh can be 1.0 (compact at limit)."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "auto_compact_thresh": 1.0,
    }
    is_valid, errors = validate_config(config)
    assert is_valid
    assert len(errors) == 0


def test_validate_config_preserve_last_n_zero() -> None:
    """Test preserve_last_n_messages can be 0 (compact all)."""
    config = {
        "base_url": "https://api.test.com",
        "api_key": "test_key",
        "model": "test_model",
        "default_effort": "m",
        "max_timeout": 240,
        "max_output_tokens": 8192,
        "preserve_last_n_messages": 0,
    }
    is_valid, errors = validate_config(config)
    assert is_valid
    assert len(errors) == 0
