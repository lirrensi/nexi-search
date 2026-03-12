"""Unit tests for config module."""

from __future__ import annotations

import json
from pathlib import Path

from nexi import config as config_module
from nexi.config import DEFAULT_CONFIG, Config, get_compaction_prompt, validate_config


def _build_config() -> Config:
    """Create a canonical config fixture."""
    return Config(
        llm_backends=["openai_default"],
        search_backends=["jina"],
        fetch_backends=["jina"],
        providers={
            "openai_default": {
                "type": "openai_compatible",
                "base_url": "https://api.test.com/v1",
                "api_key": "test_key",
                "model": "test_model",
            },
            "jina": {
                "type": "jina",
                "api_key": "test_jina",
            },
        },
        default_effort="m",
        max_output_tokens=8192,
        time_target=600,
        max_context=128000,
        auto_compact_thresh=0.9,
        compact_target_words=5000,
        preserve_last_n_messages=3,
        tokenizer_encoding="cl100k_base",
        provider_timeout=30,
        search_provider_retries=2,
        fetch_provider_retries=2,
    )


def test_config_to_dict_round_trip() -> None:
    """Config round-trips through dict serialization."""
    config = _build_config()

    config_dict = config.to_dict()
    restored = Config.from_dict(config_dict)

    assert restored == config


def test_validate_config_new_shape_valid() -> None:
    """Validation accepts the canonical provider-based config shape."""
    is_valid, errors = validate_config(_build_config().to_dict())

    assert is_valid
    assert errors == []


def test_validate_config_missing_provider_reference() -> None:
    """Validation rejects backend chains that reference missing providers."""
    config = _build_config().to_dict()
    config["search_backends"] = ["missing"]

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("search_backends references unknown provider" in error for error in errors)


def test_validate_config_requires_provider_type() -> None:
    """Validation rejects provider configs without a type field."""
    config = _build_config().to_dict()
    config["providers"]["jina"] = {"api_key": "test_jina"}

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("providers.jina.type" in error for error in errors)


def test_load_config_migrates_legacy_shape(tmp_path: Path, monkeypatch) -> None:
    """Loading a legacy config migrates it to provider chains."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_url": "https://legacy.example/v1",
                "api_key": "legacy-key",
                "model": "legacy-model",
                "jina_key": "legacy-jina",
                "default_effort": "m",
                "max_output_tokens": 2048,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_path)

    loaded = config_module.load_config()

    assert loaded.llm_backends == ["openai_default"]
    assert loaded.search_backends == ["jina"]
    assert loaded.fetch_backends == ["jina"]
    assert loaded.providers["openai_default"]["type"] == "openai_compatible"
    assert loaded.providers["openai_default"]["base_url"] == "https://legacy.example/v1"
    assert loaded.providers["openai_default"]["api_key"] == "legacy-key"
    assert loaded.providers["openai_default"]["model"] == "legacy-model"
    assert loaded.providers["jina"]["type"] == "jina"
    assert loaded.providers["jina"]["api_key"] == "legacy-jina"


def test_save_config_writes_canonical_shape(tmp_path: Path, monkeypatch) -> None:
    """Saving config writes only the canonical provider-based shape."""
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_path)

    config_module.save_config(_build_config())
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert "base_url" not in saved
    assert saved["llm_backends"] == ["openai_default"]
    assert saved["providers"]["openai_default"]["type"] == "openai_compatible"


def test_get_compaction_prompt() -> None:
    """Compaction prompt includes key interpolated values."""
    prompt = get_compaction_prompt(
        original_query="test query",
        content="test content",
        target_words=1000,
    )

    assert "test query" in prompt
    assert "test content" in prompt
    assert "1000" in prompt
    assert "PRESERVE EXACTLY" in prompt


def test_default_config_has_provider_chains() -> None:
    """Default config exposes provider chains and provider registry."""
    assert DEFAULT_CONFIG["llm_backends"] == ["openai_default"]
    assert DEFAULT_CONFIG["search_backends"] == ["jina"]
    assert DEFAULT_CONFIG["fetch_backends"] == ["jina"]
    assert DEFAULT_CONFIG["providers"]["openai_default"]["type"] == "openai_compatible"
    assert DEFAULT_CONFIG["providers"]["jina"]["type"] == "jina"
