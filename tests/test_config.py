"""Unit tests for config bootstrap and TOML serialization."""

# FILE: tests/test_config.py
# PURPOSE: Verify config defaults, bootstrap templates, and config validation behavior.
# OWNS: Template rendering expectations, config loading, and readiness checks.
# DOCS: docs/product.md, docs/arch.md, docs/provider-matrix.md, agent_chat/plan_crawl4ai_opt_in_2026-04-24.md

from __future__ import annotations

from pathlib import Path

import pytest

from nexi import config as config_module
from nexi.backends.custom_python import get_custom_provider_path
from nexi.config import (
    DEFAULT_CONFIG,
    Config,
    ConfigCreatedError,
    get_compaction_prompt,
    load_config,
    validate_config,
)
from nexi.config_doctor import build_doctor_report, build_doctor_summary, build_doctor_warnings
from nexi.config_template import render_config_toml, write_default_template


def _build_config() -> Config:
    """Create a canonical config fixture."""
    return Config(
        llm_backends=["openrouter"],
        search_backends=["jina"],
        fetch_backends=["snitchmd", "special_trafilatura", "special_playwright", "markdown_new"],
        providers={
            "openrouter": {
                "type": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
                "model": "test-model",
            },
            "jina": {
                "type": "jina",
                "api_key": "test-jina",
            },
            "special_trafilatura": {
                "type": "special_trafilatura",
            },
            "special_playwright": {
                "type": "special_playwright",
            },
            "snitchmd": {
                "type": "snitchmd",
                "mode": "precision",
            },
            "markdown_new": {
                "type": "markdown_new",
                "method": "auto",
                "retain_images": False,
            },
        },
        default_effort="m",
        max_context=128000,
        auto_compact_thresh=0.9,
        compact_target_words=5000,
        preserve_last_n_messages=3,
        tokenizer_encoding="cl100k_base",
        provider_timeout=30,
        direct_fetch_max_tokens=8000,
        search_provider_retries=2,
        fetch_provider_retries=2,
    )


def _patch_config_paths(monkeypatch, tmp_path: Path) -> Path:
    """Point config module paths at a temporary directory."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_path)
    return config_path


def test_config_to_dict_round_trip() -> None:
    """Config round-trips through dict serialization."""
    config = _build_config()
    assert Config.from_dict(config.to_dict()) == config


def test_write_default_template_writes_config_toml(tmp_path: Path) -> None:
    """The default bootstrap template is written to config.toml."""
    config_path = tmp_path / "config.toml"

    created = write_default_template(config_path)

    assert created is True
    assert config_path.exists()
    text = config_path.read_text(encoding="utf-8")
    assert "llm_backends = []" in text
    assert 'fetch_backends = ["snitchmd", "special_trafilatura", "special_playwright", "markdown_new"]' in text
    assert "[providers.markdown_new]" in text
    assert "[providers.snitchmd]" in text
    assert "[providers.special_trafilatura]" in text
    assert "[providers.special_playwright]" in text
    assert "# [providers.crawl4ai_local]" in text
    assert "# [providers.openrouter]" in text
    assert text.count("# [providers.jina]") == 1
    assert text.count("# [providers.searxng]") == 1
    assert "Define each [providers.<name>] table only once" in text
    assert '# - add "jina" to search_backends' in text
    assert '# - add "searxng" to search_backends' in text
    assert '# - add "snitchmd" to fetch_backends' in text
    assert '# - add "crawl4ai_local" to fetch_backends' in text
    assert '# - add "jina" to fetch_backends' in text
    assert '# search_backends = ["jina"]' not in text


def test_render_config_toml_reuses_one_provider_block_for_shared_provider() -> None:
    """Shared providers are defined once even when used in multiple chains."""
    text = render_config_toml(
        {
            "llm_backends": ["openrouter"],
            "search_backends": ["jina"],
            "fetch_backends": ["jina"],
            "providers": {
                "openrouter": {
                    "type": "openai_compatible",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key": "test-key",
                    "model": "test-model",
                },
                "jina": {
                    "type": "jina",
                    "api_key": "test-jina",
                },
            },
        }
    )

    assert text.count("[providers.jina]") == 1
    assert 'search_backends = ["jina"]' in text
    assert 'fetch_backends = ["jina"]' in text


def test_build_doctor_report_accepts_searxng_search_provider() -> None:
    """Doctor readiness passes with a configured SearXNG search provider."""
    config = Config(
        llm_backends=["openrouter"],
        search_backends=["searxng"],
        fetch_backends=["special_trafilatura", "special_playwright", "markdown_new"],
        providers={
            "openrouter": {
                "type": "openai_compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
                "model": "test-model",
            },
            "searxng": {
                "type": "searxng",
                "base_url": "https://search.example.org",
            },
            "special_trafilatura": {
                "type": "special_trafilatura",
            },
            "special_playwright": {
                "type": "special_playwright",
            },
            "markdown_new": {
                "type": "markdown_new",
                "method": "auto",
                "retain_images": False,
            },
        },
        default_effort="m",
    )

    report = build_doctor_report(config)

    assert report["nexi"] == []
    assert report["nexi-search"] == []


def test_build_doctor_summary_and_warnings_include_chain_health() -> None:
    """Doctor summary surfaces active chains and warns on single-provider chains."""
    config = _build_config()

    summary = build_doctor_summary(config)
    warnings = build_doctor_warnings(config)

    assert "Configured providers: 6" in summary
    assert "llm_backends (1): openrouter" in summary
    assert "search_backends (1): jina" in summary
    assert (
        "fetch_backends (4): snitchmd, special_trafilatura, special_playwright, markdown_new"
        in summary
    )
    assert "LLM models: openrouter=test-model" in summary
    assert any("Only one provider in llm_backends" in item for item in warnings)
    assert any("Only one provider in search_backends" in item for item in warnings)
    assert not any("fetch_backends" in item for item in warnings)


def test_load_config_parses_toml(tmp_path: Path, monkeypatch) -> None:
    """Loading config reads TOML and fills optional defaults."""
    config_path = _patch_config_paths(monkeypatch, tmp_path)
    config_path.write_text(
        """
llm_backends = ["openrouter"]
search_backends = ["jina"]
fetch_backends = ["special_trafilatura", "special_playwright", "markdown_new"]

default_effort = "m"
max_context = 128000
auto_compact_thresh = 0.9
compact_target_words = 5000
preserve_last_n_messages = 3
tokenizer_encoding = "cl100k_base"
provider_timeout = 30
direct_fetch_max_tokens = 8000
search_provider_retries = 2
fetch_provider_retries = 2

[providers.openrouter]
type = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
api_key = "test-key"
model = "test-model"

[providers.jina]
type = "jina"
api_key = "test-jina"

[providers.special_trafilatura]
type = "special_trafilatura"

[providers.special_playwright]
type = "special_playwright"

[providers.markdown_new]
type = "markdown_new"
method = "auto"
retain_images = false
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config()

    assert loaded.llm_backends == ["openrouter"]
    assert loaded.search_backends == ["jina"]
    assert loaded.fetch_backends == ["special_trafilatura", "special_playwright", "markdown_new"]
    assert loaded.providers["openrouter"]["type"] == "openai_compatible"
    assert loaded.provider_timeout == 30
    assert loaded.direct_fetch_max_tokens == 8000


def test_load_config_adds_hint_for_duplicate_provider_tables(tmp_path: Path, monkeypatch) -> None:
    """Duplicate provider tables get a clearer parse error message."""
    config_path = _patch_config_paths(monkeypatch, tmp_path)
    config_path.write_text(
        """
search_backends = ["jina"]
fetch_backends = ["jina"]
llm_backends = []
default_effort = "m"

[providers.jina]
type = "jina"
api_key = "a"

[providers.jina]
type = "jina"
api_key = "b"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_config()

    assert "define each [providers.NAME] table only once" in str(excinfo.value)


def test_validate_config_allows_empty_llm_and_search_chains() -> None:
    """Template-style configs are valid before readiness checks."""
    is_valid, errors = validate_config(DEFAULT_CONFIG)
    assert is_valid
    assert errors == []
    assert DEFAULT_CONFIG["fetch_backends"] == [
        "snitchmd",
        "special_trafilatura",
        "special_playwright",
        "markdown_new",
    ]


def test_ensure_config_writes_template_and_raises(tmp_path: Path, monkeypatch) -> None:
    """Missing config bootstraps the template and stops the command."""
    config_path = _patch_config_paths(monkeypatch, tmp_path)

    with pytest.raises(ConfigCreatedError) as excinfo:
        config_module.ensure_config()

    assert excinfo.value.config_path == config_path
    assert config_path.exists()
    template_text = config_path.read_text(encoding="utf-8")
    assert "# [providers.crawl4ai_local]" in template_text
    assert (
        'fetch_backends = ["snitchmd", "special_trafilatura", "special_playwright", "markdown_new"]'
        in template_text
    )


def test_save_config_writes_commented_examples(tmp_path: Path, monkeypatch) -> None:
    """Saving config preserves inactive provider examples as comments."""
    config_path = _patch_config_paths(monkeypatch, tmp_path)

    config_module.save_config(_build_config())
    text = config_path.read_text(encoding="utf-8")

    assert "[providers.openrouter]" in text
    assert "# [providers.openai]" in text
    assert "# [providers.searxng]" in text
    assert "# [providers.custom_search]" in text


def test_custom_provider_file_lookup_uses_new_config_dir(tmp_path: Path, monkeypatch) -> None:
    """Custom provider files resolve under the new config directory."""
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    assert get_custom_provider_path("provider-custom_api") == tmp_path / "custom_api.py"


def test_validate_config_accepts_existing_custom_provider(tmp_path: Path, monkeypatch) -> None:
    """provider-* types validate when the backing file exists."""
    (tmp_path / "custom_api.py").write_text(
        "from __future__ import annotations\n", encoding="utf-8"
    )
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    config = _build_config().to_dict()
    config["llm_backends"] = []
    config["search_backends"] = ["custom_search"]
    config["fetch_backends"] = []
    config["providers"] = {"custom_search": {"type": "provider-custom_api"}}

    is_valid, errors = validate_config(config)

    assert is_valid
    assert errors == []


def test_validate_config_rejects_missing_custom_provider_file(tmp_path: Path, monkeypatch) -> None:
    """provider-* types fail validation when the file is missing."""
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    config = _build_config().to_dict()
    config["llm_backends"] = []
    config["search_backends"] = ["custom_search"]
    config["fetch_backends"] = []
    config["providers"] = {"custom_search": {"type": "provider-custom_api"}}

    is_valid, errors = validate_config(config)

    assert not is_valid
    assert any("missing custom provider file" in error for error in errors)


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
