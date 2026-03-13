"""Configuration management for NEXI."""

from __future__ import annotations

import secrets
import tomllib
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nexi.backends.custom_python import get_custom_provider_path, is_custom_provider_type
from nexi.config_template import (
    ACTIVE_FETCH_PROVIDER_DEFAULTS,
    DEFAULT_CHAIN_CONFIG,
    DEFAULT_SCALAR_CONFIG,
    render_config_toml,
    write_default_template,
)

CONFIG_DIR = Path.home() / ".config" / "nexi"
CONFIG_FILE = CONFIG_DIR / "config.toml"

CONFIG_CREATED_MESSAGE = (
    "Created a config template. Fill it in and activate one LLM provider and one "
    "search provider before running NEXI."
)

EFFORT_LEVELS = {
    "s": {"max_iter": 8, "description": "Quick search"},
    "m": {"max_iter": 16, "description": "Balanced"},
    "l": {"max_iter": 32, "description": "Thorough research"},
}
INTERNAL_LLM_MAX_TOKENS = 8192

DEFAULT_CONFIG = {
    "llm_backends": deepcopy(DEFAULT_CHAIN_CONFIG["llm_backends"]),
    "search_backends": deepcopy(DEFAULT_CHAIN_CONFIG["search_backends"]),
    "fetch_backends": deepcopy(DEFAULT_CHAIN_CONFIG["fetch_backends"]),
    "providers": deepcopy(ACTIVE_FETCH_PROVIDER_DEFAULTS),
    "default_effort": DEFAULT_SCALAR_CONFIG["default_effort"],
    "max_context": DEFAULT_SCALAR_CONFIG["max_context"],
    "auto_compact_thresh": DEFAULT_SCALAR_CONFIG["auto_compact_thresh"],
    "compact_target_words": DEFAULT_SCALAR_CONFIG["compact_target_words"],
    "preserve_last_n_messages": DEFAULT_SCALAR_CONFIG["preserve_last_n_messages"],
    "tokenizer_encoding": DEFAULT_SCALAR_CONFIG["tokenizer_encoding"],
    "provider_timeout": DEFAULT_SCALAR_CONFIG["provider_timeout"],
    "search_provider_retries": DEFAULT_SCALAR_CONFIG["search_provider_retries"],
    "fetch_provider_retries": DEFAULT_SCALAR_CONFIG["fetch_provider_retries"],
}

EXTRACTOR_PROMPT_TEMPLATE = """You are a precise information extractor.

TASK: Extract content from this webpage that answers: "{query}"

{instructions}

EXTRACT:
- Key facts, claims, and data points
- Relevant quotes (keep exact wording)
- Technical details, numbers, dates, examples
- Code snippets if present
- Important caveats or contradictions

IGNORE:
- Navigation, ads, author bios, footers
- Generic introductions/conclusions
- Off-topic sections

FORMAT:
Use markdown. Keep useful headers for structure.
Target 400-600 words of the most relevant content.

If page has limited relevance, extract whatever IS relevant - even if brief.
"""

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful research assistant. Your goal is to answer the user's query thoroughly and accurately.

Current date: {current_date}
Effort level: {effort_description}

You have access to these tools:
- web_search: Search the web for information (supports multiple parallel queries)
- web_get: Fetch full content from specific URLs (supports multiple URLs)
- final_answer: Provide the final answer and end the search

Guidelines:
1. Start with web_search to gather initial information
2. Use parallel queries to explore different angles efficiently
3. Use web_get to read full pages for detailed information when needed
4. Each web_get result includes citation markers like [1], [2] for each URL
5. When providing the final answer, use [1], [2], etc. to cite sources
6. Synthesize information from multiple sources
7. Call final_answer when you have a complete answer
8. Do not end final_answer with followup questions

Always respond with a tool call.
"""

CONTINUATION_SYSTEM_PROMPT = """You are a helpful research assistant. The user is asking a follow-up question after a previous search. You have access to the conversation history. Continue the research naturally, using previous context where relevant. Use the web_search, web_get, and final_answer tools as needed."""

CHUNK_SELECTOR_PROMPT = """You are a precise content selector.

TASK: Given the query "{query}", identify which chunks contain relevant information.

INSTRUCTIONS:
- Read each chunk carefully
- Return ONLY the chunk numbers that contain useful information for answering the query
- Be selective - only include chunks with substantive, relevant content
- Ignore chunks that are navigation, headers alone, or off-topic

OUTPUT FORMAT: Just the numbers, comma-separated.
Examples: 3, 7, 12
Examples: 1, 4, 5, 8, 15
Examples: 2

If no chunks are relevant, respond with: none
"""

COMPACTION_PROMPT_TEMPLATE = """Create a dense, factual text summary of research findings.

PRESERVE EXACTLY:
- All numbers, dates, statistics
- Technical terms and proper nouns
- Direct quotes (keep verbatim)
- Specific claims with nuance

MERGE:
- Duplicate information across sources
- Related findings into coherent sections

DO NOT include URLs or links in summary - those are tracked separately.

Output plain text summary only. Target {target_words} words maximum.

Original query: {original_query}

Content to summarize:
{content}
"""


@dataclass
class Config:
    """NEXI configuration."""

    llm_backends: list[str]
    search_backends: list[str]
    fetch_backends: list[str]
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_effort: str = "m"
    max_context: int = 128000
    auto_compact_thresh: float = 0.9
    compact_target_words: int = 5000
    preserve_last_n_messages: int = 3
    tokenizer_encoding: str = "cl100k_base"
    provider_timeout: int = 30
    search_provider_retries: int = 2
    fetch_provider_retries: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create config from dictionary."""
        normalized = _normalize_config_data(data)
        return cls(**normalized)


class ConfigCreatedError(RuntimeError):
    """Raised when NEXI bootstraps a missing config template."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        super().__init__(CONFIG_CREATED_MESSAGE)


def get_config_path() -> Path:
    """Get path to config file."""
    return CONFIG_FILE


def format_config_created_message(config_path: Path, display_path: str | None = None) -> str:
    """Build the user-facing bootstrap message for a missing config."""
    shown_path = display_path or str(config_path)
    return f"Config template created at {shown_path}. {CONFIG_CREATED_MESSAGE}"


def _build_default_config() -> dict[str, Any]:
    """Create a fresh default config mapping."""
    return deepcopy(DEFAULT_CONFIG)


def _normalize_config_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize config data without injecting inactive provider tables."""
    normalized = data.copy()

    for field_name, default_value in DEFAULT_CHAIN_CONFIG.items():
        if field_name not in normalized:
            normalized[field_name] = deepcopy(default_value)

    for field_name, default_value in DEFAULT_SCALAR_CONFIG.items():
        if field_name not in normalized:
            normalized[field_name] = deepcopy(default_value)

    if "providers" not in normalized:
        normalized["providers"] = {}

    return normalized


def _validate_provider_chain(
    config: dict[str, Any],
    field_name: str,
    errors: list[str],
) -> None:
    """Validate a provider chain list against the provider registry mapping."""
    providers = config.get("providers")
    chain = config.get(field_name)

    if not isinstance(chain, list):
        errors.append(f"{field_name} must be a list")
        return

    if not all(isinstance(item, str) and item.strip() for item in chain):
        errors.append(f"{field_name} must contain non-empty provider names")
        return

    if not isinstance(providers, dict):
        return

    for provider_name in chain:
        provider_config = providers.get(provider_name)
        if provider_config is None:
            errors.append(f"{field_name} references unknown provider: {provider_name}")
            continue
        if not isinstance(provider_config, dict):
            errors.append(f"providers.{provider_name} must be an object")
            continue
        provider_type = provider_config.get("type")
        if not isinstance(provider_type, str) or not provider_type.strip():
            errors.append(f"providers.{provider_name}.type must be a non-empty string")
            continue

        if is_custom_provider_type(provider_type):
            try:
                provider_path = get_custom_provider_path(provider_type)
            except ValueError as exc:
                errors.append(f"providers.{provider_name}.type {exc}")
                continue
            if not provider_path.exists():
                errors.append(
                    f"providers.{provider_name}.type references missing custom provider file: {provider_path}"
                )


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate configuration dictionary.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        return False, ["Config must be an object"]

    required = [
        "llm_backends",
        "search_backends",
        "fetch_backends",
        "providers",
        "default_effort",
    ]
    for field_name in required:
        if field_name not in config:
            errors.append(f"Missing required field: {field_name}")

    providers = config.get("providers")
    if providers is not None and not isinstance(providers, dict):
        errors.append("providers must be an object")

    _validate_provider_chain(config, "llm_backends", errors)
    _validate_provider_chain(config, "search_backends", errors)
    _validate_provider_chain(config, "fetch_backends", errors)

    effort = config.get("default_effort", "")
    if effort not in EFFORT_LEVELS:
        errors.append(f"default_effort must be one of: {', '.join(EFFORT_LEVELS.keys())}")

    max_context = config.get("max_context")
    if max_context is not None and (not isinstance(max_context, int) or max_context <= 0):
        errors.append("max_context must be a positive integer")

    auto_compact_thresh = config.get("auto_compact_thresh")
    if auto_compact_thresh is not None:
        if not isinstance(auto_compact_thresh, int | float):
            errors.append("auto_compact_thresh must be a number")
        elif not (0.0 <= auto_compact_thresh <= 1.0):
            errors.append("auto_compact_thresh must be between 0.0 and 1.0")

    compact_target_words = config.get("compact_target_words")
    if compact_target_words is not None and (
        not isinstance(compact_target_words, int) or compact_target_words <= 0
    ):
        errors.append("compact_target_words must be a positive integer")

    preserve_last_n = config.get("preserve_last_n_messages")
    if preserve_last_n is not None and (
        not isinstance(preserve_last_n, int) or preserve_last_n < 0
    ):
        errors.append("preserve_last_n_messages must be a non-negative integer")

    tokenizer_encoding = config.get("tokenizer_encoding")
    if tokenizer_encoding is not None and (
        not isinstance(tokenizer_encoding, str) or not tokenizer_encoding.strip()
    ):
        errors.append("tokenizer_encoding must be a non-empty string")

    provider_timeout = config.get("provider_timeout")
    if provider_timeout is not None and (
        not isinstance(provider_timeout, int) or provider_timeout <= 0
    ):
        errors.append("provider_timeout must be a positive integer")

    search_provider_retries = config.get("search_provider_retries")
    if search_provider_retries is not None and (
        not isinstance(search_provider_retries, int) or search_provider_retries <= 0
    ):
        errors.append("search_provider_retries must be a positive integer")

    fetch_provider_retries = config.get("fetch_provider_retries")
    if fetch_provider_retries is not None and (
        not isinstance(fetch_provider_retries, int) or fetch_provider_retries <= 0
    ):
        errors.append("fetch_provider_retries must be a positive integer")

    return len(errors) == 0, errors


def load_config() -> Config:
    """Load configuration from disk.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    try:
        with open(CONFIG_FILE, "rb") as file_obj:
            raw_data = tomllib.load(file_obj)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(_format_toml_error(exc)) from exc

    if not isinstance(raw_data, dict):
        raise ValueError("Config must be a TOML table")

    data = _normalize_config_data(raw_data)
    is_valid, errors = validate_config(data)
    if not is_valid:
        raise ValueError(f"Invalid config: {'; '.join(errors)}")

    return Config.from_dict(data)


def save_config(config: Config) -> None:
    """Save configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(render_config_toml(config.to_dict()), encoding="utf-8")


def _format_toml_error(error: tomllib.TOMLDecodeError) -> str:
    """Format TOML parse errors with actionable hints for common mistakes."""
    message = f"Invalid TOML: {error}"
    if "Cannot declare" in str(error) and "providers" in str(error):
        return (
            f"{message}. Hint: define each [providers.NAME] table only once, even when the same "
            "provider is used in both search_backends and fetch_backends."
        )
    return message


def generate_id() -> str:
    """Generate a short random hex ID."""
    return secrets.token_hex(3)


def ensure_config() -> Config:
    """Ensure configuration exists and is valid."""
    if not CONFIG_FILE.exists():
        write_default_template(CONFIG_FILE, force=False)
        raise ConfigCreatedError(CONFIG_FILE)
    return load_config()


def get_system_prompt(effort: str, prompt_template: str | None = None) -> str:
    """Get formatted system prompt.

    Args:
        effort: Effort level (s/m/l) - affects how thorough the search should be.
        prompt_template: Optional custom prompt template. Uses default if None.

    Returns:
        Formatted system prompt with current date and effort level substituted.
    """
    template = prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE
    current_date = datetime.now().strftime("%Y-%m-%d")

    effort_descriptions = {
        "s": "small - quick search, be concise",
        "m": "medium - balanced thoroughness",
        "l": "large - exhaustive research, explore deeply",
    }
    effort_description = effort_descriptions.get(effort, effort_descriptions["m"])

    return template.format(
        current_date=current_date,
        effort_description=effort_description,
    )


def get_compaction_prompt(
    original_query: str,
    content: str,
    target_words: int = 5000,
) -> str:
    """Get formatted compaction prompt for summarizing conversation.

    Args:
        original_query: The user's original search query.
        content: The content to summarize (web_fetch results + assistant messages).
        target_words: Target word count for the summary.

    Returns:
        Formatted compaction prompt with original_query, content, and target_words substituted.
    """
    return COMPACTION_PROMPT_TEMPLATE.format(
        original_query=original_query,
        content=content,
        target_words=target_words,
    )


__all__ = [
    "CHUNK_SELECTOR_PROMPT",
    "COMPACTION_PROMPT_TEMPLATE",
    "CONFIG_CREATED_MESSAGE",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "CONTINUATION_SYSTEM_PROMPT",
    "Config",
    "ConfigCreatedError",
    "DEFAULT_CONFIG",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE",
    "EFFORT_LEVELS",
    "EXTRACTOR_PROMPT_TEMPLATE",
    "INTERNAL_LLM_MAX_TOKENS",
    "ensure_config",
    "format_config_created_message",
    "generate_id",
    "get_compaction_prompt",
    "get_config_path",
    "get_system_prompt",
    "load_config",
    "save_config",
    "validate_config",
    "write_default_template",
]
