"""Configuration management for NEXI."""

from __future__ import annotations

import json
import secrets
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import questionary

CONFIG_DIR = Path.home() / ".local" / "share" / "nexi"
CONFIG_FILE = CONFIG_DIR / "config.json"

EFFORT_LEVELS = {
    "s": {"max_iter": 8, "description": "Quick search"},
    "m": {"max_iter": 16, "description": "Balanced"},
    "l": {"max_iter": 32, "description": "Thorough research"},
}

DEFAULT_CONFIG = {
    "llm_backends": ["openai_default"],
    "search_backends": ["jina"],
    "fetch_backends": ["crawl4ai_local", "markdown_new", "jina"],
    "providers": {
        "openai_default": {
            "type": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "model": "google/gemini-2.5-flash-lite",
        },
        "markdown_new": {
            "type": "markdown_new",
            "method": "auto",
            "retain_images": False,
        },
        "crawl4ai_local": {
            "type": "crawl4ai",
            "headless": True,
        },
        "jina": {
            "type": "jina",
            "api_key": "",
        },
        "tavily": {
            "type": "tavily",
            "api_key": "",
            "search_depth": "basic",
            "topic": "general",
            "max_results": 5,
        },
        "exa": {
            "type": "exa",
            "api_key": "",
            "num_results": 5,
            "text": True,
        },
        "firecrawl": {
            "type": "firecrawl",
            "api_key": "",
            "only_main_content": True,
            "formats": ["markdown"],
            "limit": 5,
        },
        "linkup": {
            "type": "linkup",
            "api_key": "",
            "depth": "standard",
            "output_type": "searchResults",
        },
        "brave": {
            "type": "brave",
            "api_key": "",
            "count": 5,
        },
        "serpapi": {
            "type": "serpapi",
            "api_key": "",
            "engine": "google",
        },
        "serper": {
            "type": "serper",
            "api_key": "",
            "num": 5,
        },
        "perplexity_search": {
            "type": "perplexity_search",
            "api_key": "",
            "max_results": 5,
        },
    },
    "default_effort": "m",
    "max_output_tokens": 8192,
    "time_target": None,
    "max_context": 128000,
    "auto_compact_thresh": 0.9,
    "compact_target_words": 5000,
    "preserve_last_n_messages": 3,
    "tokenizer_encoding": "cl100k_base",
    "provider_timeout": 30,
    "search_provider_retries": 2,
    "fetch_provider_retries": 2,
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
Maximum iterations available: {max_iter}

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
8. You have up to {max_iter} iterations - use them wisely
9. If you reach max iterations, provide your best answer with available information
10. Do not end final_answer with followup questions

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
    max_output_tokens: int = 8192
    time_target: int | None = None
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


def get_config_path() -> Path:
    """Get path to config file."""
    return CONFIG_FILE


def _build_default_config() -> dict[str, Any]:
    """Create a fresh default config mapping."""
    return deepcopy(DEFAULT_CONFIG)


def _is_legacy_config(data: dict[str, Any]) -> bool:
    """Return True when config uses the legacy top-level provider fields."""
    legacy_fields = {"base_url", "api_key", "model", "jina_key"}
    return any(field in data for field in legacy_fields)


def _migrate_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a legacy config dictionary to the canonical provider shape."""
    migrated = _build_default_config()
    migrated["providers"]["openai_default"] = {
        "type": "openai_compatible",
        "base_url": data.get(
            "base_url",
            DEFAULT_CONFIG["providers"]["openai_default"]["base_url"],
        ),
        "api_key": data.get("api_key", ""),
        "model": data.get(
            "model",
            DEFAULT_CONFIG["providers"]["openai_default"]["model"],
        ),
    }
    migrated["providers"]["jina"] = {
        "type": "jina",
        "api_key": data.get("jina_key", ""),
    }
    migrated["default_effort"] = data.get("default_effort", DEFAULT_CONFIG["default_effort"])
    migrated["max_output_tokens"] = data.get(
        "max_output_tokens",
        DEFAULT_CONFIG["max_output_tokens"],
    )
    migrated["time_target"] = data.get("time_target", DEFAULT_CONFIG["time_target"])
    migrated["max_context"] = data.get("max_context", DEFAULT_CONFIG["max_context"])
    migrated["auto_compact_thresh"] = data.get(
        "auto_compact_thresh",
        DEFAULT_CONFIG["auto_compact_thresh"],
    )
    migrated["compact_target_words"] = data.get(
        "compact_target_words",
        DEFAULT_CONFIG["compact_target_words"],
    )
    migrated["preserve_last_n_messages"] = data.get(
        "preserve_last_n_messages",
        DEFAULT_CONFIG["preserve_last_n_messages"],
    )
    migrated["tokenizer_encoding"] = data.get(
        "tokenizer_encoding",
        DEFAULT_CONFIG["tokenizer_encoding"],
    )
    migrated["provider_timeout"] = data.get(
        "provider_timeout",
        data.get("jina_timeout", DEFAULT_CONFIG["provider_timeout"]),
    )
    migrated["search_provider_retries"] = data.get(
        "search_provider_retries",
        DEFAULT_CONFIG["search_provider_retries"],
    )
    migrated["fetch_provider_retries"] = data.get(
        "fetch_provider_retries",
        DEFAULT_CONFIG["fetch_provider_retries"],
    )
    return migrated


def _normalize_config_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize config data to the canonical shape with defaults."""
    normalized = _migrate_legacy_config(data) if _is_legacy_config(data) else data.copy()
    merged = _build_default_config()
    merged.update({key: value for key, value in normalized.items() if key != "providers"})
    if "providers" in normalized:
        merged["providers"] = normalized["providers"]
    return merged


def _validate_provider_chain(
    config: dict[str, Any],
    field_name: str,
    errors: list[str],
) -> None:
    """Validate a provider chain list against the provider registry mapping."""
    providers = config.get("providers")
    chain = config.get(field_name)

    if not isinstance(chain, list) or not chain:
        errors.append(f"{field_name} must be a non-empty list")
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
        "max_output_tokens",
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

    tokens = config.get("max_output_tokens", 0)
    if not isinstance(tokens, int) or tokens <= 0:
        errors.append("max_output_tokens must be a positive integer")

    time_target = config.get("time_target")
    if time_target is not None and (not isinstance(time_target, int) or time_target <= 0):
        errors.append("time_target must be a positive integer or null")

    max_context = config.get("max_context")
    if max_context is not None and (not isinstance(max_context, int) or max_context <= 0):
        errors.append("max_context must be a positive integer")

    auto_compact_thresh = config.get("auto_compact_thresh")
    if auto_compact_thresh is not None:
        if not isinstance(auto_compact_thresh, (int, float)):
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

    with open(CONFIG_FILE, encoding="utf-8") as file_obj:
        raw_data = json.load(file_obj)

    data = _normalize_config_data(raw_data)
    is_valid, errors = validate_config(data)
    if not is_valid:
        raise ValueError(f"Invalid config: {'; '.join(errors)}")

    return Config.from_dict(data)


def save_config(config: Config) -> None:
    """Save configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(config.to_dict(), file_obj, indent=2, ensure_ascii=False)


def generate_id() -> str:
    """Generate a short random hex ID."""
    return secrets.token_hex(3)


def _build_default_runtime_config(
    base_url: str,
    api_key: str,
    model: str,
    jina_key: str,
) -> dict[str, Any]:
    """Create the canonical config payload from first-time setup answers."""
    config = _build_default_config()
    config["providers"]["openai_default"] = {
        "type": "openai_compatible",
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }
    config["providers"]["jina"] = {
        "type": "jina",
        "api_key": jina_key,
    }
    return config


def run_first_time_setup() -> Config:
    """Run interactive first-time setup.

    Returns:
        Config object with user-provided values.
    """
    print("Welcome to NEXI! 🔍")
    print("Let's set up your configuration...\n")

    default_llm = DEFAULT_CONFIG["providers"]["openai_default"]

    base_url = questionary.text(
        "LLM Base URL:",
        default=default_llm["base_url"],
    ).ask()
    api_key = questionary.password(
        "API Key:",
        instruction="Your LLM API key (e.g., OpenRouter)",
    ).ask()
    model = questionary.text(
        "Model:",
        default=default_llm["model"],
    ).ask()
    jina_key = questionary.password(
        "Jina API Key (optional):",
        instruction="Press Enter to skip (rate-limited free tier)",
    ).ask()

    effort_choices = [
        questionary.Choice("s - Quick (8 iterations)", value="s"),
        questionary.Choice("m - Balanced (16 iterations) [default]", value="m"),
        questionary.Choice("l - Thorough (32 iterations)", value="l"),
    ]
    default_effort = questionary.select(
        "Default effort level:",
        choices=effort_choices,
        default="m",
    ).ask()

    max_output_tokens = questionary.text(
        "Max output tokens:",
        default=str(DEFAULT_CONFIG["max_output_tokens"]),
    ).ask()

    print("\n--- Context Management (Optional) ---")
    print("These settings control automatic conversation compaction to prevent context overflow.")
    print("Press Enter to use defaults for all context settings.\n")

    max_context = questionary.text(
        "Max context window (tokens):",
        default=str(DEFAULT_CONFIG["max_context"]),
        instruction="Model's context limit (e.g., 128000 for GPT-4)",
    ).ask()
    auto_compact_thresh = questionary.text(
        "Auto-compact threshold (0.0-1.0):",
        default=str(DEFAULT_CONFIG["auto_compact_thresh"]),
        instruction="Trigger compaction when context reaches this fraction (e.g., 0.9 = 90%)",
    ).ask()
    compact_target_words = questionary.text(
        "Compact target words:",
        default=str(DEFAULT_CONFIG["compact_target_words"]),
        instruction="Target word count for summaries",
    ).ask()
    preserve_last_n = questionary.text(
        "Preserve last N messages:",
        default=str(DEFAULT_CONFIG["preserve_last_n_messages"]),
        instruction="Number of recent assistant messages to keep un-compacted",
    ).ask()
    tokenizer_encoding = questionary.text(
        "Tokenizer encoding:",
        default=DEFAULT_CONFIG["tokenizer_encoding"],
        instruction="tiktoken encoding name (e.g., cl100k_base for GPT-4)",
    ).ask()
    provider_timeout = questionary.text(
        "Provider timeout (seconds):",
        default=str(DEFAULT_CONFIG["provider_timeout"]),
        instruction="Default timeout for provider API calls",
    ).ask()
    search_provider_retries = questionary.text(
        "Search provider retries:",
        default=str(DEFAULT_CONFIG["search_provider_retries"]),
        instruction="Retry attempts per search provider before failover",
    ).ask()
    fetch_provider_retries = questionary.text(
        "Fetch provider retries:",
        default=str(DEFAULT_CONFIG["fetch_provider_retries"]),
        instruction="Retry attempts per fetch provider before failover",
    ).ask()

    config_data = _build_default_runtime_config(
        base_url=base_url or default_llm["base_url"],
        api_key=api_key or "",
        model=model or default_llm["model"],
        jina_key=jina_key or "",
    )
    config_data["default_effort"] = default_effort or DEFAULT_CONFIG["default_effort"]
    config_data["max_output_tokens"] = (
        int(max_output_tokens) if max_output_tokens else DEFAULT_CONFIG["max_output_tokens"]
    )
    config_data["max_context"] = int(max_context) if max_context else DEFAULT_CONFIG["max_context"]
    config_data["auto_compact_thresh"] = (
        float(auto_compact_thresh) if auto_compact_thresh else DEFAULT_CONFIG["auto_compact_thresh"]
    )
    config_data["compact_target_words"] = (
        int(compact_target_words)
        if compact_target_words
        else DEFAULT_CONFIG["compact_target_words"]
    )
    config_data["preserve_last_n_messages"] = (
        int(preserve_last_n) if preserve_last_n else DEFAULT_CONFIG["preserve_last_n_messages"]
    )
    config_data["tokenizer_encoding"] = tokenizer_encoding or DEFAULT_CONFIG["tokenizer_encoding"]
    config_data["provider_timeout"] = (
        int(provider_timeout) if provider_timeout else DEFAULT_CONFIG["provider_timeout"]
    )
    config_data["search_provider_retries"] = (
        int(search_provider_retries)
        if search_provider_retries
        else DEFAULT_CONFIG["search_provider_retries"]
    )
    config_data["fetch_provider_retries"] = (
        int(fetch_provider_retries)
        if fetch_provider_retries
        else DEFAULT_CONFIG["fetch_provider_retries"]
    )

    config = Config.from_dict(config_data)
    save_config(config)

    print(f"\n✨ Configuration saved to {CONFIG_FILE}")
    return config


def ensure_config() -> Config:
    """Ensure configuration exists and is valid.

    If config doesn't exist or is invalid, runs first-time setup.

    Returns:
        Valid Config object.
    """
    try:
        return load_config()
    except (FileNotFoundError, ValueError):
        return run_first_time_setup()


def get_system_prompt(max_iter: int, effort: str = "m", prompt_template: str | None = None) -> str:
    """Get formatted system prompt.

    Args:
        max_iter: Maximum number of iterations allowed.
        effort: Effort level (s/m/l) - affects how thorough the search should be.
        prompt_template: Optional custom prompt template. Uses default if None.

    Returns:
        Formatted system prompt with current date, effort level, and max_iter substituted.
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
        max_iter=max_iter,
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
    "CONFIG_DIR",
    "CONFIG_FILE",
    "CONTINUATION_SYSTEM_PROMPT",
    "Config",
    "DEFAULT_CONFIG",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE",
    "EFFORT_LEVELS",
    "EXTRACTOR_PROMPT_TEMPLATE",
    "ensure_config",
    "generate_id",
    "get_compaction_prompt",
    "get_config_path",
    "get_system_prompt",
    "load_config",
    "run_first_time_setup",
    "save_config",
    "validate_config",
]
