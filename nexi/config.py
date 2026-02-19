"""Configuration management for NEXI."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import questionary

# Constants
CONFIG_DIR = Path.home() / ".local" / "share" / "nexi"
CONFIG_FILE = CONFIG_DIR / "config.json"

EFFORT_LEVELS = {
    "s": {"max_iter": 8, "description": "Quick search"},
    "m": {"max_iter": 16, "description": "Balanced"},
    "l": {"max_iter": 32, "description": "Thorough research"},
}

DEFAULT_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.5-flash-lite",
    "default_effort": "m",
    "max_output_tokens": 8192,
    "max_context": 128000,
    "auto_compact_thresh": 0.9,
    "compact_target_words": 5000,
    "preserve_last_n_messages": 3,
    "tokenizer_encoding": "cl100k_base",
    "jina_timeout": 30,
    "llm_max_retries": 3,
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

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful research assistant. Your goal is to answer the user\'s query thoroughly and accurately.

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

    base_url: str
    api_key: str
    model: str
    jina_key: str
    default_effort: str
    max_output_tokens: int
    time_target: int | None = None
    max_context: int = 128000
    auto_compact_thresh: float = 0.9
    compact_target_words: int = 5000
    preserve_last_n_messages: int = 3
    tokenizer_encoding: str = "cl100k_base"
    jina_timeout: int = 30
    llm_max_retries: int = 3

    # Backend configuration
    search_backend: str = "jina"
    content_fetcher: str = "jina"
    api_keys: dict[str, str] = field(default_factory=dict)  # {"jina": "key", "tavily": "key", ...}

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create config from dictionary."""
        return cls(**data)


def get_config_path() -> Path:
    """Get path to config file."""
    return CONFIG_FILE


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate configuration dictionary.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    required = [
        "base_url",
        "api_key",
        "model",
        "default_effort",
        "max_output_tokens",
    ]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate base_url
    base_url = config.get("base_url", "")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        errors.append("base_url must be a valid HTTP(S) URL")

    # Validate api_key
    api_key = config.get("api_key", "")
    if not isinstance(api_key, str) or not api_key.strip():
        errors.append("api_key must be a non-empty string")

    # Validate model
    model = config.get("model", "")
    if not isinstance(model, str) or not model.strip():
        errors.append("model must be a non-empty string")

    # Validate jina_key (optional but must be string if present)
    jina_key = config.get("jina_key", "")
    if jina_key is not None and not isinstance(jina_key, str):
        errors.append("jina_key must be a string or null")

    # Validate default_effort
    effort = config.get("default_effort", "")
    if effort not in EFFORT_LEVELS:
        errors.append(f"default_effort must be one of: {', '.join(EFFORT_LEVELS.keys())}")

    # Validate max_output_tokens
    tokens = config.get("max_output_tokens", 0)
    if not isinstance(tokens, int) or tokens <= 0:
        errors.append("max_output_tokens must be a positive integer")

    # Validate max_context (optional)
    max_context = config.get("max_context")
    if max_context is not None:
        if not isinstance(max_context, int) or max_context <= 0:
            errors.append("max_context must be a positive integer")

    # Validate auto_compact_thresh (optional)
    auto_compact_thresh = config.get("auto_compact_thresh")
    if auto_compact_thresh is not None:
        if not isinstance(auto_compact_thresh, (int, float)):
            errors.append("auto_compact_thresh must be a number")
        elif not (0.0 <= auto_compact_thresh <= 1.0):
            errors.append("auto_compact_thresh must be between 0.0 and 1.0")

    # Validate compact_target_words (optional)
    compact_target_words = config.get("compact_target_words")
    if compact_target_words is not None:
        if not isinstance(compact_target_words, int) or compact_target_words <= 0:
            errors.append("compact_target_words must be a positive integer")

    # Validate preserve_last_n_messages (optional)
    preserve_last_n = config.get("preserve_last_n_messages")
    if preserve_last_n is not None:
        if not isinstance(preserve_last_n, int) or preserve_last_n < 0:
            errors.append("preserve_last_n_messages must be a non-negative integer")

    # Validate tokenizer_encoding (optional)
    tokenizer_encoding = config.get("tokenizer_encoding")
    if tokenizer_encoding is not None:
        if not isinstance(tokenizer_encoding, str) or not tokenizer_encoding.strip():
            errors.append("tokenizer_encoding must be a non-empty string")

    # Validate jina_timeout (optional)
    jina_timeout = config.get("jina_timeout")
    if jina_timeout is not None:
        if not isinstance(jina_timeout, int) or jina_timeout <= 0:
            errors.append("jina_timeout must be a positive integer")

    # Validate llm_max_retries (optional)
    llm_max_retries = config.get("llm_max_retries")
    if llm_max_retries is not None:
        if not isinstance(llm_max_retries, int) or llm_max_retries <= 0:
            errors.append("llm_max_retries must be a positive integer")

    return len(errors) == 0, errors


def load_config() -> Config:
    """Load configuration from disk.

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)

    is_valid, errors = validate_config(data)
    if not is_valid:
        raise ValueError(f"Invalid config: {'; '.join(errors)}")

    return Config.from_dict(data)


def save_config(config: Config) -> None:
    """Save configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def generate_id() -> str:
    """Generate a short random hex ID."""
    return secrets.token_hex(3)


def run_first_time_setup() -> Config:
    """Run interactive first-time setup.

    Returns:
        Config object with user-provided values
    """
    print("Welcome to NEXI! 🔍")
    print("Let's set up your configuration...\n")

    # LLM Base URL
    base_url = questionary.text(
        "LLM Base URL:",
        default=DEFAULT_CONFIG["base_url"],
    ).ask()

    # API Key
    api_key = questionary.password(
        "API Key:",
        instruction="Your LLM API key (e.g., OpenRouter)",
    ).ask()

    # Model
    model = questionary.text(
        "Model:",
        default=DEFAULT_CONFIG["model"],
    ).ask()

    # Jina API Key
    jina_key = questionary.password(
        "Jina API Key (optional):",
        instruction="Press Enter to skip (rate-limited free tier)",
    ).ask()

    # Default effort level
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

    # Max output tokens
    max_output_tokens = questionary.text(
        "Max output tokens:",
        default=str(DEFAULT_CONFIG["max_output_tokens"]),
    ).ask()

    # Context management settings (optional)
    print("\n--- Context Management (Optional) ---")
    print("These settings control automatic conversation compaction to prevent context overflow.")
    print("Press Enter to use defaults for all context settings.\n")

    # Max context
    max_context = questionary.text(
        "Max context window (tokens):",
        default=str(DEFAULT_CONFIG["max_context"]),
        instruction="Model's context limit (e.g., 128000 for GPT-4)",
    ).ask()

    # Auto compact threshold
    auto_compact_thresh = questionary.text(
        "Auto-compact threshold (0.0-1.0):",
        default=str(DEFAULT_CONFIG["auto_compact_thresh"]),
        instruction="Trigger compaction when context reaches this fraction (e.g., 0.9 = 90%)",
    ).ask()

    # Compact target words
    compact_target_words = questionary.text(
        "Compact target words:",
        default=str(DEFAULT_CONFIG["compact_target_words"]),
        instruction="Target word count for summaries",
    ).ask()

    # Preserve last N messages
    preserve_last_n = questionary.text(
        "Preserve last N messages:",
        default=str(DEFAULT_CONFIG["preserve_last_n_messages"]),
        instruction="Number of recent assistant messages to keep un-compacted",
    ).ask()

    # Tokenizer encoding
    tokenizer_encoding = questionary.text(
        "Tokenizer encoding:",
        default=DEFAULT_CONFIG["tokenizer_encoding"],
        instruction="tiktoken encoding name (e.g., cl100k_base for GPT-4)",
    ).ask()

    # Jina timeout
    jina_timeout = questionary.text(
        "Jina API timeout (seconds):",
        default=str(DEFAULT_CONFIG["jina_timeout"]),
        instruction="Timeout for Jina AI API calls",
    ).ask()

    # LLM max retries
    llm_max_retries = questionary.text(
        "LLM max retries:",
        default=str(DEFAULT_CONFIG["llm_max_retries"]),
        instruction="Maximum retry attempts for LLM API calls",
    ).ask()

    # Create config
    config = Config(
        base_url=base_url or DEFAULT_CONFIG["base_url"],
        api_key=api_key or "",
        model=model or DEFAULT_CONFIG["model"],
        jina_key=jina_key or "",
        default_effort=default_effort or "m",
        max_output_tokens=int(max_output_tokens)
        if max_output_tokens
        else DEFAULT_CONFIG["max_output_tokens"],
        max_context=int(max_context) if max_context else DEFAULT_CONFIG["max_context"],
        auto_compact_thresh=float(auto_compact_thresh)
        if auto_compact_thresh
        else DEFAULT_CONFIG["auto_compact_thresh"],
        compact_target_words=int(compact_target_words)
        if compact_target_words
        else DEFAULT_CONFIG["compact_target_words"],
        preserve_last_n_messages=int(preserve_last_n)
        if preserve_last_n
        else DEFAULT_CONFIG["preserve_last_n_messages"],
        tokenizer_encoding=tokenizer_encoding or DEFAULT_CONFIG["tokenizer_encoding"],
        jina_timeout=int(jina_timeout) if jina_timeout else DEFAULT_CONFIG["jina_timeout"],
        llm_max_retries=int(llm_max_retries)
        if llm_max_retries
        else DEFAULT_CONFIG["llm_max_retries"],
    )

    # Save config
    save_config(config)

    print(f"\n✨ Configuration saved to {CONFIG_FILE}")

    return config


def ensure_config() -> Config:
    """Ensure configuration exists and is valid.

    If config doesn't exist or is invalid, runs first-time setup.

    Returns:
        Valid Config object
    """
    try:
        return load_config()
    except (FileNotFoundError, ValueError):
        return run_first_time_setup()


def get_system_prompt(max_iter: int, effort: str = "m", prompt_template: str | None = None) -> str:
    """Get formatted system prompt.

    Args:
        max_iter: Maximum number of iterations allowed
        effort: Effort level (s/m/l) - affects how thorough the search should be
        prompt_template: Optional custom prompt template. Uses default if None.

    Returns:
        Formatted system prompt with current date, effort level, and max_iter substituted
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
        original_query: The user's original search query
        content: The content to summarize (web_fetch results + assistant messages)
        target_words: Target word count for the summary

    Returns:
        Formatted compaction prompt with original_query, content, and target_words substituted
    """
    return COMPACTION_PROMPT_TEMPLATE.format(
        original_query=original_query,
        content=content,
        target_words=target_words,
    )
