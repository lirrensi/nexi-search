"""Configuration management for NEXI."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
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
    "max_timeout": 240,
    "max_output_tokens": 8192,
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
4. Synthesize information from multiple sources
5. Call final_answer when you have a complete answer
6. You have up to {max_iter} iterations - use them wisely
7. If you reach max iterations, provide your best answer with available information

Always respond with a tool call.
"""


@dataclass
class Config:
    """NEXI configuration."""

    base_url: str
    api_key: str
    model: str
    jina_key: str
    default_effort: str
    max_timeout: int
    max_output_tokens: int

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
        "max_timeout",
        "max_output_tokens",
    ]
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate base_url
    base_url = config.get("base_url", "")
    if not isinstance(base_url, str) or not base_url.startswith(
        ("http://", "https://")
    ):
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
        errors.append(
            f"default_effort must be one of: {', '.join(EFFORT_LEVELS.keys())}"
        )

    # Validate max_timeout
    timeout = config.get("max_timeout", 0)
    if not isinstance(timeout, int) or timeout <= 0:
        errors.append("max_timeout must be a positive integer")

    # Validate max_output_tokens
    tokens = config.get("max_output_tokens", 0)
    if not isinstance(tokens, int) or tokens <= 0:
        errors.append("max_output_tokens must be a positive integer")

    return len(errors) == 0, errors


def load_config() -> Config:
    """Load configuration from disk.

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
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

    # Max timeout
    max_timeout = questionary.text(
        "Max timeout (seconds):",
        default=str(DEFAULT_CONFIG["max_timeout"]),
    ).ask()

    # Max output tokens
    max_output_tokens = questionary.text(
        "Max output tokens:",
        default=str(DEFAULT_CONFIG["max_output_tokens"]),
    ).ask()

    # Create config
    config = Config(
        base_url=base_url or DEFAULT_CONFIG["base_url"],
        api_key=api_key or "",
        model=model or DEFAULT_CONFIG["model"],
        jina_key=jina_key or "",
        default_effort=default_effort or "m",
        max_timeout=int(max_timeout) if max_timeout else DEFAULT_CONFIG["max_timeout"],
        max_output_tokens=int(max_output_tokens)
        if max_output_tokens
        else DEFAULT_CONFIG["max_output_tokens"],
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


def get_system_prompt(
    max_iter: int, effort: str = "m", prompt_template: str | None = None
) -> str:
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
