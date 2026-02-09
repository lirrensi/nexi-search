# Nexi installation checklist

## Install commands
1. Prefer `uv tool install git+https://github.com/lirrensi/nexi-search.git` when the repository ships `uv.lock`.
2. Otherwise, install via `pipx install git+https://github.com/lirrensi/nexi-search.git` or `pip install --user --upgrade git+https://github.com/lirrensi/nexi-search.git` if neither `uv` nor `pipx` are already configured.
3. Verify the CLI with `nexi --version` and list the config path with `nexi --config`.

## Required credentials
- **OpenAI-compatible API**: host, API key, and preferred model (flash-lite or similar). Use `https://openrouter.ai/api/v1` by default and ensure the model supports the runway token limits.
- **Jina AI key**: free API key available from https://jina.ai; required for web search tools within Nexi.
- Optional preferences: `default_effort`, `max_timeout`, `max_output_tokens`, `max_context`, `tokenizer_encoding`, etc.

## Config template
The CLI writes `~/.local/share/nexi/config.json` (Linux/macOS) or `%LOCALAPPDATA%\nexi\config.json` (Windows). Agents can write this file directly once they have credentials:

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-...",
  "model": "google/gemini-2.5-flash-lite-preview-09-2025",
  "jina_key": "jina_...",
  "default_effort": "m",
  "max_timeout": 300,
  "max_output_tokens": 8000,
  "max_context": 128000,
  "auto_compact_thresh": 0.9,
  "compact_target_words": 5000,
  "preserve_last_n_messages": 3,
  "tokenizer_encoding": "cl100k_base"
}
```

## Validation
1. Run `nexi --plain ""` or `nexi "hello"` to let the tool authenticate both endpoints.
2. Inspect `~/.local/share/nexi/history.jsonl` (or `%LOCALAPPDATA%\nexi\history.jsonl`) for failure details whenever searches fail.
3. Document downstream commands using effort flags and piping so the bash worker can replicate them.
