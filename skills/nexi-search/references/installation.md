# NEXI Installation & Configuration

## Install

```bash
uv tool install git+https://github.com/lirrensi/nexi-search.git
```

Verify:
```bash
nexi --version
```

## Configuration

Config location:
- Linux/macOS: `~/.local/share/nexi/config.json`
- Windows: `%LOCALAPPDATA%\nexi\config.json`

Create the directory and config file:

```bash
mkdir -p ~/.local/share/nexi
```

```json
{
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-or-...",
  "model": "google/gemini-2.5-flash-lite",
  "jina_key": "jina_...",
  "default_effort": "m",
  "max_output_tokens": 8192,
  "time_target": 600
}
```

## Required Credentials

1. **OpenAI-compatible API**: `base_url`, `api_key`, `model`
   - OpenRouter is default: `https://openrouter.ai/api/v1`
   - Any OpenAI-compatible endpoint works

2. **Jina AI API Key**: `jina_key`
   - Free at https://jina.ai
   - Required for web search functionality

## Test

```bash
nexi --plain "hello world"
```

## Common Issues

| Error | Fix |
|-------|-----|
| Command not found | Add uv tool dir to PATH |
| 401 Unauthorized | Check `api_key` in config |
| No search results | Check `jina_key` is valid |
