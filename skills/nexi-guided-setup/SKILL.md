---
name: nexi-guided-setup
description: Install, configure, and verify the Nexi CLI so an agent can hand web searches to a local bash tool instead of running them itself. Use this skill whenever you need to prepare a host for Nexi, guide a user through the first-run credential setup, or script a reproducible configuration for downstream agents.
---

# Nexi Guided Setup

## Overview
Provide a concise path to install Nexi, surface the required credentials, and confirm the CLI is ready for downstream searches. This keeps agents focused on guidance—telling users or automation what to run—without the agent issuing the search itself.

## When to use this skill
- A user or automation needs a reproducible Nexi install so they can hand off browserless search queries via the CLI.
- The host requires fresh credentials, custom OpenAI-compatible endpoints, or non-interactive configuration edits before Nexi can run.
- You are designing an agent flow where every search is run by a bash-capable worker rather than by the agent, so you must document the install/config steps clearly.

## Install & prepare
1. Confirm the host has Python 3.12+, `uv`, `pipx`, or pip available.
2. If the repository already provides `uv.lock`, prefer `uv tool install git+https://github.com/lirrensi/nexi-search.git`. Otherwise, use `pipx install git+https://github.com/lirrensi/nexi-search.git` or `pip install --user --upgrade git+https://github.com/lirrensi/nexi-search.git`.
3. After installation, run `nexi --version` and `nexi --config` to confirm the executable is on `PATH` and the config path is reachable.

## Credentials & config
1. On first run `nexi`, the CLI walks through the interactive config wizard. Explain to the user that they must supply:
   - An OpenAI-compatible endpoint (e.g., `https://openrouter.ai/api/v1`) plus its API key and preferred model (flash-lite models are recommended).
   - A Jina AI API key (free at https://jina.ai).
   - Optional preferences like `default_effort`, `max_timeout`, and `tokenizer_encoding`.
2. The resulting config JSON lives at `~/.local/share/nexi/config.json` (Linux/macOS) or `%LOCALAPPDATA%\nexi\config.json` (Windows). Agents can also write this file directly if they control the host and already possess the credentials. See the reference for a complete template you can `cat` into place.
3. Use `nexi --edit-config` to re-open an editor or `nexi --config` to reveal the path so automation can edit it directly. Non-interactive automation should only edit the JSON when the file is stable (e.g., after the initial install) and validate JSON syntax with `python -m json.tool` before running Nexi.

## Running & validating
1. Once config exists, rerun `nexi --version` then `nexi --plain ""` or `nexi "test query"` to verify the tool can reach both OpenAI-compatible and Jina endpoints.
2. If the search command fails, inspect `~/.local/share/nexi/history.jsonl` (or `%LOCALAPPDATA%\nexi\history.jsonl`) for errors, then adjust credentials or timeouts accordingly.
3. Document to the user how to run searches: e.g., `nexi "your question"`, `nexi -e l "deep research"`, or `echo "your question" | nexi`.

## Hand-off guidance
Always instruct the user or bash worker to run Nexi with whichever flags suit the context (`--verbose`, `--plain`, effort levels). Avoid having the agent itself run the query—this skill exists so the agent can act as an installer/configurer while the bash tool performs the actual search.

## Agent-first usage idea
If an agent is prompted with something like "check the Nexi skill and use that for search," respond by summarizing the install/config expectations from this skill, confirm Nexi is already installed or reinitialize it via the documented commands, then hand the actual question to the bash worker. This keeps the agent in its guidance role while still satisfying the user's request through the Nexi CLI.

## References
- Installation & config cheat sheet: [references/installation.md](references/installation.md)
