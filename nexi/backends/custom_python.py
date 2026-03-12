"""Custom Python provider loader for NEXI.

Drop a Python file next to the config file and reference it with a
provider type like ``provider-my_backend`` or ``provider-my_backend.py``.
The file can expose ``search``, ``fetch``, and/or ``complete`` functions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

CUSTOM_PROVIDER_PREFIX = "provider-"

_MODULE_CACHE: dict[Path, tuple[int, ModuleType]] = {}
_CLASS_CACHE: dict[tuple[str, str, str], type[Any]] = {}


def is_custom_provider_type(provider_type: str) -> bool:
    """Return True when a provider type uses a local Python file."""
    return provider_type.startswith(CUSTOM_PROVIDER_PREFIX)


def get_custom_provider_path(provider_type: str) -> Path:
    """Resolve the provider file path for a custom provider type."""
    from nexi import config as config_module

    if not is_custom_provider_type(provider_type):
        raise ValueError(f"Unsupported custom provider type: {provider_type}")

    file_name = provider_type[len(CUSTOM_PROVIDER_PREFIX) :].strip()
    if not file_name:
        raise ValueError("Custom provider type must include a Python file name")
    if file_name != Path(file_name).name:
        raise ValueError("Custom provider files must live in the config directory")

    resolved_name = file_name if file_name.endswith(".py") else f"{file_name}.py"
    return config_module.CONFIG_DIR / resolved_name


def build_custom_provider_class(
    capability: str,
    provider_name: str,
    provider_type: str,
) -> type[Any]:
    """Build a custom provider class for the requested capability."""
    cache_key = (capability, provider_name, provider_type)
    cached = _CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    base_class = {
        "search": CustomPythonSearchProvider,
        "fetch": CustomPythonFetchProvider,
        "llm": CustomPythonLLMProvider,
    }.get(capability)
    if base_class is None:
        raise ValueError(f"Unsupported custom provider capability: {capability}")

    class_name = "".join(part.capitalize() for part in f"{provider_name}_{capability}".split("_"))
    custom_class = type(
        class_name,
        (base_class,),
        {
            "name": provider_name,
            "provider_name": provider_name,
            "provider_type": provider_type,
            "capability": capability,
        },
    )
    _CLASS_CACHE[cache_key] = custom_class
    return custom_class


class _CustomPythonProviderBase:
    """Shared behavior for Python-file-backed providers."""

    name = "custom_python"
    provider_name = ""
    provider_type = ""
    capability = ""
    entrypoint_name = ""

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that the provider file exists and exposes the capability."""
        module = _load_custom_module(self.provider_type)
        _get_entrypoint(module, self.entrypoint_name)

        validate_fn = _resolve_validate_config(module)
        if validate_fn is None:
            return

        result = _invoke_callable(
            validate_fn,
            capability=self.capability,
            config=config,
            provider_name=self.provider_name,
            provider_type=self.provider_type,
        )
        if inspect.isawaitable(result):
            raise ValueError("Custom provider validate_config must be synchronous")


class CustomPythonSearchProvider(_CustomPythonProviderBase):
    """Search provider backed by a local Python file."""

    entrypoint_name = "search"

    async def search(
        self,
        queries: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute a custom search provider."""
        module = _load_custom_module(self.provider_type)
        raw_result = await _call_capability(
            module=module,
            capability="search",
            queries=queries,
            config=config,
            timeout=timeout,
            verbose=verbose,
            provider_name=self.provider_name,
            provider_type=self.provider_type,
        )
        return _normalize_search_payload(raw_result, queries)


class CustomPythonFetchProvider(_CustomPythonProviderBase):
    """Fetch provider backed by a local Python file."""

    entrypoint_name = "fetch"

    async def fetch(
        self,
        urls: list[str],
        config: dict[str, Any],
        timeout: float,
        verbose: bool,
    ) -> dict[str, Any]:
        """Execute a custom fetch provider."""
        module = _load_custom_module(self.provider_type)
        raw_result = await _call_capability(
            module=module,
            capability="fetch",
            urls=urls,
            config=config,
            timeout=timeout,
            verbose=verbose,
            provider_name=self.provider_name,
            provider_type=self.provider_type,
        )
        return _normalize_fetch_payload(raw_result, urls)


class CustomPythonLLMProvider(_CustomPythonProviderBase):
    """LLM provider backed by a local Python file."""

    entrypoint_name = "complete"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: dict[str, Any],
        verbose: bool,
        max_tokens: int,
    ) -> Any:
        """Execute a custom LLM provider."""
        module = _load_custom_module(self.provider_type)
        raw_result = await _call_capability(
            module=module,
            capability="complete",
            messages=messages,
            tools=tools,
            config=config,
            verbose=verbose,
            max_tokens=max_tokens,
            provider_name=self.provider_name,
            provider_type=self.provider_type,
        )
        return _normalize_llm_response(raw_result)


def _load_custom_module(provider_type: str) -> ModuleType:
    """Load a custom provider module from disk with mtime-aware caching."""
    provider_path = get_custom_provider_path(provider_type)
    if not provider_path.exists():
        raise ValueError(f"Custom provider file not found: {provider_path}")

    mtime_ns = provider_path.stat().st_mtime_ns
    cached = _MODULE_CACHE.get(provider_path)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]

    module_hash = hashlib.sha1(str(provider_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"nexi_custom_provider_{module_hash}_{mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, provider_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load custom provider file: {provider_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[provider_path] = (mtime_ns, module)
    return module


def _resolve_validate_config(module: ModuleType) -> Any | None:
    """Resolve an optional validate_config callable from the module."""
    validate_fn = getattr(module, "validate_config", None)
    if callable(validate_fn):
        return validate_fn

    provider_obj = getattr(module, "provider", None)
    validate_fn = getattr(provider_obj, "validate_config", None)
    if callable(validate_fn):
        return validate_fn
    return None


def _get_entrypoint(module: ModuleType, capability: str) -> Any:
    """Resolve the callable implementing a capability."""
    entrypoint = getattr(module, capability, None)
    if callable(entrypoint):
        return entrypoint

    provider_obj = getattr(module, "provider", None)
    entrypoint = getattr(provider_obj, capability, None)
    if callable(entrypoint):
        return entrypoint

    provider_path = getattr(module, "__file__", "<custom provider>")
    raise ValueError(f"Custom provider '{provider_path}' does not define a '{capability}' callable")


async def _call_capability(module: ModuleType, capability: str, **kwargs: Any) -> Any:
    """Call a custom capability and await the result when needed."""
    entrypoint = _get_entrypoint(module, capability)
    result = _invoke_callable(entrypoint, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _invoke_callable(func: Any, **kwargs: Any) -> Any:
    """Call a function with only the kwargs it accepts."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return func(**kwargs)

    accepted_kwargs = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return func(**accepted_kwargs)


def _normalize_search_payload(raw_result: Any, _queries: list[str]) -> dict[str, Any]:
    """Normalize custom search output into the canonical payload shape."""
    if not isinstance(raw_result, dict):
        raise ValueError("Custom search provider must return an object with a 'searches' list")

    searches = raw_result.get("searches")
    if not isinstance(searches, list):
        raise ValueError("Custom search provider must return an object with a 'searches' list")

    for item in searches:
        if not isinstance(item, dict):
            raise ValueError("Custom search provider 'searches' entries must be objects")

    return raw_result


def _normalize_fetch_payload(raw_result: Any, _urls: list[str]) -> dict[str, Any]:
    """Normalize custom fetch output into the canonical payload shape."""
    if not isinstance(raw_result, dict):
        raise ValueError("Custom fetch provider must return an object with a 'pages' list")

    pages = raw_result.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Custom fetch provider must return an object with a 'pages' list")

    for item in pages:
        if not isinstance(item, dict):
            raise ValueError("Custom fetch provider 'pages' entries must be objects")

    return raw_result


def _normalize_llm_response(raw_result: Any) -> Any:
    """Normalize custom LLM output into an OpenAI-like response object."""
    if hasattr(raw_result, "choices"):
        return raw_result

    if isinstance(raw_result, str):
        return _build_llm_response(content=raw_result)

    if isinstance(raw_result, dict):
        if "choices" in raw_result:
            return _build_llm_response_from_choice_payload(raw_result)

        content = raw_result.get("content")
        if content is None:
            content = raw_result.get("text")
        if content is None:
            content = raw_result.get("answer")

        tool_calls = raw_result.get("tool_calls")
        if tool_calls is None and "tool_call" in raw_result:
            tool_calls = [raw_result["tool_call"]]

        return _build_llm_response(
            content=content or "",
            tool_calls=tool_calls,
            usage=raw_result.get("usage"),
        )

    raise ValueError("Custom llm provider must return a string, a dict, or an object with choices")


def _build_llm_response_from_choice_payload(payload: dict[str, Any]) -> Any:
    """Build an OpenAI-like response object from a choices payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Custom llm provider choices payload must contain a non-empty list")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Custom llm provider choices entries must be objects")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Custom llm provider choices must include a message object")

    return _build_llm_response(
        content=message.get("content") or "",
        tool_calls=message.get("tool_calls"),
        usage=payload.get("usage"),
    )


def _build_llm_response(
    content: str,
    tool_calls: Any = None,
    usage: Any = None,
) -> Any:
    """Build a minimal OpenAI-like response object."""
    message = SimpleNamespace(
        content=content,
        tool_calls=_build_tool_calls(tool_calls),
    )
    usage_payload = _build_usage(usage)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage_payload,
    )


def _build_tool_calls(tool_calls: Any) -> list[Any] | None:
    """Build tool call objects from a lightweight payload."""
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        raise ValueError("Custom llm provider tool_calls must be a list")

    built_tool_calls = []
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            raise ValueError("Custom llm provider tool_calls entries must be objects")

        function_payload = tool_call.get("function")
        if isinstance(function_payload, dict):
            name = function_payload.get("name")
            arguments = function_payload.get("arguments", {})
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Custom llm provider tool_calls require a function name")

        arguments_json = arguments if isinstance(arguments, str) else json.dumps(arguments)

        built_tool_calls.append(
            SimpleNamespace(
                id=str(tool_call.get("id") or f"custom_tool_call_{index}"),
                function=SimpleNamespace(name=name, arguments=arguments_json),
            )
        )

    return built_tool_calls


def _build_usage(usage: Any) -> Any:
    """Build a usage namespace when token metadata is supplied."""
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise ValueError("Custom llm provider usage must be an object")
    return SimpleNamespace(
        total_tokens=usage.get("total_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


__all__ = [
    "CUSTOM_PROVIDER_PREFIX",
    "CustomPythonFetchProvider",
    "CustomPythonLLMProvider",
    "CustomPythonSearchProvider",
    "build_custom_provider_class",
    "get_custom_provider_path",
    "is_custom_provider_type",
]
